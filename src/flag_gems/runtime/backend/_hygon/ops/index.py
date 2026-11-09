# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Compact boolean prefix masks directly, without materializing coordinates."""

import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.index import index as _generic_index
from flag_gems.runtime import torch_device_fn


@triton.jit
def _gather_tiles(
    X,
    INDEX,
    OUT,
    COUNT: tl.constexpr,
    ROWS: tl.constexpr,
    SUFFIX: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDE: tl.constexpr,
    TILE_ROWS: tl.constexpr,
    TILE_COLS: tl.constexpr,
):
    i = tl.program_id(0) * TILE_ROWS + tl.arange(0, TILE_ROWS)
    row = tl.load(INDEX + i, i < COUNT, other=0)
    tl.device_assert(
        (i >= COUNT) | ((row >= -ROWS) & (row < ROWS)), "index out of bounds"
    )
    row = tl.where(row < 0, row + ROWS, row)
    j = tl.arange(0, TILE_COLS)
    cur = j
    offset = tl.full((TILE_COLS,), 0, tl.int64)
    for d in tl.static_range(len(SHAPE) - 1, 0, -1):
        offset += cur % SHAPE[d] * STRIDE[d]
        cur = cur // SHAPE[d]
    valid = (i[:, None] < COUNT) & (j[None, :] < SUFFIX)
    value = tl.load(
        X + row[:, None] * STRIDE[0] + offset[None, :],
        valid & (row[:, None] >= 0) & (row[:, None] < ROWS),
        other=0,
    )
    tl.store(OUT + i[:, None] * SUFFIX + j[None, :], value, valid)


@triton.jit
def _gather_rows(
    X,
    INDEX,
    OUT,
    SIZE: tl.constexpr,
    ROWS: tl.constexpr,
    SUFFIX: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < SIZE
    row = tl.load(INDEX + i // SUFFIX, valid, other=0)
    tl.device_assert((~valid) | ((row >= -ROWS) & (row < ROWS)), "index out of bounds")
    row = tl.where(row < 0, row + ROWS, row)
    cur = i % SUFFIX
    offset = row * STRIDE[0]
    for d in tl.static_range(len(SHAPE) - 1, 0, -1):
        offset += cur % SHAPE[d] * STRIDE[d]
        cur = cur // SHAPE[d]
    value = tl.load(X + offset, valid & (row >= 0) & (row < ROWS), other=0)
    tl.store(OUT + i, value, valid)


@triton.jit
def _mask_counts(MASK, COUNTS, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    take = tl.load(MASK + i, i < SIZE, other=0).to(tl.int32)
    tl.store(COUNTS + tl.program_id(0), tl.sum(take, 0))


@triton.jit
def _scan_counts(COUNTS, PREFIX, TOTAL, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    count = tl.load(COUNTS + i, i < N, other=0)
    prefix = tl.cumsum(count, 0)
    tl.store(PREFIX + i, prefix - count, i < N)
    tl.store(TOTAL, tl.sum(count, 0))


@triton.jit
def _compact_mask(
    X,
    MASK,
    PREFIX,
    OUT,
    SIZE: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDE: tl.constexpr,
    SUFFIX_SHAPE: tl.constexpr,
    SUFFIX_STRIDE: tl.constexpr,
    SUFFIX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    take = tl.load(MASK + i, i < SIZE, other=0).to(tl.int32)
    rank = tl.cumsum(take, 0) - 1 + tl.load(PREFIX + tl.program_id(0))
    cur = i
    offset = tl.full((BLOCK,), 0, tl.int64)
    for d in tl.static_range(len(SHAPE) - 1, -1, -1):
        offset += cur % SHAPE[d] * STRIDE[d]
        cur = cur // SHAPE[d]
    if SUFFIX == 1:
        value = tl.load(X + offset, (i < SIZE) & (take != 0), other=0)
        tl.store(OUT + rank, value, (i < SIZE) & (take != 0))
    else:
        j = tl.arange(0, triton.next_power_of_2(SUFFIX))
        cur_j = j
        offset_j = tl.full((triton.next_power_of_2(SUFFIX),), 0, tl.int64)
        for d in tl.static_range(len(SUFFIX_SHAPE) - 1, -1, -1):
            offset_j += cur_j % SUFFIX_SHAPE[d] * SUFFIX_STRIDE[d]
            cur_j = cur_j // SUFFIX_SHAPE[d]
        valid = (i[:, None] < SIZE) & (take[:, None] != 0) & (j[None, :] < SUFFIX)
        value = tl.load(X + offset[:, None] + offset_j[None, :], valid, other=0)
        tl.store(OUT + rank[:, None] * SUFFIX + j[None, :], value, valid)


def index(inp, indices):
    if len(indices) != 1 or indices[0] is None:
        return _generic_index(inp, indices)
    mask = indices[0]
    if (
        mask.dtype in (torch.int32, torch.int64)
        and mask.ndim == 1
        and mask.is_contiguous()
        and mask.device == inp.device
        and inp.ndim
        and inp.shape[0]
        and not inp.is_complex()
    ):
        suffix = math.prod(inp.shape[1:])
        if 0 < suffix <= 128:
            out = torch.empty(
                (mask.numel(), *inp.shape[1:]), dtype=inp.dtype, device=inp.device
            )
            if out.numel():
                with torch_device_fn.device(inp.device):
                    if suffix > 16:
                        tile_rows = (
                            1
                            if mask.numel() <= 300
                            else 2 if mask.numel() <= 2048 else 16
                        )
                        _gather_tiles[(triton.cdiv(mask.numel(), tile_rows),)](
                            inp,
                            mask,
                            out,
                            mask.numel(),
                            inp.shape[0],
                            suffix,
                            tuple(inp.shape),
                            tuple(inp.stride()),
                            tile_rows,
                            triton.next_power_of_2(suffix),
                            num_warps=1 if tile_rows <= 2 else 4,
                            debug=True,
                        )
                        return out
                    block = 256 if out.numel() <= 16384 else 1024
                    _gather_rows[(triton.cdiv(out.numel(), block),)](
                        inp,
                        mask,
                        out,
                        out.numel(),
                        inp.shape[0],
                        suffix,
                        tuple(inp.shape),
                        tuple(inp.stride()),
                        block,
                        num_warps=1 if out.numel() <= 16384 else 4,
                        debug=True,
                    )
            return out
    if (
        mask.dtype != torch.bool
        or mask.ndim == 0
        or not mask.is_contiguous()
        or mask.device != inp.device
        or inp.is_complex()
        or tuple(mask.shape) != tuple(inp.shape[: mask.ndim])
    ):
        return _generic_index(inp, indices)
    suffix_shape = tuple(inp.shape[mask.ndim :])
    suffix = math.prod(suffix_shape)
    if suffix > 16 or mask.numel() > 2**30:
        return _generic_index(inp, indices)
    size = mask.numel()
    if size == 0:
        return torch.empty((0, *suffix_shape), dtype=inp.dtype, device=inp.device)
    block = 1024
    n = triton.cdiv(size, block)
    counts = torch.empty((n,), dtype=torch.int32, device=inp.device)
    prefix = torch.empty_like(counts)
    total = torch.empty((), dtype=torch.int32, device=inp.device)
    with torch_device_fn.device(inp.device):
        _mask_counts[(n,)](mask, counts, size, block, num_warps=4)
        _scan_counts[(1,)](
            counts, prefix, total, n, triton.next_power_of_2(n), num_warps=4
        )
        # Only the output length crosses to the host; compaction stays on device.
        selected = total.item()
        out = torch.empty((selected, *suffix_shape), dtype=inp.dtype, device=inp.device)
        if selected and suffix:
            _compact_mask[(n,)](
                inp,
                mask,
                prefix,
                out,
                size,
                tuple(mask.shape),
                tuple(inp.stride()[: mask.ndim]),
                suffix_shape,
                tuple(inp.stride()[mask.ndim :]),
                suffix,
                block,
                num_warps=4,
            )
    return out
