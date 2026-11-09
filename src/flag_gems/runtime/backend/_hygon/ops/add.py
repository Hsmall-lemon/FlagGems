# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Flat vector loads for contiguous add, retaining generic broadcasting."""

import torch
import triton
import triton.language as tl

from flag_gems.ops.add import add as _generic_add
from flag_gems.runtime import torch_device_fn


@triton.jit
def _transposed_add(
    A,
    B,
    O,
    M: tl.constexpr,
    N: tl.constexpr,
    AS: tl.constexpr,
    AM: tl.constexpr,
    AN: tl.constexpr,
    BS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    ALPHA: tl.constexpr,
    TILE: tl.constexpr,
):
    batch = tl.program_id(2)
    m = tl.program_id(0) * TILE + tl.arange(0, TILE)
    n = tl.program_id(1) * TILE + tl.arange(0, TILE)
    valid = (m[:, None] < M) & (n[None, :] < N)
    a = tl.load(A + batch * AS + m[:, None] * AM + n[None, :] * AN, valid, other=0)
    b = tl.load(B + batch * BS + m[:, None] * BM + n[None, :] * BN, valid, other=0)
    dtype: tl.constexpr = O.dtype.element_ty
    compute: tl.constexpr = (
        tl.float32 if (dtype == tl.float16 or dtype == tl.bfloat16) else dtype
    )
    tl.store(
        O + batch * M * N + m[:, None] * N + n[None, :],
        a.to(compute) + b.to(compute) * ALPHA,
        valid,
    )


@triton.jit
def _strided_add(
    A,
    B,
    O,
    SIZE: tl.constexpr,
    SHAPE: tl.constexpr,
    AS: tl.constexpr,
    BS: tl.constexpr,
    ALPHA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    cur = i
    ao = tl.full((BLOCK,), 0, tl.int64)
    bo = tl.full((BLOCK,), 0, tl.int64)
    for d in tl.static_range(len(SHAPE) - 1, -1, -1):
        coord = cur % SHAPE[d]
        ao += coord * AS[d]
        bo += coord * BS[d]
        cur = cur // SHAPE[d]
    dtype: tl.constexpr = O.dtype.element_ty
    compute: tl.constexpr = (
        tl.float32 if (dtype == tl.float16 or dtype == tl.bfloat16) else dtype
    )
    a = tl.load(A + ao, i < SIZE, other=0).to(compute)
    b = tl.load(B + bo, i < SIZE, other=0).to(compute)
    tl.store(O + i, a + b * ALPHA, i < SIZE)


@triton.jit
def _flat_add(
    A,
    B,
    O,
    SIZE: tl.constexpr,
    ALPHA: tl.constexpr,
    SCALAR: tl.constexpr,
    PERIOD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < SIZE
    dtype: tl.constexpr = O.dtype.element_ty
    compute: tl.constexpr = (
        tl.float32 if (dtype == tl.float16 or dtype == tl.bfloat16) else dtype
    )
    x = tl.load(A + i, valid, other=0).to(compute)
    if SCALAR:
        y = tl.full((), B, compute)
    else:
        if PERIOD == SIZE:
            j = i
        else:
            j = i % PERIOD
        y = tl.load(B + j, valid, other=0).to(compute)
    # Keep multiplication and addition separate, matching ATen add's alpha rule.
    value = x + y * ALPHA
    tl.store(O + i, value, valid)


def _launch_strided(a, b, alpha):
    shape = torch.broadcast_shapes(a.shape, b.shape)
    av, bv = a.broadcast_to(shape), b.broadcast_to(shape)
    size = av.numel()
    out = torch.empty(shape, dtype=a.dtype, device=a.device)
    if size:
        with torch_device_fn.device(a.device):
            tile = (
                len(shape) >= 2
                and a.shape == b.shape
                and a.stride(-1) == b.stride(-2) == 1
                and a.stride(-2) == b.stride(-1)
                and all(
                    a.stride(d) == a.stride(d + 1) * a.shape[d + 1]
                    and b.stride(d) == b.stride(d + 1) * b.shape[d + 1]
                    for d in range(a.ndim - 3)
                )
            )
            if tile:
                m, n = shape[-2:]
                batch = size // (m * n)
                _transposed_add[(triton.cdiv(m, 32), triton.cdiv(n, 32), batch)](
                    a,
                    b,
                    out,
                    m,
                    n,
                    a.stride(-3) if a.ndim >= 3 else 0,
                    *a.stride()[-2:],
                    b.stride(-3) if b.ndim >= 3 else 0,
                    *b.stride()[-2:],
                    alpha,
                    32,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
            else:
                block = 64 if size <= 65536 else 512
                _strided_add[(triton.cdiv(size, block),)](
                    a,
                    b,
                    out,
                    size,
                    tuple(shape),
                    tuple(av.stride()),
                    tuple(bv.stride()),
                    alpha,
                    block,
                    num_warps=1 if size <= 65536 else 2,
                    enable_fp_fusion=False,
                )
    return out


def add(a, b, *, alpha=1):
    if not isinstance(alpha, (int, float)):
        return _generic_add(a, b, alpha=alpha)
    if not isinstance(a, torch.Tensor) or a.is_complex() or a.dtype == torch.bool:
        return _generic_add(a, b, alpha=alpha)
    if not a.dtype.is_floating_point and not isinstance(alpha, int):
        return _generic_add(a, b, alpha=alpha)
    scalar = not isinstance(b, torch.Tensor)
    if scalar:
        if (
            not a.is_contiguous()
            or not isinstance(b, (int, float))
            or torch.result_type(a, b) != a.dtype
        ):
            return _generic_add(a, b, alpha=alpha)
        period = 1
    else:
        if b.device != a.device or b.dtype != a.dtype:
            return _generic_add(a, b, alpha=alpha)
        if not a.is_contiguous() or not b.is_contiguous():
            return _launch_strided(a, b, alpha)
        if b.shape == a.shape:
            period = a.numel()
        elif (b.numel() == 1 and b.ndim <= a.ndim) or (
            b.ndim == 1 and a.ndim and b.shape[0] == a.shape[-1]
        ):
            period = b.numel()
        else:
            return _launch_strided(a, b, alpha)
    if not isinstance(alpha, (int, float)):
        return _generic_add(a, b, alpha=alpha)
    out = torch.empty(a.shape, device=a.device, dtype=a.dtype)
    if a.numel():
        if a.dtype == torch.float32 and 24576 <= a.numel() <= 32768:
            block, warps = 256, 1
        elif a.numel() <= 32768:
            block, warps = 64, 1
        elif a.numel() <= 65536 or (a.dtype == torch.float64 and a.numel() >= 1048576):
            block, warps = 256, 1
        elif a.numel() <= 131072:
            block, warps = 512, 2
        else:
            block, warps = 1024, 4
        with torch_device_fn.device(a.device):
            _flat_add[(triton.cdiv(a.numel(), block),)](
                a,
                b,
                out,
                a.numel(),
                alpha,
                scalar,
                period,
                block,
                num_warps=warps,
                enable_fp_fusion=False,
            )
    return out
