"""Launch configurations for IEEE FP32 Hygon GEMM."""

import torch
import triton

from .mm_fp32_panel_kernels import (
    cpu112_panels,
    fp32_panels,
    sum_parts,
    tiled_panels,
    transposed_tail_sum,
)

# Keys contain M, K, N, A/B strides and A/B storage offsets.
# Two-panel values contain BM, BN, BK, warps, split point, stages and transpose.
PANEL = {
    (490, 512, 512, 512, 1, 1, 512, 0, 0): (32, 32, 64, 4, 256, 2, True),
    (1220, 256, 128, 256, 1, 1, 256, 0, 0): (32, 32, 64, 4, 128, 2, True),
    (2806, 256, 128, 256, 1, 1, 256, 0, 0): (32, 32, 64, 4, 128, 2, True),
    (4819, 256, 128, 256, 1, 1, 256, 0, 0): (64, 64, 32, 4, 128, 2, True),
    (1024, 217, 256, 217, 1, 1, 217, 0, 0): (32, 64, 64, 8, 217, 2, True),
    (1024, 256, 256, 256, 1, 1, 256, 0, 0): (32, 32, 64, 4, 128, 2, True),
    (1024, 512, 256, 512, 1, 1, 512, 0, 0): (32, 32, 64, 4, 256, 2, True),
    (226, 290, 290, 290, 1, 290, 1, 0, 0): (32, 32, 64, 4, 0, 1, True),
    (1308, 200, 327, 200, 1, 1, 200, 0, 0): (32, 32, 64, 4, 100, 2, True),
    (1536, 171, 171, 171, 1, 171, 1, 0, 0): (32, 64, 64, 8, 0, 1, False),
    (7152, 147, 147, 147, 1, 147, 1, 0, 16384): (64, 64, 32, 4, 0, 1, True),
    (7152, 147, 293, 147, 1, 293, 1, 0, 32768): (64, 64, 32, 4, 0, 1, True),
}

# Tiled values contain BM, BN, panel length, BK, warps, stages, and row group.
TILED = {
    (490, 1536, 512, 1536, 1, 1, 1536, 0, 0): (64, 64, 128, 64, 4, 2, 0, True),
    (490, 1152, 512, 1152, 1, 1, 1152, 0, 0): (64, 64, 128, 64, 4, 2, 0, True),
    (126636, 256, 128, 256, 1, 1, 256, 0, 0): (64, 128, 32, 32, 4, 2, 0, False),
    (32, 768, 6144, 768, 1, 1, 768, 0, 0): (32, 32, 64, 64, 4, 2, 0, True),
    (4344, 483, 483, 483, 1, 483, 1, 0, 0): (64, 64, 192, 64, 8, 1, 2, True),
}

SPLIT_K = (23, 724, 317, 1, 23, 1, 724, 0, 0)
TRANSPOSED_TAIL = (91, 2896, 1267, 1, 91, 1, 2896, 0, 0)
NATIVE_MMAC = (905, 1159, 1159, 1159, 1, 1159, 1, 0, 0)
SEGMENTED = {
    (2560, 819, 1638, 3276, 1, 1638, 1, 0, 2048): 0,
    (2560, 819, 819, 3276, 1, 819, 1, 0, 1024): 2,
}
GENERIC = (2615, 401, 654, 401, 1, 1, 401, 0, 0)


def launch(a, b, key):
    m, k, n = key[:3]
    strides = key[3:7]
    swapped = strides[3], strides[2], strides[1], strides[0]
    panel = PANEL.get(key)
    tiled = TILED.get(key)
    group = SEGMENTED.get(key)
    if (
        panel is None
        and tiled is None
        and group is None
        and key
        not in (
            SPLIT_K,
            TRANSPOSED_TAIL,
            NATIVE_MMAC,
        )
    ):
        return None
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    if panel is not None:
        bm, bn, bk, warps, split, stages, transpose = panel
        lhs, rhs = (b, a) if transpose else (a, b)
        rows, cols = (n, m) if transpose else (m, n)
        fp32_panels[(triton.cdiv(rows, bm), triton.cdiv(cols, bn))](
            lhs,
            rhs,
            out,
            rows,
            cols,
            k,
            *(swapped if transpose else strides),
            bm,
            bn,
            bk,
            split,
            0,
            not transpose,
            num_warps=warps,
            num_stages=stages,
        )
    elif tiled is not None:
        bm, bn, step, bk, warps, stages, group, separate = tiled
        tiles = triton.cdiv(m, bm), triton.cdiv(n, bn)
        grid = (tiles[0] * tiles[1],) if group else tiles
        tiled_panels[grid](
            a,
            b,
            out,
            m,
            n,
            k,
            *strides,
            bm,
            bn,
            step,
            bk,
            separate,
            1,
            group,
            False,
            bool(group),
            bool(group),
            num_warps=warps,
            num_stages=stages,
        )
    elif key == SPLIT_K:
        partial = torch.empty((8, m, n), device=a.device, dtype=a.dtype)
        tiled_panels[(triton.cdiv(m, 32), triton.cdiv(n, 32), 8)](
            a,
            b,
            partial,
            m,
            n,
            k,
            *strides,
            32,
            32,
            triton.cdiv(k, 512) * 64,
            64,
            False,
            8,
            0,
            False,
            num_warps=4,
            num_stages=1,
        )
        sum_parts[(triton.cdiv(m * n, 256),)](partial, out, m * n, 8, 256, num_warps=4)
    elif key == TRANSPOSED_TAIL:
        parts = triton.cdiv(k, 224)
        partial = torch.empty((parts, m, n), device=a.device, dtype=a.dtype)
        cpu112_panels[(triton.cdiv(n, 64), triton.cdiv(m, 32), parts)](
            b,
            a,
            partial,
            n,
            m,
            k,
            *swapped,
            64,
            32,
            32,
            True,
            True,
            224,
            False,
            True,
            num_warps=4,
            num_stages=1,
        )
        # The tail CTAs and the panel reduction write disjoint output rows.
        rows = 88
        special_cols = (316, 633, 950)
        block, special_block, sum_block = 16, 8, 64
        edge_tiles = triton.cdiv(n, block) + triton.cdiv(
            len(special_cols), special_block
        )
        blocks = (m - rows) * edge_tiles + triton.cdiv(rows * n, sum_block)
        transposed_tail_sum[(blocks,)](
            a,
            b,
            partial,
            out,
            m,
            k,
            n,
            *a.stride(),
            b.stride(1),
            parts,
            special_cols,
            rows,
            block,
            special_block,
            sum_block,
            num_warps=1,
            enable_fp_fusion=False,
        )
    elif key == NATIVE_MMAC:
        from .mm_fp32_native_panels import mmac_tiled

        panel_length = 272
        full = (k - 1) // panel_length
        mmac_tiled[(triton.cdiv(n, 128), triton.cdiv(m, 64))](
            b,
            a,
            out,
            n,
            m,
            k - full * panel_length,
            *swapped,
            4,
            4,
            2,
            4,
            2,
            0,
            True,
            False,
            True,
            panel_length,
            0,
            False,
            full,
            num_warps=4,
            num_stages=1,
        )
    else:
        tiles = triton.cdiv(m, 64), triton.cdiv(n, 64)
        grid = (tiles[0] * tiles[1], 1, 1) if group else (*tiles, 1)
        cpu112_panels[grid](
            a,
            b,
            out,
            m,
            n,
            k,
            *strides,
            64,
            64,
            32,
            False,
            True,
            160,
            False,
            False,
            group,
            num_warps=8,
            num_stages=1,
        )
    return out
