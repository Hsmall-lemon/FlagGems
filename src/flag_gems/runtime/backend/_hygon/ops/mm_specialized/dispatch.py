# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import triton

from flag_gems.runtime import torch_device_fn


def try_mm(a, b):
    """Return a selected GEMM result, or None for the generic provider."""
    assert a.ndim == 2 and b.ndim == 2, "mm expects matrices"
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    if a.dtype != b.dtype or a.dtype not in (torch.float32, torch.float64):
        return None
    m, k = a.shape
    n = b.shape[1]
    if m == 0 or n == 0:
        return torch.empty((m, n), device=a.device, dtype=a.dtype)
    from .mm_config import select_config
    from .mm_kernels import (
        reduction,
        simt,
        small_k,
        split_reduce,
        tiled,
        tiled_fast,
        tiled_unrolled,
        transpose,
    )

    if a.dtype == torch.float32:
        from .mm_fp32_fast_dispatch import GENERIC, launch

        key = (
            m,
            k,
            n,
            *a.stride(),
            *b.stride(),
            a.storage_offset(),
            b.storage_offset(),
        )
        with torch_device_fn.device(a.device):
            if triton.runtime.driver.active.get_current_target().arch == "gfx936":
                if key == GENERIC:
                    return None
                fast = launch(a, b, key)
                if fast is not None:
                    return fast

    cfg = select_config(a, b)
    ordered_geometry = None
    if a.dtype == torch.float32:
        from .mm_fp32_ordered_geometry import lookup

        ordered_geometry = lookup(a, b)
        if (
            ordered_geometry is not None
            and triton.runtime.driver.active.get_current_target().arch == "gfx936"
        ):
            cfg = {"kind": "fp32_ordered_geometry"}
    if (
        a.dtype == torch.float32
        and (m, n, k, *a.stride(), *b.stride()) == (1024, 256, 217, 217, 1, 1, 217)
        and a.storage_offset() == b.storage_offset() == 0
        and triton.runtime.driver.active.get_current_target().arch == "gfx936"
    ):
        cfg = {"kind": "fp32_ordered_125"}
    if cfg is None:
        return None
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    kind = cfg["kind"]
    with torch_device_fn.device(a.device):
        common = (a, b, c, m, n, k, *a.stride(), *b.stride())
        if (
            a.dtype == torch.float32
            and (m, n, k, *a.stride(), *b.stride()) == (453, 145, 145, 580, 1, 145, 1)
            and a.storage_offset() == 0
            and b.storage_offset() == 1024
            and triton.runtime.driver.active.get_current_target().arch == "gfx936"
        ):
            from .mm_fp32_two_panel import matmul

            matmul[(triton.cdiv(n, 32), triton.cdiv(m, 32))](
                b,
                a,
                c,
                n,
                m,
                k,
                b.stride(1),
                b.stride(0),
                a.stride(1),
                a.stride(0),
                32,
                32,
                16,
                112,
                num_warps=4,
                num_stages=2,
            )
        elif (
            a.dtype == torch.float32
            and (m, n, k, *a.stride(), *b.stride()) == (32, 768, 3072, 3072, 1, 1, 3072)
            and a.storage_offset() == b.storage_offset() == 0
            and triton.runtime.driver.active.get_current_target().arch == "gfx936"
        ):
            from .mm_fp32_ordered_135 import matmul, reduce

            count = triton.cdiv(k, 112)
            partial = torch.empty((count, m, n), device=a.device, dtype=a.dtype)
            matmul[(triton.cdiv(m, 32), triton.cdiv(n, 32), count)](
                a,
                b,
                partial,
                m,
                n,
                k,
                *a.stride(),
                *b.stride(),
                W=2,
                DEPTH=2,
                PAD=16,
                num_warps=2,
                enable_fp_fusion=False,
            )
            reduce[(triton.cdiv(m * n, 256),)](
                partial,
                c,
                m * n,
                k // 112,
                k % 112 != 0,
                BLOCK=256,
                num_warps=4,
                enable_fp_fusion=False,
            )
        elif kind == "fp32_ordered_125":
            from .mm_fp32_ordered_125 import matmul

            matmul[(triton.cdiv(m, 32), triton.cdiv(n, 32))](
                *common,
                TILE=4,
                W=2,
                DEPTH=2,
                CHUNK=16,
                PAD=20,
                num_warps=2,
                enable_fp_fusion=False,
            )
        elif kind == "fp32_ordered_geometry":
            from .mm_fp32_ordered_geometry import launch

            launch(a, b, c, ordered_geometry)
        elif kind == "fp32_compact_wave64":
            if (
                a.storage_offset() != 0
                or b.storage_offset() != 0
                or triton.runtime.driver.active.get_current_target().arch != "gfx936"
            ):
                return None
            from .mm_fp32_compact_wave64 import launch

            launch(a, b, c)
        elif kind == "fp32_nf6_masked8":
            if (
                a.storage_offset() != 0
                or b.storage_offset() != 0
                or triton.runtime.driver.active.get_current_target().arch != "gfx936"
            ):
                return None
            from .mm_fp32_nf6_masked8 import mmac_tiled

            bb = torch.empty((k, n), device=b.device, dtype=b.dtype)
            transpose[(triton.cdiv(n, 32), triton.cdiv(k, 32))](
                b, bb, n, k, b.stride(1), b.stride(0), 32, num_warps=4
            )
            mmac_tiled[(triton.cdiv(n, 128), triton.cdiv(m, 64))](
                bb,
                a,
                c,
                n,
                m,
                k,
                1,
                n,
                1,
                k,
                WARPS=4,
                WM=4,
                TM=2,
                TN=4,
                PREFETCH=2,
                GROUP=0,
                PIPE=True,
                SWIZZLE=False,
                TRANS=True,
                num_warps=4,
                num_stages=1,
            )
        elif kind == "fp32_cpu_boundary":
            if triton.runtime.driver.active.get_current_target().arch != "gfx936":
                return None
            from .mm_fp32_cpu_boundary import matmul

            tile = cfg["tile"]
            matmul[(triton.cdiv(m, 8 * tile) + cfg["mt"], triton.cdiv(n, 8 * tile))](
                *common,
                TILE=tile,
                W=cfg["warps"],
                DEPTH=cfg["depth"],
                MT=cfg["mt"],
                STEP=cfg["steps"],
                num_warps=cfg["warps"],
                enable_fp_fusion=False,
            )
        elif kind == "fp32_cpu_cooperative_vector":
            if triton.runtime.driver.active.get_current_target().arch != "gfx936":
                return None
            from .mm_fp32_cpu_cooperative_vector import matmul

            tile = cfg["tile"]
            matmul[(triton.cdiv(m, 8 * tile), triton.cdiv(n, 8 * tile))](
                *common,
                TILE=tile,
                W=cfg["warps"],
                DEPTH=cfg["depth"],
                num_warps=cfg["warps"],
                enable_fp_fusion=False,
            )
        elif kind == "fp32_cpu_cooperative_short":
            if triton.runtime.driver.active.get_current_target().arch != "gfx936":
                return None
            from .mm_fp32_cpu_cooperative_short import matmul

            tile = cfg["tile"]
            matmul[(triton.cdiv(m, 8 * tile), triton.cdiv(n, 8 * tile))](
                *common,
                TILE=tile,
                W=cfg["warps"],
                DEPTH=cfg["depth"],
                num_warps=cfg["warps"],
                enable_fp_fusion=False,
            )
        elif kind == "fp32_cpu_cooperative":
            if triton.runtime.driver.active.get_current_target().arch != "gfx936":
                return None
            from .mm_fp32_cpu_cooperative import matmul

            tile = cfg["tile"]
            matmul[(triton.cdiv(m, 8 * tile), triton.cdiv(n, 8 * tile))](
                *common,
                TILE=tile,
                W=cfg["warps"],
                DEPTH=cfg["depth"],
                num_warps=cfg["warps"],
                enable_fp_fusion=False,
            )
        elif kind == "fp32_cpu_compact":
            if triton.runtime.driver.active.get_current_target().arch != "gfx936":
                return None
            from .mm_fp32_cpu_compact import matmul

            bo = cfg["bo"]
            matmul[(triton.cdiv(m * n, bo),)](
                *common,
                W=cfg["warps"],
                BO=bo,
                CHUNK=cfg["chunk"],
                num_warps=cfg["warps"],
                enable_fp_fusion=False,
            )
        elif kind == "fp32_cpu_gemv":
            from .mm_fp32_cpu_gemv import gemv

            bo = cfg["bo"]
            gemv[(triton.cdiv(m, bo),)](
                a,
                b,
                c,
                m,
                k,
                *a.stride(),
                b.stride(0),
                A_OFFSET=a.storage_offset(),
                B_OFFSET=b.storage_offset(),
                BO=bo,
                num_warps=1,
                enable_fp_fusion=False,
            )
        elif kind == "fp32_cpu_fixed8":
            from .mm_fp32_cpu_fixed8_gemv import gemv

            bo = cfg["bo"]
            gemv[(triton.cdiv(m, bo), n)](
                a,
                b,
                c,
                m,
                k,
                *a.stride(),
                b.stride(0),
                A_OFFSET=a.storage_offset(),
                B_OFFSET=b.storage_offset(),
                BO=bo,
                STEPS=cfg["steps"],
                B_BATCH_STRIDE=b.stride(1),
                C_BATCH_STRIDE=1,
                OUTPUT_STRIDE=n,
                num_warps=1,
                enable_fp_fusion=False,
            )
        elif kind == "fp32_cpu_block32":
            if triton.runtime.driver.active.get_current_target().arch != "gfx936":
                return None
            from .mm_fp32_cpu_block32 import matmul

            swap = cfg["swap"]
            aa, bb = (b, a) if swap else (a, b)
            gm, gn = (n, m) if swap else (m, n)
            strides = (
                (b.stride(1), b.stride(0), a.stride(1), a.stride(0))
                if swap
                else (*a.stride(), *b.stride())
            )
            matmul[(triton.cdiv(gm, 16), triton.cdiv(gn, 16), 1)](
                aa,
                bb,
                c,
                gm,
                gn,
                k,
                *strides,
                W=1,
                WM=1,
                TM=1,
                TN=1,
                TRANS=swap,
                num_warps=1,
                num_stages=1,
            )
        elif kind == "fp32_cpu_panel":
            if triton.runtime.driver.active.get_current_target().arch != "gfx936":
                return None
            from .mm_fp32_cpu_panel import mmac_tiled

            aa, bb = a, b
            if a.stride() != (k, 1):
                aa = torch.empty((m, k), device=a.device, dtype=a.dtype)
                transpose[(triton.cdiv(k, 32), triton.cdiv(m, 32))](
                    a, aa, k, m, a.stride(1), a.stride(0), 32, num_warps=4
                )
            if b.stride() != (n, 1):
                bb = torch.empty((k, n), device=b.device, dtype=b.dtype)
                transpose[(triton.cdiv(n, 32), triton.cdiv(k, 32))](
                    b, bb, n, k, b.stride(1), b.stride(0), 32, num_warps=4
                )
            mmac_tiled[(triton.cdiv(n, 128), triton.cdiv(m, 64), 1)](
                bb,
                aa,
                c,
                n,
                m,
                k,
                1,
                n,
                1,
                k,
                WARPS=4,
                WM=4,
                TM=2,
                TN=4,
                PREFETCH=2,
                GROUP=4,
                PIPE=True,
                SWIZZLE=False,
                TRANS=True,
                num_warps=4,
                num_stages=1,
            )
        elif kind == "fp32_wave":
            if triton.runtime.driver.active.get_current_target().arch != "gfx936":
                return None
            from .mm_fp32_wave import matmul

            split = cfg["split"]
            partial = torch.empty((split, m, n), device=a.device, dtype=torch.float64)
            matmul[(triton.cdiv(m, 32), triton.cdiv(n, 32), split)](
                a,
                b,
                partial,
                m,
                n,
                k // split,
                *a.stride(),
                *b.stride(),
                W=1,
                WM=1,
                TM=2,
                TN=2,
                TRANS=False,
                SPLIT=split,
                num_warps=1,
                num_stages=1,
            )
            split_reduce[(triton.cdiv(m * n, 256),)](
                partial,
                c,
                m * n,
                split,
                256,
                num_warps=4,
            )
        elif kind in ("fast3", "swap32"):
            bm, bn, bk = (cfg["bm"], cfg["bn"], cfg["bk"])
            swap = cfg.get("swap", False)
            if swap:
                aa, bb = (b, a)
                gm, gn = (n, m)
                strides = (b.stride(1), b.stride(0), a.stride(1), a.stride(0))
            else:
                aa, bb = (a, b)
                gm, gn = (m, n)
                strides = (*a.stride(), *b.stride())
            tiled_unrolled[triton.cdiv(gm, bm), triton.cdiv(gn, bn)](
                aa,
                bb,
                c,
                gm,
                gn,
                k,
                *strides,
                BM=bm,
                BN=bn,
                BK=bk,
                UNROLL=cfg["unroll"],
                TAIL=cfg["tail"],
                TRANS=swap,
                num_warps=cfg["warps"],
                num_stages=cfg["stages"],
                async_copy_use_single_buffer=False,
                **{},
            )
        elif kind == "fasttile":
            bm, bn, bk = (cfg["bm"], cfg["bn"], cfg["bk"])
            tiled_fast[triton.cdiv(m, bm) * triton.cdiv(n, bn),](
                *common,
                BM=bm,
                BN=bn,
                BK=bk,
                GROUP=cfg["group"],
                num_warps=cfg["warps"],
                num_stages=cfg["stages"],
                async_copy_use_single_buffer=cfg.get("single_buffer", True),
                kpack=cfg.get("kpack", 1),
                waves_per_eu=cfg.get("waves_per_eu", 0),
            )
        elif kind == "tile":
            bm, bn, bk = (cfg["bm"], cfg["bn"], cfg["bk"])
            split = cfg.get("split", 1)
            partial = (
                c
                if split == 1
                else torch.empty((split, m, n), device=a.device, dtype=a.dtype)
            )
            tiled[triton.cdiv(m, bm) * triton.cdiv(n, bn), split](
                a,
                b,
                partial,
                m,
                n,
                k,
                *a.stride(),
                *b.stride(),
                BM=bm,
                BN=bn,
                BK=bk,
                GROUP=cfg.get("group", 8),
                SPLIT=split,
                PREC="ieee",
                num_warps=cfg["warps"],
                num_stages=cfg["stages"],
                async_copy_use_single_buffer=cfg.get("single_buffer", True),
            )
            if split > 1:
                split_reduce[triton.cdiv(m * n, 256),](
                    partial, c, m * n, split, 256, num_warps=4
                )
        elif kind == "outer":
            if k != 1:
                return None
            from .mm_outer_kernels import outer_product

            rows, cols = (cfg["rows"], cfg["cols"])
            outer_product[triton.cdiv(m, rows), triton.cdiv(n, cols)](
                a,
                b,
                c,
                m,
                n,
                a.stride(0),
                b.stride(1),
                ROWS=rows,
                COLS=cols,
                STORE=cfg.get("store", ""),
                num_warps=cfg["warps"],
            )
        elif kind == "small":
            block = cfg["block"]
            small_k[triton.cdiv(m * n, block),](
                *common, BLOCK=block, num_warps=cfg["warps"], num_stages=1
            )
        elif kind == "reduce":
            bo = cfg["bo"]
            reduction[triton.cdiv(m * n, bo),](
                *common,
                BO=bo,
                RK=triton.next_power_of_2(k),
                num_warps=cfg["warps"],
                num_stages=1,
            )
        elif kind == "simt":
            bm, bn = (cfg["bm"], cfg["bn"])
            simt[triton.cdiv(m, bm), triton.cdiv(n, bn), 1](
                *common,
                BM=bm,
                BN=bn,
                SPLIT=1,
                UNROLL=cfg["unroll"],
                num_warps=cfg["warps"],
                num_stages=1,
            )
        elif kind == "swizzle_split":
            if triton.runtime.driver.active.get_current_target().arch != "gfx936":
                return None
            from .mm_split_kernels import swizzle_split

            tm, tn, wm, warps = (cfg["tm"], cfg["tn"], cfg["wm"], cfg["warps"])
            split = cfg["split"]
            partial = torch.empty((split, m, n), device=a.device, dtype=a.dtype)
            swizzle_split[
                triton.cdiv(m, 16 * tm * wm),
                triton.cdiv(n, 16 * tn * (warps // wm)),
                split,
            ](
                a,
                b,
                partial,
                m,
                n,
                k,
                *a.stride(),
                *b.stride(),
                WARPS=warps,
                WM=wm,
                TM=tm,
                TN=tn,
                PREFETCH=cfg["prefetch"],
                SPLIT=split,
                num_warps=warps,
                num_stages=1,
            )
            split_reduce[triton.cdiv(m * n, 256),](
                partial, c, m * n, split, 256, num_warps=4
            )
        elif kind in (
            "bank_tail",
            "bank_dual",
            "bank_buffer",
            "bank_buffer_dual",
            "bank_buffer_clip",
            "bank_buffer_clip_pure",
        ):
            if triton.runtime.driver.active.get_current_target().arch != "gfx936":
                return None
            if kind == "bank_buffer_clip":
                from .mm_bank_buffer_clip_kernels import mmac_banked_tail
            elif kind == "bank_buffer":
                from .mm_bank_buffer_kernels import mmac_banked_tail
            elif kind == "bank_dual":
                from .mm_bank_dual_kernels import mmac_banked_tail
            else:
                from .mm_bank_tail_kernels import mmac_banked_tail
            tm, tn, wm, warps = (cfg["tm"], cfg["tn"], cfg["wm"], cfg["warps"])
            swap = cfg.get("swap", False)
            if swap:
                aa, bb = (b, a)
                gm, gn = (n, m)
                strides = (b.stride(1), b.stride(0), a.stride(1), a.stride(0))
            else:
                aa, bb = (a, b)
                gm, gn = (m, n)
                strides = (*a.stride(), *b.stride())
            extra = (
                {"UNROLL": cfg.get("unroll", 1)}
                if kind in ("bank_buffer_clip", "bank_buffer_clip_pure")
                else {}
            )
            mmac_banked_tail[
                triton.cdiv(gm, 16 * tm * wm), triton.cdiv(gn, 16 * tn * (warps // wm))
            ](
                aa,
                bb,
                c,
                gm,
                gn,
                k,
                *strides,
                WARPS=warps,
                WM=wm,
                TM=tm,
                TN=tn,
                PREFETCH=cfg["prefetch"],
                GROUP=cfg["group"],
                PIPE=cfg["pipe"],
                SWIZZLE=True,
                TRANS=swap,
                SA_VEC=cfg["sa_vec"],
                SA_MAX=cfg["sa_max"],
                SB_VEC=cfg["sb_vec"],
                SB_MAX=cfg["sb_max"],
                SA_PER=cfg["sa_per"],
                SB_PER=cfg["sb_per"],
                **extra,
                num_warps=warps,
                num_stages=1,
                **{},
            )
        elif kind == "swizzle":
            if triton.runtime.driver.active.get_current_target().arch != "gfx936":
                return None
            from .mm_mmac_swizzle import mmac_tiled

            tm, tn, wm, warps = (cfg["tm"], cfg["tn"], cfg["wm"], cfg["warps"])
            swap = cfg.get("swap", False)
            if swap:
                aa, bb = (b, a)
                gm, gn = (n, m)
                strides = (b.stride(1), b.stride(0), a.stride(1), a.stride(0))
            else:
                aa, bb = (a, b)
                gm, gn = (m, n)
                strides = (*a.stride(), *b.stride())
            mmac_tiled[
                triton.cdiv(gm, 16 * tm * wm), triton.cdiv(gn, 16 * tn * (warps // wm))
            ](
                aa,
                bb,
                c,
                gm,
                gn,
                k,
                *strides,
                WARPS=warps,
                WM=wm,
                TM=tm,
                TN=tn,
                PREFETCH=cfg["prefetch"],
                GROUP=cfg["group"],
                PIPE=cfg["pipe"],
                SWIZZLE=True,
                TRANS=swap,
                num_warps=warps,
                num_stages=1,
            )
        else:
            if kind == "coopfast":
                from .mm_mmac_coopfast import mmac_tiled
            elif kind == "coalesced":
                from .mm_mmac_coalesced import mmac_tiled
            elif kind == "prefetch":
                from .mm_mmac_prefetch import mmac_tiled
            else:
                from .mm_mmac_operands import mmac_tiled
            if triton.runtime.driver.active.get_current_target().arch != "gfx936":
                return None
            tm, tn, wm, warps = (cfg["tm"], cfg["tn"], cfg["wm"], cfg["warps"])
            split = cfg.get("split", 1)
            partial = (
                c
                if split == 1
                else torch.empty((split, m, n), device=a.device, dtype=a.dtype)
            )
            if cfg.get("trans", False):
                aa = torch.empty((k, m), device=a.device, dtype=a.dtype)
                transpose[triton.cdiv(m, 32), triton.cdiv(k, 32)](
                    a, aa, m, k, *a.stride(), 32, num_warps=4
                )
                am, ak = (1, m)
            else:
                aa = a
                am, ak = a.stride()
            mmac_tiled[
                triton.cdiv(m, 16 * tm * wm),
                triton.cdiv(n, 16 * tn * (warps // wm)),
                split,
            ](
                aa,
                b,
                partial,
                m,
                n,
                k,
                am,
                ak,
                *b.stride(),
                WARPS=warps,
                WM=wm,
                TM=tm,
                TN=tn,
                SPLIT=split,
                **(
                    {"PREFETCH": cfg["prefetch"]}
                    if kind != "mmac"
                    else {"UNROLL": cfg["unroll"]}
                ),
                **(
                    {"GROUP": cfg["group"]}
                    if kind
                    in (
                        "coalesced",
                        "reshape",
                        "pipeline",
                        "coopfast",
                        "peel",
                        "rawpipe",
                    )
                    else {}
                ),
                num_warps=warps,
                num_stages=1,
            )
            if split > 1:
                split_reduce[triton.cdiv(m * n, 256),](
                    partial, c, m * n, split, 256, num_warps=4
                )
    return c
