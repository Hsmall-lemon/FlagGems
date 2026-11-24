# isort: skip_file - preserve the archived provider import order
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

"""Bounded FP32 matrix route; true strides and exact public semantics.
Based on the Apache-2.0 FlagGems tiled GEMM approach. No data-dependent host reads.
Candidate changes schedule, constexpr index arithmetic and per-shape-class tiles.
"""

import torch
import triton
import triton.language as tl
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry


@libentry()
@triton.jit
def _mm_tiled(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SAM: tl.constexpr,
    SAK: tl.constexpr,
    SBK: tl.constexpr,
    SBN: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    pm = pid // tl.cdiv(N, BN)
    pn = pid % tl.cdiv(N, BN)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for step in range(tl.cdiv(K, BK)):
        kk = step * BK + rk
        av = tl.load(
            A + rm[:, None] * SAM + kk[None, :] * SAK,
            mask=(rm[:, None] < M) & (kk[None, :] < K),
            other=0.0,
        )
        bv = tl.load(
            B + kk[:, None] * SBK + rn[None, :] * SBN,
            mask=(kk[:, None] < K) & (rn[None, :] < N),
            other=0.0,
        )
        acc = tl.dot(av, bv, acc, input_precision="ieee")
    tl.store(
        C + rm[:, None] * N + rn[None, :],
        acc,
        mask=(rm[:, None] < M) & (rn[None, :] < N),
    )


@libentry()
@triton.jit
def _mm_vector(
    A,
    B,
    C,
    M: tl.constexpr,
    K: tl.constexpr,
    SAM: tl.constexpr,
    SAK: tl.constexpr,
    SBK: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rk = tl.arange(0, BK)
    av = tl.load(
        A + rm[:, None] * SAM + rk[None, :] * SAK,
        mask=(rm[:, None] < M) & (rk[None, :] < K),
        other=0.0,
    )
    bv = tl.load(B + rk * SBK, mask=rk < K, other=0.0)
    out = tl.sum(av * bv[None, :], axis=1)
    tl.store(C + rm, out, mask=rm < M)


@libentry()
@triton.jit
def _rank_one(
    A,
    B,
    Bias,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    SAM: tl.constexpr,
    SBN: tl.constexpr,
    SCM: tl.constexpr,
    SCN: tl.constexpr,
    ALPHA: tl.constexpr,
    BETA: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    EMPTY_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row = (off // N).to(tl.int64)
    col = (off % N).to(tl.int64)
    mask = off < M * N
    if EMPTY_K or ALPHA == 0:
        prod = tl.full(
            (BLOCK,), 0, tl.float64 if A.dtype.element_ty == tl.float64 else tl.float32
        )
    else:
        av = tl.load(A + row * SAM, mask=mask, other=0)
        bv = tl.load(B + col * SBN, mask=mask, other=0)
        prod = av * bv
        if ALPHA != 1:
            prod = prod * ALPHA
    if HAS_BIAS and BETA != 0:
        bias = tl.load(Bias + row * SCM + col * SCN, mask=mask, other=0)
        if BETA != 1:
            bias = bias * BETA
        result = bias if EMPTY_K or ALPHA == 0 else prod + bias
    else:
        result = prod
    tl.store(C + off, result, mask=mask)


def _rank_one_output(a, b, bias=None, alpha=1.0, beta=0.0):
    M, K = a.shape
    N = b.shape[1]
    out = torch.empty((M, N), dtype=a.dtype, device=a.device)
    if M == 0 or N == 0:
        return out
    sm = sn = 0
    if bias is not None:
        if bias.ndim == 1:
            sn = bias.stride(0) if bias.shape[0] != 1 else 0
        elif bias.ndim == 2:
            sm = bias.stride(0) if bias.shape[0] != 1 else 0
            sn = bias.stride(1) if bias.shape[1] != 1 else 0
    with torch_device_fn.device(a.device):
        _rank_one[(triton.cdiv(M * N, 256),)](
            a,
            b,
            a if bias is None else bias,
            out,
            M,
            N,
            a.stride(0),
            b.stride(1),
            sm,
            sn,
            ALPHA=alpha,
            BETA=beta,
            HAS_BIAS=bias is not None,
            EMPTY_K=K == 0,
            BLOCK=256,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return out


def _install_mm(torch, flag_gems):
    # Use the same tuned backend entry for both supported activation paths.
    from flag_gems.runtime.backend._hygon.ops.mm import mm

    flag_gems.mm = mm


"""One bounded FP32 small-batched BMM candidate; paste into variant.py.

Call _install_bmm(torch, flag_gems) AFTER the fixed baseline providers install.
No public torch API replacement, autotuning, CPU fallback or import-time install.
"""
import triton as _bmm_triton  # noqa: E402 - retained source assembly order
import triton.language as _bmm_tl  # noqa: E402 - retained source assembly order
from flag_gems.utils import libentry as _bmm_libentry  # noqa: E402


@_bmm_libentry()
@_bmm_triton.jit
def _bmm_small_kernel(
    A,
    B,
    O,
    M: _bmm_tl.constexpr,
    N: _bmm_tl.constexpr,
    K: _bmm_tl.constexpr,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    BLOCK_M: _bmm_tl.constexpr,
    BLOCK_N: _bmm_tl.constexpr,
    BLOCK_K: _bmm_tl.constexpr,
):
    # Flatten batch and spatial tiles into one launch dimension. No modulo-zero:
    # the Python entry returns before launching on an empty batch/M/N.
    pid = _bmm_tl.program_id(0)
    tiles_m = _bmm_tl.cdiv(M, BLOCK_M)
    tiles_n = _bmm_tl.cdiv(N, BLOCK_N)
    batch = (pid // (tiles_m * tiles_n)).to(_bmm_tl.int64)
    spatial = pid % (tiles_m * tiles_n)
    rm = ((spatial // tiles_n) * BLOCK_M + _bmm_tl.arange(0, BLOCK_M)).to(_bmm_tl.int64)
    rn = ((spatial % tiles_n) * BLOCK_N + _bmm_tl.arange(0, BLOCK_N)).to(_bmm_tl.int64)
    a_base = A + batch * stride_ab
    b_base = B + batch * stride_bb
    out_ptrs = O + (batch * M + rm[:, None]) * N + rn[None, :]
    out_mask = (rm[:, None] < M) & (rn[None, :] < N)

    rk = _bmm_tl.arange(0, BLOCK_K).to(_bmm_tl.int64)
    accumulator = _bmm_tl.zeros((BLOCK_M, BLOCK_N), dtype=_bmm_tl.float32)
    for start in range(_bmm_tl.cdiv(K, BLOCK_K)):
        kk = start * BLOCK_K + rk
        av = _bmm_tl.load(
            a_base + rm[:, None] * stride_am + kk[None, :] * stride_ak,
            mask=(rm[:, None] < M) & (kk[None, :] < K),
            other=0.0,
        )
        bv = _bmm_tl.load(
            b_base + kk[:, None] * stride_bk + rn[None, :] * stride_bn,
            mask=(kk[:, None] < K) & (rn[None, :] < N),
            other=0.0,
        )
        # IEEE FP32 inputs/accumulation: no TF32, tf32x3 or dtype reduction.
        # Do not sanitize NaN/Inf or skip valid zero products (0*Inf is NaN).
        accumulator = _bmm_tl.dot(
            av,
            bv,
            accumulator,
            input_precision="ieee",
            out_dtype=_bmm_tl.float32,
        )
    result = accumulator
    _bmm_tl.store(out_ptrs, result, mask=out_mask)


def _install_bmm(torch, flag_gems):
    """Patch the actual FG root; capture its current FP64/large fallback once.

    Root-code witnesses obtain flag_gems.bmm.__code__ AFTER this assignment.
    No __wrapped__ indirection hides the executed patch from that witness.
    """
    _bmm_current = flag_gems.bmm
    if getattr(_bmm_current, "_bmm_component_installed", False):
        return _bmm_current
    _bmm_fallback = _bmm_current

    def _bmm_impl(A, B):
        # Invalid/unsupported inputs retain the installed provider's checks and
        # semantics. No .contiguous(), .to(), offset reconstruction or caching.
        if not (isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor)):
            return _bmm_fallback(A, B)
        if A.ndim != 3 or B.ndim != 3:
            return _bmm_fallback(A, B)
        if (
            A.dtype != torch.float32
            or B.dtype != torch.float32
            or A.device != B.device
            or A.device.type not in ("cuda", "flagos")
        ):
            return _bmm_fallback(A, B)
        batch, M, K = A.shape
        if B.shape[0] != batch or B.shape[1] != K:
            return _bmm_fallback(A, B)
        N = B.shape[2]
        if M > 128 or N > 128 or K > 128:
            return _bmm_fallback(A, B)
        a_strides, b_strides = A.stride(), B.stride()
        if any(s < 0 for s in a_strides + b_strides):
            return _bmm_fallback(A, B)

        out = torch.empty((batch, M, N), dtype=A.dtype, device=A.device)
        if batch == 0 or M == 0 or N == 0:
            return out
        if K == 0:
            # Fresh, nonaliasing output; never dereference empty input storage.
            return out.zero_()
        if N <= 16:
            block_m, block_n, block_k = 64, 16, 32
        elif K <= 32:
            block_m, block_n, block_k = 64, 64, 16
        else:
            block_m, block_n, block_k = 64, 64, 32
        grid = (batch * _bmm_triton.cdiv(M, block_m) * _bmm_triton.cdiv(N, block_n),)
        # Tensor pointers already include storage_offset. All six original
        # strides, including zeros and interleaved batch strides, are retained.
        with flag_gems.runtime.torch_device_fn.device(A.device):
            _bmm_small_kernel[grid](
                A,
                B,
                out,
                M,
                N,
                K,
                a_strides[0],
                a_strides[1],
                a_strides[2],
                b_strides[0],
                b_strides[1],
                b_strides[2],
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                num_stages=1,
                num_warps=4,
                enable_fp_fusion=False,
            )
        return out

    _bmm_impl._bmm_component_installed = True
    flag_gems.bmm = _bmm_impl
    return _bmm_impl


def _install_addmm_edges(torch, flag_gems):
    original = flag_gems.addmm

    def addmm(bias, a, b, *, beta=1, alpha=1):
        if (
            a.ndim != 2
            or b.ndim != 2
            or a.shape[1] != b.shape[0]
            or a.dtype not in (torch.float32, torch.float64)
            or a.dtype != b.dtype
            or bias.dtype != a.dtype
            or a.device != b.device
            or bias.device != a.device
            or a.shape[1] not in (0, 1)
        ):
            return original(bias, a, b, beta=beta, alpha=alpha)
        M = a.shape[0]
        N = b.shape[1]
        valid = (
            bias.ndim == 0
            or (bias.ndim == 1 and bias.shape[0] in (1, N))
            or (bias.ndim == 2 and bias.shape[0] in (1, M) and bias.shape[1] in (1, N))
        )
        if not valid:
            return original(bias, a, b, beta=beta, alpha=alpha)
        return _rank_one_output(a, b, bias, alpha, beta)

    flag_gems.addmm = addmm


def install(torch, flag_gems):
    _install_mm(torch, flag_gems)
    _install_bmm(torch, flag_gems)
    _install_addmm_edges(torch, flag_gems)
