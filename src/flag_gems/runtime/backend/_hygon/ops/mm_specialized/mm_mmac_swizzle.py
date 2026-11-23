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

"""FP64 MMAC with bank-aware LDS layouts and optional transposed computation.

TRANS computes B.T @ A.T directly into the original contiguous output. Input
tiles are read contiguously, and the LDS layout supplies native MMAC operands.
"""

import triton
import triton.experimental.gluon.language.amd._layouts as amd_layouts
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import (
    BlockedLayout,
    DistributedLinearLayout,
    SwizzledSharedLayout,
)

from .mm_pipeline_compute import compute

if not hasattr(amd_layouts, "HCUMFMALayout"):
    amd_layouts.HCUMFMALayout = amd_layouts.AMDMFMALayout
if not hasattr(amd_layouts, "HCUWMMALayout"):
    amd_layouts.HCUWMMALayout = amd_layouts.AMDWMMALayout


@g.jit
def store_acc(
    acc,
    C,
    pm,
    pn,
    M: gl.constexpr,
    N: gl.constexpr,
    TM: gl.constexpr,
    TN: gl.constexpr,
    WARPS: gl.constexpr,
    WM: gl.constexpr,
    I: gl.constexpr,
    J: gl.constexpr,
    TRANS: gl.constexpr,
):
    off: gl.constexpr = (I * TN + J) * 4
    value = gl.join(
        gl.join(acc[off], acc[off + 2]), gl.join(acc[off + 1], acc[off + 3])
    )
    value = gl.reshape(value, (WARPS, 4, 16, 4))
    value = gl.permute(value, (0, 2, 3, 1))
    value = gl.reshape(value, (WARPS, 16, 16))
    if TRANS:
        value = gl.permute(value, (0, 2, 1))
    value = gl.reshape(value, (WARPS * 256,))
    L: gl.constexpr = BlockedLayout([2], [64], [WARPS], [0])
    value = gl.convert_layout(value, L)
    x = gl.arange(0, WARPS * 256, layout=L)
    w = x // 256
    rr = (
        pm * 16 * TM * WM
        + w % WM * 16 * TM
        + I * 16
        + (x % 16 if TRANS else x % 256 // 16)
    )
    cc = (
        pn * 16 * TN * (WARPS // WM)
        + w // WM * 16 * TN
        + J * 16
        + (x % 256 // 16 if TRANS else x % 16)
    )
    offset = cc * M + rr if TRANS else rr * N + cc
    gl.store(C + offset, value, (rr < M) & (cc < N))


@triton.constexpr_function
def levels(n):
    return n.bit_length() - 1


@g.jit
def parts(x, WARPS: gl.constexpr, PREFETCH: gl.constexpr):
    values = (x,)
    for level in gl.static_range(levels(PREFETCH)):
        out = ()
        for i in gl.static_range(2**level):
            even, odd = gl.split(
                gl.reshape(values[i], (WARPS * 64, PREFETCH // 2 ** (level + 1), 2))
            )
            out += (even, odd)
        values = out
    return values


@triton.constexpr_function
def rev(i, pf):
    n = pf.bit_length() - 1
    r = 0
    for j in range(n):
        r = r << 1 | i >> j & 1
    return r


@triton.constexpr_function
def load_layout(tm, tn, w, wm, pf, which):
    if which == 0:
        reg = [[16 << i, 0] for i in range(tm.bit_length() - 1)] + [
            [0, 4 << i] for i in range(pf.bit_length() - 1)
        ]
        lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]]
        warp = [[16 * tm * (1 << i), 0] for i in range(wm.bit_length() - 1)] + [
            [0, 0] for i in range((w // wm).bit_length() - 1)
        ]
        shape = [16 * tm * wm, 4 * pf]
    else:
        reg = [[0, 16 << i] for i in range(tn.bit_length() - 1)] + [
            [4 << i, 0] for i in range(pf.bit_length() - 1)
        ]
        lane = [[0, 1], [0, 2], [0, 4], [0, 8], [1, 0], [2, 0]]
        warp = [[0, 0] for i in range(wm.bit_length() - 1)] + [
            [0, 16 * tn * (1 << i)] for i in range((w // wm).bit_length() - 1)
        ]
        shape = [4 * pf, 16 * tn * (w // wm)]
    return DistributedLinearLayout(
        reg_bases=reg, lane_bases=lane, warp_bases=warp, block_bases=[], shape=shape
    )


@g.jit
def load(
    A,
    B,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    AM: gl.constexpr,
    AK: gl.constexpr,
    BK: gl.constexpr,
    BN: gl.constexpr,
    WARPS: gl.constexpr,
    WM: gl.constexpr,
    TM: gl.constexpr,
    TN: gl.constexpr,
    PREFETCH: gl.constexpr,
    pm,
    pn,
    step,
    FULL: gl.constexpr,
    SWIZZLE: gl.constexpr,
    TRANS: gl.constexpr,
):
    WN: gl.constexpr = WARPS // WM
    LD: gl.constexpr = BlockedLayout([2], [64], [WARPS], [0])
    if TRANS:
        ax = gl.arange(0, WM * TM * 64 * PREFETCH, layout=LD)
        rr = pm * 16 * TM * WM + ax % (16 * TM * WM)
        kk = step * 4 * PREFETCH + ax // (16 * TM * WM)
        if FULL and M % (16 * TM * WM) == 0:
            av = gl.load(A + rr * AM + kk * AK)
        elif FULL:
            av = gl.load(A + rr * AM + kk * AK, rr < M, 0)
        else:
            av = gl.load(A + rr * AM + kk * AK, (rr < M) & (kk < K), 0)
        bx = gl.arange(0, WN * TN * 64 * PREFETCH, layout=LD)
        kk = step * 4 * PREFETCH + bx % (4 * PREFETCH)
        cc = pn * 16 * TN * WN + bx // (4 * PREFETCH)
        if FULL and N % (16 * TN * WN) == 0:
            bv = gl.load(B + kk * BK + cc * BN)
        elif FULL:
            bv = gl.load(B + kk * BK + cc * BN, cc < N, 0)
        else:
            bv = gl.load(B + kk * BK + cc * BN, (cc < N) & (kk < K), 0)
        av = gl.permute(gl.reshape(av, (4 * PREFETCH, 16 * TM * WM)), (1, 0))
        bv = gl.permute(gl.reshape(bv, (16 * TN * WN, 4 * PREFETCH)), (1, 0))
    else:
        ax = gl.arange(0, WM * TM * 64 * PREFETCH, layout=LD)
        rr = pm * 16 * TM * WM + ax // (4 * PREFETCH)
        kk = step * 4 * PREFETCH + ax % (4 * PREFETCH)
        if FULL and M % (16 * TM * WM) == 0:
            av = gl.load(A + rr * AM + kk * AK)
        elif FULL:
            av = gl.load(A + rr * AM + kk * AK, rr < M, 0)
        else:
            av = gl.load(A + rr * AM + kk * AK, (rr < M) & (kk < K), 0)
        bx = gl.arange(0, WN * TN * 64 * PREFETCH, layout=LD)
        kk = step * 4 * PREFETCH + bx // (16 * TN * WN)
        cc = pn * 16 * TN * WN + bx % (16 * TN * WN)
        if FULL and N % (16 * TN * WN) == 0:
            bv = gl.load(B + kk * BK + cc * BN)
        elif FULL:
            bv = gl.load(B + kk * BK + cc * BN, cc < N, 0)
        else:
            bv = gl.load(B + kk * BK + cc * BN, (cc < N) & (kk < K), 0)
        av = gl.reshape(av, (16 * TM * WM, 4 * PREFETCH))
        bv = gl.reshape(bv, (4 * PREFETCH, 16 * TN * WN))
    SA: gl.constexpr = (
        SwizzledSharedLayout(1, 2, 8 if PREFETCH >= 2 else 4, [1, 0])
        if SWIZZLE
        else SwizzledSharedLayout(1, 1, 1, [1, 0])
    )
    SB: gl.constexpr = (
        SwizzledSharedLayout(2, 1, 8, [1, 0])
        if SWIZZLE
        else SwizzledSharedLayout(1, 1, 1, [1, 0])
    )
    LA2: gl.constexpr = load_layout(TM, TN, WARPS, WM, PREFETCH, 0)
    LB2: gl.constexpr = load_layout(TM, TN, WARPS, WM, PREFETCH, 1)
    LA: gl.constexpr = BlockedLayout(
        [1, 1, TM, 1, PREFETCH, 1],
        [1, 1, 1, 16, 1, 4],
        [WN, WM, 1, 1, 1, 1],
        [3, 5, 4, 2, 1, 0],
    )
    LB: gl.constexpr = BlockedLayout(
        [1, 1, TN, PREFETCH, 1, 1],
        [1, 1, 1, 1, 4, 16],
        [WN, WM, 1, 1, 1, 1],
        [5, 4, 3, 2, 1, 0],
    )
    sa = gl.allocate_shared_memory(
        gl.float64, (16 * TM * WM, 4 * PREFETCH), SA, value=av
    )
    sb = gl.allocate_shared_memory(
        gl.float64, (4 * PREFETCH, 16 * TN * WN), SB, value=bv
    )
    av = sa.load(LA2)
    bv = sb.load(LB2)
    av = gl.reshape(av, (1, WM, TM, 16, PREFETCH, 4))
    av = gl.convert_layout(av, LA)
    av = (
        av.to(gl.uint64, bitcast=True)
        | gl.full((WN, WM, TM, 16, PREFETCH, 4), 0, gl.uint64, LA)
    ).to(gl.float64, bitcast=True)
    av = gl.reshape(av, (WARPS, TM, 16, PREFETCH, 4))
    av = gl.permute(av, (0, 4, 2, 1, 3))
    av = gl.reshape(av, (WARPS * 64, TM * PREFETCH))
    bv = gl.permute(gl.reshape(bv, (PREFETCH, 4, WN, TN, 16)), (2, 3, 0, 1, 4))
    bv = gl.reshape(bv, (WN, 1, TN, PREFETCH, 4, 16))
    bv = gl.convert_layout(bv, LB)
    bv = (
        bv.to(gl.uint64, bitcast=True)
        | gl.full((WN, WM, TN, PREFETCH, 4, 16), 0, gl.uint64, LB)
    ).to(gl.float64, bitcast=True)
    bv = gl.reshape(bv, (WARPS, TN, PREFETCH, 64))
    bv = gl.permute(bv, (0, 3, 1, 2))
    bv = gl.reshape(bv, (WARPS * 64, TN * PREFETCH))
    ap = parts(av, WARPS, TM * PREFETCH)
    bp = parts(bv, WARPS, TN * PREFETCH)
    aa = ()
    bb = ()
    L1: gl.constexpr = BlockedLayout([1], [64], [WARPS], [0])
    for ki in gl.static_range(PREFETCH):
        for i in gl.static_range(TM):
            aa += (
                gl.convert_layout(
                    gl.reshape(
                        ap[rev(i * PREFETCH + ki, TM * PREFETCH)], (WARPS * 64,)
                    ),
                    L1,
                ),
            )
        for j in gl.static_range(TN):
            bb += (
                gl.convert_layout(
                    gl.reshape(
                        bp[rev(j * PREFETCH + ki, TN * PREFETCH)], (WARPS * 64,)
                    ),
                    L1,
                ),
            )
    return (aa, bb)


@g.jit
def mmac_tiled(
    A,
    B,
    C,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    AM: gl.constexpr,
    AK: gl.constexpr,
    BK: gl.constexpr,
    BN: gl.constexpr,
    WARPS: gl.constexpr,
    WM: gl.constexpr,
    TM: gl.constexpr,
    TN: gl.constexpr,
    PREFETCH: gl.constexpr,
    GROUP: gl.constexpr,
    PIPE: gl.constexpr,
    SWIZZLE: gl.constexpr,
    TRANS: gl.constexpr,
):
    L: gl.constexpr = BlockedLayout([1], [64], [WARPS], [0])
    nm = gl.cdiv(M, 16 * TM * WM)
    nn = gl.cdiv(N, 16 * TN * (WARPS // WM))
    p = gl.program_id(0) + gl.program_id(1) * nm
    group = p // (GROUP * nn)
    size = gl.minimum(nm - group * GROUP, GROUP)
    pm = group * GROUP + p % (GROUP * nn) % size
    pn = p % (GROUP * nn) // size
    acc = ()
    for i in gl.static_range(TM * TN * 4):
        acc += (gl.full((WARPS * 64,), 0, gl.float64, L),)
    steps: gl.constexpr = K // (4 * PREFETCH)
    if PIPE and steps > 0:
        aa, bb = load(
            A,
            B,
            M,
            N,
            K,
            AM,
            AK,
            BK,
            BN,
            WARPS,
            WM,
            TM,
            TN,
            PREFETCH,
            pm,
            pn,
            0,
            True,
            SWIZZLE,
            TRANS,
        )
        for step in range(steps - 1):
            na, nb = load(
                A,
                B,
                M,
                N,
                K,
                AM,
                AK,
                BK,
                BN,
                WARPS,
                WM,
                TM,
                TN,
                PREFETCH,
                pm,
                pn,
                step + 1,
                True,
                SWIZZLE,
                TRANS,
            )
            acc = compute(aa, bb, acc, TM, TN, PREFETCH)
            aa = na
            bb = nb
        acc = compute(aa, bb, acc, TM, TN, PREFETCH)
    else:
        for step in range(steps):
            aa, bb = load(
                A,
                B,
                M,
                N,
                K,
                AM,
                AK,
                BK,
                BN,
                WARPS,
                WM,
                TM,
                TN,
                PREFETCH,
                pm,
                pn,
                step,
                True,
                SWIZZLE,
                TRANS,
            )
            acc = compute(aa, bb, acc, TM, TN, PREFETCH)
    if K % (4 * PREFETCH) != 0:
        aa, bb = load(
            A,
            B,
            M,
            N,
            K,
            AM,
            AK,
            BK,
            BN,
            WARPS,
            WM,
            TM,
            TN,
            PREFETCH,
            pm,
            pn,
            steps,
            False,
            SWIZZLE,
            TRANS,
        )
        acc = compute(aa, bb, acc, TM, TN, PREFETCH)
    if TRANS:
        x = gl.arange(0, WARPS * 64, layout=L)
        lane = x % 64
        warp = x // 64
        for i in gl.static_range(TM):
            for j in gl.static_range(TN):
                for z in gl.static_range(4):
                    rr = pm * 16 * TM * WM + warp % WM * 16 * TM + i * 16 + lane % 16
                    cc = (
                        pn * 16 * TN * (WARPS // WM)
                        + warp // WM * 16 * TN
                        + j * 16
                        + lane // 16
                        + z * 4
                    )
                    v = acc[(i * TN + j) * 4 + z]
                    if M % (16 * TM * WM) == 0 and N % (16 * TN * (WARPS // WM)) == 0:
                        gl.store(C + rr + cc * M, v)
                    else:
                        gl.store(C + rr + cc * M, v, (rr < M) & (cc < N))
    else:
        for i in gl.static_range(TM):
            for j in gl.static_range(TN):
                store_acc(acc, C, pm, pn, M, N, TM, TN, WARPS, WM, i, j, False)
