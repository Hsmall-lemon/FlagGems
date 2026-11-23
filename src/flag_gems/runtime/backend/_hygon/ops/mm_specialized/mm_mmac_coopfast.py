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

import triton
import triton.language as tl

from .mm_tl_bank_compute import compute as compute_bank


@triton.jit
def mmac_tiled(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    WARPS: tl.constexpr,
    WM: tl.constexpr,
    TM: tl.constexpr,
    TN: tl.constexpr,
    SPLIT: tl.constexpr,
    PREFETCH: tl.constexpr,
    GROUP: tl.constexpr,
):
    tl.static_assert(SPLIT == 1)
    nm = tl.cdiv(M, 16 * TM * WM)
    nn = tl.cdiv(N, 16 * TN * (WARPS // WM))
    p = tl.program_id(0) + tl.program_id(1) * nm
    g = p // (GROUP * nn)
    gs = tl.minimum(nm - g * GROUP, GROUP)
    pm = g * GROUP + p % (GROUP * nn) % gs
    pn = p % (GROUP * nn) // gs
    acc = ()
    for i in tl.static_range(TM * TN * 4):
        acc += (tl.full((WARPS * 64,), 0, tl.float64),)
    chunk = tl.cdiv(tl.cdiv(K, 4 * PREFETCH), SPLIT) * 4 * PREFETCH
    for step in range(tl.cdiv(chunk, 4 * PREFETCH)):
        aa, bb = cooperative_load(
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
            chunk,
            step,
        )
        acc = compute_bank(aa, bb, acc, TM, TN, PREFETCH)
    for i in tl.static_range(TM):
        for j in tl.static_range(TN):
            store_acc(acc, C, pm, pn, M, N, WARPS, WM, TM, TN, i, j)


@triton.jit
def store_acc(
    acc,
    C,
    pm,
    pn,
    M: tl.constexpr,
    N: tl.constexpr,
    WARPS: tl.constexpr,
    WM: tl.constexpr,
    TM: tl.constexpr,
    TN: tl.constexpr,
    TI: tl.constexpr,
    TJ: tl.constexpr,
):
    off: tl.constexpr = (TI * TN + TJ) * 4
    vals = tl.interleave(
        tl.interleave(acc[off], acc[off + 2]), tl.interleave(acc[off + 1], acc[off + 3])
    )
    x = tl.arange(0, WARPS * 256)
    warp = x // 256
    row = x % 256 // 16
    col = x % 16
    value = tl.reshape(
        tl.trans(tl.reshape(vals, (WARPS, 4, 16, 4)), (0, 2, 3, 1)), (WARPS * 256,)
    )
    rr = pm * (16 * TM * WM) + warp % WM * 16 * TM + TI * 16 + row
    cc = pn * (16 * TN * (WARPS // WM)) + warp // WM * 16 * TN + TJ * 16 + col
    if M % (16 * TM * WM) == 0 and N % (16 * TN * (WARPS // WM)) == 0:
        tl.store(C + rr * N + cc, value)
    else:
        tl.store(C + rr * N + cc, value, (rr < M) & (cc < N))


@triton.constexpr_function
def split_levels(n):
    return n.bit_length() - 1


@triton.constexpr_function
def reverse_bits(i, n):
    r = 0
    for j in range(n):
        r = r << 1 | i >> j & 1
    return r


@triton.jit
def split_parts(at, WW: tl.constexpr, TT: tl.constexpr, PP: tl.constexpr):
    at = tl.reshape(at, (WW * 64, TT * PP))
    parts = (at,)
    for level in tl.static_range(split_levels(TT * PP)):
        new = ()
        for pi in tl.static_range(2**level):
            even, odd = tl.split(
                tl.reshape(parts[pi], (WW * 64, TT * PP // 2 ** (level + 1), 2))
            )
            new += (even, odd)
        parts = new
    return parts


@triton.jit
def cooperative_load(
    A,
    B,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    WARPS: tl.constexpr,
    WM: tl.constexpr,
    TM: tl.constexpr,
    TN: tl.constexpr,
    PREFETCH: tl.constexpr,
    pm,
    pn,
    chunk: tl.constexpr,
    step,
):
    WN: tl.constexpr = WARPS // WM
    ax = tl.arange(0, WM * TM * 64 * PREFETCH)
    aw = ax // (TM * 64 * PREFETCH)
    ar = ax // (4 * PREFETCH) % (TM * 16)
    ak = ax % (4 * PREFETCH)
    rr = pm * (16 * TM * WM) + aw * 16 * TM + ar
    kk = step * 4 * PREFETCH + ak
    if M % (16 * TM * WM) == 0 and K % (4 * PREFETCH) == 0:
        av = tl.load(A + rr * AM + kk * AK)
    else:
        av = tl.load(A + rr * AM + kk * AK, (rr < M) & (kk < K), 0)
    av = tl.reshape(av, (1, WM, TM, 16, PREFETCH, 4))
    av = tl.broadcast_to(av, (WN, WM, TM, 16, PREFETCH, 4))
    av = tl.reshape(av, (WARPS, TM, 16, PREFETCH, 4))
    av = tl.trans(av, (0, 4, 2, 1, 3))
    ap = split_parts(av, WARPS, TM, PREFETCH)
    bx = tl.arange(0, WN * TN * 64 * PREFETCH)
    bw = bx // (TN * 64 * PREFETCH)
    bj = bx // (64 * PREFETCH) % TN
    bk = bx // 16 % (4 * PREFETCH)
    bc = bx % 16
    cc = pn * (16 * TN * WN) + bw * 16 * TN + bj * 16 + bc
    kk = step * 4 * PREFETCH + bk
    if N % (16 * TN * WN) == 0 and K % (4 * PREFETCH) == 0:
        bv = tl.load(B + kk * BK + cc * BN)
    else:
        bv = tl.load(B + kk * BK + cc * BN, (cc < N) & (kk < K), 0)
    bv = tl.reshape(bv, (WN, 1, TN, PREFETCH, 4, 16))
    bv = tl.broadcast_to(bv, (WN, WM, TN, PREFETCH, 4, 16))
    bv = tl.reshape(bv, (WARPS, TN, PREFETCH, 64))
    bv = tl.trans(bv, (0, 3, 1, 2))
    bp = split_parts(bv, WARPS, TN, PREFETCH)
    aa = ()
    bb = ()
    for ki in tl.static_range(PREFETCH):
        for ti in tl.static_range(TM):
            aa += (
                tl.reshape(
                    ap[reverse_bits(ti * PREFETCH + ki, split_levels(TM * PREFETCH))],
                    (WARPS * 64,),
                ),
            )
        for tj in tl.static_range(TN):
            bb += (
                tl.reshape(
                    bp[reverse_bits(tj * PREFETCH + ki, split_levels(TN * PREFETCH))],
                    (WARPS * 64,),
                ),
            )
    return (aa, bb)
