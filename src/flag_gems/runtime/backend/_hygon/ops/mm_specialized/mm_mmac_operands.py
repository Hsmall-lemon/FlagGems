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
    UNROLL: tl.constexpr,
):
    ids = tl.arange(0, WARPS * 64)
    lane = ids % 64
    warp = ids // 64
    rbase = tl.program_id(0) * (16 * TM * WM) + warp % WM * 16 * TM
    nbase = tl.program_id(1) * (16 * TN * (WARPS // WM)) + warp // WM * 16 * TN
    acc = ()
    for i in tl.static_range(TM * TN * 4):
        acc += (tl.full((WARPS * 64,), 0, tl.float64),)
    chunk = tl.cdiv(tl.cdiv(K, 4), SPLIT) * 4
    for step in tl.range(tl.cdiv(chunk, 4), loop_unroll_factor=UNROLL):
        kk = tl.program_id(2) * chunk + step * 4 + lane // 16
        aa = ()
        bb = ()
        for i in tl.static_range(TM):
            rr = rbase + i * 16 + lane % 16
            aa += (tl.load(A + rr * AM + kk * AK, (rr < M) & (kk < K), 0),)
        for j in tl.static_range(TN):
            cc = nbase + j * 16 + lane % 16
            bb += (tl.load(B + kk * BK + cc * BN, (cc < N) & (kk < K), 0),)
        acc = compute_bank(aa, bb, acc, TM, TN, 1)
    for i in tl.static_range(TM):
        for j in tl.static_range(TN):
            for z in tl.static_range(4):
                rr = rbase + i * 16 + lane % 16
                cc = nbase + j * 16 + lane // 16 + z * 4
                tl.store(
                    C + tl.program_id(2) * M * N + rr * N + cc,
                    acc[(i * TN + j) * 4 + z],
                    (rr < M) & (cc < N),
                )
