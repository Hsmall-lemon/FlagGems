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


"""One launch for non-fused panel interiors and whole-K CPU row edges."""

import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import (
    BlockedLayout,
    SliceLayout,
    SwizzledSharedLayout,
)

from .mm_fp32_cpu_cooperative_short import compute_group
from .mm_fp32_cpu_cooperative_vector import vector_reduce_asm
from .mm_fp32_cpu_fixed8_gemv import parts
from .mm_fp32_cpu_panel import shared_pointer


@g.jit
def row_edge(rr, M: gl.constexpr, MT: gl.constexpr):
    BASE: gl.constexpr = M // MT
    REM: gl.constexpr = M % MT
    split = (BASE + 1) * REM
    part = gl.where(rr < split, rr // (BASE + 1), REM + (rr - split) // BASE)
    begin = gl.where(part < REM, part * (BASE + 1), split + (part - REM) * BASE)
    length = BASE + (part < REM).to(gl.int32)
    return rr >= begin + (length // 4) * 4


@g.jit
def edges(
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
    TILE: gl.constexpr,
    W: gl.constexpr,
    STEP: gl.constexpr,
    MT: gl.constexpr,
    PART,
    PN,
):
    BS: gl.constexpr = 8 * TILE
    L: gl.constexpr = BlockedLayout([1], [64], [W], [0])
    LP: gl.constexpr = BlockedLayout([1, STEP], [64, 1], [W, 1], [0, 1])
    x = gl.arange(0, W * 64, layout=L)
    warp, lane = x // 64, x % 64
    begin = PART * (M // MT) + gl.minimum(PART, M % MT)
    length = M // MT + (PART < M % MT).to(gl.int32)
    rr = begin + (length // 4) * 4 + warp
    cc = PN * BS + lane // 4
    chain = lane % 4
    valid = (rr < begin + length) & (cc < N) & (lane // 4 < BS)
    acc = gl.full((W * 64,), 0, gl.float32, L)
    for i in gl.static_range(4):
        aprefix = gl.load(A + rr * AM + i * AK, valid, 0)
        bprefix = gl.load(B + i * BK + cc * BN, valid, 0)
        acc = acc + aprefix * bprefix
    acc = gl.where(chain == 0, acc, 0.0)
    kr = gl.arange(0, STEP, layout=SliceLayout(0, LP))
    r = gl.convert_layout(rr, SliceLayout(1, LP))
    c = gl.convert_layout(cc, SliceLayout(1, LP))
    ch = gl.convert_layout(chain, SliceLayout(1, LP))
    active = gl.convert_layout(valid, SliceLayout(1, LP))
    END: gl.constexpr = 4 + ((K - 4) // 4) * 4
    for block in range(triton.cdiv((K - 4) // 4, STEP)):
        kval = 4 + (block * STEP + kr[None, :]) * 4 + ch[:, None]
        mask = active[:, None] & (kval < END)
        av = gl.load(A + r[:, None] * AM + kval * AK, mask, 0)
        bv = gl.load(B + kval * BK + c[:, None] * BN, mask, 0)
        prod = parts(av * bv, W * 64, STEP)
        for i in gl.static_range(STEP):
            acc = acc + gl.convert_layout(prod[i], L)
    mat = gl.reshape(acc, (W * 16, 4))
    ML: gl.constexpr = mat.type.layout
    ci = gl.arange(0, 4, layout=SliceLayout(0, ML))
    sums = ()
    for i in gl.static_range(4):
        sums += (gl.sum(gl.where(ci[None, :] == i, mat, 0.0), axis=1),)
    value = (sums[0] + sums[2]) + (sums[1] + sums[3])
    ix = gl.arange(0, W * 16, layout=value.type.layout)
    r2 = begin + (length // 4) * 4 + ix // 16
    c2 = PN * BS + ix % 16
    mask = (r2 < begin + length) & (c2 < N) & (ix % 16 < BS)
    for i in gl.static_range(END, K):
        av = gl.load(A + r2 * AM + i * AK, mask, 0)
        bv = gl.load(B + i * BK + c2 * BN, mask, 0)
        value = value + av * bv
    gl.store(C + r2 * N + c2, value, mask)


@g.jit
def mainbody(
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
    TILE: gl.constexpr,
    W: gl.constexpr,
    DEPTH: gl.constexpr,
    MT: gl.constexpr,
    PM,
    PN,
):
    BS: gl.constexpr = 8 * TILE
    LL: gl.constexpr = BlockedLayout([1], [64], [W], [0])
    x = gl.arange(0, W * 64, layout=LL)
    lane, warp = x % 64, x // 64
    row0, col0 = PM * BS, PN * BS
    L: gl.constexpr = BlockedLayout([1, 1, 4], [1, 8, 8], [W, 1, 1], [2, 1, 0])
    p = gl.arange(0, W, layout=SliceLayout(1, SliceLayout(2, L)))
    r = row0 + gl.arange(0, BS, layout=SliceLayout(0, SliceLayout(2, L)))
    c = col0 + gl.arange(0, BS, layout=SliceLayout(0, SliceLayout(2, L)))
    kk = gl.arange(0, 128, layout=SliceLayout(0, SliceLayout(1, L)))
    sa = gl.allocate_shared_memory(
        gl.float32, (W * BS, 128), SwizzledSharedLayout(4, 1, 8, [1, 0])
    )
    sb = gl.allocate_shared_memory(
        gl.float32, (W * BS, 128), SwizzledSharedLayout(4, 1, 8, [1, 0])
    )
    ap = ()
    bp = ()
    for group in gl.static_range(8):
        kvals = gl.full((W * 64,), group * 4, gl.int32, LL)
        ap += (shared_pointer(sa, warp * BS + lane // 8, kvals).to(gl.uint32),)
        bp += (shared_pointer(sb, warp * BS + lane % 8, kvals).to(gl.uint32),)
    zeros = ()
    for i in gl.static_range(TILE * TILE):
        zeros += (gl.full((W * 64,), 0, gl.float32, LL),)
    all_parts = ()
    for group in gl.static_range(triton.cdiv(triton.cdiv(K, 112), W)):
        kval = (p[:, None, None] + group * W) * 112 + kk[None, None, :]
        active = (kval < K) & (kk[None, None, :] < 112)
        av = gl.load(
            A + r[None, :, None] * AM + kval * AK, (r[None, :, None] < M) & active, 0
        )
        bv = gl.load(
            B + kval * BK + c[None, :, None] * BN, (c[None, :, None] < N) & active, 0
        )
        sa.store(gl.reshape(av, (W * BS, 128)))
        sb.store(gl.reshape(bv, (W * BS, 128)))
        all_parts += compute_group(ap, bp, zeros, TILE, DEPTH, K, W, group)
    readp = shared_pointer(sa, gl.full((W * 64,), 0, gl.int32, LL), lane).to(gl.uint32)
    writep = readp + warp * 256
    gl.static_assert(TILE * triton.cdiv(triton.cdiv(K, 112), W) <= 16)
    spec2: gl.constexpr = vector_reduce_asm(K, W, TILE)
    total = gl.inline_asm_elementwise(
        spec2[0],
        spec2[1],
        (readp, writep) + all_parts,
        dtype=(gl.float32,) * (TILE * TILE),
        is_pure=False,
        pack=1,
    )
    for i in gl.static_range(TILE):
        for j in gl.static_range(TILE):
            rr, cc = row0 + lane // 8 + i * 8, col0 + lane % 8 + j * 8
            gl.store(
                C + rr * N + cc,
                total[i * TILE + j],
                (rr < M) & (cc < N) & (warp == 0) & (~row_edge(rr, M, MT)),
            )


@g.jit
def matmul(
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
    TILE: gl.constexpr,
    W: gl.constexpr,
    DEPTH: gl.constexpr,
    MT: gl.constexpr,
    STEP: gl.constexpr,
):
    pm, pn = gl.program_id(0), gl.program_id(1)
    if pm < triton.cdiv(M, 8 * TILE):
        mainbody(A, B, C, M, N, K, AM, AK, BK, BN, TILE, W, DEPTH, MT, pm, pn)
    else:
        edges(
            A,
            B,
            C,
            M,
            N,
            K,
            AM,
            AK,
            BK,
            BN,
            TILE,
            W,
            STEP,
            MT,
            pm - triton.cdiv(M, 8 * TILE),
            pn,
        )
