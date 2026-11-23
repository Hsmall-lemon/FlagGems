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


@triton.jit
def fp32_panels(
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
    TM: tl.constexpr,
    TN: tl.constexpr,
    TK: tl.constexpr,
    SPLIT: tl.constexpr,
    REVERSE: tl.constexpr,
    DIRECT: tl.constexpr,
):
    i = tl.program_id(0) * TM + tl.arange(0, TM)
    j = tl.program_id(1) * TN + tl.arange(0, TN)
    ki = tl.arange(0, TK)
    acc0 = tl.zeros((TM, TN), tl.float32)
    acc1 = tl.zeros((TM, TN), tl.float32)
    for block in range(triton.cdiv(SPLIT, TK)):
        q = block * TK + ki
        av = tl.load(
            A + i[:, None] * AM + q[None, :] * AK,
            (i[:, None] < M) & (q[None, :] < SPLIT),
            0,
        )
        bv = tl.load(
            B + q[:, None] * BK + j[None, :] * BN,
            (j[None, :] < N) & (q[:, None] < SPLIT),
            0,
        )
        acc0 = tl.dot(av, bv, acc0, input_precision="ieee")
    for block in range(triton.cdiv(K - SPLIT, TK)):
        q = SPLIT + block * TK + ki
        if REVERSE:
            q = K - 1 - (block * TK + ki)
            valid = q >= SPLIT
        else:
            valid = q < K
        av = tl.load(
            A + i[:, None] * AM + q[None, :] * AK, (i[:, None] < M) & valid[None, :], 0
        )
        bv = tl.load(
            B + q[:, None] * BK + j[None, :] * BN, valid[:, None] & (j[None, :] < N), 0
        )
        acc1 = tl.dot(av, bv, acc1, input_precision="ieee")
    if DIRECT:
        tl.store(
            C + i[:, None] * N + j[None, :],
            acc1 + acc0,
            (i[:, None] < M) & (j[None, :] < N),
        )
    else:
        tl.store(
            C + j[None, :] * M + i[:, None],
            acc1 + acc0,
            (i[:, None] < M) & (j[None, :] < N),
        )


@triton.jit
def split_chain_steps(
    values, BLOCK: tl.constexpr, CHAINS: tl.constexpr, STEPS: tl.constexpr
):
    chunks = (values,)
    for level in tl.static_range(STEPS.bit_length() - 1):
        items = ()
        for i in tl.static_range(2**level):
            value = tl.reshape(
                chunks[i], (BLOCK, STEPS // (2 ** (level + 1)), 2, CHAINS)
            )
            left, right = tl.split(tl.permute(value, (0, 1, 3, 2)))
            items += (left, right)
        chunks = items
    return chunks


@triton.jit
def edge_chain_gemv(
    A,
    B,
    C,
    K: tl.constexpr,
    N: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BN: tl.constexpr,
    START: tl.constexpr,
    SEGMENTS: tl.constexpr,
    SPECIAL: tl.constexpr,
    BLOCK: tl.constexpr,
    CHUNK: tl.constexpr,
    PACKED: tl.constexpr,
    OFFSET: tl.constexpr = 0,
    TILE=None,
    EDGE_ROW=None,
):
    row = START + (tl.program_id(1) if EDGE_ROW is None else EDGE_ROW)
    x = ((tl.program_id(0) if TILE is None else TILE) - OFFSET) * BLOCK + tl.arange(
        0, BLOCK
    )
    col = x
    if SPECIAL:
        for i in tl.static_range(len(SEGMENTS)):
            col = tl.where(x == i, SEGMENTS[i], col)
        valid = x < len(SEGMENTS)
        PREFIX: tl.constexpr = 1
        CHAINS: tl.constexpr = 8
    else:
        valid = x < N
        for special_col in tl.static_range(len(SEGMENTS)):
            valid = valid & (x != SEGMENTS[special_col])
        PREFIX: tl.constexpr = 4
        CHAINS: tl.constexpr = 4
    lane = tl.arange(0, CHAINS)
    initial = tl.full((BLOCK,), 0, tl.float32)
    for p in tl.static_range(PREFIX):
        initial = initial + tl.load(B + p + col * BN, valid, 0) * tl.load(
            A + row * AM + p * AK
        )
    acc = tl.where(lane[None, :] == 0, initial[:, None], 0.0)
    LOOPS: tl.constexpr = (K - PREFIX) // CHAINS
    for block in tl.range(
        triton.cdiv(LOOPS, CHUNK), num_stages=2, loop_unroll_factor=2
    ):
        if PACKED:
            pos = block * CHUNK * CHAINS + tl.arange(0, CHUNK * CHAINS)
            q = PREFIX + pos
            mask = pos < LOOPS * CHAINS
            a = tl.load(A + row * AM + q * AK, mask, 0)
            b = tl.load(
                B + col[:, None] * BN + q[None, :], valid[:, None] & mask[None, :], 0
            )
            chunks = split_chain_steps(b * a[None, :], BLOCK, CHAINS, CHUNK)
            tl.static_assert(CHUNK == 8)
            for i in tl.static_range(8):
                acc = acc + tl.reshape(
                    chunks[(0, 4, 2, 6, 1, 5, 3, 7)[i]], (BLOCK, CHAINS)
                )
        else:
            for i in tl.static_range(CHUNK):
                group = block * CHUNK + i
                q = PREFIX + group * CHAINS + lane
                a = tl.load(A + row * AM + q * AK, group < LOOPS, 0)
                b = tl.load(
                    B + col[:, None] * BN + q[None, :],
                    valid[:, None] & (group < LOOPS),
                    0,
                )
                acc = acc + b * a[None, :]
    vals = ()
    for i in tl.static_range(CHAINS):
        vals += (tl.sum(tl.where(lane[None, :] == i, acc, 0.0), 1),)
    if SPECIAL:
        total = vals[0]
        for i in tl.static_range(1, CHAINS):
            total = total + vals[i]
    else:
        total = (vals[0] + vals[2]) + (vals[1] + vals[3])
    for q in tl.static_range(PREFIX + LOOPS * CHAINS, K):
        total = total + tl.load(B + q + col * BN, valid, 0) * tl.load(
            A + row * AM + q * AK
        )
    tl.store(C + row * N + col, total, valid)


@triton.jit
def transposed_tail_sum(
    a,
    b,
    partial,
    out,
    m: tl.constexpr,
    k: tl.constexpr,
    n: tl.constexpr,
    am: tl.constexpr,
    ak: tl.constexpr,
    bn: tl.constexpr,
    parts: tl.constexpr,
    cols: tl.constexpr,
    rows: tl.constexpr,
    block: tl.constexpr,
    special_block: tl.constexpr,
    sum_block: tl.constexpr,
):
    regular: tl.constexpr = triton.cdiv(n, block)
    edge_tiles: tl.constexpr = regular + triton.cdiv(len(cols), special_block)
    edge_ctas: tl.constexpr = (m - rows) * edge_tiles
    pid = tl.program_id(0)
    if pid < edge_ctas:
        tile, row = pid % edge_tiles, pid // edge_tiles
        if tile < regular:
            edge_chain_gemv(
                a,
                b,
                out,
                k,
                n,
                am,
                ak,
                bn,
                rows,
                cols,
                False,
                block,
                8,
                False,
                0,
                tile,
                row,
            )
        else:
            edge_chain_gemv(
                a,
                b,
                out,
                k,
                n,
                am,
                ak,
                bn,
                rows,
                cols,
                True,
                special_block,
                8,
                False,
                regular,
                tile,
                row,
            )
    else:
        x = (pid - edge_ctas) * sum_block + tl.arange(0, sum_block)
        value = tl.full((sum_block,), 0, tl.float32)
        for part in tl.static_range(parts):
            value += tl.load(partial + part * m * n + x, x < rows * n, 0)
        tl.store(out + x, value, x < rows * n)


@triton.jit
def cpu112_dot(
    A,
    B,
    I,
    J,
    OFFSET,
    M: tl.constexpr,
    N: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    TM: tl.constexpr,
    TN: tl.constexpr,
    TK: tl.constexpr,
    LENGTH: tl.constexpr,
    MIXED: tl.constexpr,
):
    acc = tl.full((TM, TN), 0, tl.float32)
    if MIXED:
        low = tl.full((TM, TN), 0, tl.float32)
    for block in tl.static_range(triton.cdiv(LENGTH, TK)):
        r = block * TK + tl.arange(0, TK)
        q = OFFSET + r
        a = tl.load(
            A + I[:, None] * AM + q[None, :] * AK,
            (I[:, None] < M) & (r[None, :] < LENGTH),
            0,
        )
        b = tl.load(
            B + q[:, None] * BK + J[None, :] * BN,
            (r[:, None] < LENGTH) & (J[None, :] < N),
            0,
        )
        if MIXED:
            ah, bh = a.to(tl.float16), b.to(tl.float16)
            al = ((a - ah.to(tl.float32)) * 2048.0).to(tl.float16)
            bl = ((b - bh.to(tl.float32)) * 2048.0).to(tl.float16)
            acc = tl.dot(ah, bh, acc)
            low = tl.dot(al, bh, low)
            low = tl.dot(ah, bl, low)
        else:
            acc = tl.dot(a, b, acc, input_precision="ieee")
    if MIXED:
        acc += low * 0.00048828125
    return acc


@triton.jit
def cpu112_panels(
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
    TM: tl.constexpr,
    TN: tl.constexpr,
    TK: tl.constexpr,
    PARALLEL: tl.constexpr,
    PEEL: tl.constexpr,
    PANEL: tl.constexpr = 112,
    MIXED: tl.constexpr = False,
    TRANSPOSE: tl.constexpr = False,
    GROUP: tl.constexpr = 0,
    SUM_MODE: tl.constexpr = 0,
):
    if GROUP:
        tiles_m: tl.constexpr = triton.cdiv(M, TM)
        tiles_n: tl.constexpr = triton.cdiv(N, TN)
        pid = tl.program_id(0)
        first_m = pid // (GROUP * tiles_n) * GROUP
        group_size = tl.minimum(tiles_m - first_m, GROUP)
        local = pid % (GROUP * tiles_n)
        mi = first_m + local % group_size
        ni = local // group_size
    else:
        mi, ni = tl.program_id(0), tl.program_id(1)
    i = mi * TM + tl.arange(0, TM)
    j = ni * TN + tl.arange(0, TN)
    z = tl.arange(0, TK)
    if PEEL:
        total = tl.full((TM, TN), 0, tl.float32)
        HAS_TAIL: tl.constexpr = K % PANEL != 0
        if PARALLEL:
            part = tl.program_id(2)
            C += part * M * N
            if HAS_TAIL and part == 0:
                total = cpu112_dot(
                    A,
                    B,
                    i,
                    j,
                    K // PANEL * PANEL,
                    M,
                    N,
                    AM,
                    AK,
                    BK,
                    BN,
                    TM,
                    TN,
                    TK,
                    K % PANEL,
                    MIXED,
                )
            else:
                total = cpu112_dot(
                    A,
                    B,
                    i,
                    j,
                    (part - HAS_TAIL) * PANEL,
                    M,
                    N,
                    AM,
                    AK,
                    BK,
                    BN,
                    TM,
                    TN,
                    TK,
                    PANEL,
                    MIXED,
                )
        else:
            if HAS_TAIL:
                total = cpu112_dot(
                    A,
                    B,
                    i,
                    j,
                    K // PANEL * PANEL,
                    M,
                    N,
                    AM,
                    AK,
                    BK,
                    BN,
                    TM,
                    TN,
                    TK,
                    K % PANEL,
                    MIXED,
                )
            correction = tl.full((TM, TN), 0, tl.float32)
            for part in range(K // PANEL):
                index = K // PANEL - 1 - part if SUM_MODE == 2 else part
                acc = cpu112_dot(
                    A,
                    B,
                    i,
                    j,
                    index * PANEL,
                    M,
                    N,
                    AM,
                    AK,
                    BK,
                    BN,
                    TM,
                    TN,
                    TK,
                    PANEL,
                    MIXED,
                )
                if SUM_MODE == 1:
                    value = acc - correction
                    updated = total + value
                    correction = (updated - total) - value
                    total = updated
                else:
                    total = total + acc
        index = (
            j[None, :] * M + i[:, None] if TRANSPOSE else i[:, None] * N + j[None, :]
        )
        tl.store(C + index, total, (i[:, None] < M) & (j[None, :] < N))
        return
    if PARALLEL:
        start = tl.program_id(2)
        stop = start + 1
    else:
        start = 0
        stop = triton.cdiv(K, 112)
    total = tl.full((TM, TN), 0, tl.float32)
    for p in range(start, stop):
        if K % 112:
            offset = tl.where(p == 0, K // 112, p - 1) * 112
        else:
            offset = p * 112
        acc = tl.full((TM, TN), 0, tl.float32)
        for t in range(triton.cdiv(112, TK)):
            r = t * TK + z
            q = offset + r
            valid = (q < K) & (r < 112)
            a = tl.load(
                A + i[:, None] * AM + q[None, :] * AK,
                (i[:, None] < M) & valid[None, :],
                0,
            )
            b = tl.load(
                B + q[:, None] * BK + j[None, :] * BN,
                valid[:, None] & (j[None, :] < N),
                0,
            )
            acc = tl.dot(a, b, acc, input_precision="ieee")
        total = tl.inline_asm_elementwise(
            "v_add_f32 $0, $1, $2",
            "=v,v,v",
            [total, acc],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    if PARALLEL:
        C += start * M * N
    tl.store(
        C + i[:, None] * N + j[None, :], total, (i[:, None] < M) & (j[None, :] < N)
    )


@triton.jit
def cpu112_sum(
    P,
    C,
    SIZE: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
    MODE: tl.constexpr,
):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    result = tl.full((BLOCK,), 0, tl.float32)
    correction = tl.full((BLOCK,), 0, tl.float32)
    for part in range(PARTS):
        index = PARTS - 1 - part if MODE == 2 else part
        value = tl.load(P + index * SIZE + x, x < SIZE, 0)
        if MODE == 1:
            value = value - correction
            updated = result + value
            correction = (updated - result) - value
            result = updated
        else:
            result = result + value
    tl.store(C + x, result, x < SIZE)


@triton.jit
def sum_parts(P, C, SIZE: tl.constexpr, PARTS: tl.constexpr, BLOCK: tl.constexpr):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    p = tl.arange(0, PARTS)
    values = tl.load(P + p[:, None] * SIZE + x[None, :], x[None, :] < SIZE, 0)
    tl.store(C + x, tl.sum(values, axis=0), x < SIZE)


@triton.jit
def tiled_panels(
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
    TM: tl.constexpr,
    TN: tl.constexpr,
    STEP: tl.constexpr,
    TK: tl.constexpr,
    SEPARATE: tl.constexpr,
    PARTS: tl.constexpr,
    GROUP: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    PEEL_TAIL: tl.constexpr = False,
    UNROLL: tl.constexpr = False,
    CACHE_A: tl.constexpr = "",
    CACHE_B: tl.constexpr = "",
    TAIL_K: tl.constexpr = 0,
):
    if GROUP:
        nblocks: tl.constexpr = triton.cdiv(N, TN)
        mblocks: tl.constexpr = triton.cdiv(M, TM)
        pid = tl.program_id(0)
        start_m = pid // (GROUP * nblocks) * GROUP
        width = tl.minimum(GROUP, mblocks - start_m)
        local = pid % (GROUP * nblocks)
        block_m = start_m + local % width
        block_n = local // width
    else:
        block_m = tl.program_id(0)
        block_n = tl.program_id(1)
    i = block_m * TM + tl.arange(0, TM)
    j = block_n * TN + tl.arange(0, TN)
    kk = tl.arange(0, TK)
    total = tl.full((TM, TN), 0, tl.float32)
    if PARTS > 1:
        first = tl.program_id(2)
        last = first + 1
    else:
        first = 0
        last = K // STEP if PEEL_TAIL else triton.cdiv(K, STEP)
    for panel in tl.range(first, last, loop_unroll_factor=4 if UNROLL else 1):
        part = tl.full((TM, TN), 0, tl.float32)
        for block in tl.static_range(STEP // TK):
            q = panel * STEP + block * TK + kk
            am = i[:, None] < M
            bm = j[None, :] < N
            if not PEEL_TAIL and K % (STEP * PARTS):
                am = am & (q[None, :] < K)
                bm = bm & (q[:, None] < K)
            a = tl.load(
                A + i[:, None] * AM + q[None, :] * AK, am, 0, cache_modifier=CACHE_A
            )
            b = tl.load(
                B + q[:, None] * BK + j[None, :] * BN, bm, 0, cache_modifier=CACHE_B
            )
            part = tl.dot(a, b, part, input_precision="ieee")
        if SEPARATE:
            total = tl.inline_asm_elementwise(
                "v_add_f32 $0, $1, $2",
                constraints="=v,v,v",
                args=[total, part],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
        else:
            total += part
    if PEEL_TAIL and K % STEP:
        part = tl.full((TM, TN), 0, tl.float32)
        TAIL_BLOCK: tl.constexpr = TAIL_K if TAIL_K else TK
        for block in tl.static_range(triton.cdiv(K % STEP, TAIL_BLOCK)):
            q = K // STEP * STEP + block * TAIL_BLOCK + tl.arange(0, TAIL_BLOCK)
            a = tl.load(
                A + i[:, None] * AM + q[None, :] * AK,
                (i[:, None] < M) & (q[None, :] < K),
                0,
            )
            b = tl.load(
                B + q[:, None] * BK + j[None, :] * BN,
                (j[None, :] < N) & (q[:, None] < K),
                0,
            )
            part = tl.dot(a, b, part, input_precision="ieee")
        total = total + part
    if PARTS > 1:
        C += first * M * N
    if TRANSPOSE:
        positions = j[None, :] * M + i[:, None]
    else:
        positions = i[:, None] * N + j[None, :]
    tl.store(C + positions, total, (i[:, None] < M) & (j[None, :] < N))
