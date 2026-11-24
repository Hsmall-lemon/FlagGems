"""Experimental exact CPU-panel SIMT: redundant direct global loads, no LDS.

Each wave owns a tile and one 112-term CPU panel. Duplicate operand addresses
within the wave are intentional: global coalescing supplies the broadcast,
removing LDS staging/bpermute. Four-term loads are double buffered.
"""

import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SwizzledSharedLayout

from .cooperative_incremental_streamed import reduce_group
from .mm_fp32_cpu_panel import shared_pointer


@triton.constexpr_function
def assembly(tile, k, depth, ak, bk):
    count = tile * tile
    ga = (count + 3) // 4 * 4
    ao = ga + depth * 2 * tile * 4
    bo = ao + tile
    ki = bo + tile
    ia, ib = count, count + 1
    iao, ibo, karg = count + 2, count + 2 + tile, count + 2 + 2 * tile
    code = [f"s_mov_b64 s[32:33], ${ia}", f"s_mov_b64 s[34:35], ${ib}"]
    for i in range(tile):
        code += [f"v_mov_b32 v{ao+i}, ${iao+i}", f"v_mov_b32 v{bo+i}, ${ibo+i}"]
    code += [
        f"v_readfirstlane_b32 s44, ${karg}",
        f"s_cmp_ge_u32 s44, {k}",
        "s_cbranch_scc1 3f",
    ]

    def nload(q):
        return tile * (
            (1 if ak == 1 and q == 4 else q) + (1 if bk == 1 and q == 4 else q)
        )

    def prefetch(chunk, length):
        q = min(4, length - chunk * 4)
        slot = chunk % depth
        out = []
        for i in range(tile):
            ar = ga + slot * 2 * tile * 4 + i * 4
            br = ga + slot * 2 * tile * 4 + tile * 4 + i * 4
            for reg, offset, stride, base in [
                (ar, ao + i, ak, 32),
                (br, bo + i, bk, 34),
            ]:
                if stride == 1 and q == 4:
                    out += [
                        f"global_load_dwordx4 v[{reg}:{reg+3}], v{offset}, s[{base}:{base+1}] offset:{chunk*16}"
                    ]
                else:
                    for j in range(q):
                        byte = (chunk * 4 + j) * stride * 4
                        if byte <= 4095:
                            out += [
                                f"global_load_dword v{reg+j}, v{offset}, s[{base}:{base+1}] offset:{byte}"
                            ]
                        else:
                            out += [
                                f"v_add_u32 v{ki}, {byte}, v{offset}",
                                f"global_load_dword v{reg+j}, v{ki}, s[{base}:{base+1}]",
                            ]
        return out

    def pipeline(length):
        out = []
        chunks = (length + 3) // 4
        for chunk in range(min(depth, chunks)):
            out += prefetch(chunk, length)
        for chunk in range(chunks):
            remain = 0
            for pending in range(chunk + 1, min(chunk + depth, chunks)):
                remain += nload(min(4, length - pending * 4))
            out += [f"s_waitcnt vmcnt({min(remain,63)})"]
            slot = chunk % depth
            for q in range(min(4, length - chunk * 4)):
                for i in range(tile):
                    for j in range(tile):
                        ar = ga + slot * 2 * tile * 4 + i * 4 + q
                        br = ga + slot * 2 * tile * 4 + tile * 4 + j * 4 + q
                        out += [f"v_mac_f32 v{i*tile+j}, v{ar}, v{br}"]
            if chunk + depth < chunks:
                out += prefetch(chunk + depth, length)
        out += ["s_waitcnt vmcnt(0)"]
        return out

    if k % 112:
        code += [f"s_cmp_eq_u32 s44, {k//112*112}", "s_cbranch_scc1 1f"]
    if k >= 112:
        code += pipeline(112)
    code += ["s_branch 3f"]
    if k % 112:
        code += ["1:"] + pipeline(k % 112)
    code += ["3:"]
    cons = (
        [f"=&{{v{i}}}" for i in range(count)]
        + ["s", "s"]
        + ["v"] * (2 * tile + 1)
        + [str(i) for i in range(count)]
        + [f"~{{v{i}}}" for i in range(ga, ki + 1)]
        + [f"~{{s{i}}}" for i in [32, 33, 34, 35, 44]]
        + ["~{scc}", "~{memory}"]
    )
    return "\n".join(code), ",".join(cons)


@g.jit
def consume(
    A,
    B,
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
    row0,
    col0,
    panel,
):
    L: gl.constexpr = BlockedLayout([1], [64], [W], [0])
    x = gl.arange(0, W * 64, layout=L)
    lane = x % 64
    kval = panel * 112
    aa = ()
    bb = ()
    zeros = ()
    for i in gl.static_range(TILE):
        aa += ((gl.minimum(row0 + lane // 8 + i * 8, M - 1) * AM + kval * AK) * 4,)
        bb += ((gl.minimum(col0 + lane % 8 + i * 8, N - 1) * BN + kval * BK) * 4,)
    for i in gl.static_range(TILE * TILE):
        zeros += (gl.full((W * 64,), 0, gl.float32, L),)
    spec: gl.constexpr = assembly(TILE, K, DEPTH, AK, BK)
    return gl.inline_asm_elementwise(
        spec[0],
        spec[1],
        (A.to(gl.uint64), B.to(gl.uint64)) + aa + bb + (kval,) + zeros,
        dtype=(gl.float32,) * (TILE * TILE),
        is_pure=False,
        pack=1,
    )


@g.jit
def panels(
    A,
    B,
    P,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    AM: gl.constexpr,
    AK: gl.constexpr,
    BK: gl.constexpr,
    BN: gl.constexpr,
    TILE: gl.constexpr = 2,
    W: gl.constexpr = 4,
    DEPTH: gl.constexpr = 2,
):
    gl.static_assert(AK > 0 and BK > 0 and M > 0 and N > 0)
    gl.static_assert(DEPTH == 1 or DEPTH == 2 or DEPTH == 3)
    L: gl.constexpr = BlockedLayout([1], [64], [W], [0])
    x = gl.arange(0, W * 64, layout=L)
    lane, warp = x % 64, x // 64
    row0, col0 = gl.program_id(0) * 8 * TILE, gl.program_id(1) * 8 * TILE
    panel = gl.program_id(2) * W + warp
    value = consume(A, B, M, N, K, AM, AK, BK, BN, TILE, W, DEPTH, row0, col0, panel)
    for i in gl.static_range(TILE):
        for j in gl.static_range(TILE):
            rr, cc = row0 + lane // 8 + i * 8, col0 + lane % 8 + j * 8
            gl.store(
                P + panel * M * N + rr * N + cc,
                value[i * TILE + j],
                (panel < triton.cdiv(K, 112)) & (rr < M) & (cc < N),
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
    TILE: gl.constexpr = 2,
    W: gl.constexpr = 4,
    DEPTH: gl.constexpr = 2,
):
    gl.static_assert(AK > 0 and BK > 0 and M > 0 and N > 0)
    gl.static_assert(DEPTH == 1 or DEPTH == 2 or DEPTH == 3)
    L: gl.constexpr = BlockedLayout([1], [64], [W], [0])
    x = gl.arange(0, W * 64, layout=L)
    lane, warp = x % 64, x // 64
    row0, col0 = gl.program_id(0) * 8 * TILE, gl.program_id(1) * 8 * TILE
    if W > 1:
        WORDS: gl.constexpr = TILE * TILE * W * 64
        shared = gl.allocate_shared_memory(
            gl.float32, (1, WORDS), SwizzledSharedLayout(1, 1, 1, [1, 0])
        )
        z = gl.full((W * 64,), 0, gl.int32, L)
        base = shared_pointer(shared, z, z).to(gl.uint32)
        readp = base + lane * 4
        writep = readp + warp * 256
    total = ()
    for i in gl.static_range(TILE * TILE):
        total += (gl.full((W * 64,), 0, gl.float32, L),)
    for group in gl.static_range(triton.cdiv(triton.cdiv(K, 112), W)):
        pi = warp + group * W
        panel = gl.where(pi == 0, K // 112, pi - 1) if K % 112 else pi
        panel = gl.where(pi < triton.cdiv(K, 112), panel, triton.cdiv(K, 112))
        part = consume(A, B, M, N, K, AM, AK, BK, BN, TILE, W, DEPTH, row0, col0, panel)
        if W == 1:
            for i in gl.static_range(TILE * TILE):
                total = total[:i] + (total[i] + part[i],) + total[i + 1 :]
        else:
            total = reduce_group(
                readp,
                writep,
                part,
                total,
                warp,
                TILE,
                W,
                min(W, triton.cdiv(K, 112) - group * W),
            )
    for i in gl.static_range(TILE):
        for j in gl.static_range(TILE):
            rr, cc = row0 + lane // 8 + i * 8, col0 + lane % 8 + j * 8
            gl.store(
                C + rr * N + cc, total[i * TILE + j], (warp == 0) & (rr < M) & (cc < N)
            )
