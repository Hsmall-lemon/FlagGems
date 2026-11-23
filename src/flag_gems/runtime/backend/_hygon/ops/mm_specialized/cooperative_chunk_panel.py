"""Cooperating CPU panels with smaller LDS stages to improve occupancy."""

import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import (
    BlockedLayout,
    SliceLayout,
    SwizzledSharedLayout,
)

from .cooperative_vector_reduce import vector_reduce_asm
from .mm_fp32_cpu_panel import shared_pointer


@triton.constexpr_function
def assembly(tile, depth, length, chunk):
    count = tile * tile
    scratch = count
    groups = triton.cdiv(length, 4)
    code = ["s_waitcnt lgkmcnt(0)", "s_barrier"]

    def loads(group):
        slot = group % depth
        start = scratch + slot * 8 * tile
        out = []
        for i in range(tile):
            a, b = start + i * 4, start + 4 * tile + i * 4
            offset = (i * 8 * chunk + group // 8 * 32) * 4
            out += [
                f"ds_read_b128 v[{a}:{a+3}], ${count+group%8} offset:{offset}",
                f"ds_read_b128 v[{b}:{b+3}], ${count+8+group%8} offset:{offset}",
            ]
        return out

    for group in range(min(groups, depth)):
        code += loads(group)
    for group in range(groups):
        start = scratch + (group % depth) * 8 * tile
        code += [f"s_waitcnt lgkmcnt({min(15,2*tile*min(depth-1,groups-group-1))})"]
        for k in range(min(4, length - group * 4)):
            for i in range(tile):
                for j in range(tile):
                    code += [
                        f"v_mac_f32 v{i*tile+j}, v{start+4*i+k}, v{start+4*tile+4*j+k}"
                    ]
        if group + depth < groups:
            code += loads(group + depth)
    code += ["s_barrier"]
    cons = (
        [f"=&{{v{i}}}" for i in range(count)]
        + ["v"] * 16
        + [str(i) for i in range(count)]
        + [f"~{{v{i}}}" for i in range(scratch, scratch + 8 * tile * depth)]
        + ["~{memory}"]
    )
    return "\n".join(code), ",".join(cons)


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
    CHUNK: gl.constexpr,
    SPLIT: gl.constexpr = False,
):
    BS: gl.constexpr = 8 * TILE
    LL: gl.constexpr = BlockedLayout([1], [64], [W], [0])
    x = gl.arange(0, W * 64, layout=LL)
    lane, warp = x % 64, x // 64
    row0, col0 = gl.program_id(0) * BS, gl.program_id(1) * BS
    L: gl.constexpr = BlockedLayout([1, 1, 4], [1, 8, 8], [W, 1, 1], [2, 1, 0])
    p = gl.arange(0, W, layout=SliceLayout(1, SliceLayout(2, L)))
    r = row0 + gl.arange(0, BS, layout=SliceLayout(0, SliceLayout(2, L)))
    c = col0 + gl.arange(0, BS, layout=SliceLayout(0, SliceLayout(2, L)))
    kk = gl.arange(0, CHUNK, layout=SliceLayout(0, SliceLayout(1, L)))
    sa = gl.allocate_shared_memory(
        gl.float32, (W * BS, CHUNK), SwizzledSharedLayout(4, 1, 8, [1, 0])
    )
    sb = gl.allocate_shared_memory(
        gl.float32, (W * BS, CHUNK), SwizzledSharedLayout(4, 1, 8, [1, 0])
    )
    ap = ()
    bp = ()
    for group in gl.static_range(8):
        kvals = gl.full((W * 64,), group * 4, gl.int32, LL)
        ap += (shared_pointer(sa, warp * BS + lane // 8, kvals).to(gl.uint32),)
        bp += (shared_pointer(sb, warp * BS + lane % 8, kvals).to(gl.uint32),)
    zero = ()
    for i in gl.static_range(TILE * TILE):
        zero += (gl.full((W * 64,), 0, gl.float32, LL),)
    spec: gl.constexpr = assembly(TILE, DEPTH, CHUNK, CHUNK)
    all_parts = ()
    for group in gl.static_range(1 if SPLIT else triton.cdiv(triton.cdiv(K, 112), W)):
        partial = zero
        for chunk in gl.static_range(triton.cdiv(112, CHUNK)):
            kval = (
                (p[:, None, None] + (gl.program_id(2) * W if SPLIT else group * W))
                * 112
                + chunk * CHUNK
                + kk[None, None, :]
            )
            active = (kval < K) & (chunk * CHUNK + kk[None, None, :] < 112)
            av = gl.load(
                A + r[None, :, None] * AM + kval * AK,
                (r[None, :, None] < M) & active,
                0,
            )
            bv = gl.load(
                B + kval * BK + c[None, :, None] * BN,
                (c[None, :, None] < N) & active,
                0,
            )
            sa.store(gl.reshape(av, (W * BS, CHUNK)))
            sb.store(gl.reshape(bv, (W * BS, CHUNK)))
            partial = gl.inline_asm_elementwise(
                spec[0],
                spec[1],
                ap + bp + partial,
                dtype=(gl.float32,) * (TILE * TILE),
                is_pure=False,
                pack=1,
            )
        all_parts += partial
    if SPLIT:
        panel = gl.program_id(2) * W + warp
        for i in gl.static_range(TILE):
            for j in gl.static_range(TILE):
                rr, cc = row0 + lane // 8 + i * 8, col0 + lane % 8 + j * 8
                gl.store(
                    C + panel * M * N + rr * N + cc,
                    all_parts[i * TILE + j],
                    (rr < M) & (cc < N) & (panel < triton.cdiv(K, 112)),
                )
    else:
        SIZE: gl.constexpr = triton.next_power_of_2(
            TILE * TILE * triton.cdiv(triton.cdiv(K, 112), W) * W * 64
        )
        sr = gl.allocate_shared_memory(
            gl.float32, (SIZE // 64, 64), SwizzledSharedLayout(1, 1, 1, [1, 0])
        )
        readp = shared_pointer(sr, gl.full((W * 64,), 0, gl.int32, LL), lane).to(
            gl.uint32
        )
        writep = readp + warp * 256
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
                    (rr < M) & (cc < N) & (warp == 0),
                )
