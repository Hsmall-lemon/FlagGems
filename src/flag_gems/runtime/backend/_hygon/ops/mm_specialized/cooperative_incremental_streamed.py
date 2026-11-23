"""Reduce CPU panels immediately, retaining only the cumulative output tile."""

import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import (
    BlockedLayout,
    SliceLayout,
    SwizzledSharedLayout,
)

from .cooperative_chunk_panel import assembly
from .mm_fp32_cpu_panel import shared_pointer


@triton.constexpr_function
def incremental_reduce(tile, w, count):
    num = tile * tile
    depth = 2
    code = []
    for q in range(num):
        code.append(f"ds_write_b32 ${num+1}, ${num+2+q} offset:{q*w*256}")
    code += [
        "s_waitcnt lgkmcnt(0)",
        "s_barrier",
        f"v_readfirstlane_b32 s32, ${2*num+2}",
        "s_cmp_eq_u32 s32, 0",
        "s_cbranch_scc0 1f",
    ]

    def loads(p):
        return [
            f"ds_read_b32 v{num+q*depth+p%depth}, ${num} offset:{(q*w+p)*256}"
            for q in range(num)
        ]

    for p in range(min(depth, count)):
        code += loads(p)
    for p in range(count):
        pending = min(depth - 1, count - p - 1) * num
        code.append(f"s_waitcnt lgkmcnt({min(15,pending)})")
        for q in range(num):
            code.append(f"v_add_f32 v{q}, v{q}, v{num+q*depth+p%depth}")
        if p + depth < count:
            code += loads(p + depth)
    code += ["1:", "s_waitcnt lgkmcnt(0)", "s_barrier"]
    cons = (
        [f"=&{{v{i}}}" for i in range(num)]
        + ["v"] * (3 + num)
        + [str(i) for i in range(num)]
        + [f"~{{v{i}}}" for i in range(num, num + num * depth)]
        + ["~{s32}", "~{scc}", "~{memory}"]
    )
    return "\n".join(code), ",".join(cons)


@g.jit
def reduce_group(
    readp,
    writep,
    part,
    previous,
    warp,
    TILE: gl.constexpr,
    W: gl.constexpr,
    COUNT: gl.constexpr,
):
    spec: gl.constexpr = incremental_reduce(TILE, W, COUNT)
    return gl.inline_asm_elementwise(
        spec[0],
        spec[1],
        (readp, writep) + part + (warp,) + previous,
        dtype=(gl.float32,) * (TILE * TILE),
        is_pure=False,
        pack=1,
    )


@g.jit
def compute_group(
    ap,
    bp,
    zeros,
    TILE: gl.constexpr,
    DEPTH: gl.constexpr,
    K: gl.constexpr,
    W: gl.constexpr,
    GROUP: gl.constexpr,
):
    LENGTH: gl.constexpr = (
        (K % 112)
        if (
            K % 112
            and triton.cdiv(K, 112) % W == 1
            and GROUP == triton.cdiv(triton.cdiv(K, 112), W) - 1
        )
        else 112
    )
    SPEC: gl.constexpr = assembly(TILE, DEPTH, LENGTH)
    return gl.inline_asm_elementwise(
        SPEC[0],
        SPEC[1],
        ap + bp + zeros,
        dtype=(gl.float32,) * (TILE * TILE),
        is_pure=False,
        pack=1,
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
    CHUNK: gl.constexpr = 16,
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
    zeros = ()
    for i in gl.static_range(TILE * TILE):
        zeros += (gl.full((W * 64,), 0, gl.float32, LL),)
    spec: gl.constexpr = assembly(TILE, DEPTH, CHUNK, CHUNK)
    total = zeros
    if CHUNK >= 64:
        readp = shared_pointer(sa, gl.full((W * 64,), 0, gl.int32, LL), lane).to(
            gl.uint32
        )
    else:
        sr = gl.allocate_shared_memory(
            gl.float32, (TILE * TILE * W, 64), SwizzledSharedLayout(1, 1, 1, [1, 0])
        )
        readp = shared_pointer(sr, gl.full((W * 64,), 0, gl.int32, LL), lane).to(
            gl.uint32
        )
    writep = readp + warp * 256
    for group in gl.static_range(triton.cdiv(triton.cdiv(K, 112), W)):
        pi = p[:, None, None] + group * W
        panel = gl.where(pi == 0, K // 112, pi - 1) if K % 112 else pi
        part = zeros
        for chunk in gl.static_range(triton.cdiv(112, CHUNK)):
            kval = panel * 112 + chunk * CHUNK + kk[None, None, :]
            active = (
                (kval < K)
                & (chunk * CHUNK + kk[None, None, :] < 112)
                & (pi < triton.cdiv(K, 112))
            )
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
            part = gl.inline_asm_elementwise(
                spec[0],
                spec[1],
                ap + bp + part,
                dtype=(gl.float32,) * (TILE * TILE),
                is_pure=False,
                pack=1,
            )
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
                C + rr * N + cc, total[i * TILE + j], (rr < M) & (cc < N) & (warp == 0)
            )
