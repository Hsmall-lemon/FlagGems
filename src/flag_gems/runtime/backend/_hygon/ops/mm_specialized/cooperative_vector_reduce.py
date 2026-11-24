"""Cooperating complete CPU K panels with one LDS stage per workgroup."""

import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import (
    BlockedLayout,
    SliceLayout,
    SwizzledSharedLayout,
)

from .cpu_simt_full_panel import assembly
from .mm_fp32_cpu_panel import shared_pointer


@triton.constexpr_function
def vector_reduce_asm(k, w, tile):
    count = triton.cdiv(k, 112)
    groups = triton.cdiv(count, w)
    num = tile * tile
    readp, writep = num, num + 1
    code = []
    for qi in range(num):
        for group in range(groups):
            code.append(
                f"ds_write_b32 ${writep}, ${num+2+group*num+qi} offset:{(qi*groups*w+group*w)*256}"
            )
    code += ["s_waitcnt lgkmcnt(0)", "s_barrier"]
    for qi in range(num):
        for p in range(count):
            code.append(
                f"ds_read_b32 v{num+qi*count+p}, ${readp} offset:{(qi*groups*w+p)*256}"
            )
    code += ["s_waitcnt lgkmcnt(0)"]
    for qi in range(num):
        code += [
            (
                f"v_mov_b32 v{qi}, v{num+qi*count+count-1}"
                if k % 112
                else f"v_mov_b32 v{qi}, 0"
            )
        ]
        for p in range(k // 112):
            code.append(f"v_add_f32 v{qi}, v{qi}, v{num+qi*count+p}")
    return "\n".join(code), ",".join(
        [f"=&{{v{i}}}" for i in range(num)]
        + ["v"] * (2 + groups * num)
        + [f"~{{v{i}}}" for i in range(num, num + num * count)]
        + ["~{memory}"]
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
                C + rr * N + cc, total[i * TILE + j], (rr < M) & (cc < N) & (warp == 0)
            )
