"""Stage complete 112-term panels, then consume four K values per LDS read."""

import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import (
    BlockedLayout,
    SliceLayout,
    SwizzledSharedLayout,
)

from .mm_fp32_cpu_panel import shared_pointer


@triton.constexpr_function
def assembly(tile, depth, length):
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
            offset = (i * 8 * 128 + group // 8 * 32) * 4
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
    constraints = (
        [f"=&{{v{i}}}" for i in range(count)]
        + ["v"] * 16
        + [str(i) for i in range(count)]
        + [f"~{{v{i}}}" for i in range(scratch, scratch + 8 * tile * depth)]
        + ["~{memory}"]
    )
    return "\n".join(code), ",".join(constraints)


@g.jit
def consume(
    A,
    B,
    sa,
    sb,
    ap,
    bp,
    rr,
    cc,
    kk,
    zero,
    start,
    M: gl.constexpr,
    N: gl.constexpr,
    AM: gl.constexpr,
    AK: gl.constexpr,
    BK: gl.constexpr,
    BN: gl.constexpr,
    TILE: gl.constexpr,
    DEPTH: gl.constexpr,
    LENGTH: gl.constexpr,
):
    av = gl.load(
        A + rr[:, None] * AM + (start + kk[None, :]) * AK,
        (rr[:, None] < M) & (kk[None, :] < LENGTH),
        0,
    )
    bv = gl.load(
        B + (start + kk[None, :]) * BK + cc[:, None] * BN,
        (cc[:, None] < N) & (kk[None, :] < LENGTH),
        0,
    )
    sa.store(av)
    sb.store(bv)
    spec: gl.constexpr = assembly(TILE, DEPTH, LENGTH)
    return gl.inline_asm_elementwise(
        spec[0],
        spec[1],
        ap + bp + zero,
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
    TILE: gl.constexpr = 2,
    W: gl.constexpr = 4,
    WM: gl.constexpr = 2,
    DEPTH: gl.constexpr = 2,
    SPLIT: gl.constexpr = False,
):
    BM: gl.constexpr = 8 * TILE * WM
    BC: gl.constexpr = 8 * TILE * (W // WM)
    pm, pn = gl.program_id(0), gl.program_id(1)
    L: gl.constexpr = BlockedLayout([1], [64], [W], [0])
    x = gl.arange(0, W * 64, layout=L)
    lane, warp = x % 64, x // 64
    ar = (warp % WM) * 8 * TILE + lane // 8
    br = (warp // WM) * 8 * TILE + lane % 8
    sa = gl.allocate_shared_memory(
        gl.float32, (BM, 128), SwizzledSharedLayout(4, 1, 8, [1, 0])
    )
    sb = gl.allocate_shared_memory(
        gl.float32, (BC, 128), SwizzledSharedLayout(4, 1, 8, [1, 0])
    )
    ap = ()
    bp = ()
    for group in gl.static_range(8):
        ks = gl.full((W * 64,), group * 4, gl.int32, L)
        ap += (shared_pointer(sa, ar, ks).to(gl.uint32),)
        bp += (shared_pointer(sb, br, ks).to(gl.uint32),)
    LL: gl.constexpr = BlockedLayout([1, 4], [8, 8], [1, W], [1, 0])
    rr = pm * BM + gl.arange(0, BM, layout=SliceLayout(1, LL))
    cc = pn * BC + gl.arange(0, BC, layout=SliceLayout(1, LL))
    kk = gl.arange(0, 128, layout=SliceLayout(0, LL))
    zero = ()
    for i in gl.static_range(TILE * TILE):
        zero += (gl.full((W * 64,), 0, gl.float32, L),)
    total = zero
    if SPLIT:
        panel = gl.program_id(2)
        if panel < K // 112:
            total = consume(
                A,
                B,
                sa,
                sb,
                ap,
                bp,
                rr,
                cc,
                kk,
                zero,
                panel * 112,
                M,
                N,
                AM,
                AK,
                BK,
                BN,
                TILE,
                DEPTH,
                112,
            )
        elif K % 112:
            total = consume(
                A,
                B,
                sa,
                sb,
                ap,
                bp,
                rr,
                cc,
                kk,
                zero,
                panel * 112,
                M,
                N,
                AM,
                AK,
                BK,
                BN,
                TILE,
                DEPTH,
                K % 112,
            )
    else:
        if K % 112:
            total = consume(
                A,
                B,
                sa,
                sb,
                ap,
                bp,
                rr,
                cc,
                kk,
                zero,
                K // 112 * 112,
                M,
                N,
                AM,
                AK,
                BK,
                BN,
                TILE,
                DEPTH,
                K % 112,
            )
        for panel in range(K // 112):
            part = consume(
                A,
                B,
                sa,
                sb,
                ap,
                bp,
                rr,
                cc,
                kk,
                zero,
                panel * 112,
                M,
                N,
                AM,
                AK,
                BK,
                BN,
                TILE,
                DEPTH,
                112,
            )
            for i in gl.static_range(TILE * TILE):
                total = total[:i] + (total[i] + part[i],) + total[i + 1 :]
    for i in gl.static_range(TILE):
        for j in gl.static_range(TILE):
            r = pm * BM + ar + i * 8
            c = pn * BC + br + j * 8
            gl.store(
                C + (gl.program_id(2) * M * N if SPLIT else 0) + r * N + c,
                total[i * TILE + j],
                (r < M) & (c < N),
            )
