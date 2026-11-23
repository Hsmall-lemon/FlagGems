"""Candidate: 32x32 per-wave CPU-order panels with explicit LDS partitions.

For medium matrices, 16 independent accumulators reuse each staged operand
more than the 16x16 parent. Full112 panels omit masked-load initialization.
No numerical or performance claim is made until measured by the main runner.
"""

import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SwizzledSharedLayout

from .mm_fp32_cpu_panel import shared_pointer


@triton.constexpr_function
def original_reduce_assembly(tile, w, count):
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


@triton.constexpr_function
def guarded_reduce_assembly(tile, w, count):
    spec = original_reduce_assembly(tile, w, count)
    return "s_waitcnt lgkmcnt(0)\ns_barrier\n" + spec[0], spec[1]


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
    spec: gl.constexpr = guarded_reduce_assembly(TILE, W, COUNT)
    return gl.inline_asm_elementwise(
        spec[0],
        spec[1],
        (readp, writep) + part + (warp,) + previous,
        dtype=(gl.float32,) * (TILE * TILE),
        is_pure=False,
        pack=1,
    )


@triton.constexpr_function
def assembly(depth, k, pad, am, bn):
    count = 16
    ga, gb = 16, 24
    ar, br = 32, 32 + 16 * depth
    ao, bo, ki = 32 + 32 * depth, 33 + 32 * depth, 34 + 32 * depth
    ao1, bo1 = ki + 1, ki + 2
    ia, ib = 16, 17
    iao, ibo, wa, wb, ra, rb, karg = 18, 19, 20, 21, 22, 23, 24
    prologue = [
        f"s_mov_b64 s[32:33], ${ia}",
        f"s_mov_b64 s[34:35], ${ib}",
        f"v_mov_b32 v{ao}, ${iao}",
        f"v_mov_b32 v{bo}, ${ibo}",
        f"v_mov_b32 v{ki}, ${karg}",
        f"v_add_u32 v{ao1}, {16*am*4}, v{ao}",
        f"v_add_u32 v{bo1}, {16*bn*4}, v{bo}",
    ]

    def prefetch(full):
        seq = [] if full else [f"v_mov_b32 v{i}, 0" for i in range(ga, gb + 8)]
        if not full:
            seq += [
                f"v_cmp_gt_u32 vcc, {k-k%4}, v{ki}",
                "s_and_saveexec_b64 s[40:41], vcc",
            ]
        for i in range(2):
            aoff, bof = (ao, bo) if i == 0 else (ao1, bo1)
            seq += [
                f"global_load_dwordx4 v[{ga+4*i}:{ga+4*i+3}], v{aoff}, s[32:33]",
                f"global_load_dwordx4 v[{gb+4*i}:{gb+4*i+3}], v{bof}, s[34:35]",
            ]
        if not full:
            seq += ["s_mov_b64 exec, s[40:41]"]
            if k % 4:
                seq += [
                    f"v_cmp_eq_u32 vcc, {k-k%4}, v{ki}",
                    "s_and_saveexec_b64 s[40:41], vcc",
                ]
                for i in range(2):
                    aoff, bof = (ao, bo) if i == 0 else (ao1, bo1)
                    for j in range(k % 4):
                        seq += [
                            f"global_load_dword v{ga+4*i+j}, v{aoff}, s[32:33] offset:{j*4}",
                            f"global_load_dword v{gb+4*i+j}, v{bof}, s[34:35] offset:{j*4}",
                        ]
                seq += ["s_mov_b64 exec, s[40:41]"]
        return seq

    def lds(group):
        slot = group % depth
        seq = []
        for i in range(4):
            a, b = ar + slot * 16 + i * 4, br + slot * 16 + i * 4
            offset = (i * 8 * pad + group * 4) * 4
            seq += [
                f"ds_read_b128 v[{a}:{a+3}], ${ra} offset:{offset}",
                f"ds_read_b128 v[{b}:{b+3}], ${rb} offset:{offset}",
            ]
        return seq

    def pipeline(full, length):
        code = prefetch(full or length >= 16)
        chunks = (length + 15) // 16
        for chunk in range(chunks):
            code += ["s_waitcnt vmcnt(0)"]
            for i in range(2):
                offset = i * 16 * pad * 4
                code += [
                    f"ds_write_b128 ${wa}, v[{ga+4*i}:{ga+4*i+3}] offset:{offset}",
                    f"ds_write_b128 ${wb}, v[{gb+4*i}:{gb+4*i+3}] offset:{offset}",
                ]
            code += ["s_waitcnt lgkmcnt(0)"]
            if chunk + 1 < chunks:
                for r in (ao, bo, ao1, bo1):
                    code += [f"v_add_u32 v{r}, 64, v{r}"]
                code += [f"v_add_u32 v{ki}, 16, v{ki}"]
                code += prefetch(full or (chunk + 2) * 16 <= length)
            terms = min(16, length - chunk * 16)
            groups = (terms + 3) // 4
            for group in range(min(depth, groups)):
                code += lds(group)
            for group in range(groups):
                slot = group % depth
                code += [f"s_waitcnt lgkmcnt({min(15,8*min(depth-1,groups-group-1))})"]
                for q in range(min(4, terms - group * 4)):
                    for i in range(4):
                        for j in range(4):
                            code += [
                                f"v_mac_f32 v{i*4+j}, v{ar+slot*16+i*4+q}, v{br+slot*16+j*4+q}"
                            ]
                if group + depth < groups:
                    code += lds(group + depth)
            code += ["s_waitcnt lgkmcnt(0)"]
        return code

    code = prologue + [
        f"v_readfirstlane_b32 s44, ${karg}",
        f"s_cmp_ge_u32 s44, {k}",
        "s_cbranch_scc1 3f",
    ]
    if k % 112:
        code += [f"s_cmp_eq_u32 s44, {k//112*112}", "s_cbranch_scc1 1f"]
    if k >= 112:
        code += pipeline(True, 112)
    code += ["s_branch 3f"]
    if k % 112:
        code += ["1:"] + pipeline(False, k % 112)
    code += ["3:"]
    cons = (
        [f"=&{{v{i}}}" for i in range(count)]
        + ["s", "s"]
        + ["v"] * 7
        + [str(i) for i in range(count)]
        + [f"~{{v{i}}}" for i in range(ga, bo1 + 1)]
        + [f"~{{s{i}}}" for i in [32, 33, 34, 35, 40, 41, 44]]
        + ["~{exec}", "~{vcc}", "~{scc}", "~{memory}"]
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
    TILE: gl.constexpr = 4,
    W: gl.constexpr = 2,
    DEPTH: gl.constexpr = 2,
    CHUNK: gl.constexpr = 16,
    PAD: gl.constexpr = 20,
):
    gl.static_assert(TILE == 4 and CHUNK == 16 and AK == 1 and BK == 1)
    gl.static_assert(
        M == 1024
        and N == 256
        and K == 217
        and AM == 217
        and BN == 217
        and W == 2
        and DEPTH == 2
        and PAD == 20
    )
    gl.static_assert(M % 32 == 0 and N % 32 == 0)
    gl.static_assert(DEPTH == 1 or DEPTH == 2)
    L: gl.constexpr = BlockedLayout([1], [64], [W], [0])
    x = gl.arange(0, W * 64, layout=L)
    lane, warp = x % 64, x // 64
    row0, col0 = gl.program_id(0) * 32, gl.program_id(1) * 32
    SPAN: gl.constexpr = triton.next_power_of_2(PAD)
    sa = gl.allocate_shared_memory(
        gl.float32, (W * 32, SPAN), SwizzledSharedLayout(1, 1, 1, [1, 0])
    )
    sb = gl.allocate_shared_memory(
        gl.float32, (W * 32, SPAN), SwizzledSharedLayout(1, 1, 1, [1, 0])
    )
    z = gl.full((W * 64,), 0, gl.int32, L)
    abase = shared_pointer(sa, z, z).to(gl.uint32)
    bbase = shared_pointer(sb, z, z).to(gl.uint32)
    wa = abase + ((warp * 32 + lane // 4) * PAD + (lane % 4) * 4) * 4
    wb = bbase + ((warp * 32 + lane // 4) * PAD + (lane % 4) * 4) * 4
    ra = abase + ((warp * 32 + lane // 8) * PAD) * 4
    rb = bbase + ((warp * 32 + lane % 8) * PAD) * 4
    sr = gl.allocate_shared_memory(
        gl.float32, (16 * W, 64), SwizzledSharedLayout(1, 1, 1, [1, 0])
    )
    readp = shared_pointer(sr, z, lane).to(gl.uint32)
    writep = readp + warp * 256
    zeros = ()
    for _ in gl.static_range(16):
        zeros += (gl.full((W * 64,), 0, gl.float32, L),)
    total = zeros
    spec: gl.constexpr = assembly(DEPTH, K, PAD, AM, BN)
    for group in gl.static_range(triton.cdiv(triton.cdiv(K, 112), W)):
        pi = warp + group * W
        panel = gl.where(pi == 0, K // 112, pi - 1) if K % 112 else pi
        kval = gl.where(pi < triton.cdiv(K, 112), panel * 112, K) + (lane % 4) * 4
        ao = ((row0 + lane // 4) * AM + kval) * 4
        bo = ((col0 + lane // 4) * BN + kval) * 4
        part = gl.inline_asm_elementwise(
            spec[0],
            spec[1],
            (A.to(gl.uint64), B.to(gl.uint64), ao, bo, wa, wb, ra, rb, kval) + zeros,
            dtype=(gl.float32,) * 16,
            is_pure=False,
            pack=1,
        )
        total = reduce_group(
            readp,
            writep,
            part,
            total,
            warp,
            4,
            W,
            min(W, triton.cdiv(K, 112) - group * W),
        )
    for i in gl.static_range(4):
        for j in gl.static_range(4):
            rr, cc = row0 + lane // 8 + i * 8, col0 + lane % 8 + j * 8
            gl.store(C + rr * N + cc, total[i * 4 + j], warp == 0)
