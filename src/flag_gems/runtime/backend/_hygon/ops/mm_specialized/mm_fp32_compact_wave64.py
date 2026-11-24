"""Experimental single-launch cbnd with private16K streamed K panels.

Original physical112 panel indices and original ordered vector reducer/edges.
The only new body mechanism is compact opaque global/LDS consumption in place
of whole112 descriptor staging. No correctness/performance claim before tests.
"""

import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SwizzledSharedLayout

from .mm_fp32_cpu_boundary import edges, row_edge
from .mm_fp32_cpu_cooperative_vector import vector_reduce_asm
from .mm_fp32_cpu_panel import shared_pointer


@triton.constexpr_function
def guarded_vector_reduce(k, w, tile):
    code, constraints = vector_reduce_asm(k, w, tile)
    return "s_waitcnt lgkmcnt(0)\ns_barrier\n" + code, constraints


@triton.constexpr_function
def assembly(depth, k, pad):
    count = 4
    ga, gb = 4, 8
    ar, br = 12, 12 + 8 * depth
    ao, bo, ki = 12 + 16 * depth, 13 + 16 * depth, 14 + 16 * depth
    ia, ib = 4, 5
    iao, ibo, wa, wb, ra, rb, karg = 6, 7, 8, 9, 10, 11, 12
    code = [
        f"s_mov_b64 s[32:33], ${ia}",
        f"s_mov_b64 s[34:35], ${ib}",
        f"v_mov_b32 v{ao}, ${iao}",
        f"v_mov_b32 v{bo}, ${ibo}",
        f"v_mov_b32 v{ki}, ${karg}",
    ]

    def prefetch():
        return [
            f"global_load_dwordx4 v[{ga}:{ga+3}], v{ao}, s[32:33]",
            f"global_load_dwordx4 v[{gb}:{gb+3}], v{bo}, s[34:35]",
        ]

    def lds(group):
        slot = group % depth
        seq = []
        for i in range(2):
            a, b = ar + slot * 8 + i * 4, br + slot * 8 + i * 4
            offset = (i * 8 * pad + group * 4) * 4
            seq += [
                f"ds_read_b128 v[{a}:{a+3}], ${ra} offset:{offset}",
                f"ds_read_b128 v[{b}:{b+3}], ${rb} offset:{offset}",
            ]
        return seq

    def pipeline(length):
        assert length % 16 == 0
        complete = length // 16
        out = prefetch() + [
            f"s_mov_b32 s45, {complete}",
            "4:",
            "s_waitcnt vmcnt(0)",
            f"ds_write_b128 ${wa}, v[{ga}:{ga+3}]",
            f"ds_write_b128 ${wb}, v[{gb}:{gb+3}]",
            "s_waitcnt lgkmcnt(0)",
            "s_cmp_eq_u32 s45, 1",
            "s_cbranch_scc1 5f",
            f"v_add_u32 v{ao}, 64, v{ao}",
            f"v_add_u32 v{bo}, 64, v{bo}",
            f"v_add_u32 v{ki}, 16, v{ki}",
        ]
        out += prefetch() + ["5:"]
        for group in range(min(depth, 4)):
            out += lds(group)
        for group in range(4):
            slot = group % depth
            out += [f"s_waitcnt lgkmcnt({4*min(depth-1,3-group)})"]
            for q in range(4):
                for i in range(2):
                    for j in range(2):
                        out += [
                            f"v_mac_f32 v{i*2+j}, v{ar+slot*8+i*4+q}, v{br+slot*8+j*4+q}"
                        ]
            if group + depth < 4:
                out += lds(group + depth)
        out += [
            "s_waitcnt lgkmcnt(0)",
            "s_sub_u32 s45, s45, 1",
            "s_cmp_lg_u32 s45, 0",
            "s_cbranch_scc1 4b",
        ]
        return out

    # This finite implementation has only112/64-term panels, both aligned16.
    assert k == 512
    code += [
        f"v_readfirstlane_b32 s44, ${karg}",
        f"s_cmp_ge_u32 s44, {k}",
        "s_cbranch_scc1 3f",
        f"s_cmp_eq_u32 s44, {k//112*112}",
        "s_cbranch_scc1 1f",
    ]
    code += pipeline(112) + ["s_branch 3f", "1:"] + pipeline(64) + ["3:"]
    cons = (
        [f"=&{{v{i}}}" for i in range(count)]
        + ["s", "s"]
        + ["v"] * 7
        + [str(i) for i in range(count)]
        + [f"~{{v{i}}}" for i in range(ga, ki + 1)]
        + [f"~{{s{i}}}" for i in [32, 33, 34, 35, 44, 45]]
        + ["~{scc}", "~{memory}"]
    )
    return "\n".join(code), ",".join(cons)


@g.jit
def body(
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
    PM,
    PN,
    DEPTH: gl.constexpr = 2,
):
    W: gl.constexpr = 4
    L: gl.constexpr = BlockedLayout([1], [64], [W], [0])
    x = gl.arange(0, W * 64, layout=L)
    lane, warp = x % 64, x // 64
    row0, col0 = PM * 16, PN * 16
    PAD: gl.constexpr = 20
    sa = gl.allocate_shared_memory(
        gl.float32, (64, 32), SwizzledSharedLayout(1, 1, 1, [1, 0])
    )
    sb = gl.allocate_shared_memory(
        gl.float32, (64, 32), SwizzledSharedLayout(1, 1, 1, [1, 0])
    )
    z = gl.full((W * 64,), 0, gl.int32, L)
    abase = shared_pointer(sa, z, z).to(gl.uint32)
    bbase = shared_pointer(sb, z, z).to(gl.uint32)
    wa = abase + ((warp * 16 + lane // 4) * PAD + (lane % 4) * 4) * 4
    wb = bbase + ((warp * 16 + lane // 4) * PAD + (lane % 4) * 4) * 4
    ra = abase + ((warp * 16 + lane // 8) * PAD) * 4
    rb = bbase + ((warp * 16 + lane % 8) * PAD) * 4
    zeros = ()
    for _ in gl.static_range(4):
        zeros += (gl.full((W * 64,), 0, gl.float32, L),)
    spec: gl.constexpr = assembly(DEPTH, K, PAD)
    all_parts = ()
    for group in gl.static_range(2):
        pi = warp + group * W
        kval = gl.minimum(pi * 112, K) + (lane % 4) * 4
        ao = (gl.minimum(row0 + lane // 4, M - 1) * AM + kval) * 4
        bo = ((col0 + lane // 4) * BN + kval) * 4
        part = gl.inline_asm_elementwise(
            spec[0],
            spec[1],
            (A.to(gl.uint64), B.to(gl.uint64), ao, bo, wa, wb, ra, rb, kval) + zeros,
            dtype=(gl.float32,) * 4,
            is_pure=False,
            pack=1,
        )
        all_parts += part
    # Separate descriptor; compiler may reuse dead operand storage. The guard
    # before every reducer write makes either physical allocation safe.
    sr = gl.allocate_shared_memory(
        gl.float32, (32, 64), SwizzledSharedLayout(1, 1, 1, [1, 0])
    )
    readp = shared_pointer(sr, z, lane).to(gl.uint32)
    writep = readp + warp * 256
    reduce_spec: gl.constexpr = guarded_vector_reduce(K, W, 2)
    total = gl.inline_asm_elementwise(
        reduce_spec[0],
        reduce_spec[1],
        (readp, writep) + all_parts,
        dtype=(gl.float32,) * 4,
        is_pure=False,
        pack=1,
    )
    for i in gl.static_range(2):
        for j in gl.static_range(2):
            rr, cc = row0 + lane // 8 + i * 8, col0 + lane % 8 + j * 8
            gl.store(
                C + rr * N + cc,
                total[i * 2 + j],
                (rr < M) & (cc < N) & (warp == 0) & (~row_edge(rr, M, 4)),
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
    DEPTH: gl.constexpr = 2,
):
    gl.static_assert(M == 490 and N == 64 and K == 512)
    gl.static_assert(AM == 512 and AK == 1 and BK == 1 and BN == 512)
    gl.static_assert(DEPTH == 1 or DEPTH == 2)
    pm, pn = gl.program_id(0), gl.program_id(1)
    if pm < triton.cdiv(M, 16):
        body(A, B, C, M, N, K, AM, AK, BK, BN, pm, pn, DEPTH)
    else:
        edges(
            A, B, C, M, N, K, AM, AK, BK, BN, 2, 4, 32, 4, pm - triton.cdiv(M, 16), pn
        )


def launch(a, b, out, depth=2):
    if tuple(a.shape) != (490, 512) or tuple(b.shape) != (512, 64):
        raise ValueError("Recorded model64 shape required")
    if tuple(a.stride()) != (512, 1) or tuple(b.stride()) != (1, 512):
        raise ValueError("Recorded model64 strides required")
    if a.storage_offset() != 0 or b.storage_offset() != 0:
        raise ValueError("Recorded zero-offset inputs required")
    if str(a.dtype) != "torch.float32" or str(b.dtype) != "torch.float32":
        raise ValueError("FP32 required")
    return matmul[(35, 4)](
        a,
        b,
        out,
        490,
        64,
        512,
        *a.stride(),
        *b.stride(),
        DEPTH=depth,
        num_warps=4,
        enable_fp_fusion=False,
    )
