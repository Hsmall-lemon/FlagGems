"""row135 W2 original swizzle/pipeline; precomputed B group addresses in consumed tied argument registers."""

import triton
import triton.language as tl
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SwizzledSharedLayout

from .mm_fp32_cpu_panel import shared_pointer


@triton.constexpr_function
def assembly(depth, k, pad, bn):
    assert (depth, k, pad, bn) == (2, 3072, 16, 3072)
    ga, gb, ar, br, ao, bo = 8, 12, 20, 24, 56, 57
    ia, ib, iao, ibo, wa, wb, karg = 13, 14, 59, 60, 15, 16, 58
    # Outputs8/9 are tied mutable RA/RB at v16/17, never ordinary inputs.
    prologue = [
        f"s_mov_b64 s[32:33], ${ia}",
        f"s_mov_b64 s[34:35], ${ib}",
        f"v_mov_b32 v{ao}, v{iao}",
        f"v_mov_b32 v{bo}, v{ibo}",
    ]

    def prefetch():
        return [
            f"global_load_dwordx4 v[{ga}:{ga+3}], v{ao}, s[32:33]",
            f"global_load_dwordx4 v[{gb}:{gb+3}], v{bo}, s[34:35]",
        ]

    def advance_and_prefetch():
        registers = (ao, bo)
        return [f"v_add_u32 v{r}, 64, v{r}" for r in registers] + prefetch()

    def write_stage(bank):
        # A belongs only to this wave and overwrites after its group3 LGKM0.
        # B uses the other shared bank, protected by current CTA publication.
        return [
            "s_waitcnt vmcnt(0)",
            f"ds_write_b128 ${wa}, v[{ga}:{ga+3}] offset:0",
            f"ds_write_b128 ${wb}, v[{gb}:{gb+3}] offset:{bank*2048}",
        ]

    def publish_stage():
        return ["s_waitcnt lgkmcnt(0)", "s_barrier"]

    def lds(group, bank):
        slot = group % 2
        # Inout group state0,1,2,3 uses XOR deltas16,48,16 bytes.
        out = []
        if group:
            delta = (group ^ (group - 1)) * 16
            out = [f"v_xor_b32 v16, {delta}, v16"]
        out += [f"ds_read2_b32 v[{ar+slot*2}:{ar+slot*2+1}], v16 offset0:0 offset1:128"]
        for j in range(4):
            r = br + slot * 16 + j * 4
            # B bank is always an ADD immediate; never XOR4096 on absolute RB.
            out += [
                f"ds_read_b128 v[{r}:{r+3}], v{(17,58,59,60)[group]} offset:{bank*2048+j*512}"
            ]
        return out

    def consume(bank, reset, write_next=False):
        out = lds(0, bank) + lds(1, bank)
        for group in range(4):
            slot = group % 2
            out += [f"s_waitcnt lgkmcnt({5*min(1,3-group)})"]
            if group == 3 and write_next:
                # All current-bank reads have retired. The other bank was
                # released by the current publication barrier. Stage global
                # data now, leaving32 ordered NF DPP to cover its DS latency.
                # Publication and next global loads stay after these products.
                out += write_stage(1 - bank)
            for q in range(4):
                for i in range(2):
                    for j in range(4):
                        out += [
                            f"v_mac_f32 v{i*4+j}, v{ar+slot*2+i}, v{br+slot*16+j*4+q} quad_perm:[{q},{q},{q},{q}]"
                        ]
            if group + 2 < 4:
                out += lds(group + 2, bank)
        if reset:
            # Last group already waited0; all reads retired. Reset group3->0.
            # A remains in this wave single bank; B bank is an ADD immediate.
            out += ["v_xor_b32 v16, 48, v16"]
        return out

    def role_body():
        # Both full112 and tail48 share one32K pair body plus a final16K body.
        out = (
            [
                f"v_readfirstlane_b32 s44, v{karg}",
                f"s_cmp_ge_u32 s44, {k}",
                "s_cbranch_scc1 3f",
                "v_xor_b32 v58, 16, v17",
                "v_xor_b32 v59, 32, v17",
                "v_xor_b32 v60, 48, v17",
                "s_mov_b32 s45, 3",
                "s_cmp_eq_u32 s44, 3024",
                "s_cbranch_scc0 2f",
                "s_mov_b32 s45, 1",
                "2:",
            ]
            + prefetch()
            + write_stage(0)
            + publish_stage()
            + ["4:"]
        )
        out += advance_and_prefetch() + consume(0, True, True) + publish_stage()
        out += advance_and_prefetch() + consume(1, True, True) + publish_stage()
        out += ["s_sub_u32 s45, s45, 1", "s_cmp_lg_u32 s45, 0", "s_cbranch_scc1 4b"]
        out += consume(0, False) + ["3:"]
        return out

    code = prologue + role_body()
    # The r523 bounded32-bit output address and external reducer are unchanged.
    code += [f"v_mov_b32 v{ao}, $18"]
    for i in range(2):
        if i:
            code += [f"v_add_u32 v{ao}, 24576, v{ao}"]
        for j in range(4):
            code += [f"global_store_dword v{ao}, v{i*4+j}, $17 offset:{j*32}"]
    # Outputs10..12 are true inouts initialized to Karg/AO/BO in v58/59/60.
    # AO/BO are copied and Karg consumed before these registers become B addresses.
    cons = (
        [f"=&{{v{i}}}" for i in range(8)]
        + ["=&{v16}", "=&{v17}", "=&{v58}", "=&{v59}", "=&{v60}"]
        + ["s", "s", "v", "v", "s", "v"]
        + [str(i) for i in range(13)]
        + [f"~{{v{i}}}" for i in range(8, 60) if i not in (16, 17, 18, 19, 58, 59)]
        + [f"~{{s{i}}}" for i in (32, 33, 34, 35, 40, 41, 44, 45)]
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
    W: gl.constexpr = 2,
    DEPTH: gl.constexpr = 2,
    PAD: gl.constexpr = 16,
):
    gl.static_assert(
        M == 32
        and N == 768
        and K == 3072
        and AM == 3072
        and AK == 1
        and BK == 1
        and BN == 3072
    )
    gl.static_assert(W == 2 and DEPTH == 2 and PAD == 16)
    L: gl.constexpr = BlockedLayout([1], [64], [W], [0])
    x = gl.arange(0, 128, layout=L)
    lane, warp = x % 64, x // 64
    row0, col0 = gl.program_id(0) * 32, gl.program_id(1) * 32
    # Both allocation shapes are powers of two; metadata must be exactly6KiB.
    ashared = gl.allocate_shared_memory(
        gl.float32, (1, 512), SwizzledSharedLayout(1, 1, 1, [1, 0])
    )
    bshared = gl.allocate_shared_memory(
        gl.float32, (1, 1024), SwizzledSharedLayout(1, 1, 1, [1, 0])
    )
    z = gl.full((128,), 0, gl.int32, L)
    abase = shared_pointer(ashared, z, z).to(gl.uint32)
    bbase = shared_pointer(bshared, z, z).to(gl.uint32)
    # A16 rows perwave have no crosswave readers; B32 rows are shared.
    awave = abase + warp * 1024
    arow = lane // 4
    brow = warp * 16 + lane // 4
    wa = awave + ((arow * 16 + (lane % 4) * 4) ^ ((arow % 8) * 4)) * 4
    wb = bbase + ((brow * 16 + (lane % 4) * 4) ^ ((brow % 8) * 4)) * 4
    arow = lane // 8
    brow = lane % 8
    ra = awave + ((arow * 16 + lane % 4) ^ ((arow % 8) * 4)) * 4
    rb = bbase + ((brow * 16) ^ ((brow % 8) * 4)) * 4
    zeros = ()
    for _ in gl.static_range(8):
        zeros += (gl.full((128,), 0, gl.float32, L),)
    pi = gl.program_id(2)
    panel = gl.where(pi == 0, K // 112, pi - 1)
    kval = gl.where(pi < 28, panel * 112, K) + (lane % 4) * 4
    ao = ((row0 + (warp % 2) * 16 + lane // 4) * AM + kval) * 4
    bo = ((col0 + warp * 16 + lane // 4) * BN + kval) * 4
    co = (
        panel * M * N
        + ((warp % 2) * 16 + lane // 8) * N
        + col0
        + (warp // 2) * 32
        + lane % 8
    ) * 4
    spec: gl.constexpr = assembly(DEPTH, K, PAD, BN)
    gl.inline_asm_elementwise(
        spec[0],
        spec[1],
        (A.to(gl.uint64), B.to(gl.uint64), wa, wb, C.to(gl.uint64), co)
        + zeros
        + (ra, rb, kval, ao, bo),
        dtype=(gl.float32,) * 8 + (gl.uint32,) * 5,
        is_pure=False,
        pack=1,
    )


@triton.jit
def reduce(
    P,
    C,
    SIZE: tl.constexpr,
    FULL: tl.constexpr,
    HAS_TAIL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = (
        tl.load(P + FULL * SIZE + x, x < SIZE, 0)
        if HAS_TAIL
        else tl.full((BLOCK,), 0, tl.float32)
    )
    for p in range(FULL):
        part = tl.load(P + p * SIZE + x, x < SIZE, 0)
        acc = acc + part
    tl.store(C + x, acc, x < SIZE)
