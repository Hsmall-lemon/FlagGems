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

"""FP32 32-term prefetch with ordered 112-term panel accumulation."""

import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SwizzledSharedLayout

from .mm_fp32_cpu_panel import shared_pointer, store_acc


@triton.constexpr_function
def asm_spec(w, wm, tm, tn, am, ak, bk, bn, m, n, steps, panel, accurate):
    assert w == 1 and wm == 1 and tm == 1 and tn == 1 and not accurate
    k = steps * 16
    end = k // 112 * 112
    assert k % 32 == 0 and end % 32 == 0
    starts = list(range(end, k, 32)) + list(range(0, end, 32))
    bm = 32
    bnn = 32
    ia, ib = 8, 9
    aoff, boff, aw, bw, ra, rb = 10, 11, 12, 13, 14, 15
    # 8 accumulators, 32 operand prefetch registers, 16 global-load registers.
    ar, br, ga, gb = 8, 24, 40, 48
    code = [
        f"s_mov_b64 s[32:33], ${ia}",
        f"s_mov_b64 s[34:35], ${ib}",
        "s_mov_b32 m0, 0xffff",
    ]

    def load():
        return [
            f"global_load_dwordx4 v[{ga}:{ga+3}], ${aoff}, s[32:33]",
            f"global_load_dwordx4 v[{ga+4}:{ga+7}], ${aoff}, s[32:33] offset:{16*ak*4}",
            f"global_load_dwordx4 v[{gb}:{gb+3}], ${boff}, s[34:35]",
            f"global_load_dwordx4 v[{gb+4}:{gb+7}], ${boff}, s[34:35] offset:{16*bk*4}",
        ]

    code += load()
    for step, start in enumerate(starts):
        code += ["s_waitcnt vmcnt(0)"]
        for part in range(2):
            for j in range(4):
                code += [
                    f"ds_write_b32 ${aw}, v{ga+part*4+j} offset:{(part*16*bm+j*bm)*4}",
                    f"ds_write_b32 ${bw}, v{gb+part*4+j} offset:{(part*16*bnn+j*bnn)*4}",
                ]
        if step + 1 < len(starts):
            delta = starts[step + 1] - start
            code += [
                f"s_add_u32 s32, s32, {(delta*ak*4)&0xffffffff}",
                f"s_addc_u32 s33, s33, {-1 if delta<0 else 0}",
                f"s_add_u32 s34, s34, {(delta*bk*4)&0xffffffff}",
                f"s_addc_u32 s35, s35, {-1 if delta<0 else 0}",
            ]
            code += load()
        code += ["s_waitcnt lgkmcnt(0)"]
        for ki in range(4):
            code += [
                f"ds_read_m32x8_b32 v[{ar+ki*4}:{ar+ki*4+3}], ${ra} offset:{ki*8*bm*4}",
                f"ds_read_m32x8_b32 v[{br+ki*4}:{br+ki*4+3}], ${rb} offset:{ki*8*bnn*4}",
            ]
        for ki in range(4):
            code += [
                f"s_waitcnt lgkmcnt({(3-ki)*2})",
                f"v_mmac_16x16x8_f32 v[0:3], v[{ar+ki*4}:{ar+ki*4+1}], v[{br+ki*4}:{br+ki*4+1}], v[0:3]",
            ]
            end_k = start + (ki + 1) * 8
            if (start >= end and end_k == k) or (start < end and end_k % 112 == 0):
                code += [f"v_add_f32 v{4+q}, v{4+q}, v{q}" for q in range(4)]
                code += [f"v_mov_b32 v{q}, 0" for q in range(4)]
    code += [f"v_mov_b32 v{q}, v{4+q}" for q in range(4)]
    constraints = (
        [f"=&{{v{i}}}" for i in range(8)]
        + ["s", "s"]
        + ["v"] * 6
        + [str(i) for i in range(8)]
        + [f"~{{v{i}}}" for i in range(8, 56)]
        + [f"~{{s{i}}}" for i in range(32, 36)]
        + ["~{memory}", "~{m0}", "~{scc}"]
    )
    return "\n".join(code), ",".join(constraints)


@g.jit
def native(
    A,
    B,
    sa,
    sb,
    outer,
    pm,
    pn,
    M: gl.constexpr,
    N: gl.constexpr,
    AM: gl.constexpr,
    AK: gl.constexpr,
    BK: gl.constexpr,
    BN: gl.constexpr,
    W: gl.constexpr,
    WM: gl.constexpr,
    TM: gl.constexpr,
    TN: gl.constexpr,
    STEPS: gl.constexpr,
    PANEL: gl.constexpr,
    ACCURATE: gl.constexpr,
):
    BM: gl.constexpr = 16 * (TM if TM > 2 else 2) * WM
    BNN: gl.constexpr = 16 * (TN if TN > 2 else 2) * (W // WM)
    L: gl.constexpr = BlockedLayout([1], [64], [W], [0])
    x = gl.arange(0, W * 64, layout=L)
    lane, warp = x % 64, x // 64
    zero = gl.full((W * 64,), 0, gl.int32, L)
    ap = shared_pointer(sa, zero, zero).to(gl.uint32)
    bp = shared_pointer(sb, zero, zero).to(gl.uint32)
    args = (A.to(gl.uint64), B.to(gl.uint64))
    aw = ()
    bw = ()
    for i in gl.static_range(16 * (16 * TM * WM) // (W * 64 * 4)):
        t = (x + i * W * 64) * 4
        if AM == 1:
            r, kk = t % BM, t // BM
        else:
            r, kk = t // 16, t % 16
        # Main loop reads only full K panels; clamp padding along output axes.
        r = gl.minimum(pm * 16 * TM * WM + r, M - 1)
        args += ((r * AM + kk * AK) * 4,)
        aw += (ap + (kk * BM + (t % BM if AM == 1 else t // 16)) * 4,)
    for i in gl.static_range(16 * (16 * TN * (W // WM)) // (W * 64 * 4)):
        t = (x + i * W * 64) * 4
        if BN == 1:
            kk, c = t // BNN, t % BNN
        else:
            kk, c = t % 16, t // 16
        c = gl.minimum(pn * 16 * TN * (W // WM) + c, N - 1)
        args += ((kk * BK + c * BN) * 4,)
        bw += (bp + (kk * BNN + (t % BNN if BN == 1 else t // 16)) * 4,)
    ra = (
        ap
        + ((lane // 8) * BM + (warp % WM) * 16 * (TM if TM > 2 else 2) + (lane % 8) * 4)
        * 4
    )
    rb = (
        bp
        + (
            (lane // 8) * BNN
            + (warp // WM) * 16 * (TN if TN > 2 else 2)
            + (lane % 8) * 4
        )
        * 4
    )
    partial = ()
    for i in gl.static_range(TM * TN * 4):
        partial += (gl.full((W * 64,), 0, gl.float32, L),)
    args += aw + bw + (ra, rb) + (outer + partial if ACCURATE else partial + outer)
    spec: gl.constexpr = asm_spec(
        W, WM, TM, TN, AM, AK, BK, BN, M, N, STEPS, PANEL, ACCURATE
    )
    result = gl.inline_asm_elementwise(
        spec[0],
        spec[1],
        args,
        dtype=(gl.float32,) * (TM * TN * 8),
        is_pure=False,
        pack=1,
    )
    if ACCURATE:
        corrected = ()
        for i in gl.static_range(TM * TN * 4):
            if ACCURATE == 5:
                corrected += (result[i] + result[TM * TN * 4 + i],)
            else:
                corrected += (
                    result[i].to(gl.float64) - result[TM * TN * 4 + i].to(gl.float64),
                )
        return corrected
    else:
        return result[: TM * TN * 4]


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
    W: gl.constexpr,
    WM: gl.constexpr,
    TM: gl.constexpr,
    TN: gl.constexpr,
    TRANS: gl.constexpr,
    PANEL: gl.constexpr = True,
    ACCURATE: gl.constexpr = False,
    SPLIT: gl.constexpr = 1,
):
    A += gl.program_id(2) * K * AK
    B += gl.program_id(2) * K * BK
    C += gl.program_id(2) * M * N
    pm, pn = gl.program_id(0), gl.program_id(1)
    L: gl.constexpr = BlockedLayout([1], [64], [W], [0])
    zero = ()
    for i in gl.static_range(TM * TN * 4):
        zero += (gl.full((W * 64,), 0, gl.float32, L),)
    sa = gl.allocate_shared_memory(
        gl.float32,
        (32, 16 * (TM if TM > 2 else 2) * WM),
        SwizzledSharedLayout(1, 1, 1, [1, 0]),
    )
    sb = gl.allocate_shared_memory(
        gl.float32,
        (32, 16 * (TN if TN > 2 else 2) * (W // WM)),
        SwizzledSharedLayout(1, 1, 1, [1, 0]),
    )
    gl.static_assert(K % 32 == 0 and (K // 112 * 112) % 32 == 0)
    gl.static_assert(AK == 1 and BK == 1 and AM > 1 and BN > 1)
    start: gl.constexpr = K // 112 * 112 if K % 112 else 0
    acc = native(
        A + start * AK,
        B + start * BK,
        sa,
        sb,
        zero,
        pm,
        pn,
        M,
        N,
        AM,
        AK,
        BK,
        BN,
        W,
        WM,
        TM,
        TN,
        K // 16,
        True,
        False,
    )
    for i in gl.static_range(TM):
        for j in gl.static_range(TN):
            store_acc(acc, C, pm, pn, M, N, TM, TN, W, WM, i, j, TRANS)
