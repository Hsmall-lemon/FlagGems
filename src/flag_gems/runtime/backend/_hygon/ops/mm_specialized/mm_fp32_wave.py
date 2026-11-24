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

"""FP32 loads with native FP64 MMAC and a single explicitly constrained loop."""

import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SwizzledSharedLayout
from triton.experimental.gluon.language import _core as gc

from .mm_mmac_swizzle import store_acc


@gc.builtin
def shared_pointer(desc, rows, cols, _semantic=None):
    ptr = gc.pointer_type(gl.float32, 3)
    ty = gc.distributed_type(ptr, rows.shape, rows.type.layout)
    op = _semantic.builder.create_local_pointers(
        ty.to_ir(_semantic.builder), desc.handle, rows.handle, cols.handle
    )
    integer_ty = gc.distributed_type(gl.uint64, rows.shape, rows.type.layout)
    value = _semantic.builder.create_ptr_to_int(
        op.get_result(0), integer_ty.to_ir(_semantic.builder)
    )
    return gc.tensor(value, integer_ty)


@g.jit
def pack64(lo, hi):
    return (lo.to(gl.uint64) | hi.to(gl.uint64) << 32).to(gl.float64, bitcast=True)


@triton.constexpr_function
def asm_spec(w, wm, tm, tn, am, ak, bk, bn, steps):
    bm, bnn = 16 * tm * wm, 16 * tn * (w // wm)
    count = tm * tn * 8
    ar = count
    br = ar + 4 * tm
    cv = br + 4 * tn
    ga = cv + 2 * (tm + tn)
    na = 16 * bm // (w * 64 * 4)
    nb = 16 * bnn // (w * 64 * 4)
    gb = ga + 4 * na
    limit = gb + 4 * nb
    ia, ib = count, count + 1
    vec = count + 2
    aoff = [vec + i for i in range(na)]
    boff = [vec + na + i for i in range(nb)]
    aw = [vec + na + nb + i for i in range(na)]
    bw = [vec + 2 * na + nb + i for i in range(nb)]
    ra, rb = vec + 2 * na + 2 * nb, vec + 2 * na + 2 * nb + 1
    code = [
        f"s_mov_b64 s[32:33], ${ia}",
        f"s_mov_b64 s[34:35], ${ib}",
        f"s_mov_b32 s36, {steps}",
        "s_mov_b32 m0, 0xffff",
    ]

    def load():
        return [
            f"global_load_dwordx4 v[{ga+4*i}:{ga+4*i+3}], ${aoff[i]}, s[32:33]"
            for i in range(na)
        ] + [
            f"global_load_dwordx4 v[{gb+4*i}:{gb+4*i+3}], ${boff[i]}, s[34:35]"
            for i in range(nb)
        ]

    code += load() + [".Lpromote_loop:", "s_waitcnt vmcnt(0)", "s_barrier"]
    for i in range(na):
        if am == 1:
            code += [f"ds_write_b128 ${aw[i]}, v[{ga+4*i}:{ga+4*i+3}]"]
        else:
            code += [
                f"ds_write_b32 ${aw[i]}, v{ga+4*i+j} offset:{j*bm*8}" for j in range(4)
            ]
    for i in range(nb):
        if bn == 1:
            code += [f"ds_write_b128 ${bw[i]}, v[{gb+4*i}:{gb+4*i+3}]"]
        else:
            code += [
                f"ds_write_b32 ${bw[i]}, v{gb+4*i+j} offset:{j*bnn*8}" for j in range(4)
            ]
    code += [
        "s_cmp_gt_u32 s36, 1",
        "s_cbranch_scc0 .Lpromote_prefetch_done",
        f"s_add_u32 s32, s32, {16*ak*4}",
        "s_addc_u32 s33, s33, 0",
        f"s_add_u32 s34, s34, {16*bk*4}",
        "s_addc_u32 s35, s35, 0",
    ]
    code += load() + [".Lpromote_prefetch_done:", "s_waitcnt lgkmcnt(0)", "s_barrier"]
    for ki in range(2):
        for i in range(tm // 2):
            dst = ar + ki * 2 * tm + 4 * i
            code += [
                f"ds_read_m32x8_b32 v[{dst}:{dst+3}], ${ra} offset:{(ki*8*bm+i*32)*4}"
            ]
        for j in range(tn // 2):
            dst = br + ki * 2 * tn + 4 * j
            code += [
                f"ds_read_m32x8_b32 v[{dst}:{dst+3}], ${rb} offset:{(ki*8*bnn+j*32)*4}"
            ]
    code += ["s_waitcnt lgkmcnt(0)"]
    for ki in range(2):
        for half in range(2):
            for i in range(tm):
                dst = cv + 2 * i
                code += [f"v_cvt_f64_f32 v[{dst}:{dst+1}], v{ar+ki*2*tm+i*2+half}"]
            for j in range(tn):
                dst = cv + 2 * tm + 2 * j
                code += [f"v_cvt_f64_f32 v[{dst}:{dst+1}], v{br+ki*2*tn+j*2+half}"]
            for i in range(tm):
                for j in range(tn):
                    dst = (i * tn + j) * 8
                    aa = cv + 2 * i
                    bb = cv + 2 * tm + 2 * j
                    code += [
                        f"v_mmac_16x16x4_f64 v[{dst}:{dst+7}], v[{aa}:{aa+1}], v[{bb}:{bb+1}], v[{dst}:{dst+7}]"
                    ]
    code += [
        "s_sub_u32 s36, s36, 1",
        "s_cmp_lg_u32 s36, 0",
        "s_cbranch_scc1 .Lpromote_loop",
    ]
    cons = (
        [f"=&{{v{i}}}" for i in range(count)]
        + ["s", "s"]
        + ["v"] * (2 * na + 2 * nb + 2)
        + [str(i) for i in range(count)]
        + [f"~{{v{i}}}" for i in range(ar, limit)]
        + [f"~{{s{i}}}" for i in range(32, 37)]
        + ["~{memory}", "~{m0}", "~{scc}"]
    )
    return "\n".join(
        [line for line in code if w != 1 or line != "s_barrier"]
    ), ",".join(cons)


@g.jit
def native(
    A,
    B,
    sa,
    sb,
    acc,
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
):
    BM: gl.constexpr = 16 * TM * WM
    BNN: gl.constexpr = 16 * TN * (W // WM)
    L: gl.constexpr = BlockedLayout([1], [64], [W], [0])
    x = gl.arange(0, W * 64, layout=L)
    lane, warp = x % 64, x // 64
    zero = gl.full((W * 64,), 0, gl.int32, L)
    ap = shared_pointer(sa, zero, zero).to(gl.uint32)
    bp = shared_pointer(sb, zero, zero).to(gl.uint32)
    args = (A, B)
    aw, bw = (), ()
    for i in gl.static_range(16 * BM // (W * 64 * 4)):
        t = (x + i * W * 64) * 4
        if AM == 1:
            r, kk = t % BM, t // BM
        else:
            r, kk = t // 16, t % 16
        sk = (kk // 8) * 8 + (kk % 4) * 2 + (kk % 8) // 4
        r = gl.minimum(pm * BM + r, M - 1)
        args += ((r * AM + kk * AK) * 4,)
        aw += (ap + (sk * BM + (t % BM if AM == 1 else t // 16)) * 4,)
    for i in gl.static_range(16 * BNN // (W * 64 * 4)):
        t = (x + i * W * 64) * 4
        if BN == 1:
            kk, c = t // BNN, t % BNN
        else:
            kk, c = t % 16, t // 16
        sk = (kk // 8) * 8 + (kk % 4) * 2 + (kk % 8) // 4
        c = gl.minimum(pn * BNN + c, N - 1)
        args += ((kk * BK + c * BN) * 4,)
        bw += (bp + (sk * BNN + (t % BNN if BN == 1 else t // 16)) * 4,)
    ra = ap + ((lane // 8) * BM + (warp % WM) * 16 * TM + (lane % 8) * 4) * 4
    rb = bp + ((lane // 8) * BNN + (warp // WM) * 16 * TN + (lane % 8) * 4) * 4
    args += aw + bw + (ra, rb) + acc
    spec: gl.constexpr = asm_spec(W, WM, TM, TN, AM, AK, BK, BN, STEPS)
    return gl.inline_asm_elementwise(
        spec[0],
        spec[1],
        args,
        dtype=(gl.uint32,) * (TM * TN * 8),
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
    W: gl.constexpr,
    WM: gl.constexpr,
    TM: gl.constexpr,
    TN: gl.constexpr,
    TRANS: gl.constexpr,
    SPLIT: gl.constexpr = 1,
    TOTAL_K: gl.constexpr = 0,
):
    gl.static_assert(W == 1, "single-wave synchronization is required")
    split = gl.program_id(2)
    A += split * K * AK
    B += split * K * BK
    C += split * M * N
    pm, pn = gl.program_id(0), gl.program_id(1)
    L: gl.constexpr = BlockedLayout([1], [64], [W], [0])
    acc = ()
    for i in gl.static_range(TM * TN * 8):
        acc += (gl.full((W * 64,), 0, gl.uint32, L),)
    sa = gl.allocate_shared_memory(
        gl.float32, (16, 16 * TM * WM), SwizzledSharedLayout(1, 1, 1, [1, 0])
    )
    sb = gl.allocate_shared_memory(
        gl.float32, (16, 16 * TN * (W // WM)), SwizzledSharedLayout(1, 1, 1, [1, 0])
    )
    end: gl.constexpr = (K // 16) * 16
    if end > 0:
        acc = native(
            A, B, sa, sb, acc, pm, pn, M, N, AM, AK, BK, BN, W, WM, TM, TN, end // 16
        )
    gl.static_assert(K % 16 == 0, "promotion path requires complete K panels")
    vals = ()
    for i in gl.static_range(TM * TN * 4):
        vals += (pack64(acc[2 * i], acc[2 * i + 1]),)
    for i in gl.static_range(TM):
        for j in gl.static_range(TN):
            store_acc(vals, C, pm, pn, M, N, TM, TN, W, WM, i, j, TRANS)
