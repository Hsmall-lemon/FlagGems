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

"""Candidate: native fast FP32 pipeline plus ordered nonfused residual.

The native loop preserves every FP32 input bit. Contiguous transposed views
and sufficient K-tail storage permit padded reads for unused output rows;
unused output columns duplicate valid source columns. The final K panel
uses masked loads, and output stores always preserve logical bounds.
"""

import triton
import triton.experimental.gluon.language.amd._layouts as amd_layouts
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SwizzledSharedLayout
from triton.experimental.gluon.language import _core as gc

if not hasattr(amd_layouts, "HCUMFMALayout"):
    amd_layouts.HCUMFMALayout = amd_layouts.AMDMFMALayout
if not hasattr(amd_layouts, "HCUWMMALayout"):
    amd_layouts.HCUWMMALayout = amd_layouts.AMDWMMALayout


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
def store_acc(
    acc,
    C,
    pm,
    pn,
    M: gl.constexpr,
    N: gl.constexpr,
    TM: gl.constexpr,
    TN: gl.constexpr,
    WARPS: gl.constexpr,
    WM: gl.constexpr,
    I: gl.constexpr,
    J: gl.constexpr,
    TRANS: gl.constexpr,
):
    off: gl.constexpr = (I * TN + J) * 4
    value = gl.join(
        gl.join(acc[off], acc[off + 2]), gl.join(acc[off + 1], acc[off + 3])
    )
    value = gl.reshape(value, (WARPS, 4, 16, 4))
    value = gl.permute(value, (0, 2, 3, 1))
    value = gl.reshape(value, (WARPS, 16, 16))
    if TRANS:
        value = gl.permute(value, (0, 2, 1))
    value = gl.reshape(value, (WARPS * 256,))
    L: gl.constexpr = BlockedLayout([2], [64], [WARPS], [0])
    value = gl.convert_layout(value, L)
    x = gl.arange(0, WARPS * 256, layout=L)
    w = x // 256
    rr = (
        pm * 16 * TM * WM
        + (w % WM) * 16 * TM
        + I * 16
        + (x % 16 if TRANS else x % 256 // 16)
    )
    cc = (
        pn * 16 * TN * (WARPS // WM)
        + (w // WM) * 16 * TN
        + J * 16
        + (x % 256 // 16 if TRANS else x % 16)
    )
    offset = cc * M + rr if TRANS else rr * N + cc
    gl.store(C + offset, value, (rr < M) & (cc < N))


@triton.constexpr_function
def read_compute_asm(tm, tn, pf, bm, bn):
    count = tm * tn * 4
    code = []
    for ki in range(pf):
        for i in range(tm // 2):
            dst = count + 4 * i
            code.append(
                f"ds_read_m32x8_b32 v[{dst}:{dst + 3}], ${count} offset:{(ki * 8 * bm + i * 32) * 4}"
            )
        for j in range(tn // 2):
            dst = count + 2 * tm + 4 * j
            code.append(
                f"ds_read_m32x8_b32 v[{dst}:{dst + 3}], ${count + 1} offset:{(ki * 8 * bn + j * 32) * 4}"
            )
        code.append("s_waitcnt lgkmcnt(0)")
        for i in range(tm):
            for j in range(tn):
                dst = (i * tn + j) * 4
                aa = count + 2 * i
                bb = count + 2 * tm + 2 * j
                code.append(
                    f"v_mmac_16x16x8_f32 v[{dst}:{dst + 3}], v[{aa}:{aa + 1}], v[{bb}:{bb + 1}], v[{dst}:{dst + 3}]"
                )
    return "\n".join(code)


@triton.constexpr_function
def read_compute_constraints(tm, tn):
    count = tm * tn * 4
    return ",".join(
        [f"=&{{v{i}}}" for i in range(count)]
        + ["v", "v"]
        + [str(i) for i in range(count)]
        + [f"~{{v{count + i}}}" for i in range(2 * (tm + tn))]
        + ["~{memory}"]
    )


@g.jit
def matrix_compute(
    ap,
    bp,
    acc,
    TM: gl.constexpr,
    TN: gl.constexpr,
    PF: gl.constexpr,
    BM: gl.constexpr,
    BN: gl.constexpr,
):
    DT: gl.constexpr = (gl.float32,) * (TM * TN * 4)
    return gl.inline_asm_elementwise(
        asm=read_compute_asm(TM, TN, PF, BM, BN),
        constraints=read_compute_constraints(TM, TN),
        args=(ap, bp) + acc,
        dtype=DT,
        is_pure=False,
        pack=1,
    )


@g.jit
def global_prefetch(
    A,
    B,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    AM: gl.constexpr,
    AK: gl.constexpr,
    BK: gl.constexpr,
    BN: gl.constexpr,
    WARPS: gl.constexpr,
    WM: gl.constexpr,
    TM: gl.constexpr,
    TN: gl.constexpr,
    PREFETCH: gl.constexpr,
    pm,
    pn,
    step,
    FULL: gl.constexpr,
    SWIZZLE: gl.constexpr,
    TRANS: gl.constexpr,
):
    WN: gl.constexpr = WARPS // WM
    LD: gl.constexpr = BlockedLayout([2], [64], [WARPS], [0])
    ax = gl.arange(0, WM * TM * 128 * PREFETCH, layout=LD)
    bx = gl.arange(0, WN * TN * 128 * PREFETCH, layout=LD)
    if TRANS:
        rr = (pm * 16 * TM * WM + ax % (16 * TM * WM)) % M
        ak = step * 8 * PREFETCH + ax // (16 * TM * WM)
        bk = step * 8 * PREFETCH + bx % (8 * PREFETCH)
        cc = (pn * 16 * TN * WN + bx // (8 * PREFETCH)) % N
    else:
        rr = (pm * 16 * TM * WM + ax // (8 * PREFETCH)) % M
        ak = step * 8 * PREFETCH + ax % (8 * PREFETCH)
        bk = step * 8 * PREFETCH + bx // (16 * TN * WN)
        cc = (pn * 16 * TN * WN + bx % (16 * TN * WN)) % N
    if FULL:
        av = gl.load(A + rr * AM + ak * AK)
        bv = gl.load(B + bk * BK + cc * BN)
    else:
        av = gl.load(A + rr * AM + ak * AK, ak < K, 0)
        bv = gl.load(B + bk * BK + cc * BN, bk < K, 0)
    return av, bv


@g.jit
def shared_compute(
    sa,
    sb,
    av,
    bv,
    acc,
    WARPS: gl.constexpr,
    WM: gl.constexpr,
    TM: gl.constexpr,
    TN: gl.constexpr,
    PREFETCH: gl.constexpr,
    TRANS: gl.constexpr,
):
    WN: gl.constexpr = WARPS // WM
    L0: gl.constexpr = BlockedLayout([1], [64], [WARPS], [0])
    token = gl.full((WARPS * 64,), 0, gl.int32, L0)
    gl.inline_asm_elementwise(
        asm="s_barrier\nv_mov_b32 $0, $1",
        constraints="=v,v,~{memory}",
        args=[token],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )
    if TRANS:
        av = gl.reshape(av, (8 * PREFETCH, 16 * TM * WM))
        bv = gl.permute(gl.reshape(bv, (16 * TN * WN, 8 * PREFETCH)), (1, 0))
    else:
        av = gl.permute(gl.reshape(av, (16 * TM * WM, 8 * PREFETCH)), (1, 0))
        bv = gl.reshape(bv, (8 * PREFETCH, 16 * TN * WN))
    sa.store(av)
    sb.store(bv)
    L: gl.constexpr = BlockedLayout([1], [64], [WARPS], [0])
    x = gl.arange(0, WARPS * 64, layout=L)
    lane = x % 64
    warp = x // 64
    zeros = gl.full((WARPS * 64,), 0, gl.int32, L)
    ap = shared_pointer(sa, zeros, zeros).to(gl.uint32)
    bp = shared_pointer(sb, zeros, zeros).to(gl.uint32)
    ap, bp = gl.inline_asm_elementwise(
        asm="s_waitcnt lgkmcnt(0)\ns_barrier\ns_mov_b32 m0, 0xffff\nv_mov_b32 $0, $2\nv_mov_b32 $1, $3",
        constraints="=&v,=&v,v,v,~{memory},~{m0}",
        args=[ap, bp],
        dtype=(gl.uint32, gl.uint32),
        is_pure=False,
        pack=1,
    )
    ap += ((lane // 8) * (16 * TM * WM) + (warp % WM) * 16 * TM + (lane % 8) * 4) * 4
    bp += ((lane // 8) * (16 * TN * WN) + (warp // WM) * 16 * TN + (lane % 8) * 4) * 4
    return matrix_compute(ap, bp, acc, TM, TN, PREFETCH, 16 * TM * WM, 16 * TN * WN)


@triton.constexpr_function
def native_pipeline_asm(ak, bk, steps):
    # Two 128x8 A panels and one 16x64 B panel per workgroup.
    # Inputs: scalar bases A/B, three vector global offsets, two LDS write
    # addresses, and two LDS matrix-read addresses. All FP32 bits are retained.
    code = [
        "s_mov_b32 s37, 7",
        "s_mov_b64 s[32:33], $32",
        "s_mov_b64 s[34:35], $33",
        f"s_mov_b32 s36, {steps}",
        "s_mov_b32 m0, 0xffff",
        "global_load_dwordx4 v[56:59], $34, s[32:33]",
        "global_load_dwordx4 v[60:63], $35, s[32:33]",
        "global_load_dwordx4 v[64:67], $36, s[34:35]",
        ".Lmatrix_native_loop:",
        "s_waitcnt vmcnt(0)",
        "s_barrier",
        "ds_write_b128 $37, v[56:59]",
        "ds_write_b128 $37, v[60:63] offset:4096",
        "ds_write_b32 $38, v64",
        "ds_write_b32 $38, v65 offset:256",
        "ds_write_b32 $38, v66 offset:512",
        "ds_write_b32 $38, v67 offset:768",
        "s_cmp_gt_u32 s36, 1",
        "s_cbranch_scc0 .Lmatrix_native_prefetch_done",
        f"s_add_u32 s32, s32, {16 * ak * 4}",
        "s_addc_u32 s33, s33, 0",
        f"s_add_u32 s34, s34, {16 * bk * 4}",
        "s_addc_u32 s35, s35, 0",
        "global_load_dwordx4 v[56:59], $34, s[32:33]",
        "global_load_dwordx4 v[60:63], $35, s[32:33]",
        "global_load_dwordx4 v[64:67], $36, s[34:35]",
        ".Lmatrix_native_prefetch_done:",
        "s_waitcnt lgkmcnt(0)",
        "s_barrier",
    ]
    for ki in range(2):
        code += [
            f"ds_read_m32x8_b32 v[{32 + ki * 4}:{35 + ki * 4}], $39 offset:{ki * 4096}",
            f"ds_read_m32x8_b32 v[{40 + ki * 8}:{43 + ki * 8}], $40 offset:{ki * 2048}",
            f"ds_read_m32x8_b32 v[{44 + ki * 8}:{47 + ki * 8}], $40 offset:{ki * 2048 + 128}",
        ]
    for ki in range(2):
        code.append("s_waitcnt lgkmcnt(3)" if ki == 0 else "s_waitcnt lgkmcnt(0)")
        for i in range(2):
            for j in range(4):
                dst = (i * 4 + j) * 4
                ar = 32 + ki * 4 + i * 2
                br = 40 + ki * 8 + j * 2
                code.append(
                    f"v_mmac_16x16x8_f32 v[{dst}:{dst + 3}], v[{ar}:{ar + 1}], v[{br}:{br + 1}], v[{dst}:{dst + 3}]"
                )
    code += [
        "s_sub_u32 s37, s37, 1",
        "s_cmp_eq_u32 s37, 0",
        "s_cbranch_scc0 .Lpanel112_continue",
    ]
    for i in range(32):
        code += [f"v_add_f32 v{68+i}, v{68+i}, v{i}", f"v_mov_b32 v{i}, 0"]
    code += ["s_mov_b32 s37, 7", ".Lpanel112_continue:"]
    code += [
        "s_sub_u32 s36, s36, 1",
        "s_cmp_lg_u32 s36, 0",
        "s_cbranch_scc1 .Lmatrix_native_loop",
    ]
    for i in range(32):
        code += [f"v_mov_b32 v{i}, v{68+i}"]
    for pos in range(len(code)):
        for index in range(40, 31, -1):
            code[pos] = code[pos].replace("$" + str(index), "$" + str(index + 32))
    return "\n".join(code)


@triton.constexpr_function
def native_pipeline_constraints():
    return ",".join(
        [f"=&{{v{i}}}" for i in range(32)]
        + [f"=&{{v{68+i}}}" for i in range(32)]
        + ["s", "s"]
        + ["v"] * 7
        + [str(i) for i in range(64)]
        + [f"~{{v{i}}}" for i in range(32, 68)]
        + [f"~{{s{i}}}" for i in range(32, 38)]
        + ["~{memory}", "~{m0}", "~{scc}"]
    )


@g.jit
def native_pipeline(
    A,
    B,
    sa,
    sb,
    acc,
    pm,
    pn,
    N: gl.constexpr,
    AK: gl.constexpr,
    BN: gl.constexpr,
    STEPS: gl.constexpr,
):
    L: gl.constexpr = BlockedLayout([1], [64], [4], [0])
    x = gl.arange(0, 256, layout=L)
    lane = x % 64
    warp = x // 64
    zero = gl.full((256,), 0, gl.int32, L)
    ap = shared_pointer(sa, zero, zero).to(gl.uint32)
    bp = shared_pointer(sb, zero, zero).to(gl.uint32)
    ao = ((pm * 128 + (x * 4) % 128) + ((x * 4) // 128) * AK) * 4
    ah = ao + 8 * AK * 4
    bo = (((x * 4) % 16) + gl.minimum(pn * 64 + (x * 4) // 16, N - 1) * BN) * 4
    wa = ap + x * 16
    wb = bp + (((x * 4) % 16) * 64 + (x * 4) // 16) * 4
    ra = ap + ((lane // 8) * 128 + warp * 32 + (lane % 8) * 4) * 4
    rb = bp + ((lane // 8) * 64 + (lane % 8) * 4) * 4
    DT: gl.constexpr = (gl.float32,) * 64
    partial = ()
    for i in gl.static_range(32):
        partial += (gl.full((256,), 0, gl.float32, L),)
    result = gl.inline_asm_elementwise(
        asm=native_pipeline_asm(AK, 1, STEPS),
        constraints=native_pipeline_constraints(),
        args=(A.to(gl.uint64), B.to(gl.uint64), ao, ah, bo, wa, wb, ra, rb)
        + partial
        + acc,
        dtype=DT,
        is_pure=False,
        pack=1,
    )
    return result[:32]


@triton.constexpr_function
def residual_assembly(tm, tn, bm, bn, length, depth):
    assert tm == 2 and tn == 4 and depth == 2
    count = tm * tn * 4
    ar = count
    br = ar + tm * depth
    limit = br + 4 * tn * depth
    pa, pb = limit, limit + 1
    code = ["s_waitcnt lgkmcnt(0)", "s_barrier"]

    def loads(k):
        out = [
            f"v_add_u32 v{pa}, {(k//2*2*bm)*4}, ${count}",
            f"v_add_u32 v{pb}, {(k//4*4*bn)*4}, ${count+1}",
        ]
        dst = ar + (k % depth) * tm
        offset = k % 2 * bm
        out.append(
            f"ds_read2_b32 v[{dst}:{dst+1}], v{pa} offset0:{offset} offset1:{offset+16}"
        )
        for j in range(tn):
            for q in (0, 2):
                dst = br + (k % depth) * tn * 4 + j * 4 + q
                offset = k % 4 * bn + j * 16 + q * 4
                assert offset + 4 <= 255
                out.append(
                    f"ds_read2_b32 v[{dst}:{dst+1}], v{pb} offset0:{offset} offset1:{offset+4}"
                )
        return out

    for k in range(min(length, depth)):
        code += loads(k)
    for k in range(length):
        # Nine ds_read2 instructions per term. The threshold never exceeds
        # the nine pending instructions for the following prefetched term.
        code.append(f"s_waitcnt lgkmcnt({9*min(depth-1,length-k-1)})")
        for i in range(tm):
            for j in range(tn):
                for q in range(4):
                    dst = (i * tn + j) * 4 + q
                    aa = ar + (k % depth) * tm + i
                    bb = br + (k % depth) * tn * 4 + j * 4 + q
                    code.append(f"v_mac_f32 v{dst}, v{aa}, v{bb}")
        if k + depth < length:
            code += loads(k + depth)
    code += ["s_waitcnt lgkmcnt(0)", "s_barrier"]
    constraints = (
        [f"=&{{v{i}}}" for i in range(count)]
        + ["v", "v"]
        + [str(i) for i in range(count)]
        + [f"~{{v{i}}}" for i in range(count, limit + 2)]
        + ["~{memory}"]
    )
    return "\n".join(code), ",".join(constraints)


@triton.constexpr_function
def finish4_assembly(tm, tn):
    count = tm * tn * 4
    ar = count
    br = ar + tm
    code = ["s_waitcnt lgkmcnt(0)"]
    for i in range(tm):
        code.append(f"ds_read_b32 v{ar+i}, ${count} offset:{i*16*4}")
    for j in range(tn):
        code.append(f"ds_read_b32 v{br+j}, ${count+1} offset:{j*16*4}")
    code.append("s_waitcnt lgkmcnt(0)")
    for i in range(tm):
        for j in range(tn):
            dst = (i * tn + j) * 4
            code.append(
                f"v_mmac_16x16x4_f32 v[{dst}:{dst+3}], v{ar+i}, v{br+j}, v[{dst}:{dst+3}]"
            )
    code += ["s_waitcnt lgkmcnt(0)", "s_barrier"]
    cons = (
        [f"=&{{v{i}}}" for i in range(count)]
        + ["v"] * 2
        + [str(i) for i in range(count)]
        + [f"~{{v{i}}}" for i in range(count, br + tn)]
        + ["~{memory}"]
    )
    return "\n".join(code), ",".join(cons)


@g.jit
def finish4(ap, bp, acc, TM: gl.constexpr, TN: gl.constexpr):
    spec: gl.constexpr = finish4_assembly(TM, TN)
    return gl.inline_asm_elementwise(
        spec[0],
        spec[1],
        (ap, bp) + acc,
        dtype=(gl.float32,) * (TM * TN * 4),
        is_pure=False,
        pack=1,
    )


@g.jit
def shared_compute_nonfused(
    sa,
    sb,
    av,
    bv,
    acc,
    WARPS: gl.constexpr,
    WM: gl.constexpr,
    TM: gl.constexpr,
    TN: gl.constexpr,
    PREFETCH: gl.constexpr,
    TRANS: gl.constexpr,
    LENGTH: gl.constexpr,
):
    WN: gl.constexpr = WARPS // WM
    L: gl.constexpr = BlockedLayout([1], [64], [WARPS], [0])
    token = gl.full((WARPS * 64,), 0, gl.int32, L)
    gl.inline_asm_elementwise(
        "s_barrier\nv_mov_b32 $0, $1",
        "=v,v,~{memory}",
        [token],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )
    if TRANS:
        av = gl.reshape(av, (8 * PREFETCH, 16 * TM * WM))
        bv = gl.permute(gl.reshape(bv, (16 * TN * WN, 8 * PREFETCH)), (1, 0))
    else:
        av = gl.permute(gl.reshape(av, (16 * TM * WM, 8 * PREFETCH)), (1, 0))
        bv = gl.reshape(bv, (8 * PREFETCH, 16 * TN * WN))
    sa.store(av)
    sb.store(bv)
    x = gl.arange(0, WARPS * 64, layout=L)
    lane, warp = x % 64, x // 64
    zeros = gl.full((WARPS * 64,), 0, gl.int32, L)
    ap = shared_pointer(sa, zeros, (warp % WM) * 16 * TM + lane % 16).to(gl.uint32)
    bp = shared_pointer(sb, zeros, (warp // WM) * 16 * TN + lane // 16).to(gl.uint32)
    gl.static_assert(LENGTH == 16)
    spec: gl.constexpr = residual_assembly(TM, TN, 16 * TM * WM, 16 * TN * WN, 12, 2)
    acc = gl.inline_asm_elementwise(
        spec[0],
        spec[1],
        (ap, bp) + acc,
        dtype=(gl.float32,) * (TM * TN * 4),
        is_pure=False,
        pack=1,
    )
    aq = shared_pointer(sa, 12 + lane // 16, (warp % WM) * 16 * TM + lane % 16).to(
        gl.uint32
    )
    bq = shared_pointer(sb, 12 + lane // 16, (warp // WM) * 16 * TN + lane % 16).to(
        gl.uint32
    )
    return finish4(aq, bq, acc, TM, TN)


@g.jit
def stage_native_tail(
    A,
    B,
    sa,
    sb,
    pm,
    pn,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    AK: gl.constexpr,
    BN: gl.constexpr,
    END: gl.constexpr,
):
    # Accumulators are initialized after global issue and before its wait,
    # preserving +0 bits while allowing these independent moves to overlap.
    # First LDS use in this kernel: no prior reader requires a pre-write barrier.
    # Own writes complete before return; residual_assembly's retained entry
    # barrier publishes all writers before any cross-wave residual LDS read.
    # Deliberately limited geometry: every 128-row A tile is in-bounds.
    # B rows clamp to the final logical row; all 16 residual K are in-bounds.
    gl.static_assert(K == 128 and END == 112 and M % 128 == 0 and BN == K)
    L: gl.constexpr = BlockedLayout([1], [64], [4], [0])
    x = gl.arange(0, 256, layout=L)
    zero = gl.full((256,), 0, gl.int32, L)
    ap = shared_pointer(sa, zero, zero).to(gl.uint32)
    bp = shared_pointer(sb, zero, zero).to(gl.uint32)
    ao = ((pm * 128 + (x * 4) % 128) + ((x * 4) // 128) * AK) * 4
    ah = ao + 8 * AK * 4
    bo = (((x * 4) % 16) + gl.minimum(pn * 64 + (x * 4) // 16, N - 1) * BN) * 4
    wa = ap + x * 16
    wb = bp + (((x * 4) % 16) * 64 + (x * 4) // 16) * 4
    result = gl.inline_asm_elementwise(
        asm="""s_mov_b64 s[32:33], $34
s_mov_b64 s[34:35], $35
s_mov_b32 m0, 0xffff
global_load_dwordx4 v[32:35], $36, s[32:33]
global_load_dwordx4 v[36:39], $37, s[32:33]
global_load_dwordx4 v[40:43], $38, s[34:35]
v_mov_b32 v0, 0
v_mov_b32 v1, 0
v_mov_b32 v2, 0
v_mov_b32 v3, 0
v_mov_b32 v4, 0
v_mov_b32 v5, 0
v_mov_b32 v6, 0
v_mov_b32 v7, 0
v_mov_b32 v8, 0
v_mov_b32 v9, 0
v_mov_b32 v10, 0
v_mov_b32 v11, 0
v_mov_b32 v12, 0
v_mov_b32 v13, 0
v_mov_b32 v14, 0
v_mov_b32 v15, 0
v_mov_b32 v16, 0
v_mov_b32 v17, 0
v_mov_b32 v18, 0
v_mov_b32 v19, 0
v_mov_b32 v20, 0
v_mov_b32 v21, 0
v_mov_b32 v22, 0
v_mov_b32 v23, 0
v_mov_b32 v24, 0
v_mov_b32 v25, 0
v_mov_b32 v26, 0
v_mov_b32 v27, 0
v_mov_b32 v28, 0
v_mov_b32 v29, 0
v_mov_b32 v30, 0
v_mov_b32 v31, 0
s_waitcnt vmcnt(0)
ds_write_b128 $39, v[32:35]
ds_write_b128 $39, v[36:39] offset:4096
ds_write_b32 $40, v40
ds_write_b32 $40, v41 offset:256
ds_write_b32 $40, v42 offset:512
ds_write_b32 $40, v43 offset:768
s_waitcnt lgkmcnt(0)
v_mov_b32 $32, $41
v_mov_b32 $33, $42""",
        constraints=(
            "=&{v0},=&{v1},=&{v2},=&{v3},=&{v4},=&{v5},=&{v6},=&{v7},"
            "=&{v8},=&{v9},=&{v10},=&{v11},=&{v12},=&{v13},=&{v14},=&{v15},"
            "=&{v16},=&{v17},=&{v18},=&{v19},=&{v20},=&{v21},=&{v22},=&{v23},"
            "=&{v24},=&{v25},=&{v26},=&{v27},=&{v28},=&{v29},=&{v30},=&{v31},"
            "=&v,=&v,s,s,v,v,v,v,v,v,v,~{v32},~{v33},~{v34},~{v35},~{v36},~{v37},"
            "~{v38},~{v39},~{v40},~{v41},~{v42},~{v43},~{s32},~{s33},~{s34},~{s35},~{m0},~{memory}"
        ),
        args=(
            (A + END * AK).to(gl.uint64),
            (B + END).to(gl.uint64),
            ao,
            ah,
            bo,
            wa,
            wb,
            ap,
            bp,
        ),
        dtype=(gl.float32,) * 32 + (gl.uint32, gl.uint32),
        is_pure=False,
        pack=1,
    )
    return result[32], result[33], result[:32]


@triton.constexpr_function
def hybrid6_assembly(tm, tn):
    assert (tm, tn) == (2, 4)
    first, _ = residual_assembly(tm, tn, 128, 64, 5, 2)
    code = first.splitlines()  # Original ordered NF relativeK0..4.

    def matrix8(start, which):
        out = [
            f"v_add_u32 v68, {start*128*4}, $37",
            f"v_add_u32 v69, {start*64*4}, $38",
            "ds_read_m32x8_b32 v[32:35], v68",
            "ds_read_m32x8_b32 v[36:39], v69",
            "ds_read_m32x8_b32 v[40:43], v69 offset:128",
            "s_waitcnt lgkmcnt(0)",
        ]
        if which == "five_to_nine":
            # start4: low K4/6/8/10 keeps kz1,2; highK5/7/9/11 keeps0,1,2.
            out += ["v_add_u32 v74, 0xffffffff, $36", "v_cmp_lt_u32 vcc, v74, 2"]
        else:
            assert which == "eleven_to_fifteen"
            # start8: low K8/10/12/14 keepskz2,3; highK9/11/13/15 keeps1,2,3.
            out += ["v_cmp_lt_u32 vcc, $36, 2"]
        for r in range(32, 44, 2):
            if which == "five_to_nine":
                out.append(f"v_cndmask_b32 v{r}, 0, v{r}, vcc")
            else:
                out.append(f"v_cndmask_b32 v{r}, v{r}, 0, vcc")
        if which == "five_to_nine":
            out += ["v_cmp_lt_u32 vcc, $36, 3"]
        else:
            out += ["v_cmp_ne_u32 vcc, 0, $36"]
        for r in range(33, 44, 2):
            out.append(f"v_cndmask_b32 v{r}, 0, v{r}, vcc")
        for i in range(2):
            for j in range(4):
                dst = (i * 4 + j) * 4
                aa = 32 + i * 2
                bb = 36 + j * 2
                out.append(
                    f"v_mmac_16x16x8_f32 v[{dst}:{dst+3}], v[{aa}:{aa+1}], v[{bb}:{bb+1}], v[{dst}:{dst+3}]"
                )
        return out

    code += ["s_mov_b32 m0, 0xffff"]
    code += matrix8(4, "five_to_nine")
    # Single original NF relativeK10 between the two valid fused windows.
    code += ["v_add_u32 v74, 5120, $32", "v_add_u32 v75, 2560, $33"]
    one, _ = residual_assembly(tm, tn, 128, 64, 1, 2)
    one = one.splitlines()
    assert one[:2] == ["s_waitcnt lgkmcnt(0)", "s_barrier"]
    assert one[-2:] == ["s_waitcnt lgkmcnt(0)", "s_barrier"]
    code += [line.replace("$32", "v74").replace("$33", "v75") for line in one[2:-2]]
    code += matrix8(8, "eleven_to_fifteen")
    code += ["s_waitcnt lgkmcnt(0)", "s_barrier"]
    constraints = (
        [f"=&{{v{i}}}" for i in range(32)]
        + ["v"] * 7
        + [str(i) for i in range(32)]
        + [f"~{{v{i}}}" for i in range(32, 76)]
        + ["~{vcc}", "~{m0}", "~{memory}"]
    )
    return "\n".join(code), ",".join(constraints)


@g.jit
def shared_compute_prepared(sa, sb, ap0, bp0, acc, TM: gl.constexpr, TN: gl.constexpr):
    L: gl.constexpr = BlockedLayout([1], [64], [4], [0])
    x = gl.arange(0, 256, layout=L)
    lane, warp = x % 64, x // 64
    kz = lane // 16
    ap = ap0 + (warp * 16 * TM + lane % 16) * 4
    bp = bp0 + kz * 4
    aq = ap0 + (kz * 128 + warp * 16 * TM + lane % 16) * 4
    bq = bp0 + (kz * 64 + lane % 16) * 4
    # Original native matrix-read base mappings, now used by residual windows.
    ra = ap0 + ((lane // 8) * 128 + warp * 32 + (lane % 8) * 4) * 4
    rb = bp0 + ((lane // 8) * 64 + (lane % 8) * 4) * 4
    spec: gl.constexpr = hybrid6_assembly(TM, TN)
    return gl.inline_asm_elementwise(
        spec[0],
        spec[1],
        (ap, bp, aq, bq, kz, ra, rb) + acc,
        dtype=(gl.float32,) * (TM * TN * 4),
        is_pure=False,
        pack=1,
    )


@g.jit
def mmac_tiled(
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
    WARPS: gl.constexpr,
    WM: gl.constexpr,
    TM: gl.constexpr,
    TN: gl.constexpr,
    PREFETCH: gl.constexpr,
    GROUP: gl.constexpr,
    PIPE: gl.constexpr,
    SWIZZLE: gl.constexpr,
    TRANS: gl.constexpr,
    SPLIT: gl.constexpr = 1,
):
    gl.static_assert(SPLIT == 1)
    L: gl.constexpr = BlockedLayout([1], [64], [WARPS], [0])
    nm = gl.cdiv(M, 16 * TM * WM)
    nn = gl.cdiv(N, 16 * TN * (WARPS // WM))
    p = gl.program_id(0) + gl.program_id(1) * nm
    if GROUP == 0:
        pm = gl.program_id(0)
        pn = gl.program_id(1)
    else:
        group = p // (GROUP * nn)
        size = gl.minimum(nm - group * GROUP, GROUP)
        pm = group * GROUP + (p % (GROUP * nn)) % size
        pn = (p % (GROUP * nn)) // size
    acc = ()
    for i in gl.static_range(TM * TN * 4):
        acc += (gl.full((WARPS * 64,), 0, gl.float32, L),)
    gl.static_assert(
        WARPS == 4 and WM == 4 and TM == 2 and TN == 4 and PREFETCH == 2 and TRANS
    )
    gl.static_assert(AM == 1 and BK == 1 and AK == M and BN >= K)
    end: gl.constexpr = (K // 112) * 112
    tail: gl.constexpr = K - end
    gl.static_assert(tail == 16)
    gl.static_assert(M % 128 == 0 or tail >= (127 + M - 1) // M)
    sa = gl.allocate_shared_memory(
        gl.float32, (16, 128), SwizzledSharedLayout(1, 1, 1, [1, 0])
    )
    sb = gl.allocate_shared_memory(
        gl.float32, (16, 64), SwizzledSharedLayout(1, 1, 1, [1, 0])
    )
    gl.static_assert(K == 128 and end == 112 and M % 128 == 0 and BN == K)
    ap0, bp0, acc = stage_native_tail(A, B, sa, sb, pm, pn, M, N, K, AK, BN, end)
    acc = shared_compute_prepared(sa, sb, ap0, bp0, acc, TM, TN)
    if end > 0:
        acc = native_pipeline(A, B, sa, sb, acc, pm, pn, N, AK, BN, end // 16)
    if TRANS:
        x = gl.arange(0, WARPS * 64, layout=L)
        lane = x % 64
        warp = x // 64
        for i in gl.static_range(TM):
            for j in gl.static_range(TN):
                for z in gl.static_range(4):
                    rr = pm * 16 * TM * WM + (warp % WM) * 16 * TM + i * 16 + lane % 16
                    cc = (
                        pn * 16 * TN * (WARPS // WM)
                        + (warp // WM) * 16 * TN
                        + j * 16
                        + lane // 16
                        + z * 4
                    )
                    v = acc[(i * TN + j) * 4 + z]
                    if M % (16 * TM * WM) == 0 and N % (16 * TN * (WARPS // WM)) == 0:
                        gl.store(C + rr + cc * M, v)
                    else:
                        gl.store(C + rr + cc * M, v, (rr < M) & (cc < N))
    else:
        for i in gl.static_range(TM):
            for j in gl.static_range(TN):
                store_acc(acc, C, pm, pn, M, N, TM, TN, WARPS, WM, i, j, False)
