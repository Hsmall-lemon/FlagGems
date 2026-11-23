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


"""Preserve FP32 CPU panel and boundary accumulation in one device kernel."""

import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout

from .mm_fp32_cpu_fixed8_gemv import parts


@triton.constexpr_function
def chunk_asm(chunk):
    q = 4 + 2 * chunk
    code = [f"v_cmp_eq_u32 vcc, 0, ${q}"]
    for start in range(0, chunk, 4):
        code += [
            f"v_mac_f32 v0, ${4+start}, ${4+chunk+start}",
            "s_and_saveexec_b64 s[32:33], vcc",
        ]
        for i in range(start + 1, start + 4):
            code += [f"v_mac_f32 v0, ${4+i}, ${4+chunk+i}"]
        code += ["s_xor_b64 exec, s[32:33], exec"]
        for i in range(start + 1, start + 4):
            code += [f"v_mac_f32 v{i%4}, ${4+i}, ${4+chunk+i}"]
        code += ["s_mov_b64 exec, s[32:33]"]
    cons = (
        [f"=&{{v{i}}}" for i in range(4)]
        + ["v"] * (2 * chunk + 1)
        + [str(i) for i in range(4)]
        + ["~{s32}", "~{s33}", "~{scc}", "~{vcc}"]
    )
    return "\n".join(code), ",".join(cons)


@g.jit
def tuple_convert(values, layout: gl.constexpr):
    out = ()
    for i in gl.static_range(len(values)):
        out += (gl.convert_layout(values[i], layout),)
    return out


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
    BO: gl.constexpr = 64,
    CHUNK: gl.constexpr = 16,
):
    P: gl.constexpr = triton.next_power_of_2(triton.cdiv(K, 112))
    O: gl.constexpr = min(BO, 64)
    L: gl.constexpr = BlockedLayout([1, 1], [64 // O, O], [W, 1], [1, 0])
    LP: gl.constexpr = BlockedLayout(
        [1, 1, CHUNK], [64 // O, O, 1], [W, 1, 1], [1, 0, 2]
    )
    p = gl.arange(0, P, layout=SliceLayout(1, L))
    x = gl.program_id(0) * BO + gl.arange(0, BO, layout=SliceLayout(0, L))
    cn = x % N
    # CPU GEMM panels use one chain; the four-column gaps use four chains.
    quad = (cn >= 16) if N == 20 else (cn % 28 >= 24)
    pp = gl.convert_layout(p, SliceLayout(1, SliceLayout(2, LP)))
    xx = gl.convert_layout(x, SliceLayout(0, SliceLayout(2, LP)))
    kk = gl.arange(0, CHUNK, layout=SliceLayout(0, SliceLayout(0, LP)))
    kk3 = gl.expand_dims(gl.expand_dims(kk, 0), 0)
    FL: gl.constexpr = BlockedLayout([1], [64], [W], [0])
    spec: gl.constexpr = chunk_asm(CHUNK)
    part0 = gl.full((P, BO), 0, gl.float32, L)
    part1 = gl.full((P, BO), 0, gl.float32, L)
    part2 = gl.full((P, BO), 0, gl.float32, L)
    part3 = gl.full((P, BO), 0, gl.float32, L)
    qmat = gl.where(
        quad[None, :],
        gl.full((P, BO), 1, gl.int32, L),
        gl.full((P, BO), 0, gl.int32, L),
    )
    qflat = gl.convert_layout(gl.reshape(qmat, (P * BO,)), FL)
    for chunk in range(triton.cdiv(112, CHUNK)):
        k = pp[:, None, None] * 112 + chunk * CHUNK + kk3
        r = xx[None, :, None] // N
        c = xx[None, :, None] % N
        valid = (r < M) & (k < K) & (chunk * CHUNK + kk3 < 112)
        av = gl.load(A + r * AM + k * AK, valid, 0)
        bv = gl.load(B + k * BK + c * BN, valid, 0)
        aseq = parts(gl.reshape(av, (P * BO, CHUNK)), P * BO, CHUNK)
        bseq = parts(gl.reshape(bv, (P * BO, CHUNK)), P * BO, CHUNK)
        aseq = tuple_convert(aseq, FL)
        bseq = tuple_convert(bseq, FL)
        flat0 = gl.convert_layout(gl.reshape(part0, (P * BO,)), FL)
        flat1 = gl.convert_layout(gl.reshape(part1, (P * BO,)), FL)
        flat2 = gl.convert_layout(gl.reshape(part2, (P * BO,)), FL)
        flat3 = gl.convert_layout(gl.reshape(part3, (P * BO,)), FL)
        acc = gl.inline_asm_elementwise(
            spec[0],
            spec[1],
            aseq + bseq + (qflat, flat0, flat1, flat2, flat3),
            dtype=(gl.float32,) * 4,
            is_pure=True,
            pack=1,
        )
        part0 = gl.convert_layout(gl.reshape(acc[0], (P, BO)), L)
        part1 = gl.convert_layout(gl.reshape(acc[1], (P, BO)), L)
        part2 = gl.convert_layout(gl.reshape(acc[2], (P, BO)), L)
        part3 = gl.convert_layout(gl.reshape(acc[3], (P, BO)), L)
    part = gl.where(quad[None, :], (part0 + part2) + (part1 + part3), part0)
    total = (
        gl.sum(gl.where(p[:, None] == K // 112, part, 0.0), axis=0)
        if K % 112
        else gl.full((BO,), 0, gl.float32, SliceLayout(0, L))
    )
    for panel in gl.static_range(K // 112):
        value = gl.sum(gl.where(p[:, None] == panel, part, 0.0), axis=0)
        total = total + value
    gl.store(C + x, total, x < M * N)
