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

"""Declare a complete FP64 MMAC bank in one compiler-visible asm operation."""

import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl


@triton.constexpr_function
def bank_asm(tm, tn, pf):
    count = tm * tn * 8
    instructions = []
    for ki in range(pf):
        base = count + ki * 2 * (tm + tn)
        for i in range(tm):
            for j in range(tn):
                dst = (i * tn + j) * 8
                aa, bb = (base + 2 * i, base + 2 * tm + 2 * j)
                instructions.append(
                    f"v_mmac_16x16x4_f64 v[{dst}:{dst + 7}], v[{aa}:{aa + 1}], v[{bb}:{bb + 1}], v[{dst}:{dst + 7}]"
                )
    return "\n".join(instructions)


@triton.constexpr_function
def bank_constraints(tm, tn, pf):
    count = tm * tn * 8
    aa = [
        f"{{v{count + ki * 2 * (tm + tn) + 2 * i + half}}}"
        for ki in range(pf)
        for i in range(tm)
        for half in range(2)
    ]
    bb = [
        f"{{v{count + ki * 2 * (tm + tn) + 2 * tm + 2 * j + half}}}"
        for ki in range(pf)
        for j in range(tn)
        for half in range(2)
    ]
    return ",".join(
        [f"=&{{v{i}}}" for i in range(count)]
        + aa
        + bb
        + [str(i) for i in range(count)]
        + ["~{memory}"]
    )


@g.jit
def split_bits(value):
    bits = value.to(gl.uint64, bitcast=True)
    return (bits.to(gl.uint32), (bits >> 32).to(gl.uint32))


@g.jit
def compute(aa, bb, acc, TM: gl.constexpr, TN: gl.constexpr, PREFETCH: gl.constexpr):
    operands = ()
    for i in gl.static_range(TM * PREFETCH):
        lo, hi = split_bits(aa[i])
        operands += (lo, hi)
    for i in gl.static_range(TN * PREFETCH):
        lo, hi = split_bits(bb[i])
        operands += (lo, hi)
    for i in gl.static_range(TM * TN * 4):
        lo, hi = split_bits(acc[i])
        operands += (lo, hi)
    DT: gl.constexpr = (gl.uint32,) * (TM * TN * 8)
    values = gl.inline_asm_elementwise(
        bank_asm(TM, TN, PREFETCH),
        bank_constraints(TM, TN, PREFETCH),
        operands,
        dtype=DT,
        is_pure=False,
        pack=1,
    )
    result = ()
    for i in gl.static_range(TM * TN * 4):
        bits = values[2 * i].to(gl.uint64) | values[2 * i + 1].to(gl.uint64) << 32
        result += (bits.to(gl.float64, bitcast=True),)
    return result
