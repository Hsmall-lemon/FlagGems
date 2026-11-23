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


"""FP32 vector products with the published CPU reference reduction order."""

import triton
import triton.language as tl


@triton.jit
def gemv(
    A,
    B,
    C,
    M: tl.constexpr,
    K: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BK: tl.constexpr,
    A_OFFSET: tl.constexpr,
    B_OFFSET: tl.constexpr,
    BO: tl.constexpr,
    A_START: tl.constexpr = 0,
    B_START: tl.constexpr = 0,
    C_START: tl.constexpr = 0,
    B_BATCH_STRIDE: tl.constexpr = 0,
    C_BATCH_STRIDE: tl.constexpr = 0,
    OUTPUT_STRIDE: tl.constexpr = 1,
):
    batch = tl.program_id(1)
    A += A_START
    B += B_START + batch * B_BATCH_STRIDE
    C += C_START + batch * C_BATCH_STRIDE
    b_offset = B_OFFSET + batch * B_BATCH_STRIDE
    x = tl.arange(0, BO * 8)
    row = tl.program_id(0) * BO + x // 8
    lane = x % 8
    edge = row >= (M // 4) * 4
    last = tl.minimum((row // 4) * 4 + 3, M - 1)
    prefix = 4 - ((A_OFFSET + last * AM) % 4)
    prefix = tl.where(edge, 4 - b_offset % 4, prefix)
    parts = tl.where(edge & ((row % 2) == 0), 8, 4)
    if AK != 1 and AM == 1:
        prefix = tl.where(edge, prefix, 0)
        parts = tl.where(edge, parts, 1)
    end = prefix + ((K - prefix) // parts) * parts
    acc = tl.full((BO * 8,), 0, tl.float32)
    for prefix_k in tl.static_range(4):
        av = tl.load(A + row * AM + prefix_k * AK, (row < M) & (prefix_k < prefix), 0)
        bv = tl.load(
            B + prefix_k * BK + tl.zeros((BO * 8,), tl.int32), prefix_k < prefix, 0
        )
        acc = acc + av * bv
    acc = tl.where(lane == 0, acc, 0.0)
    for step in range(K if AK != 1 and AM == 1 else K // 4):
        kk = prefix + step * parts + lane
        active = (row < M) & (lane < parts) & (kk < end)
        av = tl.load(A + row * AM + kk * AK, active, 0)
        bv = tl.load(B + kk * BK, active, 0)
        acc = acc + av * bv
    mat = tl.reshape(acc, (BO, 8))
    values = ()
    for i in tl.static_range(8):
        index = tl.full((BO, 1), i, tl.int32)
        values += (tl.sum(tl.gather(mat, index, axis=1), axis=1),)
    value = ((values[0] + values[4]) + (values[2] + values[6])) + (
        (values[1] + values[5]) + (values[3] + values[7])
    )
    rr = tl.program_id(0) * BO + tl.arange(0, BO)
    last = tl.minimum((rr // 4) * 4 + 3, M - 1)
    prefix = 4 - ((A_OFFSET + last * AM) % 4)
    edge = rr >= (M // 4) * 4
    prefix = tl.where(edge, 4 - b_offset % 4, prefix)
    parts = tl.where(edge & ((rr % 2) == 0), 8, 4)
    if AK != 1 and AM == 1:
        prefix = tl.where(edge, prefix, 0)
        parts = tl.where(edge, parts, 1)
    end = prefix + ((K - prefix) // parts) * parts
    for t in tl.static_range(7):
        kk = end + t
        av = tl.load(A + rr * AM + kk * AK, (rr < M) & (kk < K), 0)
        bv = tl.load(B + kk * BK, kk < K, 0)
        value = value + av * bv
    tl.store(C + rr * OUTPUT_STRIDE, value, rr < M)
