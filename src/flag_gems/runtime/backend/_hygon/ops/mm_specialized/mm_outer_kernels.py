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

"""K=1 product tiles that reuse the input row and column vectors."""

import triton
import triton.language as tl


@triton.jit
def outer_product(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    AM: tl.constexpr,
    BN: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    STORE: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.program_id(1) * COLS + tl.arange(0, COLS)
    a = tl.load(A + rows * AM, rows < M, 0)
    b = tl.load(B + cols * BN, cols < N, 0)
    tl.store(
        C + rows[:, None] * N + cols[None, :],
        a[:, None] * b[None, :],
        (rows[:, None] < M) & (cols[None, :] < N),
        cache_modifier=STORE,
    )
