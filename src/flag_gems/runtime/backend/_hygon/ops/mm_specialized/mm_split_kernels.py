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

"""Split the existing FP64 swizzle computation along K."""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from .mm_mmac_swizzle import mmac_tiled


@g.jit
def swizzle_split(
    A,
    B,
    P,
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
    SPLIT: gl.constexpr,
):
    gl.static_assert(K >= SPLIT)
    part = gl.program_id(2)
    chunk: gl.constexpr = K // SPLIT
    tail: gl.constexpr = K - (SPLIT - 1) * chunk
    aa = A + part * chunk * AK
    bb = B + part * chunk * BK
    pp = P + part * M * N
    if tail == chunk:
        mmac_tiled(
            aa,
            bb,
            pp,
            M,
            N,
            chunk,
            AM,
            AK,
            BK,
            BN,
            WARPS,
            WM,
            TM,
            TN,
            PREFETCH,
            1,
            True,
            True,
            False,
        )
    elif part == SPLIT - 1:
        mmac_tiled(
            aa,
            bb,
            pp,
            M,
            N,
            tail,
            AM,
            AK,
            BK,
            BN,
            WARPS,
            WM,
            TM,
            TN,
            PREFETCH,
            1,
            True,
            True,
            False,
        )
    else:
        mmac_tiled(
            aa,
            bb,
            pp,
            M,
            N,
            chunk,
            AM,
            AK,
            BK,
            BN,
            WARPS,
            WM,
            TM,
            TN,
            PREFETCH,
            1,
            True,
            True,
            False,
        )
