# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Vectorized dense conversions for gfx936."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout


@gluon.jit
def convert_vector(value, Y):
    if Y.dtype.element_ty == gl.bfloat16:
        bits = value.to(gl.uint32, bitcast=True)
        bias = 0x7FFF + ((bits >> 16) & 1)
        rounded = (bits + bias) >> 16
        rounded = gl.where((bits & 0x7FFFFFFF) > 0x7F800000, 0x7FC0, rounded)
        converted = rounded.to(gl.uint16).to(gl.bfloat16, bitcast=True)
    elif value.dtype == gl.bfloat16:
        bits = value.to(gl.uint16, bitcast=True).to(gl.uint32) << 16
        converted = bits.to(gl.float32, bitcast=True)
    else:
        converted = value.to(gl.float32)
    return converted


@gluon.jit
def cast_vector(X, Y, N: gl.constexpr, BLOCK: gl.constexpr, WARPS: gl.constexpr):
    offsets = gl.program_id(0) * BLOCK + gl.arange(
        0, BLOCK, layout=BlockedLayout([4], [64], [WARPS], [0])
    )
    if N % BLOCK == 0:
        gl.store(Y + offsets, convert_vector(gl.load(X + offsets), Y))
    elif gl.program_id(0) < N // BLOCK:
        gl.store(Y + offsets, convert_vector(gl.load(X + offsets), Y))
    else:
        value = gl.load(X + offsets, offsets < N, other=0)
        gl.store(Y + offsets, convert_vector(value, Y), offsets < N)
