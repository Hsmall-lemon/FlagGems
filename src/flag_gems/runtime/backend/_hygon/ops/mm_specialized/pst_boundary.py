"""Shape-based boundary partition for the CPU SGEMM reference arithmetic."""

import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout

from .cpu_order_fixed8_gemv import parts


@g.jit
def columns(
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
    GAPS: gl.constexpr,
    ROWS: gl.constexpr,
    WIDTH: gl.constexpr,
    W: gl.constexpr,
    STEP: gl.constexpr,
    MODE: gl.constexpr,
):
    L: gl.constexpr = BlockedLayout([1], [64], [W], [0])
    LP: gl.constexpr = BlockedLayout([1, STEP], [64, 1], [W, 1], [0, 1])
    ix = gl.program_id(0) * W * 64 + gl.arange(0, W * 64, layout=L)
    rr, gap = ix // WIDTH, ix % WIDTH
    cc = gl.full((W * 64,), 0, gl.int32, L)
    cg = cc
    cw = cc
    cursor: gl.constexpr = 0
    for gi in gl.static_range(len(GAPS)):
        start, width = GAPS[gi]
        cc = gl.where(
            (gap >= cursor) & (gap < cursor + width), start + gap - cursor, cc
        )
        cg = gl.where((gap >= cursor) & (gap < cursor + width), start, cg)
        cw = gl.where((gap >= cursor) & (gap < cursor + width), width, cw)
        cursor += width
    valid = gl.full((W * 64,), False, gl.int1, L)
    for ri in gl.static_range(len(ROWS)):
        start, length = ROWS[ri]
        valid = valid | ((rr >= start) & (rr < start + length))
    valid = valid & (rr < M)
    r = gl.convert_layout(rr, SliceLayout(1, LP))
    c = gl.convert_layout(cc, SliceLayout(1, LP))
    active = gl.convert_layout(valid, SliceLayout(1, LP))
    ki = gl.arange(0, STEP, layout=SliceLayout(0, LP))
    total = gl.full((W * 64,), 0, gl.float32, L)
    pref = (4 - (rr * N + cg) % 4) % 4
    pre = (cw >= pref + 4) & (cc - cg < pref)
    for panel in gl.static_range(triton.cdiv(K, 112)):
        START = (
            K // 112 * 112
            if K % 112 and panel == 0
            else (panel - 1) * 112 if K % 112 else panel * 112
        )
        LENGTH = K % 112 if K % 112 and panel == 0 else 112
        value = (
            total if MODE == 0 or MODE == 3 else gl.full((W * 64,), 0, gl.float32, L)
        )
        a0 = gl.full((W * 64,), 0, gl.float32, L)
        a1 = a0
        a2 = a0
        a3 = a0
        for block in range((LENGTH + STEP - 1) // STEP):
            kval = START + block * STEP + ki[None, :]
            av = gl.load(
                A + r[:, None] * AM + kval * AK,
                active[:, None] & (block * STEP + ki[None, :] < LENGTH),
                0,
            )
            bv = gl.load(
                B + kval * BK + c[:, None] * BN,
                active[:, None] & (block * STEP + ki[None, :] < LENGTH),
                0,
            )
            prod = parts(av * bv, W * 64, STEP)
            for t in gl.static_range(STEP):
                p = gl.convert_layout(prod[t], L)
                if MODE == 2:
                    if t % 4 == 0:
                        a0 = a0 + p
                    elif t % 4 == 1:
                        a1 = a1 + p
                    elif t % 4 == 2:
                        a2 = a2 + p
                    else:
                        a3 = a3 + p
                elif MODE == 3:
                    if t % 8 == 0:
                        secondary = gl.full((W * 64,), 0, gl.float32, L)
                    if t % 8 < 4:
                        value = value + p
                    else:
                        secondary = secondary + p
                        value = gl.where(pre, value, value + p)
                    if t % 8 == 7:
                        value = gl.where(pre, value + secondary, value)
                else:
                    value = value + p
        if MODE == 2:
            value = (a0 + a2) + (a1 + a3)
        if MODE == 1 or MODE == 2:
            total = total + value
        else:
            total = value
    gl.store(C + rr * N + cc, total, valid)
