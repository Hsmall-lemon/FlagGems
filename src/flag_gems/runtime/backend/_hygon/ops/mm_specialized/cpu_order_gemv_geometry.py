"""Experimental GEMV with geometry-derived scalar and strided SSE tail orders."""

import triton
import triton.experimental.gluon.language.amd._layouts as amd_layouts

if not hasattr(amd_layouts, "HCUMFMALayout"):
    amd_layouts.HCUMFMALayout = amd_layouts.AMDMFMALayout
if not hasattr(amd_layouts, "HCUWMMALayout"):
    amd_layouts.HCUWMMALayout = amd_layouts.AMDWMMALayout
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout


@triton.constexpr_function
def reverse(i, n):
    out = 0
    for _ in range(n.bit_length() - 1):
        out = (out << 1) | (i & 1)
        i >>= 1
    return out


@g.jit
def parts(x, ROWS: gl.constexpr, LEN: gl.constexpr):
    values = (x,)
    for level in gl.static_range(LEN.bit_length() - 1):
        out = ()
        for i in gl.static_range(2**level):
            a, b = gl.split(gl.reshape(values[i], (ROWS, LEN // (2 ** (level + 1)), 2)))
            out += (a, b)
        values = out
    result = ()
    for i in gl.static_range(LEN):
        result += (gl.reshape(values[reverse(i, LEN)], (ROWS,)),)
    return result


@g.jit
def gemv(
    A,
    B,
    C,
    M: gl.constexpr,
    K: gl.constexpr,
    AM: gl.constexpr,
    AK: gl.constexpr,
    BK: gl.constexpr,
    A_OFFSET: gl.constexpr,
    B_OFFSET: gl.constexpr,
    BO: gl.constexpr,
    STEPS: gl.constexpr = 16,
    A_START: gl.constexpr = 0,
    B_START: gl.constexpr = 0,
    C_START: gl.constexpr = 0,
    B_BATCH_STRIDE: gl.constexpr = 0,
    C_BATCH_STRIDE: gl.constexpr = 0,
    OUTPUT_STRIDE: gl.constexpr = 1,
):
    batch = gl.program_id(1)
    A += A_START
    B += B_START + batch * B_BATCH_STRIDE
    C += C_START + batch * C_BATCH_STRIDE
    b_offset = B_OFFSET + batch * B_BATCH_STRIDE
    L: gl.constexpr = BlockedLayout([1], [64], [1], [0])
    LP: gl.constexpr = BlockedLayout([1, STEPS], [64, 1], [1, 1], [0, 1])
    x = gl.arange(0, BO * 8, layout=L)
    row, lane = gl.program_id(0) * BO + x // 8, x % 8
    edge = row >= (M // 4) * 4
    last = gl.minimum((row // 4) * 4 + 3, M - 1)
    prefix = 4 - ((A_OFFSET + last * AM) % 4)
    prefix = gl.where(edge, 4 - b_offset % 4, prefix)
    lanes = gl.where(edge & ((row % 2) == 0), 8, 4)
    if AK != 1 and AM == 1:
        prefix = gl.where(edge, prefix, 0)
        lanes = gl.where(edge, lanes, 1)
    if AK != 1 and AM == 1:
        # SSE trans=N updates each output in increasing K order, including tails.
        prefix = gl.full((BO * 8,), 0, gl.int32, L)
        lanes = gl.full((BO * 8,), 1, gl.int32, L)
    elif BK != 1:
        # xsgemv b4043ef: seed chain 0 with product 0, then eight chains
        # for k=1..8, k=9..16; b40453c..57b folds the chains in order.
        prefix = gl.where(edge, 1, prefix)
        lanes = gl.where(edge, 8, lanes)
    end = prefix + ((K - prefix) // lanes) * lanes
    acc = gl.full((BO * 8,), 0, gl.float32, L)
    for pki in gl.static_range(4):
        apref = gl.load(A + row * AM + pki * AK, (row < M) & (pki < prefix), 0)
        bpref = gl.load(
            B + pki * BK + gl.zeros((BO * 8,), gl.int32, L), pki < prefix, 0
        )
        acc = acc + apref * bpref
    acc = gl.where(lane == 0, acc, 0.0)
    st = gl.arange(0, STEPS, layout=SliceLayout(0, LP))
    rr = gl.convert_layout(row, SliceLayout(1, LP))
    pp = gl.convert_layout(prefix, SliceLayout(1, LP))
    ll = gl.convert_layout(lane, SliceLayout(1, LP))
    ls = gl.convert_layout(lanes, SliceLayout(1, LP))
    ee = gl.convert_layout(end, SliceLayout(1, LP))
    for step in range(triton.cdiv(K, STEPS if AK != 1 and AM == 1 else STEPS * 4)):
        ck = pp[:, None] + (step * STEPS + st[None, :]) * ls[:, None] + ll[:, None]
        active = (rr[:, None] < M) & (ll[:, None] < ls[:, None]) & (ck < ee[:, None])
        av = gl.load(A + rr[:, None] * AM + ck * AK, active, 0)
        bv = gl.load(B + ck * BK, active, 0)
        product = parts(av * bv, BO * 8, STEPS)
        for i in gl.static_range(STEPS):
            pv = gl.convert_layout(product[i], L)
            if AK != 1 and AM == 1:
                # xsgemv b404855..8d6: before output alignment, each
                # eight-K update is (old+p0+p1+p2+p3)+(p4+p5+p6+p7).
                yp = (4 - (C_START + batch * C_BATCH_STRIDE) % 4) % 4
                scalar = (row < yp) & (M >= yp + 8) & (step * STEPS + i < K // 8 * 8)
                if i % 8 == 4:
                    secondary = gl.full((BO * 8,), 0, gl.float32, L)
                if i % 8 >= 4:
                    secondary = secondary + pv
                    acc = gl.where(scalar, acc, acc + pv)
                    if i % 8 == 7:
                        acc = gl.where(scalar, acc + secondary, acc)
                else:
                    acc = acc + pv
            else:
                acc = acc + pv
    mat = gl.reshape(acc, (BO, 8))
    ML: gl.constexpr = mat.type.layout
    ix = gl.arange(0, 8, layout=SliceLayout(0, ML))
    sums = ()
    for i in gl.static_range(8):
        sums += (gl.sum(gl.where(ix[None, :] == i, mat, 0.0), axis=1),)
    value = ((sums[0] + sums[4]) + (sums[2] + sums[6])) + (
        (sums[1] + sums[5]) + (sums[3] + sums[7])
    )
    if BK != 1 and AK == 1:
        seq = sums[0]
        for ci in gl.static_range(1, 8):
            seq = seq + sums[ci]
        mr = gl.program_id(0) * BO + gl.arange(0, BO, layout=value.type.layout)
        value = gl.where(mr >= (M // 4) * 4, seq, value)
    LL: gl.constexpr = value.type.layout
    rr2 = gl.program_id(0) * BO + gl.arange(0, BO, layout=LL)
    last2 = gl.minimum((rr2 // 4) * 4 + 3, M - 1)
    prefix2 = 4 - ((A_OFFSET + last2 * AM) % 4)
    edge2 = rr2 >= (M // 4) * 4
    prefix2 = gl.where(edge2, 4 - b_offset % 4, prefix2)
    lanes2 = gl.where(edge2 & ((rr2 % 2) == 0), 8, 4)
    if AK != 1 and AM == 1:
        prefix2 = gl.where(edge2, prefix2, 0)
        lanes2 = gl.where(edge2, lanes2, 1)
    if AK != 1 and AM == 1:
        prefix2 = gl.full((BO,), 0, gl.int32, LL)
        lanes2 = gl.full((BO,), 1, gl.int32, LL)
    elif BK != 1:
        prefix2 = gl.where(edge2, 1, prefix2)
        lanes2 = gl.where(edge2, 8, lanes2)
    end2 = prefix2 + ((K - prefix2) // lanes2) * lanes2
    for t in gl.static_range(7):
        tk = end2 + t
        atail = gl.load(A + rr2 * AM + tk * AK, (rr2 < M) & (tk < K), 0)
        btail = gl.load(B + tk * BK, tk < K, 0)
        value = value + atail * btail
    gl.store(C + rr2 * OUTPUT_STRIDE, value, rr2 < M)
