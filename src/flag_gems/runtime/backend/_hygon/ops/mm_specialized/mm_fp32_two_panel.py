"""FP32 GEMM with two independently accumulated K panels."""

import triton
import triton.language as tl


@triton.jit
def matmul(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    TM: tl.constexpr,
    TN: tl.constexpr,
    TK: tl.constexpr,
    SPLIT: tl.constexpr,
):
    i = tl.program_id(0) * TM + tl.arange(0, TM)
    j = tl.program_id(1) * TN + tl.arange(0, TN)
    k = tl.arange(0, TK)
    first = tl.zeros((TM, TN), tl.float32)
    tail = tl.zeros((TM, TN), tl.float32)
    for block in range(SPLIT // TK):
        q = block * TK + k
        a = tl.load(A + i[:, None] * AM + q[None, :] * AK, i[:, None] < M, 0)
        b = tl.load(B + q[:, None] * BK + j[None, :] * BN, j[None, :] < N, 0)
        first = tl.dot(a, b, first, input_precision="ieee")
    for block in range(triton.cdiv(K - SPLIT, TK)):
        q = SPLIT + block * TK + k
        a = tl.load(
            A + i[:, None] * AM + q[None, :] * AK,
            (i[:, None] < M) & (q[None, :] < K),
            0,
        )
        b = tl.load(
            B + q[:, None] * BK + j[None, :] * BN,
            (q[:, None] < K) & (j[None, :] < N),
            0,
        )
        tail = tl.dot(a, b, tail, input_precision="ieee")
    tl.store(
        C + j[None, :] * M + i[:, None],
        tail + first,
        (i[:, None] < M) & (j[None, :] < N),
    )
