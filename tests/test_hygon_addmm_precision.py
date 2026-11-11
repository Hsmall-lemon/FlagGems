# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Long FP32 reductions use an independently checked high-precision oracle."""

import math

import pytest
import torch

import flag_gems

pytestmark = [
    pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon kernels"),
]


@pytest.mark.addmm
def test_hygon_addmm_r186():
    k = 3145728
    specs = [((1,), (1,)), ((1, k), (48, 1)), ((k, 1), (1, 48))]
    host, device = [], []
    for i, (shape, stride) in enumerate(specs, start=1):
        size = 1 + sum((n - 1) * s for n, s in zip(shape, stride))
        generator = torch.Generator().manual_seed(10186000 + i)
        base = torch.empty(size).normal_(0, 0.25, generator=generator)
        host.append(torch.as_strided(base, shape, stride))
        device.append(torch.as_strided(base.to(flag_gems.device), shape, stride))
    bias, a, b = host
    products = a.double().flatten() * b.double().flatten()
    exact = math.fsum((math.fsum(products.tolist()), bias.item()))
    expected64 = torch.addmm(bias.double(), a.double(), b.double())
    torch.testing.assert_close(
        expected64,
        torch.tensor([[exact]], dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    # CPU FP32 addmm accumulates error at this K; keep the default FP32
    # tolerances and round an independently verified FP64 reference once.
    actual = flag_gems.addmm(*device).cpu()
    torch.testing.assert_close(actual, expected64.float())


@pytest.mark.parametrize(
    "shape", [(1, 4096, 3), (3, 4097, 1), (1, 8193, 1), (5, 12289, 1)]
)
@pytest.mark.parametrize("alpha,beta", [(1, 1), (-0.5, 1.25), (1, 0), (0, 1)])
@pytest.mark.parametrize("out_variant", [False, True])
@pytest.mark.addmm
def test_hygon_addmm_long_vector_layouts(shape, alpha, beta, out_variant):
    m, k, n = shape
    generator = torch.Generator().manual_seed(18601)
    a_base = torch.randn(m, 2 * k + 1, generator=generator) * 0.25
    b_base = torch.randn(n, 2 * k + 1, generator=generator) * 0.25
    a, b = a_base[:, 1::2], b_base[:, 1::2].t()
    bias = torch.randn(m, n, generator=generator) * 0.25
    if beta == 0:
        bias.fill_(float("nan"))
    expected = torch.addmm(
        bias.double(), a.double(), b.double(), alpha=alpha, beta=beta
    ).float()
    ag = a_base.to(flag_gems.device)[:, 1::2]
    bg = b_base.to(flag_gems.device)[:, 1::2].t()
    ig = bias.to(flag_gems.device)
    if out_variant:
        backing = torch.empty(m, 2 * n + 1, device=flag_gems.device)
        out = backing[:, 1::2]
        result = flag_gems.addmm_out(ig, ag, bg, alpha=alpha, beta=beta, out=out)
        assert result.data_ptr() == out.data_ptr()
    else:
        result = flag_gems.addmm(ig, ag, bg, alpha=alpha, beta=beta)
    torch.testing.assert_close(result.cpu(), expected)


@pytest.mark.parametrize("k", [4096, 4097, 8193])
@pytest.mark.addmm
def test_hygon_addmm_long_vector_cancellation(k):
    # Exact sum is small while FP32 accumulation loses the unit terms.
    a = torch.ones(1, k)
    a[:, 0] = 2**24
    a[:, -1] = -(2**24)
    b = torch.ones(k, 1)
    bias = torch.tensor([0.25])
    expected = torch.tensor([[k - 2 + 0.25]], dtype=torch.float32)
    result = flag_gems.addmm(
        bias.to(flag_gems.device), a.to(flag_gems.device), b.to(flag_gems.device)
    )
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
