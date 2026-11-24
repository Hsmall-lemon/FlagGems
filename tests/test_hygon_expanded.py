# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for expanded Hygon layouts and precise reductions."""

import pytest
import torch

import flag_gems

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name != "hygon", reason="Hygon kernels"
)


@pytest.mark.addmm
@pytest.mark.parametrize(
    "shape", [(1, 1024, 37), (1, 8193, 1), (17, 4097, 1), (333, 83, 63)]
)
@pytest.mark.parametrize("beta", [0, 1])
def test_expanded_addmm_precision_and_output_stride(shape, beta):
    m, k, n = shape
    g = torch.Generator().manual_seed(702)
    a = torch.randn(m, k, generator=g) * 0.25
    b = torch.randn(k, n, generator=g) * 0.25
    bias = torch.full((n,), float("nan")) if beta == 0 else torch.randn(n, generator=g)
    expected = torch.addmm(
        bias.double(), a.double(), b.double(), alpha=2, beta=beta
    ).float()
    ad, bd, cd = [t.to(flag_gems.device) for t in (a, b, bias)]
    out = torch.empty((m, 2 * n), device=flag_gems.device)[:, ::2]
    with torch.no_grad():
        actual = flag_gems.addmm(cd, ad, bd, alpha=2, beta=beta)
        returned = flag_gems.addmm_out(cd, ad, bd, alpha=2, beta=beta, out=out)
    assert returned is out
    torch.testing.assert_close(actual.cpu(), expected)
    torch.testing.assert_close(out.cpu(), expected)


@pytest.mark.add
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "layout",
    ["transpose_broadcast", "transpose_dense", "offset_rows", "broadcast_plane"],
)
def test_expanded_add_layouts(dtype, layout):
    g = torch.Generator().manual_seed(703)
    if layout.startswith("transpose"):
        backing = torch.randn(2, 129, 1025, generator=g, dtype=dtype)
        a = backing.transpose(1, 2)
        b = torch.randn(
            (2, 1, 129) if layout.endswith("broadcast") else (2, 1025, 129),
            generator=g,
            dtype=dtype,
        )
        ad = backing.to(flag_gems.device).transpose(1, 2)
        bd = b.to(flag_gems.device)
    elif layout == "offset_rows":
        backing = torch.randn(257, 4, 1159, generator=g, dtype=dtype)
        a, b = backing[:, 1:2, :], backing[:, 2:3, :]
        storage = backing.to(flag_gems.device)
        ad, bd = storage[:, 1:2, :], storage[:, 2:3, :]
    else:
        a = torch.randn(7, 197, 199, generator=g, dtype=dtype)
        b = torch.randn(7, 1, 1, generator=g, dtype=dtype)
        ad, bd = a.to(flag_gems.device), b.to(flag_gems.device)
    with torch.no_grad():
        actual = flag_gems.add(ad, bd, alpha=2).cpu()
    torch.testing.assert_close(actual, torch.add(a, b, alpha=2))


@pytest.mark.add
def test_expanded_add_int64_scalar():
    a = torch.tensor([-(2**40), 2**40], dtype=torch.int64)
    actual = flag_gems.add(a.to(flag_gems.device), 3, alpha=2).cpu()
    assert torch.equal(actual, a + 6)


@pytest.mark.index
@pytest.mark.parametrize("column", [False, True])
def test_expanded_index_negative_strided_indices(column):
    x = torch.arange(35, dtype=torch.float32).reshape(7, 5)
    storage = torch.tensor([9, -1, 9, 0, 9, 2, 9])
    ix = storage[1::2]
    indices = [None, ix] if column else [ix]
    device_ix = storage.to(flag_gems.device)[1::2]
    actual = flag_gems.index(
        x.to(flag_gems.device), [None, device_ix] if column else [device_ix]
    )
    assert torch.equal(actual.cpu(), torch.ops.aten.index.Tensor(x, indices))


@pytest.mark.index
@pytest.mark.parametrize("suffix", [4, 33])
@pytest.mark.parametrize("empty", [False, True])
def test_expanded_index_broadcast_mask(suffix, empty):
    x = torch.arange(3 * 5 * suffix, dtype=torch.float32).reshape(3, 5, suffix)
    base = (
        torch.zeros(5, dtype=torch.bool)
        if empty
        else torch.tensor([True, False, True, False, True])
    )
    mask = base.expand(3, 5)
    actual = flag_gems.index(
        x.to(flag_gems.device), [base.to(flag_gems.device).expand(3, 5)]
    )
    assert torch.equal(actual.cpu(), torch.ops.aten.index.Tensor(x, [mask]))


@pytest.mark.index
def test_expanded_index_empty_integer():
    x = torch.arange(4, device=flag_gems.device)
    actual = flag_gems.index(
        x, [torch.empty(0, dtype=torch.int64, device=flag_gems.device)]
    )
    assert actual.shape == (0,)


@pytest.mark.addmm
@pytest.mark.parametrize("beta", [0, 1])
def test_expanded_addmm_aligned_k(beta):
    generator = torch.Generator().manual_seed(20261109)
    m, n, k = 4096, 512, 1025
    a = (torch.randn(m * k + 7, generator=generator) * 0.125)[3 : 3 + m * k].view(m, k)
    b = (
        (torch.randn(n * k + 9, generator=generator) * 0.125)[5 : 5 + n * k]
        .view(n, k)
        .t()
    )
    bias = torch.randn(n, generator=generator) * 0.125
    if beta == 0:
        bias.fill_(float("nan"))
    expected = torch.addmm(bias, a, b, alpha=0.5, beta=beta)
    ad, bd, cd = [t.to(flag_gems.device) for t in (a, b, bias)]
    out = torch.empty((m, 2 * n), device=flag_gems.device)[:, ::2]
    with torch.no_grad():
        actual = flag_gems.addmm(cd, ad, bd, alpha=0.5, beta=beta)
        returned = flag_gems.addmm_out(cd, ad, bd, alpha=0.5, beta=beta, out=out)
    assert returned is out
    torch.testing.assert_close(actual.cpu(), expected)
    torch.testing.assert_close(out.cpu(), expected)


@pytest.mark.addmm
@pytest.mark.parametrize(
    "kind,alpha,beta",
    [("vector", 1, 1), ("scalar", 0.5, -0.75), ("row", -1, 1), ("nan", 2, 0)],
)
def test_expanded_addmm_transposed_bias_and_out(kind, alpha, beta):
    m, k, n = 4097, 1025, 513
    g = torch.Generator().manual_seed(745)
    a = torch.randn(m, k, generator=g) * 0.25
    b = (torch.randn(n, k, generator=g) * 0.25).t()
    if kind == "scalar":
        bias = torch.tensor(0.25)
    elif kind == "row":
        bias = torch.randn(m, 1, generator=g)
    elif kind == "nan":
        bias = torch.full((n,), float("nan"))
    else:
        bias = torch.randn(n, generator=g)
    expected = torch.addmm(bias, a, b, alpha=alpha, beta=beta)
    ad, bd, cd = [t.to(flag_gems.device) for t in (a, b, bias)]
    out = torch.empty((m, 2 * n + 3), device=flag_gems.device)[:, 1 : 2 * n + 1 : 2]
    with torch.no_grad():
        actual = flag_gems.addmm(cd, ad, bd, alpha=alpha, beta=beta)
        returned = flag_gems.addmm_out(cd, ad, bd, alpha=alpha, beta=beta, out=out)
    assert returned is out
    torch.testing.assert_close(actual.cpu(), expected)
    torch.testing.assert_close(out.cpu(), expected)


@pytest.mark.addmm
@pytest.mark.parametrize(
    "shape", [(512, 1024, 512), (800, 1024, 1024), (1024, 1024, 2048)]
)
@pytest.mark.parametrize("seed", [915, 916, 917])
def test_expanded_addmm_short_rowmajor_precision(shape, seed):
    m, k, n = shape
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(m, k, generator=g) * 0.25
    b = torch.randn(k, n, generator=g) * 0.25
    bias = torch.randn(n, generator=g) * 0.25
    expected = torch.addmm(bias, a, b)
    ad, bd, cd = [t.to(flag_gems.device) for t in (a, b, bias)]
    with torch.no_grad():
        actual = flag_gems.addmm(cd, ad, bd)
    torch.testing.assert_close(actual.cpu(), expected)
