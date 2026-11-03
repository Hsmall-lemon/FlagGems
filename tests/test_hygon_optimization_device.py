# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Explicit device smoke; the archived qualification contracts remain required."""

import os

import pytest

GROUP = os.environ.get("FLAGGEMS_HYGON_GROUP")
if GROUP not in {"matrix", "sort", "under90", "index"}:
    pytest.skip(
        "set FLAGGEMS_HYGON_GROUP explicitly in a fresh process",
        allow_module_level=True,
    )
if GROUP in {"under90", "index"}:
    pytest.importorskip("torch_fl")

torch = pytest.importorskip("torch")
flag_gems = pytest.importorskip("flag_gems")
if flag_gems.vendor_name != "hygon":
    pytest.skip("requires Hygon hardware", allow_module_level=True)

from flag_gems.runtime.backend._hygon.experimental import install  # noqa: E402

DEVICE = "flagos" if GROUP in {"under90", "index"} else flag_gems.device


@pytest.fixture(scope="module", autouse=True)
def install_group():
    install(groups=(GROUP,), serialized_eager=GROUP == "under90")
    with torch.inference_mode():
        yield


def check(actual, expected):
    actual = actual.cpu()
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5, equal_nan=True)
    if expected.is_floating_point():
        zeros = expected == 0
        assert torch.equal(torch.signbit(actual[zeros]), torch.signbit(expected[zeros]))


@pytest.mark.skipif(GROUP != "matrix", reason="separate fresh process per group")
@pytest.mark.parametrize("k", [0, 1, 3, 32])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_matrix_edges_and_strides(k, dtype):
    generator = torch.Generator().manual_seed(73)
    a = torch.randn((k, 7), generator=generator, dtype=dtype).t()
    b = torch.randn((5, k), generator=generator, dtype=dtype).t()
    da, db = a.to(DEVICE), b.to(DEVICE)
    check(flag_gems.mm(da, db), torch.mm(a, b))
    check(da, a)
    check(db, b)
    if k <= 1:
        bias = torch.arange(5, dtype=dtype)
        for alpha, beta in [(1, 1), (0, 1), (1, 0), (0.5, -0.5)]:
            check(
                flag_gems.addmm(bias.to(DEVICE), da, db, alpha=alpha, beta=beta),
                torch.addmm(bias, a, b, alpha=alpha, beta=beta),
            )


@pytest.mark.skipif(GROUP != "matrix", reason="separate fresh process per group")
@pytest.mark.parametrize("k", [0, 1, 16])
def test_bmm_edges(k):
    a = torch.arange(2 * 7 * k, dtype=torch.float32).reshape(2, 7, k) / 17
    b = torch.arange(2 * k * 5, dtype=torch.float32).reshape(2, k, 5) / 23
    check(flag_gems.bmm(a.to(DEVICE), b.to(DEVICE)), torch.bmm(a, b))


@pytest.mark.skipif(GROUP != "sort", reason="separate fresh process per group")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int64])
def test_stable_sort_specials_and_duplicates(dtype):
    if dtype.is_floating_point:
        a = torch.tensor(
            [float("nan"), -0.0, 0.0, -float("inf"), 3.0, 3.0, float("inf")],
            dtype=dtype,
        )
    else:
        a = torch.tensor([-(2**62), 3, 3, 0, 2**62, -1, 0], dtype=dtype)
    for descending in [False, True]:
        expected = torch.sort(a, stable=True, descending=descending)
        actual = flag_gems.sort_stable(a.to(DEVICE), stable=True, descending=descending)
        check(actual[0], expected.values)
        assert torch.equal(actual[1].cpu(), expected.indices)


@pytest.mark.skipif(GROUP != "under90", reason="separate fresh process per group")
def test_under90_specials_and_empty():
    a = torch.tensor(
        [
            -float("inf"),
            -1.0,
            -0.0,
            0.0,
            torch.finfo(torch.float32).tiny / 8,
            1.0,
            float("inf"),
            float("nan"),
        ],
        dtype=torch.float32,
    )
    a = a.repeat(32768)
    da = a.to(DEVICE)
    check(flag_gems.rsqrt(da), torch.rsqrt(a))
    for mode in ["none", "tanh"]:
        check(
            flag_gems.gelu(da, approximate=mode),
            torch.nn.functional.gelu(a, approximate=mode),
        )
    check(
        flag_gems.linspace(0, 1, 0, dtype=torch.float32, device=DEVICE),
        torch.linspace(0, 1, 0, dtype=torch.float32),
    )
    check(da, a)


@pytest.mark.skipif(GROUP != "index", reason="separate fresh process per group")
def test_index_duplicates_negative_and_input_preservation():
    a = torch.arange(63, dtype=torch.float32).reshape(7, 9).t()
    indices = torch.tensor([0, -1, 0, 3], dtype=torch.int64)
    da, di = a.to(DEVICE), indices.to(DEVICE)
    check(flag_gems.index(da, [None, di]), a[:, indices])
    check(da, a)
    assert torch.equal(di.cpu(), indices)
