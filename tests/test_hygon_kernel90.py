# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Numerical and layout regressions for the Hygon small-kernel paths."""

import pytest
import torch

import flag_gems

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name != "hygon", reason="Hygon kernels"
)


@pytest.mark.bmm
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize(
    "shape",
    [
        (7, 1, 3, 10),
        (33, 3, 10, 2),
        (129, 1, 16, 48),
        (3, 5, 3, 128),
        (2, 1, 48, 7),
        (4, 8, 128, 1),
        (3, 1, 0, 4),
        (0, 1, 3, 7),
    ],
)
def test_hygon_small_bmm(dtype, shape):
    batch, m, k, n = shape
    torch.manual_seed(2301)
    a = torch.randn(batch, k, m, dtype=dtype).transpose(1, 2)
    b = torch.randn(batch, n, k, dtype=dtype).transpose(1, 2)
    reference = (
        torch.bmm(a.float(), b.float()).to(dtype)
        if dtype in (torch.float16, torch.bfloat16)
        else torch.bmm(a, b)
    )
    result = flag_gems.bmm(a.to(flag_gems.device), b.to(flag_gems.device)).cpu()
    torch.testing.assert_close(result, reference)


@pytest.mark.bmm
@pytest.mark.parametrize(
    "shape",
    [
        (3, 33, 16, 41),
        (3, 101, 16, 101),
        (3, 101, 101, 16),
        (2, 65, 3, 67),
        (3, 300, 3, 300),
        (2, 129, 8, 257),
    ],
)
@pytest.mark.parametrize("strided_output", [False, True])
def test_hygon_short_matrix_bmm_out(shape, strided_output):
    batch, m, k, n = shape
    a = torch.randn(batch, k, m).transpose(1, 2)
    b = torch.randn(batch, n, k).transpose(1, 2)
    expected = torch.bmm(a, b)
    if strided_output:
        out = torch.empty(batch, n, m, device=flag_gems.device).transpose(1, 2)
    else:
        out = torch.empty(batch, m, n, device=flag_gems.device)
    result = flag_gems.bmm_out(a.to(flag_gems.device), b.to(flag_gems.device), out)
    assert result.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out.cpu(), expected)


@pytest.mark.bmm
@pytest.mark.parametrize(
    "shape", [(7, 32, 32), (115, 48, 48), (17, 63, 57), (3, 64, 64)]
)
@pytest.mark.parametrize("layout", ["plain", "strided_input", "strided_output"])
def test_hygon_single_row_bmm(shape, layout):
    batch, k, n = shape
    generator = torch.Generator().manual_seed(6701)
    a_base = torch.randn(batch, 1, 2 * k + 1, generator=generator) * 0.25
    b_base = torch.randn(batch, k, n + 3, generator=generator) * 0.25
    if layout == "strided_input":
        a, b = a_base[:, :, 1::2], b_base[:, :, 1 : n + 1]
        ag_base, bg_base = a_base.to(flag_gems.device), b_base.to(flag_gems.device)
        ag, bg = ag_base[:, :, 1::2], bg_base[:, :, 1 : n + 1]
    else:
        a, b = a_base[:, :, :k].contiguous(), b_base[:, :, :n].contiguous()
        ag, bg = a.to(flag_gems.device), b.to(flag_gems.device)
    expected = torch.bmm(a, b)
    if layout == "strided_output":
        out = torch.empty(batch, 1, 2 * n, device=flag_gems.device)[:, :, ::2]
        actual = flag_gems.bmm_out(ag, bg, out)
        assert actual.data_ptr() == out.data_ptr()
    else:
        actual = flag_gems.bmm(ag, bg)
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.add
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int64])
@pytest.mark.parametrize("shape", [(), (6,), (33, 78), (542, 30), (101, 3, 33)])
@pytest.mark.parametrize("kind", ["scalar", "tensor", "vector"])
def test_hygon_flat_add(dtype, shape, kind):
    a = torch.randint(-20, 20, shape).to(dtype)
    b = (
        3
        if kind == "scalar"
        else torch.randint(-20, 20, shape if kind == "tensor" else shape[-1:]).to(dtype)
    )
    alpha = -2
    reference = torch.add(a, b, alpha=alpha)
    result = flag_gems.add(
        a.to(flag_gems.device),
        b.to(flag_gems.device) if isinstance(b, torch.Tensor) else b,
        alpha=alpha,
    ).cpu()
    torch.testing.assert_close(result, reference)


@pytest.mark.add
@pytest.mark.parametrize("kind", ["transpose", "broadcast", "slice"])
def test_hygon_strided_add(kind):
    base = torch.randn(2, 3, 37, 41, dtype=torch.float64)
    a = base[:, :, :33, :33]
    if kind == "transpose":
        b = a.transpose(-1, -2)
    elif kind == "broadcast":
        b = a[:, :, :1, :]
    else:
        b = base[:, :, 1:34, 1:34]
    gpu_base = base.to(flag_gems.device)
    gpu_a = torch.as_strided(gpu_base, a.shape, a.stride(), a.storage_offset())
    gpu_b = torch.as_strided(gpu_base, b.shape, b.stride(), b.storage_offset())
    result = flag_gems.add(gpu_a, gpu_b, alpha=-3)
    torch.testing.assert_close(result.cpu(), torch.add(a, b, alpha=-3))


@pytest.mark.add
@pytest.mark.parametrize("size", [24575, 24576, 32768, 32769])
@pytest.mark.parametrize("kind", ["scalar", "tensor", "broadcast"])
def test_hygon_medium_flat_add(size, kind):
    generator = torch.Generator().manual_seed(6801)
    a = torch.randn(size, generator=generator) * 0.25
    b = (
        0
        if kind == "scalar"
        else torch.randn(1 if kind == "broadcast" else size, generator=generator) * 0.25
    )
    bg = b.to(flag_gems.device) if isinstance(b, torch.Tensor) else b
    actual = flag_gems.add(a.to(flag_gems.device), bg, alpha=-2)
    torch.testing.assert_close(actual.cpu(), torch.add(a, b, alpha=-2))


@pytest.mark.index
@pytest.mark.parametrize("density", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("suffix", [None, 1, 3, 0])
@pytest.mark.parametrize("broadcast", [False, True])
def test_hygon_mask_compaction(density, suffix, broadcast):
    torch.manual_seed(7301)
    shape = (1, 5, 17, 19)
    if broadcast:
        a = torch.arange(5, dtype=torch.int64).reshape(1, 5, 1, 1).expand(shape)
    else:
        a = torch.randn(shape, dtype=torch.float64)
    gpu = a.to(flag_gems.device)
    if broadcast:
        gpu = (
            torch.arange(5, dtype=torch.int64, device=flag_gems.device)
            .reshape(1, 5, 1, 1)
            .expand(shape)
        )
    if suffix is not None:
        a = a.unsqueeze(-1).expand(*shape, suffix)
        gpu = gpu.unsqueeze(-1).expand(*shape, suffix)
    mask = torch.rand(shape) < density
    expected = a[mask]
    result = flag_gems.index(gpu, [mask.to(flag_gems.device)]).cpu()
    assert torch.equal(result, expected)


@pytest.mark.clone
@pytest.mark.parametrize(
    "memory_format", [torch.preserve_format, torch.contiguous_format]
)
@pytest.mark.parametrize("kind", ["transpose", "slice", "broadcast", "empty"])
def test_hygon_clone_layout(memory_format, kind):
    base = torch.randn((7, 13, 4), dtype=torch.float64)
    if kind == "transpose":
        a = base.transpose(1, 2)
    elif kind == "slice":
        a = base[:, ::2, 1:]
    elif kind == "broadcast":
        a = base[:1].expand(5, 13, 4)
    else:
        a = base[:0]
    expected = torch.clone(a, memory_format=memory_format)
    # Rebuild the original view on device, including its holes and zero strides.
    gpu_base = base.to(flag_gems.device)
    gpu = torch.as_strided(gpu_base, a.shape, a.stride(), a.storage_offset())
    result = flag_gems.clone(gpu, memory_format=memory_format)
    assert result.shape == expected.shape
    assert result.stride() == expected.stride()
    assert torch.equal(result.cpu(), expected)


@pytest.mark.index
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("suffix", [4, 48, 128])
def test_hygon_gather_negative_indices(dtype, suffix):
    base = torch.arange(7 * (suffix * 2 + 1), dtype=torch.float64).reshape(
        7, suffix * 2 + 1
    )
    a = base[:, 1::2]
    i = torch.tensor([-1, -7, 0, 5, 5, 2], dtype=dtype)
    device_base = base.to(flag_gems.device)
    device_a = torch.as_strided(device_base, a.shape, a.stride(), a.storage_offset())
    result = flag_gems.index(device_a, [i.to(flag_gems.device)])
    assert torch.equal(result.cpu(), a[i])


@pytest.mark.addmm
@pytest.mark.parametrize("shape", [(33, 112, 48), (63, 128, 31)])
@pytest.mark.parametrize("kind", ["plain", "strided", "scaled"])
def test_hygon_addmm_small_fp32_dispatch(shape, kind):
    m, k, n = shape
    g = torch.Generator().manual_seed(3191)
    a = (
        torch.randn(k, m, generator=g).t()
        if kind == "strided"
        else torch.randn(m, k, generator=g)
    ) * 0.25
    b = torch.randn(n, k, generator=g).t() * 0.25
    bias = torch.randn(n, generator=g) * 0.25
    alpha, beta = (0.5, -2) if kind == "scaled" else (1, 1)
    expected = torch.addmm(bias, a, b, alpha=alpha, beta=beta)
    ag, bg, ig = (t.to(flag_gems.device) for t in (a, b, bias))
    if kind == "strided":
        out = torch.empty((m, n * 2), device=flag_gems.device)[:, ::2]
        result = flag_gems.addmm_out(ig, ag, bg, out=out)
        assert result.data_ptr() == out.data_ptr()
    else:
        result = flag_gems.addmm(ig, ag, bg, alpha=alpha, beta=beta)
    torch.testing.assert_close(result.cpu(), expected)


@pytest.mark.addmm
@pytest.mark.parametrize("seed", [2901, 2902])
@pytest.mark.parametrize("kind", ["dual", "strided_out", "scaled"])
def test_hygon_addmm_large_bf16_dispatch(seed, kind):
    # An incomplete M tile checks discarded-row loads in the dual accumulator.
    m, k, n = 8195, 2048, 2048
    g = torch.Generator().manual_seed(seed)
    a = (torch.randn(m, k, generator=g) * 0.25).to(torch.bfloat16)
    b = (torch.randn(n, k, generator=g) * 0.25).to(torch.bfloat16).t()
    bias = (torch.randn(n, generator=g) * 0.25).to(torch.bfloat16)
    alpha, beta = (0.5, -1) if kind == "scaled" else (1, 1)
    expected = torch.addmm(
        bias.float(), a.float(), b.float(), alpha=alpha, beta=beta
    ).to(torch.bfloat16)
    ag, bg, ig = (t.to(flag_gems.device) for t in (a, b, bias))
    if kind == "strided_out":
        out = torch.empty((m, n * 2), dtype=torch.bfloat16, device=flag_gems.device)[
            :, ::2
        ]
        result = flag_gems.addmm_out(ig, ag, bg, out=out)
        assert result.data_ptr() == out.data_ptr()
    else:
        result = flag_gems.addmm(ig, ag, bg, alpha=alpha, beta=beta)
    torch.testing.assert_close(result.cpu(), expected)


@pytest.mark.addmm
@pytest.mark.parametrize("n", [4096, 6144])
@pytest.mark.parametrize("seed", [3901, 3902])
def test_hygon_addmm_bf16_quarter_stores(n, seed):
    m, k = 8195, 2048
    g = torch.Generator().manual_seed(seed)
    a = (torch.randn(m, k, generator=g) * 0.25).to(torch.bfloat16)
    b = (torch.randn(n, k, generator=g) * 0.25).to(torch.bfloat16).t()
    bias = (torch.randn(n, generator=g) * 0.25).to(torch.bfloat16)
    expected = torch.addmm(bias.float(), a.float(), b.float()).to(torch.bfloat16)
    result = flag_gems.addmm(
        bias.to(flag_gems.device), a.to(flag_gems.device), b.to(flag_gems.device)
    )
    torch.testing.assert_close(result.cpu(), expected)


@pytest.mark.addmm
@pytest.mark.parametrize(
    "shape", [(8195, 2112, 2304), (8195, 4096, 4096), (8447, 5120, 2048)]
)
@pytest.mark.parametrize("seed", [4901, 4902])
def test_hygon_addmm_bf16_long_k_half_stores(shape, seed):
    m, k, n = shape
    g = torch.Generator().manual_seed(seed)
    a = (torch.randn(m, k, generator=g) * 0.25).to(torch.bfloat16)
    b = (torch.randn(n, k, generator=g) * 0.25).to(torch.bfloat16).t()
    bias = (torch.randn(n, generator=g) * 0.25).to(torch.bfloat16)
    expected = torch.addmm(bias.float(), a.float(), b.float()).to(torch.bfloat16)
    result = flag_gems.addmm(
        bias.to(flag_gems.device), a.to(flag_gems.device), b.to(flag_gems.device)
    )
    torch.testing.assert_close(result.cpu(), expected)


@pytest.mark.addmm
@pytest.mark.parametrize("seed", [158000, 158001, 158002])
def test_hygon_addmm_fp32_accumulation(seed):
    g = torch.Generator().manual_seed(seed)
    bias = torch.randn(1152, generator=g) * 0.25
    a = torch.randn(400, 1280, generator=g) * 0.25
    b = (torch.randn(1152, 1280, generator=g) * 0.25).t()
    expected = torch.addmm(bias, a, b)
    result = flag_gems.addmm(
        bias.to(flag_gems.device), a.to(flag_gems.device), b.to(flag_gems.device)
    ).cpu()
    torch.testing.assert_close(result, expected)


@pytest.mark.addmm
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("beta", [0, 1])
def test_hygon_addmm_low_precision_tails(dtype, beta):
    torch.manual_seed(905)
    a = (torch.randn(1031, 64) * 0.25).to(dtype)
    b = (torch.randn(257, 64) * 0.25).to(dtype).t()
    bias = (
        torch.full((257,), float("nan"), dtype=dtype)
        if beta == 0
        else torch.randn(257).to(dtype)
    )
    expected = torch.addmm(bias.float(), a.float(), b.float(), beta=beta).to(dtype)
    result = flag_gems.addmm(
        bias.to(flag_gems.device),
        a.to(flag_gems.device),
        b.to(flag_gems.device),
        beta=beta,
    )
    torch.testing.assert_close(result.cpu(), expected)


@pytest.mark.addmm
@pytest.mark.parametrize(
    "shape", [(128, 256, 256), (257, 289, 317), (303, 320, 320), (511, 511, 512)]
)
@pytest.mark.parametrize("scale", [0.25, 1.0])
@pytest.mark.parametrize("strided_output", [False, True])
@pytest.mark.parametrize("transpose_a", [False, True])
def test_hygon_addmm_medium_precise(shape, scale, strided_output, transpose_a):
    m, k, n = shape
    g = torch.Generator().manual_seed(1143)
    ac = (
        torch.randn(k, m, generator=g).t()
        if transpose_a
        else torch.randn(m, k, generator=g)
    ) * scale
    bc = torch.randn(n, k, generator=g).t() * scale
    bias = torch.randn(1, n, generator=g) * scale
    a = ac.to(flag_gems.device)
    b = bc.to(flag_gems.device)
    gpu_bias = bias.to(flag_gems.device)
    # Use double precision on CPU so the reference does not introduce FP32 GEMM error.
    expected = torch.addmm(
        bias.double(), ac.double(), bc.double(), alpha=-0.5, beta=1.25
    ).float()
    if strided_output:
        out = torch.empty(n, m, device=flag_gems.device).t()
        result = flag_gems.addmm_out(gpu_bias, a, b, alpha=-0.5, beta=1.25, out=out)
        assert result.data_ptr() == out.data_ptr()
    else:
        result = flag_gems.addmm(gpu_bias, a, b, alpha=-0.5, beta=1.25)
    torch.testing.assert_close(result.cpu(), expected)
    nan_bias = torch.full((n,), float("nan"))
    result = flag_gems.addmm(nan_bias.to(flag_gems.device), a, b, beta=0)
    expected = torch.addmm(nan_bias.double(), ac.double(), bc.double(), beta=0).float()
    torch.testing.assert_close(result.cpu(), expected)


@pytest.mark.addmm
@pytest.mark.parametrize(
    "unsafe", ["none", "register_use", "missing_final_wait", "global_store"]
)
def test_hygon_addmm_schedule_dependencies(unsafe):
    from flag_gems.runtime.backend._hygon.ops.addmm_schedule import _defer_prefetch_wait

    assembly = """loop: ; Inner Loop Header
    buffer_load_dwordx4 v[140:143], v172, s[0:3], 0 offen
    s_waitcnt vmcnt(0) lgkmcnt(0)
    s_barrier
    ds_read_b128 v[144:147], v134
    buffer_load_dwordx4 v[168:171], v176, s[4:7], 0 offen
    buffer_load_dwordx4 v[172:175], v172, s[0:3], 0 offen offset:64
    buffer_load_dwordx4 v[176:179], v176, s[4:7], 0 offen offset:64
    v_mmac_f32_16x16x16_bf16 v[61:64], v[144:145], v[148:149], v[61:64]
    s_waitcnt vmcnt(0) lgkmcnt(0)
    s_barrier
    ds_write_b128 v137, v[140:143]
    ds_write_b128 v137, v[168:171] offset:16384
    ds_write_b128 v137, v[172:175] offset:32768
    ds_write_b128 v137, v[176:179] offset:49152
    s_cbranch_scc1 loop
"""
    if unsafe == "register_use":
        assembly = assembly.replace(
            "    ds_read_b128", "    v_mov_b32 v145, v140\n    ds_read_b128"
        )
    elif unsafe == "missing_final_wait":
        at = assembly.rindex("s_waitcnt vmcnt(0)")
        assembly = assembly[:at] + assembly[at:].replace("vmcnt(0)", "vmcnt(1)", 1)
    elif unsafe == "global_store":
        assembly = assembly.replace(
            "    ds_read_b128",
            "    buffer_store_dword v145, v172, s[0:3], 0 offen\n    ds_read_b128",
        )
    result = _defer_prefetch_wait(assembly)
    if unsafe != "none":
        assert result == assembly
    else:
        assert result == assembly.replace(
            "s_waitcnt vmcnt(0) lgkmcnt(0)", "s_waitcnt lgkmcnt(0)", 1
        )
        assert result.count("s_barrier") == 2


@pytest.mark.addmm
@pytest.mark.parametrize("unaligned", ["bias", "out"])
def test_hygon_addmm_long_k_unaligned(unaligned):
    m, k, n = 8195, 2112, 2304
    g = torch.Generator().manual_seed(6301)
    a = (torch.randn(m, k, generator=g) * 0.25).to(torch.bfloat16)
    b = (torch.randn(n, k, generator=g) * 0.25).to(torch.bfloat16).t()
    bias = (torch.randn(n, generator=g) * 0.25).to(torch.bfloat16)
    expected = torch.addmm(bias.float(), a.float(), b.float()).to(torch.bfloat16)
    ag, bg, ig = (t.to(flag_gems.device) for t in (a, b, bias))
    if unaligned == "bias":
        ig = torch.empty(n + 1, dtype=torch.bfloat16, device=flag_gems.device)[1:]
        ig.copy_(bias)
    extra = 1 if unaligned == "out" else 0
    out = torch.empty(m * n + extra, dtype=torch.bfloat16, device=flag_gems.device)[
        extra:
    ].view(m, n)
    result = flag_gems.addmm_out(ig, ag, bg, out=out)
    assert result.data_ptr() == out.data_ptr()
    torch.testing.assert_close(result.cpu(), expected)
