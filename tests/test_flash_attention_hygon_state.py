import pytest
import torch

import flag_gems


@pytest.mark.scaled_dot_product_flash_attention
@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon STATE layout")
@pytest.mark.parametrize(
    "scale,pattern",
    [
        (None, "random"),
        (128**-0.5, "random"),
        (0.08838834764831843, "random"),
        (0.125, "random"),
        (-(128**-0.5), "random"),
        (None, "mixed_large"),
        (None, "zero_qk"),
        (0.08838834764831843, "zero_q_large_k"),
        (0.08838834764831843, "large_q_zero_k"),
        (0.08838834764831843, "large_v"),
    ],
)
def test_state_flash_attention_interleaved_qkv(scale, pattern):
    generator = torch.Generator().manual_seed(20261105)
    storage = torch.randn(
        (8, 2049, 3, 16, 128), dtype=torch.bfloat16, generator=generator
    )
    if pattern == "mixed_large":
        # Exercise stable fallback and bounded tiles in the same invocation.
        storage[0].mul_(8)
    elif pattern == "zero_qk":
        storage[:, :, 0].zero_()
        storage[:, :, 1].zero_()
    elif pattern == "zero_q_large_k":
        storage[:, :, 0].zero_()
        storage[:, :, 1].fill_(1e20)
    elif pattern == "large_q_zero_k":
        storage[:, :, 0].fill_(1e20)
        storage[:, :, 1].zero_()
    elif pattern == "large_v":
        storage[:, :, 2].mul_(100)
    device_storage = storage.to(flag_gems.device)
    shape = (8, 16, 2049, 128)
    stride = (12589056, 128, 6144, 1)
    cpu_inputs = [
        storage.as_strided(shape, stride, offset) for offset in (0, 2048, 4096)
    ]
    inputs = [
        device_storage.as_strided(shape, stride, offset) for offset in (0, 2048, 4096)
    ]
    if flag_gems.device == "flagos":
        # This Torch-FL build skips codegen for the exact flash ATen entry;
        # explicitly bind its PrivateUse1 entry for this test's lifetime.
        lib = torch.library.Library("aten", "IMPL", "PrivateUse1")
        lib.impl(
            "_scaled_dot_product_flash_attention",
            flag_gems._scaled_dot_product_flash_attention,
        )
        result = torch.ops.aten._scaled_dot_product_flash_attention.default(
            *inputs, scale=scale
        )
    else:
        with flag_gems.use_gems(include=["_scaled_dot_product_flash_attention"]):
            result = torch.ops.aten._scaled_dot_product_flash_attention.default(
                *inputs, scale=scale
            )

    # Every batch and head, interior tile boundaries and the final odd row.
    rows = torch.tensor([0, 63, 64, 127, 128, 1024, 2047, 2048])
    q, k, v = [x.float() for x in cpu_inputs]
    actual_scale = 128**-0.5 if scale is None else scale
    scores = q[:, :, rows] @ k.transpose(-1, -2) * actual_scale
    reference = torch.softmax(scores, dim=-1) @ v
    reference_lse = torch.logsumexp(scores, dim=-1)
    torch.testing.assert_close(
        result[0].cpu().float()[:, :, rows], reference, atol=0.002, rtol=0.02
    )
    torch.testing.assert_close(
        result[1].cpu()[:, :, rows], reference_lse, atol=2e-5, rtol=2e-5
    )
    assert len(result) == 9
    assert result[2] is None and result[3] is None
    assert result[4:6] == (2049, 2049)
    assert result[0].shape == shape and result[0].dtype == torch.bfloat16
    assert result[1].shape == shape[:3] and result[1].dtype == torch.float32
    assert result[6].shape == (2,) and result[6].dtype == torch.int64
    assert result[7].ndim == 0 and result[7].dtype == torch.int64
    assert result[8] is None


@pytest.mark.scaled_dot_product_flash_attention
@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon STATE layout")
@pytest.mark.parametrize(
    "n,seed,pattern",
    [
        (n, seed, pattern)
        for n, seed in (
            (3, 5058009176373978791),
            (23, 6792970978253416049),
            (362, 8147618770989597313),
        )
        for pattern in (
            "normal",
            "large_qk",
            "zero_q_large_k",
            "large_q_zero_k",
            "large_v",
            "large_scores_large_v",
        )
    ]
    + [
        (362, 8147618770989597313, pattern)
        for pattern in (
            "signed_v8",
            "tiny_v",
            "large_finite_v",
            "negative_scores",
            "changing_scores",
        )
    ],
)
def test_state_flash_attention_expanded_independent_backings(n, seed, pattern):
    torch.manual_seed(seed)
    shape = (2, 16, n, 128)
    stride = (n * 6144, 128, 6144, 1)
    offsets = (0, 2048, 4096)
    backings = [
        torch.randn((2 * n * 6144 - 4096 + offset,), dtype=torch.bfloat16)
        for offset in offsets
    ]
    cpu = [x.as_strided(shape, stride, o) for x, o in zip(backings, offsets)]
    if pattern == "large_qk":
        cpu[0].mul_(4)
        cpu[1].mul_(4)
    elif pattern == "zero_q_large_k":
        cpu[0].zero_()
        cpu[1].fill_(1e20)
    elif pattern == "large_q_zero_k":
        cpu[0].fill_(1e20)
        cpu[1].zero_()
    elif pattern == "large_v":
        cpu[2].mul_(100)
    elif pattern == "large_scores_large_v":
        cpu[0].fill_(1.75)
        cpu[1].fill_(1.75)
        cpu[2].fill_(1e30)
    elif pattern == "signed_v8":
        cpu[2].copy_(torch.where(cpu[2] >= 0, 8.0, -8.0))
    elif pattern == "tiny_v":
        cpu[2].mul_(1e-20)
    elif pattern == "large_finite_v":
        cpu[2].mul_(10000)
    elif pattern == "negative_scores":
        cpu[0].fill_(1.75)
        cpu[1].fill_(-1.75)
    elif pattern == "changing_scores":
        cpu[0].fill_(1)
        cpu[1].fill_(0.01)
        cpu[1][:, :, :16].fill_(-100)
    stores = [x.to(flag_gems.device) for x in backings]
    dev = [x.as_strided(shape, stride, o) for x, o in zip(stores, offsets)]
    scale = 0.08838834764831843
    if flag_gems.device == "flagos":
        lib = torch.library.Library("aten", "IMPL", "PrivateUse1")
        lib.impl(
            "_scaled_dot_product_flash_attention",
            flag_gems._scaled_dot_product_flash_attention,
        )
        result = torch.ops.aten._scaled_dot_product_flash_attention.default(
            *dev, scale=scale
        )
    else:
        with flag_gems.use_gems(include=["_scaled_dot_product_flash_attention"]):
            result = torch.ops.aten._scaled_dot_product_flash_attention.default(
                *dev, scale=scale
            )
    q, k, v = [x.float() for x in cpu]
    scores = q @ k.transpose(-1, -2) * scale
    expected = torch.softmax(scores, dim=-1) @ v
    expected_lse = torch.logsumexp(scores, dim=-1)
    torch.testing.assert_close(result[0].cpu().float(), expected, atol=0.002, rtol=0.02)
    torch.testing.assert_close(result[1].cpu(), expected_lse, atol=2e-5, rtol=2e-5)
    assert len(result) == 9 and result[4:6] == (n, n)
