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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.native_batch_norm
@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon-specific path")
@pytest.mark.parametrize(
    "rows,affine,training,eps",
    [
        (1, True, False, 1e-5),
        (8, True, False, 1e-5),
        (9, False, False, 2e-3),
        (9, True, True, 1e-5),
    ],
    ids=["one-row", "full-tile", "tail-no-affine", "training-fallback"],
)
def test_native_batch_norm_hygon(rows, affine, training, eps, caplog):
    channels = 128
    dtype = torch.float32
    cpu_inputs = (
        torch.linspace(-2, 2, rows * channels, dtype=dtype, device="cpu").reshape(
            rows, channels
        ),
        (
            torch.linspace(0.5, 1.5, channels, dtype=dtype, device="cpu")
            if affine
            else None
        ),
        (
            torch.linspace(-0.25, 0.25, channels, dtype=dtype, device="cpu")
            if affine
            else None
        ),
        torch.linspace(-1, 1, channels, dtype=dtype, device="cpu"),
        torch.linspace(0.5, 2, channels, dtype=dtype, device="cpu"),
    )
    inputs = tuple(
        t.to(flag_gems.device, copy=True) if t is not None else None for t in cpu_inputs
    )
    op = torch.ops.aten.native_batch_norm.default
    reference = op(*cpu_inputs, training, 0.1, eps)

    with (
        caplog.at_level(
            "DEBUG", logger=utils.gems_log_logger(flag_gems.native_batch_norm)
        ),
        flag_gems.use_gems(include=["native_batch_norm"]),
    ):
        result = op(*inputs, training, 0.1, eps)

    route = "GEMS_HYGON NATIVE_BATCH_NORM"
    fast_route = route + " INFERENCE"
    if training:
        assert route in caplog.messages and fast_route not in caplog.messages
    else:
        assert fast_route in caplog.messages

    assert len(result) == len(reference) == 3
    for actual, expected in zip(result, reference):
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert actual.device == inputs[0].device
        utils.gems_assert_close(
            actual, expected, dtype, reduce_dim=rows if training else 1
        )
    if not training:
        assert result[1].numel() == result[2].numel() == 0
    for actual, expected in zip(inputs[-2:], cpu_inputs[-2:]):
        if training:
            utils.gems_assert_close(actual, expected, dtype)
        else:
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
