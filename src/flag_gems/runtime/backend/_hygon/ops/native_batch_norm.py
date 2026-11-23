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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.native_batch_norm import (
    native_batch_norm as _generic_native_batch_norm,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import tl_extra_shim

logger = logging.getLogger(__name__)
rsqrt = tl_extra_shim.rsqrt

_CHANNELS = 128
_BLOCK_ROWS = 8
# Keep signed 32-bit byte offsets valid even for the masked part of the last tile.
_MAX_ROWS = (1 << 31) // (_BLOCK_ROWS * _CHANNELS * 4) * _BLOCK_ROWS


@triton.jit
def native_batch_norm_inference_kernel(
    X,
    W,
    B,
    MEAN,
    VAR,
    Y,
    n_rows,
    eps,
    C: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    channels = tl.arange(0, C)
    mean = tl.load(MEAN + channels)
    inv_std = rsqrt(tl.load(VAR + channels) + eps)
    if W is not None:
        weight = tl.load(W + channels)
    else:
        weight = tl.full((C,), 1.0, tl.float32)
    if B is not None:
        bias = tl.load(B + channels)
    else:
        bias = tl.full((C,), 0.0, tl.float32)

    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    offsets = rows[:, None] * C + channels[None, :]
    mask = rows[:, None] < n_rows
    x = tl.load(X + offsets, mask, other=0.0)
    y = weight[None, :] * (x - mean[None, :]) * inv_std[None, :] + bias[None, :]
    tl.store(Y + offsets, y, mask)


def _is_channel_vector(tensor, device):
    return (
        isinstance(tensor, torch.Tensor)
        and tensor.device == device
        and tensor.dtype == torch.float32
        and tensor.layout == torch.strided
        and not tensor.is_nested
        and tensor.shape == (_CHANNELS,)
        and tensor.is_contiguous()
        and not tensor.is_neg()
    )


def native_batch_norm(
    input,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    training=False,
    momentum=0.1,
    eps=1e-5,
):
    device = input.device
    if (
        training
        or device.type not in ("cuda", "flagos")
        or input.dtype != torch.float32
        or input.layout != torch.strided
        or input.is_nested
        or input.ndim != 2
        or input.shape[1] != _CHANNELS
        or not 0 < input.shape[0] <= _MAX_ROWS
        or not input.is_contiguous()
        or input.is_neg()
        or not _is_channel_vector(running_mean, device)
        or not _is_channel_vector(running_var, device)
        or (weight is not None and not _is_channel_vector(weight, device))
        or (bias is not None and not _is_channel_vector(bias, device))
    ):
        logger.debug("GEMS_HYGON NATIVE_BATCH_NORM")
        return _generic_native_batch_norm(
            input, weight, bias, running_mean, running_var, training, momentum, eps
        )

    logger.debug("GEMS_HYGON NATIVE_BATCH_NORM INFERENCE")
    output = torch.empty_like(input)
    n_rows = input.shape[0]
    # Eight rows amortize the channel loads without sacrificing grid size on
    # small batches; the last dimension keeps adjacent channels contiguous.
    with torch_device_fn.device(device):
        native_batch_norm_inference_kernel[(triton.cdiv(n_rows, _BLOCK_ROWS),)](
            input,
            weight,
            bias,
            running_mean,
            running_var,
            output,
            n_rows,
            eps,
            C=_CHANNELS,
            BLOCK_ROWS=_BLOCK_ROWS,
            num_warps=2,
            enable_fp_fusion=False,
        )

    save_mean = torch.empty((0,), dtype=torch.float32, device=device)
    save_invstd = torch.empty((0,), dtype=torch.float32, device=device)
    return output, save_mean, save_invstd
