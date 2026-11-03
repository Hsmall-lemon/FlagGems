# isort: skip_file - preserve the archived provider import order
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

"""F08 final family trial: retain measured rsqrt launch, remove no-gain pow override.

Main owns GPU validation. No power attribute/registration/kernel is changed.
"""

import triton
import triton.language as tl
from flag_gems.runtime import torch_device_fn

TRIAL_ID = "f08-r2-native-rsqrt-alignment-repair"
RSQRT_MIN_NUMEL = 65536
MAX_NUMEL = (1 << 31) - 1
_DEBUG = False
_HITS = {"rsqrt_fast": 0, "rsqrt_fallback": 0}


@triton.jit
def _rsqrt_kernel(input_ptr, output_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(input_ptr + offsets, mask=offsets < N, other=1.0)
    bits = x.to(tl.int32, bitcast=True)
    magnitude = bits & 0x7FFFFFFF
    mantissa = bits & 0x007FFFFF
    subnormal = (magnitude < 0x00800000) & (mantissa != 0)
    # Integer mantissa conversion avoids hardware flushing the subnormal input.
    # x = mantissa * 2**-149, hence rsqrt(x) = rsqrt(mantissa) * 2**74.5.
    argument = tl.where(subnormal, mantissa.to(tl.float32), x.to(tl.float32))
    result = tl.rsqrt(argument)
    result = tl.where(subnormal, result * 2.6713738906281536e22, result)
    result = tl.where(subnormal & (bits < 0), float("nan"), result)
    result = tl.where(
        magnitude == 0, tl.where(bits < 0, -float("inf"), float("inf")), result
    )
    tl.store(output_ptr + offsets, result, mask=offsets < N)


def set_debug(enabled=True):
    """Untimed hit diagnostics only; leave disabled for measurements."""
    global _DEBUG
    _DEBUG = bool(enabled)
    for key in _HITS:
        _HITS[key] = 0


def install(torch, flag_gems):
    previous = getattr(flag_gems, "_under90_f08_numeric", None)
    if previous is not None:
        if previous["trial_id"] != TRIAL_ID:
            raise RuntimeError("F08 trial already installed; use an isolated snapshot")
        if flag_gems.rsqrt is not previous["wrappers"]["rsqrt"]:
            raise RuntimeError("F08 installed provider changed; use a fresh process")
        return previous

    config = flag_gems._FULL_CONFIG
    entries = [entry for entry in config if entry[0] == "rsqrt"]
    if len(entries) != 1:
        raise RuntimeError("F08 expected one rsqrt registration")
    original_rsqrt = flag_gems.rsqrt
    is_hygon = getattr(flag_gems, "vendor_name", None) == "hygon"

    def eligible(A):
        return (
            is_hygon
            and isinstance(A, torch.Tensor)
            and A.dtype == torch.float32
            and A.device.type == "flagos"
            and A.layout == torch.strided
            and RSQRT_MIN_NUMEL <= A.numel() <= MAX_NUMEL
            and A.is_contiguous()
            and not A.requires_grad
            and not torch.is_grad_enabled()
            and not A.is_conj()
            and not A.is_neg()
        )

    def rsqrt(A):
        if eligible(A):
            if _DEBUG:
                _HITS["rsqrt_fast"] += 1
            output = torch.empty_like(A)
            n = A.numel()
            with torch_device_fn.device(A.device):
                _rsqrt_kernel[(triton.cdiv(n, 1024),)](
                    A,
                    output,
                    n,
                    BLOCK=1024,
                    num_warps=4,
                    num_stages=2,
                    enable_fp_fusion=True,
                    allow_flush_denorm=False,
                )
            return output
        if _DEBUG:
            _HITS["rsqrt_fallback"] += 1
        return original_rsqrt(A)

    rsqrt.__wrapped__ = original_rsqrt
    rsqrt.__doc__ = "F08 retained rsqrt launch; exact active-provider fallback."
    old_entry = entries[0]
    new_entry = (old_entry[0], rsqrt, *old_entry[2:])
    updated = tuple(new_entry if entry[0] == "rsqrt" else entry for entry in config)
    updated_cache = {
        name: [new_entry if entry[0] == "rsqrt" else entry for entry in values]
        for name, values in flag_gems.FULL_CONFIG_BY_FUNC.items()
    }
    flag_gems._FULL_CONFIG = updated
    flag_gems.FULL_CONFIG_BY_FUNC = updated_cache
    flag_gems.rsqrt = rsqrt
    state = {
        "trial_id": TRIAL_ID,
        "originals": {"rsqrt": original_rsqrt},
        "wrappers": {"rsqrt": rsqrt},
        "hits": _HITS,
        "set_debug": set_debug,
        "debug_default": False,
        "rsqrt_numel_bounds": (RSQRT_MIN_NUMEL, MAX_NUMEL),
        "rsqrt_launch": {"block": 1024, "num_warps": 4, "num_stages": 2},
    }
    flag_gems._under90_f08_numeric = state
    return state
