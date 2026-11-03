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

"""F04 trial 1: contiguous FP32 functional masked_fill, one Triton launch.

Pinned gate: design-f04.json, snapshot 20261102-162036-adc63b87dd14.
No vendor calls or tensor-content assumptions. Unsupported calls use the active
FlagGems function captured at install time. nan_to_num/inplace are untouched.
Main copies this exact self-contained source to snapshot/f04.py.
"""

import math
import sys

import triton
import triton.language as tl

IS_BASELINE = False
TRIAL_ID = "f04-t1-contig-1024-w4"
BLOCK_SIZE = 1024
NUM_WARPS = 4
MIN_RANK = 1
MAX_RANK = 8
MAX_NUMEL = 1 << 20
_FP32_MAX = 3.4028234663852886e38
_TORCH = None
_DEVICE_FN = None
_INSTALLED_ROOT = None
_ORIGINAL = None
_PATCHES = ()


@triton.jit
def _f04_masked_fill_contiguous(
    input_ptr, mask_ptr, output_ptr, value, BLOCK: tl.constexpr
):
    # Guarded lengths are positive multiples of BLOCK; every lane is valid.
    # Offsets are contiguous, allowing the backend to vectorize loads/stores.
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(input_ptr + offset)
    mask = tl.load(mask_ptr + offset)
    y = tl.where(mask, value.to(tl.float32), x)
    tl.store(output_ptr + offset, y)


def _eligible(inp, mask, value):
    torch = _TORCH
    if type(value) is not float:
        return False
    if math.isfinite(value) and abs(value) > _FP32_MAX:
        return False
    if type(inp) is not torch.Tensor or type(mask) is not torch.Tensor:
        return False
    if inp.dtype != torch.float32 or mask.dtype != torch.bool:
        return False
    if inp.layout != torch.strided or mask.layout != torch.strided:
        return False
    if inp.device.type not in ("flagos", "cuda") or mask.device != inp.device:
        return False
    if inp.requires_grad or mask.requires_grad:
        return False
    if tuple(mask.shape) != tuple(inp.shape):
        return False
    if not MIN_RANK <= len(inp.shape) <= MAX_RANK:
        return False
    n = inp.numel()
    if n <= 0 or n > MAX_NUMEL or n % BLOCK_SIZE != 0:
        return False
    if not inp.is_contiguous() or not mask.is_contiguous():
        return False
    # Generic contiguous flattening: no square/full-shape specialization.
    # Zero-offset storage is retained; no alignment hints are asserted here.
    # Offset views, general striding and broadcasts retain original behavior.
    if inp.storage_offset() != 0 or mask.storage_offset() != 0:
        return False
    if inp.is_conj() or inp.is_neg() or mask.is_conj() or mask.is_neg():
        return False
    return True


def masked_fill(inp, mask, value):
    """Identifiable root entry; guard misses call the captured active original."""
    if _ORIGINAL is None:
        raise RuntimeError("F04 requires install(torch, flag_gems) before use")
    if not _eligible(inp, mask, value):
        return _ORIGINAL(inp, mask, value)
    output = _TORCH.empty_like(inp)
    n = inp.numel()
    with _DEVICE_FN.device(inp.device):
        _f04_masked_fill_contiguous[(n // BLOCK_SIZE,)](
            inp, mask, output, value, BLOCK=BLOCK_SIZE, num_warps=NUM_WARPS
        )
    return output


def install(torch, flag_gems):
    """Compose with active patches; install before dispatcher registration.

    Preserve all unrelated root/alias/config objects. Matching aliases are
    replaced only when their function is exactly the captured root original.
    Existing dispatcher registrations are never overridden here.
    """
    global _TORCH, _DEVICE_FN, _INSTALLED_ROOT, _ORIGINAL, _PATCHES
    if _INSTALLED_ROOT is not None:
        if _INSTALLED_ROOT is not flag_gems:
            raise RuntimeError("F04 already belongs to another FlagGems root")
        if all(getattr(module, name) is new for module, name, old, new in _PATCHES):
            return
        raise RuntimeError("F04 installation changed; use a fresh variant process")
    original = flag_gems.masked_fill
    if original is masked_fill or getattr(original, "_f04_trial_id", None) is not None:
        raise RuntimeError(
            "F04 already installed from another module; use a fresh process"
        )
    # Import only the existing device guard; real kernel source stays in this file.
    from flag_gems.runtime import torch_device_fn

    patches = [(flag_gems, "masked_fill", original, masked_fill)]
    for module_name, module in tuple(sys.modules.items()):
        if (
            module is None
            or module is flag_gems
            or not (module_name == "flag_gems" or module_name.startswith("flag_gems."))
        ):
            continue
        namespace = getattr(module, "__dict__", {})
        if namespace.get("masked_fill") is original:
            patches.append((module, "masked_fill", original, masked_fill))
    # The registrar consumes _FULL_CONFIG, not the updated Python aliases.
    # Only functional masking entries are changed; extra tuple fields survive.
    config = getattr(flag_gems, "_FULL_CONFIG", None)
    if config is not None:
        changed = []
        for item in config:
            if (
                item[0] in ("masked_fill.Scalar", "masked_fill.Tensor")
                and item[1] is original
            ):
                changed.append((item[0], masked_fill, *item[2:]))
            else:
                changed.append(item)
        if any(a is not b for a, b in zip(config, changed)):
            new_config = tuple(changed) if isinstance(config, tuple) else changed
            patches.append((flag_gems, "_FULL_CONFIG", config, new_config))
    _TORCH, _DEVICE_FN, _ORIGINAL = torch, torch_device_fn, original
    applied = []
    try:
        for module, name, old, new in patches:
            setattr(module, name, new)
            applied.append((module, name, old, new))
    except BaseException:
        for module, name, old, new in reversed(applied):
            setattr(module, name, old)
        _TORCH = _DEVICE_FN = _ORIGINAL = None
        raise
    _PATCHES = tuple(patches)
    _INSTALLED_ROOT = flag_gems


masked_fill._f04_trial_id = TRIAL_ID
