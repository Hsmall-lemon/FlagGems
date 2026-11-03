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

# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""F07 round2 final trial2: exact GELU selector and small tanh math changes.

GELU 1024/4 preserves its CPU +Inf-to-NaN contract. Small tanh 512/2 uses
exp2 and gfx936 native reciprocal. Neg and GELU tanh-approximation stay intact.
No input tensor values reach the host; kernels and scalar helpers live here.
Fresh source-bound process required; install(torch, flag_gems) is idempotent.
"""

import sys

import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import tl_extra_shim

TRIAL_ID = "f07_r2_exactgelu_naninput_tanh_exp2_native_rcp_v2"
BLOCK_SIZE = 1024
NUM_WARPS = 4
# Keep BLOCK_SIZE/NUM_WARPS unchanged: the inherited neg uses them.
_GELU_BLOCK_SIZE = 1024
_GELU_NUM_WARPS = 4
_TANH_BLOCK_SIZE = 512
_TANH_NUM_WARPS = 2
_TANH_MAX_NUMEL = 65536
_INT32_MAX = 2147483647
_erf = tl_extra_shim.erf
_tanh = tl_extra_shim.tanh
_TORCH = None
_ORIGINALS = None
_INSTALLED_ROOT = None
_DEVICE_TYPES = ()
_PATCHES = ()


@triton.jit
def _f07_gelu_none(x):
    x_fp32 = x.to(tl.float32)
    # Pinned CPU: +Inf gives NaN; finite multiply-before-half preserves overflow.
    z = tl.where(x_fp32 == float("inf"), float("nan"), x_fp32)
    scale: tl.constexpr = 0.7071067811
    return (z * (1 + _erf(z * scale))) * 0.5


@triton.jit
def _f07_gelu_tanh(x):
    x_fp32 = x.to(tl.float32)
    positive_inf = x_fp32 == float("inf")
    safe_x = tl.where(positive_inf, 0.0, x_fp32)
    x_sq = safe_x * safe_x
    output = 0.5 * safe_x * (1 + _tanh(safe_x * 0.79788456 * (1 + 0.044715 * x_sq)))
    return tl.where(positive_inf, x_fp32, output)


@triton.jit
def _f07_gelu_forward_contiguous_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    use_tanh: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    if use_tanh:
        output = _f07_gelu_tanh(x)
    else:
        output = _f07_gelu_none(x)
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def _f07_neg_signbit_kernel(
    input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    bits = x.to(tl.uint32, bitcast=True)
    flipped = bits ^ tl.full((), 0x80000000, tl.uint32)
    output = flipped.to(tl.float32, bitcast=True)
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def _f07_native_rcp_f32(x):
    # Approved gfx936-only trial: expose unsupported assembly as a JIT failure.
    # Denominator is [1, 2] for finite input. No software division or fallback.
    return tl.inline_asm_elementwise(
        "v_rcp_f32 $0, $1",
        constraints="=v,v",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f07_tanh_signstable(x):
    x_fp32 = x.to(tl.float32)
    e = tl.exp2(-2.8853900817779268 * tl.abs(x_fp32))
    y = (1.0 - e) * _f07_native_rcp_f32(1.0 + e)
    magnitude_bits = y.to(tl.uint32, bitcast=True) & 0x7FFFFFFF
    sign_bits = x_fp32.to(tl.uint32, bitcast=True) & 0x80000000
    signed_y = (magnitude_bits | sign_bits).to(tl.float32, bitcast=True)
    # Preserve input NaNs even if the target exp2 intrinsic canonicalizes them.
    return tl.where(x_fp32 != x_fp32, x_fp32, signed_y)


@triton.jit
def _f07_tanh_contiguous_kernel(
    input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    output = _f07_tanh_signstable(x)
    tl.store(output_ptr + offsets, output, mask=mask)


def _can_use_contiguous_fp32(A):
    # Metadata only; reject lazy view bits, tensor subclasses and grad inputs.
    return (
        type(A) is _TORCH.Tensor
        and A.dtype == _TORCH.float32
        and A.layout == _TORCH.strided
        and A.device.type in _DEVICE_TYPES
        and not A.requires_grad
        and not A.is_conj()
        and not A.is_neg()
        and A.is_contiguous()
        and A.numel() <= _INT32_MAX
    )


def gelu(A, *, approximate="none"):
    """Identifiable root aten::gelu entry; unsupported domains retain baseline."""
    if _ORIGINALS is None:
        raise RuntimeError("F07 trial2 requires install(torch, flag_gems)")
    if approximate not in ("none", "tanh") or not _can_use_contiguous_fp32(A):
        return _ORIGINALS["gelu"](A, approximate=approximate)
    output = _TORCH.empty_like(A)
    n_elements = A.numel()
    if n_elements == 0:
        return output
    with torch_device_fn.device(A.device.index):
        _f07_gelu_forward_contiguous_kernel[
            (triton.cdiv(n_elements, _GELU_BLOCK_SIZE),)
        ](
            A,
            output,
            n_elements,
            use_tanh=approximate == "tanh",
            BLOCK_SIZE=_GELU_BLOCK_SIZE,
            num_warps=_GELU_NUM_WARPS,
        )
    return output


def tanh(A):
    """Identifiable aten::tanh root; larger/unsupported input keeps its original."""
    if _ORIGINALS is None:
        raise RuntimeError("F07 round2 requires install(torch, flag_gems)")
    if not _can_use_contiguous_fp32(A) or A.numel() > _TANH_MAX_NUMEL:
        return _ORIGINALS["tanh"](A)
    output = _TORCH.empty_like(A)
    n_elements = A.numel()
    if n_elements == 0:
        return output
    with torch_device_fn.device(A.device.index):
        _f07_tanh_contiguous_kernel[(triton.cdiv(n_elements, _TANH_BLOCK_SIZE),)](
            A,
            output,
            n_elements,
            BLOCK_SIZE=_TANH_BLOCK_SIZE,
            num_warps=_TANH_NUM_WARPS,
        )
    return output


def neg(A):
    """Identifiable root aten::neg entry; XOR toggles sign without 0-x math."""
    if _ORIGINALS is None:
        raise RuntimeError("F07 trial2 requires install(torch, flag_gems)")
    if not _can_use_contiguous_fp32(A):
        return _ORIGINALS["neg"](A)
    output = _TORCH.empty_like(A)
    n_elements = A.numel()
    if n_elements == 0:
        return output
    with torch_device_fn.device(A.device.index):
        _f07_neg_signbit_kernel[(triton.cdiv(n_elements, BLOCK_SIZE),)](
            A,
            output,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=NUM_WARPS,
        )
    return output


def _replace_entries(entries, originals, wrappers):
    changed = []
    for item in entries:
        name = item[0] if len(item) >= 2 else None
        if name in wrappers and item[1] is originals[name]:
            changed.append((name, wrappers[name], *item[2:]))
        else:
            changed.append(item)
    if all(new is old for new, old in zip(changed, entries)):
        return entries
    return tuple(changed) if isinstance(entries, tuple) else changed


def install(torch, flag_gems):
    """Patch exact GELU/neg/tanh aliases and registrar entries, preserving others.

    Main must witness actual root calls and named kernels in a fresh process.
    No live torch.library registrations are removed or re-registered here.
    """
    global _TORCH, _ORIGINALS, _INSTALLED_ROOT, _DEVICE_TYPES, _PATCHES
    if _INSTALLED_ROOT is not None:
        if _INSTALLED_ROOT is not flag_gems:
            raise RuntimeError("F07 trial2 belongs to a different FlagGems root")
        if all(getattr(module, name) is new for module, name, old, new in _PATCHES):
            return
        raise RuntimeError("F07 bindings changed; use a fresh source-bound process")
    if flag_gems.runtime.device.vendor_name != "hygon":
        raise RuntimeError("F07 trial2 is approved only for pinned Hygon")
    wrappers = {"gelu": gelu, "neg": neg, "tanh": tanh}
    originals = {name: getattr(flag_gems, name) for name in wrappers}
    for name, original in originals.items():
        if not callable(original):
            raise TypeError("F07 requires active callable: " + name)
        if (
            original is wrappers[name]
            or getattr(original, "_under90_f07_trial", None) is not None
        ):
            raise RuntimeError("F07 already active; use a fresh source-bound process")
    get_private_name = getattr(torch._C, "_get_privateuse1_backend_name", None)
    private_name = (
        get_private_name() if get_private_name is not None else "privateuseone"
    )
    device_types = tuple(dict.fromkeys(("cuda", private_name)))
    patches = [
        (flag_gems, name, originals[name], wrapper)
        for name, wrapper in wrappers.items()
    ]
    owners = {getattr(fn, "__module__", "") for fn in originals.values()}
    for module_name, module in tuple(sys.modules.items()):
        if module is None or module is flag_gems:
            continue
        if not (
            module_name.startswith("flag_gems.")
            or module_name.startswith("_hygon.")
            or module_name in owners
        ):
            continue
        namespace = getattr(module, "__dict__", {})
        for name, wrapper in wrappers.items():
            if namespace.get(name) is originals[name]:
                patches.append((module, name, originals[name], wrapper))
    config = getattr(flag_gems, "_FULL_CONFIG", None)
    if config is not None:
        new_config = _replace_entries(config, originals, wrappers)
        if new_config is not config:
            patches.append((flag_gems, "_FULL_CONFIG", config, new_config))
    by_func = getattr(flag_gems, "FULL_CONFIG_BY_FUNC", None)
    if by_func is not None:
        new_by_func = by_func.copy()
        changed = False
        for name, entries in by_func.items():
            replacement = _replace_entries(entries, originals, wrappers)
            if replacement is not entries:
                new_by_func[name] = replacement
                changed = True
        if changed:
            patches.append((flag_gems, "FULL_CONFIG_BY_FUNC", by_func, new_by_func))
    _TORCH, _ORIGINALS, _DEVICE_TYPES = torch, originals, device_types
    applied = []
    try:
        for module, name, old, new in patches:
            setattr(module, name, new)
            applied.append((module, name, old, new))
    except BaseException:
        for module, name, old, new in reversed(applied):
            setattr(module, name, old)
        _TORCH, _ORIGINALS, _DEVICE_TYPES = None, None, ()
        raise
    _PATCHES = tuple(patches)
    _INSTALLED_ROOT = flag_gems
    for name, wrapper in wrappers.items():
        wrapper._under90_f07_original = originals[name]


gelu._under90_f07_trial = TRIAL_ID
neg._under90_f07_trial = TRIAL_ID
tanh._under90_f07_trial = TRIAL_ID
