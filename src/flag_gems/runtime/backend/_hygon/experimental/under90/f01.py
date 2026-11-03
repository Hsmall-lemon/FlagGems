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

"""F01 final trial 2: fuse float32 truth conversion with contiguous row AND.

Load this real file as snapshot/f01.py and call install(torch, flag_gems)
before the first ATen invocation. Inputs outside the metadata guard retain
the captured FlagGems implementation. No vendor delegation or value sampling.
"""

import torch as _torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

_ORIGINAL_ALL = None
_ORIGINAL_ALL_DIM = None
_ORIGINAL_ALL_DIMS = None
_INSTALLED_GEMS = None
_PRIVATE_DEVICE_TYPE = "flagos"

TRIAL = 2
BLOCK_M = 4


@triton.jit
def _logical_and(left, right):
    return left & right


@libentry()
@triton.jit
def _all_f32_rows(
    inp,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    offsets = rows[:, None] * N + cols[None, :]
    valid = (rows[:, None] < M) & (cols[None, :] < N)
    # True padding is AND's identity, including a partial final row group.
    values = tl.load(inp + offsets, mask=valid, other=1.0)
    truth = values != 0.0
    reduced = tl.reduce(truth, axis=1, combine_fn=_logical_and)
    tl.store(out + rows, reduced, mask=rows < M)


def _fast_metadata(inp, dim, keepdim):
    """Metadata only. Never inspect values, pointers, or benchmark case IDs."""
    if type(dim) is not int or type(keepdim) is not bool:
        return False
    if inp.dtype != _torch.float32 or inp.layout != _torch.strided:
        return False
    if inp.device.type != _PRIVATE_DEVICE_TYPE:
        return False
    rank = inp.ndim
    if rank < 1 or rank > 8 or dim not in (-1, rank - 1):
        return False
    if not inp.is_contiguous():
        return False
    n = inp.shape[-1]
    if type(n) is not int or n < 128 or n > 1024:
        return False
    count = inp.numel()
    if type(count) is not int or count == 0:
        return False
    m = count // n
    # Bound indexing and launch size; no tensor-value or shape-result inference.
    return 1 <= m <= 65536 and count <= 2147483647


_NO_REPAIR = object()


def _normalized_axes(rank, dim):
    if dim is None:
        return tuple(range(rank))
    if type(dim) is int:
        requested = (dim,)
    elif isinstance(dim, (tuple, list)):
        requested = tuple(dim)
    else:
        raise TypeError("all(): dim must be an int, tuple/list of ints, or None")
    limit = max(rank, 1)
    axes = []
    for axis in requested:
        if type(axis) is not int:
            raise TypeError("all(): dimension must be an int")
        if axis < -limit or axis >= limit:
            raise IndexError("Dimension out of range")
        axis %= limit
        if axis in axes:
            raise RuntimeError("dim appears multiple times in the list of dims")
        axes.append(axis)
    # A scalar accepts dim=0/-1 but has no physical axis to remove or retain.
    return tuple(axes) if rank else ()


def _semantic_repair(inp, dim, keepdim):
    """Repair only metadata-selected edge cases; never read tensor values on host."""
    if inp.layout != _torch.strided:
        return _NO_REPAIR
    empty_axes = isinstance(dim, (tuple, list)) and len(dim) == 0
    if (
        inp.numel() != 0
        and inp.dtype != _torch.uint8
        and inp.ndim != 0
        and not empty_axes
    ):
        return _NO_REPAIR
    if type(keepdim) is not bool:
        raise TypeError("all(): keepdim must be bool")
    axes = _normalized_axes(inp.ndim, dim)
    dtype = _torch.uint8 if inp.dtype == _torch.uint8 else _torch.bool
    if keepdim:
        out_shape = tuple(1 if i in axes else n for i, n in enumerate(inp.shape))
    else:
        out_shape = tuple(n for i, n in enumerate(inp.shape) if i not in axes)
    with torch_device_fn.device(inp.device):
        if inp.numel() == 0:
            # All existing output coordinates reduce an empty set. If a
            # surviving dimension is zero, this is correctly an empty output.
            return _torch.ones(out_shape, dtype=dtype, device=inp.device)
        if not axes:
            # dim=[] means elementwise truth, including scalar dim=0/-1.
            # clone ensures fresh output even when bool conversion is a no-op.
            return inp.to(dtype=_torch.bool).to(dtype=dtype).clone()
        # Remaining repaired case is uint8. Reduce a canonical bool tensor
        # through the captured FlagGems implementation, then restore uint8 0/1.
        # Canonicalization also avoids the original full-reduction stride bug.
        truth = inp.to(dtype=_torch.bool).contiguous()
        if dim is None:
            result = _ORIGINAL_ALL(truth)
            if keepdim:
                result = result.reshape(out_shape)
        elif type(dim) is int:
            result = _ORIGINAL_ALL_DIM(truth, dim=axes[0], keepdim=keepdim)
        else:
            result = _ORIGINAL_ALL_DIMS(truth, dim=list(axes), keepdim=keepdim)
        return result.to(dtype=dtype)


def all(inp):
    repaired = _semantic_repair(inp, None, False)
    if repaired is not _NO_REPAIR:
        return repaired
    return _ORIGINAL_ALL(inp)


def all_dim(inp, dim=None, keepdim=False):
    if not _fast_metadata(inp, dim, keepdim):
        repaired = _semantic_repair(inp, dim, keepdim)
        if repaired is not _NO_REPAIR:
            return repaired
        return _ORIGINAL_ALL_DIM(inp, dim=dim, keepdim=keepdim)
    shape = tuple(inp.shape)
    n = shape[-1]
    m = inp.numel() // n
    out_shape = shape[:-1] + ((1,) if keepdim else ())
    # The guard is float32-only, so the schema's result dtype is always bool.
    with torch_device_fn.device(inp.device):
        out = _torch.empty(out_shape, dtype=_torch.bool, device=inp.device)
        _all_f32_rows[(triton.cdiv(m, BLOCK_M),)](
            inp,
            out,
            m,
            n,
            BLOCK_M,
            triton.next_power_of_2(n),
            num_warps=4,
            num_stages=2,
            allow_flush_denorm=False,
        )
    return out


def all_dims(inp, dim=None, keepdim=False):
    repaired = _semantic_repair(inp, dim, keepdim)
    if repaired is not _NO_REPAIR:
        return repaired
    return _ORIGINAL_ALL_DIMS(inp, dim=dim, keepdim=keepdim)


def install(torch, flag_gems):
    """Patch the three root exports; preserve original callable identities."""
    global _torch, _ORIGINAL_ALL, _ORIGINAL_ALL_DIM, _ORIGINAL_ALL_DIMS
    global _INSTALLED_GEMS, _PRIVATE_DEVICE_TYPE
    if _INSTALLED_GEMS is not None:
        if (
            _INSTALLED_GEMS is flag_gems
            and flag_gems.all is all
            and flag_gems.all_dim is all_dim
            and flag_gems.all_dims is all_dims
        ):
            return
        raise RuntimeError("F01 candidate must be installed once in a fresh process")
    originals = (flag_gems.all, flag_gems.all_dim, flag_gems.all_dims)
    if not _torch.__name__ == torch.__name__:
        raise RuntimeError("F01 requires the active torch module")
    _torch = torch
    _PRIVATE_DEVICE_TYPE = torch._C._get_privateuse1_backend_name()
    _ORIGINAL_ALL, _ORIGINAL_ALL_DIM, _ORIGINAL_ALL_DIMS = originals
    flag_gems.all = all
    flag_gems.all_dim = all_dim
    flag_gems.all_dims = all_dims
    _INSTALLED_GEMS = flag_gems
