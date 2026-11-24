# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Load the optional gfx936 packed dtype conversion kernel."""

import ctypes
import fcntl
import functools
import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _load_dtype_cast():
    compiler = shutil.which("hipcc")
    if compiler is None:
        return None

    try:
        source = Path(__file__).with_suffix(".cpp")
        flags = ("-x", "hip", "-O3", "-fPIC", "-shared", "--offload-arch=gfx936")
        runtime_paths = (os.environ.get("ROCM_PATH"), os.environ.get("HIP_PATH"))
        toolchain = (compiler, Path(compiler).stat().st_mtime_ns, flags, runtime_paths)
        digest = hashlib.sha256(
            source.read_bytes() + repr(toolchain).encode()
        ).hexdigest()
        cache_root = os.environ.get("FLAGOS_DEV_CACHE") or os.environ.get(
            "XDG_CACHE_HOME", str(Path.home() / ".cache")
        )
        cache = Path(cache_root) / "flag_gems" / "hygon_dtype_cast" / digest
        cache.mkdir(parents=True, exist_ok=True, mode=0o700)
        binary = cache / "cast.so"

        with (cache / "build.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not binary.is_file():
                with tempfile.TemporaryDirectory(dir=cache) as work:
                    output = Path(work) / binary.name
                    subprocess.run(
                        [compiler, *flags, str(source), "-o", str(output)],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )
                    output.replace(binary)

        library = ctypes.CDLL(str(binary))
        cast = library.launch_dtype_cast
        cast.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        cast.restype = ctypes.c_int
        return cast
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("gfx936 packed dtype conversion unavailable: %s", exc)
        return None


def launch_dtype_cast(x, out, threads, vectors):
    cast = _load_dtype_cast()
    if cast is None:
        return False
    stream = torch_device_fn.current_stream(x.device).cuda_stream
    status = cast(
        x.data_ptr(),
        out.data_ptr(),
        x.numel(),
        x.dtype == torch.bfloat16,
        threads,
        vectors,
        stream,
    )
    if status:
        raise RuntimeError(
            f"gfx936 packed dtype conversion launch failed: HIP {status}"
        )
    return True
