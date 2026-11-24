import hashlib
import importlib
import json
import os
import socket
import uuid
from importlib.metadata import version
from pathlib import Path

from accuracy_gate import (
    CONTRACT,
    CONTRACT_SHA256,
    check_full,
    cpu_reference,
    require_eligible_baselines,
)
from probe import Timer, flag_gems, make_inputs, torch, torch_fl

from flag_gems.runtime.backend._hygon.attention_state_hip import _load_state_kernel

# The extension requires libtorch to have been loaded by probe first.
fa = importlib.import_module("flash_attn_2_cuda")

inputs, cpu = make_inputs()
views = [torch_fl._C._flagos_to_cuda_view(x) for x in inputs]
wv = [x.transpose(1, 2) for x in views]
wheel = lambda: fa.fwd(
    *wv,
    None,
    None,
    0.0,
    CONTRACT["scale"],
    False,
    -1,
    -1,
    0.0,
    False,
    None,
    False,
    None,
    0.0,
)
native = lambda: torch.ops.aten._scaled_dot_product_flash_attention.default(
    *views, scale=CONTRACT["scale"]
)
repo = lambda: flag_gems._scaled_dot_product_flash_attention(
    *inputs, scale=CONTRACT["scale"]
)
library = _load_state_kernel()
assert library is not None
binding = torch.library.Library("aten", "IMPL", "PrivateUse1")
binding.impl(
    "_scaled_dot_product_flash_attention", flag_gems._scaled_dot_product_flash_attention
)
repo = lambda: torch.ops.aten._scaled_dot_product_flash_attention.default(
    *inputs, scale=CONTRACT["scale"]
)
root = Path(flag_gems.__file__).resolve().parents[2]
files = [
    "src/flag_gems/runtime/backend/_hygon/attention_state_hip.py",
    "src/flag_gems/runtime/backend/_hygon/attention_state_hip.cpp",
    "src/flag_gems/runtime/backend/_hygon/ops/attention_state.py",
    "src/flag_gems/runtime/backend/_hygon/ops/_scaled_dot_product_flash_attention.py",
    "tests/test_flash_attention_hygon_state.py",
]
hashes = {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in files}
native_libraries = sorted(
    {
        line.split()[-1]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "flash_att" in line and line.split()[-1].startswith("/")
    }
)
result = {
    "schema": 2,
    "run_id": uuid.uuid4().hex,
    "contract": CONTRACT,
    "contract_sha256": CONTRACT_SHA256,
    "input_sha256": hashlib.sha256(
        cpu[0]._base.view(torch.uint8).numpy().tobytes()
    ).hexdigest(),
    "device": {
        "node": socket.gethostname(),
        "physical_card": os.environ.get("HIP_VISIBLE_DEVICES"),
        "pid": os.getpid(),
    },
    "native": {
        "version": version("flash-attn"),
        "options": {
            name: os.environ.get(name)
            for name in ("FA_FWD_NO_MLS", "FLASH_ATTENTION_FWD_PREFETCH")
        },
        "module": fa.__file__,
        "module_sha256": hashlib.sha256(Path(fa.__file__).read_bytes()).hexdigest(),
        "loaded_libraries_sha256": {
            path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in native_libraries
        },
    },
    "source_sha256": hashes,
    "binary": library._name,
    "binary_sha256": hashlib.sha256(Path(library._name).read_bytes()).hexdigest(),
    "timings": {},
    "full_accuracy": {},
    "acceptance_timing_started": False,
}

receipt_path = Path("repo_final_profile_validation.json")
receipt_path.write_text(json.dumps(result, indent=2))
reference = cpu_reference(cpu)
for label, fn in (("wheel", wheel), ("aten", native), ("repo", repo)):
    y = fn()
    torch.flagos.synchronize()
    if label == "wheel":
        out = torch_fl._C._cuda_to_flagos_view(y[0]).transpose(1, 2)
        lse = torch_fl._C._cuda_to_flagos_view(y[5])
    elif label == "aten":
        out, lse = [torch_fl._C._cuda_to_flagos_view(t) for t in y[:2]]
    else:
        out, lse = y[:2]
    try:
        result["full_accuracy"][label] = check_full(out, lse, reference)
    except (ValueError, RuntimeError) as exc:
        result["full_accuracy"][label] = {"error": str(exc)}
    receipt_path.write_text(json.dumps(result, indent=2))
    del y, out, lse
require_eligible_baselines(result)
del reference

# No timing range is emitted until all three arms pass the same full gate.
result["acceptance_timing_started"] = True
timer = Timer()
for label, fn in [
    ("wheel", wheel),
    ("aten", native),
    ("repo", repo),
    ("wheel_repeat", wheel),
    ("aten_repeat", native),
    ("repo_repeat", repo),
]:
    result["timings"][label] = timer.bench(fn, result["run_id"] + ":" + label, rounds=5)
    print("TIMING", label, result["timings"][label], flush=True)
Path("repo_final_profile_validation.json").write_text(json.dumps(result, indent=2))
