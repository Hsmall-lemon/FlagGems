"""Full finite-input boundary checks; reject ineligible native comparisons."""

import hashlib
import importlib
import json
import os
from importlib.metadata import version
from pathlib import Path

from accuracy_gate import CONTRACT, check_full, cpu_reference

torch_fl = importlib.import_module("torch_fl")
torch = importlib.import_module("torch")
fa = importlib.import_module("flash_attn_2_cuda")
flag_gems = importlib.import_module("flag_gems")
torch.set_num_threads(16)
root = Path(flag_gems.__file__).resolve().parents[2]
sources = [
    "src/flag_gems/runtime/backend/_hygon/attention_state_hip.cpp",
    "src/flag_gems/runtime/backend/_hygon/ops/attention_state.py",
    "tests/test_flash_attention_hygon_state.py",
]
result = {
    "card": os.environ["HIP_VISIBLE_DEVICES"],
    "contract": CONTRACT,
    "native_version": version("flash-attn"),
    "native_module": fa.__file__,
    "source_sha256": {
        p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sources
    },
    "cases": {},
}
assert result["card"] in ("3", "4")
for pattern in ("zero_q_large_k", "large_q_zero_k", "large_v"):
    torch.manual_seed(20261105)
    backing = torch.randn((8, 2049, 3, 16, 128), dtype=torch.bfloat16)
    if pattern == "zero_q_large_k":
        backing[:, :, 0].zero_()
        backing[:, :, 1].fill_(1e20)
    elif pattern == "large_q_zero_k":
        backing[:, :, 0].fill_(1e20)
        backing[:, :, 1].zero_()
    else:
        backing[:, :, 2].mul_(100)
    storage = backing.to("flagos")
    cpu = [
        backing.as_strided(CONTRACT["output_shape"], CONTRACT["stride"], o)
        for o in CONTRACT["offsets"]
    ]
    dev = [
        storage.as_strided(CONTRACT["output_shape"], CONTRACT["stride"], o)
        for o in CONTRACT["offsets"]
    ]
    views = [torch_fl._C._flagos_to_cuda_view(x).transpose(1, 2) for x in dev]
    reference = cpu_reference(cpu)
    case = {}
    for arm in ("repo", "native"):
        runs = []
        first = None
        for repeat in range(3 if arm == "repo" else 1):
            if arm == "repo":
                y = flag_gems._scaled_dot_product_flash_attention(
                    *dev, scale=CONTRACT["scale"]
                )
                out, lse = y[:2]
            else:
                y = fa.fwd(
                    *views,
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
                out = torch_fl._C._cuda_to_flagos_view(y[0]).transpose(1, 2)
                lse = torch_fl._C._cuda_to_flagos_view(y[5])
            torch.flagos.synchronize()
            check = check_full(out, lse, reference)
            passed = all(
                x["failed"] == 0 and x["nonfinite"] == 0 for x in check.values()
            )
            if arm == "repo" or pattern != "large_v":
                assert passed, (pattern, arm, check)
            actual = (out.cpu(), lse.cpu())
            if first is None:
                first = actual
            equal = all(torch.equal(a, b) for a, b in zip(actual, first))
            assert equal
            runs.append(
                {
                    "passed": passed,
                    "full_accuracy": check,
                    "bitwise_equal_to_first": equal,
                }
            )
            del y, out, lse, actual
        case[arm] = runs
        print("EDGE_FULL_CHECK", pattern, arm, passed, flush=True)
        del first
    case["native_eligible_for_performance"] = case["native"][0]["passed"]
    result["cases"][pattern] = case
    Path("edge-cases.json").write_text(json.dumps(result, indent=2))
    del backing, storage, cpu, dev, views, reference
print("EDGE_FULL_CHECK_COMPLETE", flush=True)
