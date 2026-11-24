"""Full output/LSE checks on multiple seeds and two deterministic input generators."""

import hashlib
import importlib
import json
import os
import socket
from pathlib import Path

from accuracy_gate import check_full, cpu_reference
from workbook_inputs import MANIFEST, make_case

torch_fl = importlib.import_module("torch_fl")
torch = importlib.import_module("torch")
flag_gems = importlib.import_module("flag_gems")
torch.set_num_threads(16)
torch.set_grad_enabled(False)
binding = torch.library.Library("aten", "IMPL", "PrivateUse1")
binding.impl(
    "_scaled_dot_product_flash_attention", flag_gems._scaled_dot_product_flash_attention
)
root = Path(flag_gems.__file__).resolve().parents[2]
paths = [
    "src/flag_gems/runtime/backend/_hygon/attention_state_hip.py",
    "src/flag_gems/runtime/backend/_hygon/attention_state_hip.cpp",
    "src/flag_gems/runtime/backend/_hygon/ops/attention_state.py",
    "src/flag_gems/runtime/backend/_hygon/ops/_scaled_dot_product_flash_attention.py",
    "tests/test_flash_attention_hygon_state.py",
]
receipt = {
    "device": {
        "node": socket.gethostname(),
        "physical_card": os.environ["HIP_VISIBLE_DEVICES"],
    },
    "source_sha256": {
        p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths
    },
    "cases": {},
    "original_model_acceptance_verified": False,
}
for case in MANIFEST["cases"]:
    records = []
    receipt["cases"][case["id"]] = records
    specs = [
        ("normal", seed, fp32)
        for fp32 in (False, True)
        for seed in (case["seed"], 0, 65537)
    ]
    if case["id"] == "expanded_large":
        specs += [
            (pattern, case["seed"], False)
            for pattern in (
                "large_qk",
                "zero_q_large_k",
                "large_q_zero_k",
                "large_v",
                "large_scores_large_v",
                "signed_v8",
                "tiny_v",
                "large_finite_v",
                "negative_scores",
                "changing_scores",
            )
        ]
    for pattern, seed, fp32 in specs:
        cpu, dev, kwargs, identity = make_case(
            case, torch, seed=seed, fp32_generation=fp32
        )
        stores, backings = None, None
        if pattern != "normal":
            del dev
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
            backings = [
                x.as_strided((size,), (1,), 0)
                for x, size in zip(cpu, identity["backing_elements"])
            ]
            stores = [x.to("flagos") for x in backings]
            dev = [
                store.as_strided(x.shape, x.stride(), x.storage_offset())
                for store, x in zip(stores, cpu)
            ]
            h = hashlib.sha256()
            for backing in backings:
                h.update(backing.view(torch.uint8).numpy().tobytes())
            identity["base_backing_sha256"] = identity["backing_sha256"]
            identity["backing_sha256"] = h.hexdigest()
        reference = cpu_reference(cpu)
        first = None
        checks = []
        for repeat in range(3):
            result = torch.ops.aten._scaled_dot_product_flash_attention.default(
                *dev, **kwargs
            )
            torch.flagos.synchronize()
            accuracy = check_full(*result[:2], reference)
            assert all(
                s["failed"] == 0 and s["nonfinite"] == 0 for s in accuracy.values()
            ), accuracy
            current = [x.cpu() for x in result[:2]]
            if first is None:
                first = current
            else:
                assert all(
                    torch.equal(a, b) for a, b in zip(first, current)
                ), "Unstable output/LSE"
            checks.append(accuracy)
            del result, current
        records.append(
            {
                "pattern": pattern,
                "identity": identity,
                "checks": checks,
                "bitwise_stable": True,
            }
        )
        Path("workbook-stability.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print("STABILITY_PASSED", case["id"], pattern, seed, fp32, flush=True)
        del cpu, dev, reference, first, stores, backings
print("WORKBOOK_STABILITY_COMPLETE", flush=True)
