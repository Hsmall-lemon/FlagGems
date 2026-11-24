"""Recompute device acceptance from full precision receipts and raw GPU traces."""

import argparse
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--bundle",
    type=Path,
    default=Path(__file__).resolve().parents[1] / "release2610-verification",
)
args = parser.parse_args()
root = args.bundle.resolve() / "evidence"
repo = Path(__file__).resolve().parents[4]
expected_native = {
    "flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so": (
        "f130f8a659a91b67a9776f8cd2a2746d645db6e20697ff4e16bd25dc12e91663"
    ),
    "flash_attn/lib/libflash_attention.so": "69060dc997524cdc803a5bd691d698ae6e97a125cea767d37b941f536787f00b",
}
versions = {
    "torch": "2.13.0+dtk2610.2610261432.gefb4e1",
    "triton": "3.6.0+dtk2604.torch2100.2610090934.g328354",
    "flash-attn": "2.8.4+dtk2610.torch2130.2610211904.g0f1a5a",
}
manifest = json.loads((root / "inputs/manifest.json").read_text())
summary = {
    "scope": manifest["scope"],
    "workbook_sha256": manifest["workbook_sha256"],
    "cards": {},
}
source_hashes = None
input_identities = None


def read_trace(run_dir, sample):
    """Accept a raw trace or its lossless projection onto all GPU kernels."""
    trace_path = run_dir / sample["trace"]
    if trace_path.exists():
        assert (
            hashlib.sha256(trace_path.read_bytes()).hexdigest()
            == sample["trace_sha256"]
        )
    else:
        index = json.loads((run_dir / "trace-index.json").read_text())
        projection = index[sample["trace"]]
        assert projection["raw_sha256"] == sample["trace_sha256"]
        assert Path(projection["file"]).name == projection["file"]
        trace_path = run_dir / projection["file"]
        assert (
            hashlib.sha256(trace_path.read_bytes()).hexdigest() == projection["sha256"]
        )
    with gzip.open(trace_path, "rt") as file:
        return json.load(file)


for gpu in (3, 4):
    run_dir = root / f"benchmark36-final-gpu{gpu}"
    receipt = json.loads((run_dir / "benchmark.json").read_text())
    assert receipt["physical_card"] == str(gpu) and receipt["node"] == "a01r4n15"
    assert receipt["packages"] == versions, receipt["packages"]
    assert receipt["baseline_libraries_sha256"] == expected_native
    assert receipt["sampling"] == {
        "arms": ["wheel", "aten", "repo"],
        "calls": 1000,
        "rounds": 5,
        "warmup": 100,
    }
    assert receipt["compiler_environment"] == {"AMDGCN_USE_BUFFER_OPS": "0"}
    assert (
        receipt["workbook_sha256"]
        == summary["workbook_sha256"]
        == "d2596052c66d7145af47b35fda45198d1f213b69825aeb91d10112dfd8871597"
    )
    assert receipt["tolerances"] == {
        "output_atol": 0.002,
        "output_rtol": 0.02,
        "lse_atol": 2e-5,
        "lse_rtol": 2e-5,
    }
    assert (
        set(receipt["cases"]) == set(manifest["cases"]) and len(receipt["cases"]) == 4
    )
    if source_hashes is None:
        source_hashes = receipt["repo_source_sha256"]
        for name, sha in source_hashes.items():
            assert (
                hashlib.sha256(
                    (repo / "src/flag_gems/runtime/backend/_hygon" / name).read_bytes()
                ).hexdigest()
                == sha
            ), name
        input_identities = {
            key: c["input_identity"] for key, c in receipt["cases"].items()
        }
    assert receipt["repo_source_sha256"] == source_hashes
    cards = summary["cards"][str(gpu)] = {}
    for case_id, case in receipt["cases"].items():
        assert (
            case["input_identity"]
            == input_identities[case_id]
            == manifest["cases"][case_id]["identity"]
        )
        specs = manifest["cases"][case_id]["case"]["specification"][0]
        assert case["input_layouts"] == [
            {"shape": x["shape"], "stride": x["stride"], "offset": x["storage_offset"]}
            for x in specs
        ]
        shape = specs[0]["shape"]
        for arm in ("wheel", "aten", "repo"):
            gate = case["precision"][arm]
            assert gate["stable"] is True and len(gate["checks"]) == 3
            for check in gate["checks"]:
                for kind, expected_shape in (("output", shape), ("lse", shape[:3])):
                    stat = check[kind]
                    assert stat["shape"] == expected_shape and stat[
                        "elements"
                    ] == math.prod(expected_shape)
                    assert stat["failed"] == 0 and stat["nonfinite"] == 0
            samples = case["samples"][arm]
            assert len(samples) == 5
            for sample in samples:
                assert Path(sample["trace"]).name == sample["trace"]
                trace = read_trace(run_dir, sample)
                events = [
                    e
                    for e in trace["traceEvents"]
                    if e.get("cat") == "kernel" and e.get("ph") == "X"
                ]
                assert all(event["dur"] > 0 for event in events)
                count = 3 if arm == "repo" and case_id == "model_state" else 1
                assert len(events) == count * 1000 == sample["kernel_count"]
                actual_us = sum(e["dur"] for e in events) / 1000
                assert math.isclose(actual_us, sample["us_per_call"], rel_tol=1e-12)
        medians = {
            arm: statistics.median(s["us_per_call"] for s in samples)
            for arm, samples in case["samples"].items()
        }
        means = {
            arm: statistics.mean(s["us_per_call"] for s in samples)
            for arm, samples in case["samples"].items()
        }
        ratios = [
            min(
                case["samples"]["wheel"][i]["us_per_call"],
                case["samples"]["aten"][i]["us_per_call"],
            )
            / case["samples"]["repo"][i]["us_per_call"]
            for i in range(5)
        ]
        median_ratio = min(medians["wheel"], medians["aten"]) / medians["repo"]
        mean_ratio = min(means["wheel"], means["aten"]) / means["repo"]
        assert min(ratios + [median_ratio, mean_ratio]) >= 0.9, (
            gpu,
            case_id,
            ratios,
            median_ratio,
            mean_ratio,
        )
        cards[case_id] = {
            "median_us": medians,
            "median_ratio": median_ratio,
            "mean_ratio": mean_ratio,
            "round_ratios": ratios,
        }
    regression_dir = root / f"regression36-gpu{gpu}"
    regression = json.loads((regression_dir / "regression.json").read_text())
    assert regression["physical_card"] == str(gpu)
    assert regression["packages"] == versions
    assert regression["sources"] == source_hashes
    assert regression["pytest_exit_code"] == 0
    assert regression["full_workbook_stability_complete"] is True
    assert "33 passed" in (regression_dir / "run.log").read_text()
    stability_path = regression_dir / "workbook-stability.json"
    assert (
        hashlib.sha256(stability_path.read_bytes()).hexdigest()
        == regression["stability_sha256"]
    )
    stability = json.loads(stability_path.read_text())
    assert set(stability["cases"]) == set(manifest["cases"])
    for case_id, fixtures in stability["cases"].items():
        assert len(fixtures) == (16 if case_id == "expanded_large" else 6)
        for fixture in fixtures:
            assert fixture["bitwise_stable"] is True
            assert len(fixture["checks"]) == 3
            for check in fixture["checks"]:
                for stat in check.values():
                    assert stat["failed"] == 0 and stat["nonfinite"] == 0
summary["source_sha256"] = source_hashes
summary["scope_excludes_original_model_payload_and_external_frozen_gate"] = True
summary["per_card_regressions"] = {
    "pytest": 33,
    "full_stability_fixtures": 34,
    "repeats": 3,
}
summary["complete"] = True
assert json.loads((root.parent / "device-acceptance.json").read_text()) == summary
inventory_path = root.parent / "inventory.json"
if inventory_path.exists():
    inventory = json.loads(inventory_path.read_text())
    for name, expected in inventory["files"].items():
        path = root.parent / name
        assert path.resolve().is_relative_to(root.parent)
        payload = path.read_bytes()
        assert len(payload) == expected["bytes"]
        assert hashlib.sha256(payload).hexdigest() == expected["sha256"], name
print(json.dumps(summary, indent=2))
