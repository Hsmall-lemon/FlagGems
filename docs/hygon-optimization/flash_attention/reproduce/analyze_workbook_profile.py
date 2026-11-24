"""Accept only complete, currently qualified worksheet device traces."""

import argparse
import collections
import gzip
import hashlib
import json
import statistics
from pathlib import Path

from workbook_gate import require_case
from workbook_inputs import MANIFEST


def analyze(receipt, trace):
    sampling = receipt.get("sampling", {})
    if sampling != {"calls": 1000, "rounds": 5, "warmup": 100}:
        raise ValueError(
            "Acceptance requires the worksheet's 100 warmups and 5 x 1000 calls"
        )
    cases = MANIFEST["cases"]
    if set(receipt.get("cases", {})) != {c["id"] for c in cases}:
        raise ValueError("Missing worksheet cases")
    for case in cases:
        require_case(receipt, case)
        record = receipt["cases"][case["id"]]
        if (
            record.get("timing_started") is not True
            or record.get("timing_completed") is not True
        ):
            raise ValueError("Incomplete gated timing")
    events = trace["traceEvents"]
    device = [e for e in events if e.get("cat") == "HIPOPS" and e.get("ph") == "X"]
    rois = [
        e
        for e in events
        if e.get("cat") == "HIPTX" and e.get("name", "").startswith("workbook:")
    ]
    expected = {
        f"workbook:{receipt['run_id']}:{case['id']}:{arm}:round={i}"
        for case in cases
        for arm in ("wheel", "aten", "repo")
        for i in range(5)
    }
    if len(rois) != len(expected) or {r["name"] for r in rois} != expected:
        raise ValueError("Missing, duplicate, historical or unrelated timing ranges")
    results = {}
    for case in cases:
        arms = {}
        for arm in ("wheel", "aten", "repo"):
            samples = []
            for i in range(5):
                label = f"workbook:{receipt['run_id']}:{case['id']}:{arm}:round={i}"
                roi = next(r for r in rois if r["name"] == label)
                kernels = [
                    e
                    for e in device
                    if roi["ts"] <= e["ts"]
                    and e["ts"] + e["dur"] <= roi["ts"] + roi["dur"]
                ]
                per_call = {"model_state": 3}.get(case["id"], 1)
                count = 1000 * (per_call if arm == "repo" else 1)
                if len(kernels) != count or any(e["dur"] <= 0 for e in kernels):
                    raise ValueError(
                        f"Incomplete device work for {label}: {len(kernels)} vs {count}"
                    )
                samples.append(
                    {
                        "round": i,
                        "calls": 1000,
                        "us": sum(e["dur"] for e in kernels) / 1000,
                        "kernels": dict(
                            collections.Counter(e["name"] for e in kernels)
                        ),
                    }
                )
            arms[arm] = samples
        timing = {
            arm: statistics.mean(r["us"] for r in samples)
            for arm, samples in arms.items()
        }
        ratios = [
            min(arms["wheel"][i]["us"], arms["aten"][i]["us"]) / arms["repo"][i]["us"]
            for i in range(5)
        ]
        ratio_of_means = min(timing["wheel"], timing["aten"]) / timing["repo"]
        results[case["id"]] = {
            "sheet": case["sheet"],
            "row": case["row"],
            "timing_us": timing,
            "per_round_native_over_repo": ratios,
            "mean_native_over_repo": statistics.mean(ratios),
            "ratio_of_means": ratio_of_means,
            "meets_90": statistics.mean(ratios) >= 0.9 and ratio_of_means >= 0.9,
            "arms": arms,
        }
    return {
        "schema": 5,
        "run_id": receipt["run_id"],
        "device": receipt["device"],
        "workbook_sha256": receipt["workbook_sha256"],
        "source_sha256": receipt["source_sha256"],
        "native": receipt["native"],
        "method": (
            "hipprof HIPOPS duration sum per call, including every helper kernel; "
            "5 rounds x 1000 calls after 100 warmups per arm/round; "
            "faster qualified native denominator"
        ),
        "scope": "Four deterministic worksheet metadata reconstructions, not original model payload replay",
        "external_frozen_gate_verified": False,
        "original_model_acceptance_verified": False,
        "all_reconstructed_cases_meet_90": all(r["meets_90"] for r in results.values()),
        "cases": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", type=Path)
    parser.add_argument("--gate", type=Path, default=Path("workbook-gate.json"))
    parser.add_argument("--output", type=Path, default=Path("workbook-summary.json"))
    args = parser.parse_args()
    data = args.profile.read_bytes()
    if args.profile.suffix == ".gz":
        data = gzip.decompress(data)
    result = analyze(json.loads(args.gate.read_text()), json.loads(data))
    result["gate_sha256"] = hashlib.sha256(args.gate.read_bytes()).hexdigest()
    result["trace_sha256"] = hashlib.sha256(args.profile.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))
    for name, record in result["cases"].items():
        print(
            name,
            record["timing_us"],
            record["mean_native_over_repo"],
            record["meets_90"],
        )
