# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Audit current dtype scope, launch preservation and every measured ROI on CPU."""

import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path

root = Path(__file__).resolve().parent
repo = root.parents[3]
data = json.loads(gzip.decompress((root / "device-observations.json.gz").read_bytes()))


def canonical(path):
    return path.read_bytes().replace(b"\r\n", b"\n")


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def verify_provider(record):
    samples = record["samples"]
    assert len(samples) == 15
    assert {(s["aa"], s["roi"]) for s in samples} == {
        (a, i) for a in range(3) for i in range(5)
    }
    medians = [
        statistics.median(s["device_us"] for s in samples if s["aa"] == a)
        for a in range(3)
    ]
    drift = max(medians) / min(medians) - 1 if max(medians) else 0
    assert medians == record["aa_medians_us"]
    assert drift == record["aa_max_drift"]
    assert statistics.median(medians) == record["device_us"]
    assert record["stable"] == (drift <= 0.05)
    for sample in samples:
        assert (
            abs(sample["device_us"] - sample["kernel_us"] - sample["copy_us"]) < 1e-10
        )
        if "same_queue_as_gate" in sample:
            assert sample["same_queue_as_gate"]


for name, sha in data["canonical_source_sha256"].items():
    assert digest(canonical(repo / name)) == sha, name

historical_path = root.parent / "extended-20261110/device-observations.json.gz"
assert digest(historical_path.read_bytes()) == data["historical_archive_sha256"]
historical = json.loads(gzip.decompress(historical_path.read_bytes()))
assert historical["scope"]["total"] == len(historical["rows"]) == 59
assert data["scope"]["required_rows"] == 46
assert data["scope"]["dtype_rows"] == 10
assert data["scope"]["copy_rows"] == 36
assert data["scope"]["all_exact_metadata_matches"]
assert len(data["scope"]["rows"]) == 46
assert all(row["matched_cases"] for row in data["scope"]["rows"])

prefix = "src/flag_gems/runtime/backend/_hygon/ops/"
old = canonical(root / "baseline-v1-to.py.txt").decode()
assert digest(old.encode()) == historical["source_sha256"][prefix + "to.py"]
current = canonical(repo / prefix / "to.py").decode()
restored = current.replace(
    "block, warps = (512, 1) if n < 524288 else (2048, 2)",
    "block, warps = 2048, 2",
).replace(
    "and (n >= _PACKED_MIN_ELEMENTS or 32768 <= n <= 262144)",
    "and n >= _PACKED_MIN_ELEMENTS",
)
restored = restored.replace(
    "block, warps = (1024, 1) if n < _PACKED_MIN_ELEMENTS else (2048, 2)\n"
    "                if (\n"
    "                    n >= _PACKED_MIN_ELEMENTS\n"
    "                    and n % 4 == 0\n"
    "                    and x.storage_offset() % 4 == 0\n"
    "                ):",
    "block, warps = 2048, 2\n"
    "                if n % 4 == 0 and x.storage_offset() % 4 == 0:",
)
assert restored == old
for name, sha in historical["source_sha256"].items():
    if name != prefix + "to.py":
        assert digest(canonical(repo / name)) == sha
assert len(data["route_preservation"]["unchanged_controls"]) == 13
assert all(c["numel"] < 32768 for c in data["route_preservation"]["unchanged_controls"])
affected = {1711, 1713, 1718, 1719, 940018, 940034}
assert set(data["route_preservation"]["affected_workbook_cases"]) == affected
workbook_cases = json.loads((root.parent / "cases.json").read_text(encoding="utf-8"))
workbook_cases += json.loads(
    (root.parent / "extended-cases-20261110.json").read_text(encoding="utf-8")
)
recomputed = set()
for case in workbook_cases:
    if "row" in case:
        meta, target = case["metadata"][0]
        case_id = case["row"]
    else:
        meta = case["metadata"][0][0]
        target = case["metadata"][1]["dtype"]
        case_id = case["case"]
    n = math.prod(meta["shape"])
    medium_bf16 = (
        meta["dtype"] == "torch.bfloat16"
        and target == "torch.float32"
        and 262144 < n < 524288
    )
    medium_fp32 = (
        meta["dtype"] == "torch.float32"
        and target == "torch.bfloat16"
        and 32768 <= n <= 262144
        and n % 2 == 0
        and meta.get("storage_offset", 0) % 2 == 0
    )
    if medium_bf16 or medium_fp32:
        recomputed.add(case_id)
assert recomputed == affected

for capture in historical["captures"].values():
    keep = capture["preservation"]
    assert keep["completed"] and keep["sqlite_equals_official_json"]
    for record in capture["providers"].values():
        verify_provider(record)

for name, capture in data["captures"].items():
    keep = capture["preservation"]
    assert keep["completed"] and keep["sqlite_equals_official_json"]
    fp = capture["fingerprint"]
    assert fp["visible_device_count"] == 1
    assert (fp["node"], fp["pci_bus"].lower()) in {
        ("a01r3n13", "0000:77:00.0"),
        ("a14r3n03", "0000:09:00.0"),
        ("a14r3n03", "0000:36:00.0"),
    }
    source = data["source_manifests"]["source-v2" if "-v2-" in name else "source-v1"]
    for path, sha in fp["source_files"].items():
        if path in source["files"]:
            assert source["files"][path] == sha
    official = {(r["case"], r["provider"]): r for r in capture["official"]["results"]}
    native = {(r["case"], r["provider"]): r for r in capture["native"]["results"]}
    assert official.keys() == native.keys()
    for key, record in official.items():
        other = native[key]
        verify_provider(record)
        verify_provider(other)
        assert other["same_queue_as_gate"]
        for field in [
            "device_us",
            "aa_medians_us",
            "aa_max_drift",
            "stable",
            "correctness",
        ]:
            assert record[field] == other[field]
        for a, b in zip(record["samples"], other["samples"]):
            for field in [
                "device_us",
                "kernel_us",
                "copy_us",
                "kernels_per_call",
                "copies_per_call",
            ]:
                assert a[field] == b[field]
            assert a["input_pointer_shared"] and b["same_queue_as_gate"]
            if (
                record["provider"] in ["new_public", "routing_only", "published_public"]
                and not a["noop"]
            ):
                assert a["primary_route_calls"] > 0

for row in data["candidate_v2"]["rows"]:
    assert row["accepted"] and row["das_over_new"] >= 0.9
assert len(data["candidate_v2"]["rows"]) == 18
assert len(data["baseline_v1"]["rows"]) == 30
assert all(r["device_target_pass"] for r in data["baseline_v1"]["rows"])
assert data["candidate_v2"]["numerical_suites"] == {
    "n13": 106,
    "n03-gpu0": 106,
    "n03-gpu1": 106,
}

passed = no_work = 0
for row in data["rows"]:
    assert row["accepted"]
    if row["case"] in affected:
        assert row["source"] == "source-v2"
        capture = data["captures"][row["capture"]]
        records = {
            r["provider"]: r
            for r in capture["official"]["results"]
            if r["case"] == row["case"]
        }
        new = records["new_public"]
        native = records["das_native_hip_public"]
    else:
        assert row["source"] == "source-v1-preserved-route"
        old_row = next(r for r in historical["rows"] if r["case"] == row["case"])
        assert row["capture"] == old_row["latest_capture"]
        records = historical["captures"][row["capture"]]["providers"]
        new = records[f"{row['case']}:new_public"]
        native = records.get(
            f"{row['case']}:das", records.get(f"{row['case']}:das_native_hip_public")
        )
    assert native is not None
    assert (
        new["correctness"]
        and native["correctness"]
        and new["stable"]
        and native["stable"]
    )
    if row["das_over_new"] is None:
        assert new["device_us"] == native["device_us"] == 0
        no_work += 1
    else:
        assert row["das_over_new"] == native["device_us"] / new["device_us"]
        assert row["das_over_new"] >= 0.9
        passed += 1
assert (passed, no_work) == (56, 3)
assert len(data["rows"]) == 59
assert data["original_model_operands_replayed"] is False
assert data["uniform_speedup_over_old_fg_claimed"] is False
print(
    "Verified all 59 dtype/copy inputs, 18 affected-input captures and 318 candidate numerical tests."
)
