# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Recompute dtype evidence and source-route preservation using the CPU only."""

import ast
import gzip
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
repo = root.parents[3]
data = json.loads(gzip.decompress((root / "device-observations.json.gz").read_bytes()))
assert data["scope"]["total"] == len(data["rows"]) == 59


def canonical_bytes(path):
    return path.read_bytes().replace(b"\r\n", b"\n")


to_path = "src/flag_gems/runtime/backend/_hygon/ops/to.py"
if (
    hashlib.sha256(canonical_bytes(repo / to_path)).hexdigest()
    != data["source_sha256"][to_path]
):
    # The later verifier retains this complete archive and proves unchanged
    # routes, while independently accepting every affected input again.
    latest = root.parent / "recheck-20261111/verify_evidence.py"
    assert latest.exists(), "Current source requires its matching evidence"
    subprocess.run([sys.executable, str(latest)], check=True)
    print("Historical 2026-11-10 evidence checked with the current launch overlay.")
    raise SystemExit(0)


for name, sha in data["source_sha256"].items():
    assert hashlib.sha256(canonical_bytes(repo / name)).hexdigest() == sha, name

proof = json.loads((root / "source-route-proof.json").read_text(encoding="utf-8"))
prefix = "src/flag_gems/runtime/backend/_hygon/ops/"
for name in ["to.py", "to_kernels.py"]:
    baseline = root / "baseline" / (name + ".txt")
    assert (
        hashlib.sha256(canonical_bytes(baseline)).hexdigest()
        == proof["canonical_baseline_sha256"][name]
    )
    old = ast.parse(baseline.read_text(encoding="utf-8"))
    new = ast.parse((repo / prefix / name).read_text(encoding="utf-8"))
    if name == "to.py":
        imports = next(
            n
            for n in new.body
            if isinstance(n, ast.ImportFrom) and n.module == "to_kernels"
        )
        imports.names = [n for n in imports.names if n.name != "cast_fp32_bf16_quad"]
        branch = next(
            n
            for n in ast.walk(new)
            if isinstance(n, ast.If)
            and ast.unparse(n.test) == "n % 4 == 0 and x.storage_offset() % 4 == 0"
        )
        parent = next(
            n
            for n in ast.walk(new)
            if isinstance(getattr(n, "body", None), list) and branch in n.body
        )
        parent.body.remove(branch)
    else:
        new.body = [
            n
            for n in new.body
            if not (isinstance(n, ast.FunctionDef) and n.name == "cast_fp32_bf16_quad")
        ]
    assert ast.dump(old, include_attributes=False) == ast.dump(
        new, include_attributes=False
    )

for capture in data["captures"].values():
    assert (
        capture["preservation"]["completed"]
        and capture["preservation"]["sqlite_equals_official_json"]
    )
    for record in capture["providers"].values():
        samples = record["samples"]
        assert len(samples) == 15
        assert len({(s["aa"], s["roi"]) for s in samples}) == 15
        for sample in samples:
            assert sample["same_queue_as_gate"]
            assert math.isclose(
                sample["device_us"],
                sample["kernel_us"] + sample["copy_us"],
                abs_tol=1e-10,
            )
        medians = [
            statistics.median(s["device_us"] for s in samples if s["aa"] == a)
            for a in range(3)
        ]
        drift = max(medians) / min(medians) - 1 if max(medians) else 0
        assert medians == record["aa_medians_us"]
        assert drift == record["aa_max_drift"]
        assert statistics.median(medians) == record["device_us"]
        assert record["stable"] == (drift <= 0.05)

passed = no_work = 0
for row in data["rows"]:
    capture = data["captures"][row["latest_capture"]]
    providers = capture["providers"]
    new = providers[f"{row['case']}:new_public"]
    native = providers.get(
        f"{row['case']}:das", providers.get(f"{row['case']}:das_native_hip_public")
    )
    assert native is not None
    assert (
        native["correctness"]
        and new["correctness"]
        and native["stable"]
        and new["stable"]
    )
    if new["device_us"] == native["device_us"] == 0:
        assert row["status"] == "no_device_work" and row["das_over_new"] is None
        no_work += 1
    else:
        assert native["device_us"] / new["device_us"] == row["das_over_new"] >= 0.9
        passed += 1
assert (passed, no_work) == (56, 3)
assert (
    data["submission_allowed"] is False
    and data["original_model_operands_replayed"] is False
)
print(
    f"Verified {passed} device-work inputs and {no_work} no-work inputs; all historical failures retained."
)
