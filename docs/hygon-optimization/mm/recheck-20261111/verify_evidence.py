# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Verify final MM source, scope, device observations and preserved routes on CPU."""

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
docs = root.parent
repo = root.parents[3]
prefix = "src/flag_gems/runtime/backend/_hygon/ops/"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def compressed(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def canonical(path):
    return path.read_bytes().replace(b"\r\n", b"\n")


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (a, b)


data = compressed(root / "device-observations.json.gz")
status = read(root / "status.json")
assert status["candidate_acceptance_passed"] and status["pr_draft"]
assert not status["original_reviewer_operands_replayed"]
assert not status["original_report_root_cause_resolved"]
extended = compressed(docs / "extended-20261110/device-observations.json.gz")
historical = read(docs / "correction-20261109/submission-r4n15/status.json")
old_observations = compressed(
    docs / "correction-20261109/submission-r4n15/device-observations.json.gz"
)
for script in [
    docs / "extended-20261110/verify_evidence.py",
    docs / "correction-20261109/submission-r4n15/verify_evidence.py",
]:
    subprocess.run([sys.executable, str(script)], check=True)
assert len(data["scope"]["rows"]) == len(extended["rows"]) == 197
assert data["scope"]["rows"] == [
    {k: r[k] for k in ["sheet", "excel_row", "metadata"]} for r in extended["rows"]
]
assert data["original_reviewer_operands_replayed"] is False
assert data["root_cause_claimed"] is False
for path, expected in data["canonical_source_sha256"].items():
    assert sha(canonical(repo / path)) == expected, path

# The 119-key source qualified before the workbook extension. Preserve every
# old route except two K=1 launches independently qualified in the new scope.
old = {}
new = {}
exec((root / "baseline119/mm_config.py.txt").read_text(encoding="utf-8"), old)
exec((repo / prefix / "mm_config.py").read_text(encoding="utf-8"), new)
assert len(old["CONFIGS"]) == 119 and len(new["CONFIGS"]) == 175
assert not any(c["kind"] == "peel" for c in new["CONFIGS"].values())
assert not any(c.get("launch_options") for c in old["CONFIGS"].values())
changed = {k for k, v in old["CONFIGS"].items() if new["CONFIGS"].get(k) != v}
assert changed == {
    ("torch.float64", 24576, 48, 1, 1, 128, 48, 1),
    ("torch.float64", 24576, 16, 1, 1, 128, 16, 1),
}
old_selector = ast.parse(
    (root / "baseline119/mm_config.py.txt").read_text(encoding="utf-8")
)
new_selector = ast.parse((repo / prefix / "mm_config.py").read_text(encoding="utf-8"))
assert [ast.dump(n) for n in old_selector.body if isinstance(n, ast.FunctionDef)] == [
    ast.dump(n) for n in new_selector.body if isinstance(n, ast.FunctionDef)
]


class RemoveAdded(ast.NodeTransformer):
    def visit_If(self, node):
        additions = [
            "outer",
            "cache3",
            "fast_split",
            "swizzle_split",
            "bank_tail",
            "banked",
            "native_fp32",
        ]
        if ast.unparse(node.test) in [f"kind == '{kind}'" for kind in additions]:
            assert len(node.orelse) == 1
            return self.visit(node.orelse[0])
        return self.generic_visit(node)

    def visit_Call(self, node):
        node = self.generic_visit(node)
        node.keywords = [
            k
            for k in node.keywords
            if not (
                k.arg is None
                and ast.unparse(k.value) == "cfg.get('launch_options', {})"
            )
        ]
        return node


before = ast.parse((root / "baseline119/mm.py.txt").read_text(encoding="utf-8"))
after = ast.parse((repo / prefix / "mm.py").read_text(encoding="utf-8"))
assert ast.dump(before) == ast.dump(RemoveAdded().visit(after))
proof = read(root / "historical-source-proof.json")
previous_kernels = ["mm_kernels.py", "mm_w8a8_int8.py"] + [
    p.name for p in (repo / prefix).glob("mm_mmac_*.py")
]
for filename in previous_kernels:
    # Historical hashes recorded the Windows CRLF bytes; Git stores LF bytes.
    assert sha(canonical(repo / prefix / filename).replace(b"\n", b"\r\n")) == (
        proof["current_files_sha256"][filename]
    ), filename
for filename in ["mm.py", "mm_config.py"]:
    assert sha(
        canonical(root / "baseline119" / (filename + ".txt")).replace(b"\n", b"\r\n")
    ) == (proof["baseline_files_sha256"][filename])


def key(metadata):
    a, b = metadata[0]
    return (
        a["dtype"],
        a["shape"][0],
        b["shape"][1],
        a["shape"][1],
        *a["stride"],
        *b["stride"],
    )


extended_keys = {key(r["metadata"]) for r in extended["rows"]}
assert changed <= extended_keys
for row in historical["rows"]:
    if "metadata" in row:
        launch = key(row["metadata"])
    elif "added_input_metadata" in row:
        meta = row["added_input_metadata"]
        launch = (
            meta["dtype"],
            meta["a_shape"][0],
            meta["b_shape"][1],
            meta["a_shape"][1],
            *meta["a_stride"],
            *meta["b_stride"],
        )
    else:
        sample = next(
            g["samples"][0]
            for d in old_observations.values()
            for g in d["results"]
            if g["case"] == row["case"] and g["provider"] == "new_public"
        )
        launch = (
            sample["dtype"],
            sample["a_shape"][0],
            sample["b_shape"][1],
            sample["a_shape"][1],
            *sample["a_stride"],
            *sample["b_stride"],
        )
    if launch not in changed:
        assert new["CONFIGS"].get(launch) == old["CONFIGS"].get(launch), row["case"]

# Recompute every final ROI from all native kernel/copy DurationNs, including
# irregular allocator initialization copies. No outliers or A/A blocks dropped.
roi_count = 0
for name, capture in data["captures"].items():
    assert capture["run_status"] == {"exit_code": 0, "limit_hit": False}
    assert capture["preservation"]["completed"]
    assert capture["preservation"]["sqlite_equals_official_json"]
    assert capture["fingerprint"]["visible_devices"] == 1
    kernel_hashes = {
        p: h for p, h in data["canonical_source_sha256"].items() if p.startswith(prefix)
    }
    assert capture["fingerprint"]["source_files"] == kernel_hashes
    assert len(capture["stress"]) == 16
    assert all(r["accuracy"]["ok"] and r["route_witness"] for r in capture["stress"])
    regimes = {}
    for stress in capture["stress"]:
        regimes[stress["regime"]] = regimes.get(stress["regime"], 0) + 1
        expected = (
            (1e-10, 1e-10)
            if "fp64" in name
            else (2e-4, 1e-4) if "fp32" in name else (1e-5, 0.016)
        )
        assert (stress["accuracy"]["atol"], stress["accuracy"]["rtol"]) == expected
    assert regimes == {
        "random": 8,
        "basis": 2,
        "low_bits": 2,
        "dyadic_scaled": 2,
        "cancellation": 2,
    }
    if "fp64" in name:
        hashes = {}
        for stress in capture["stress"]:
            pair = (stress["seed"], stress["regime"])
            if pair in hashes:
                assert hashes[pair] == stress["input_hash"]
            hashes[pair] = stress["input_hash"]
    native = {r["message"]: r for r in capture["native_rois"]}
    for group in capture["summary"]["results"]:
        samples = group["samples"]
        assert {(s["aa"], s["roi"]) for s in samples} == {
            (a, b) for a in range(3) for b in range(5)
        }
        for sample in samples:
            message = (
                f"correction:case={group['case']}:provider={group['provider']}:"
                f"aa={sample['aa']}:roi={sample['roi']}"
            )
            observed = native.pop(message)
            calls = sample["calls"]
            assert observed["gate_duration_ns"] > 10000000 and observed["same_queue"]
            assert sample["same_queue_as_gate"] and sample["accuracy"]["ok"]
            close(
                sum(observed["kernel_duration_ns"]) / calls / 1000, sample["kernel_us"]
            )
            close(sum(observed["copy_duration_ns"]) / calls / 1000, sample["copy_us"])
            close(sample["kernel_us"] + sample["copy_us"], sample["device_us"])
            close(
                len(observed["kernel_duration_ns"]) / calls, sample["kernels_per_call"]
            )
            close(len(observed["copy_duration_ns"]) / calls, sample["copies_per_call"])
            assert sample["source_sha256"] == kernel_hashes
            if group["provider"] == "new_public":
                assert sample["primary_route_calls"] == 1
                launch = (
                    sample["dtype"],
                    sample["a_shape"][0],
                    sample["b_shape"][1],
                    sample["a_shape"][1],
                    *sample["a_stride"],
                    *sample["b_stride"],
                )
                assert sample["config"] == new["CONFIGS"][launch]
            roi_count += 1
        medians = [
            statistics.median(s["device_us"] for s in samples if s["aa"] == a)
            for a in range(3)
        ]
        assert medians == group["aa_medians_us"]
        close(statistics.median(medians), group["device_us"])
        drift = max(medians) / min(medians) - 1
        close(drift, group["aa_max_drift"])
        assert group["stable"] == (drift <= 0.05)
    assert not native
    if name in data["selected_captures"]:
        groups = capture["summary"]["results"]
        for case in {g["case"] for g in groups}:
            paired = {g["provider"]: g for g in groups if g["case"] == case}
            assert all(
                paired[p]["stable"] and paired[p]["correctness"]
                for p in ["das", "new_public"]
            )
            ratio = paired["das"]["device_us"] / paired["new_public"]["device_us"]
            assert ratio >= 0.9
            print(case, "DAS/new =", round(100 * ratio, 4), "%")
assert roi_count == 225
assert (
    status["minimum_workbook_das_over_new"] == extended["minimum_das_over_new"] >= 0.9
)
assert status["workbook_rows"] == 197 and status["historical_records_preserved"] == 179
assert status["pytest_passed"] == 410 and status["distinct_fresh_stress_passed"] == 48
assert (
    data["selected_captures"]
    == status["selected_captures"]
    == ["accept-fp64-r2", "accept-fp32-r2", "accept-bf16-r1"]
)

for name, capture in data["accuracy"].items():
    assert capture["exit"]["exit_code"] == 0
    assert ("97 passed" if name == "mm-accuracy-n13" else "313 passed") in capture[
        "log"
    ]
    for path, expected in capture["fingerprint"]["source_files"].items():
        file = repo / path
        if not file.exists():
            file = root / "harness" / (Path(path).name + ".txt")
        payload = canonical(file)
        assert expected in [sha(payload), sha(payload.replace(b"\n", b"\r\n"))], path
captured = ast.parse(
    (root / "harness/test_extended_mm_native_fp32_20261110.py.txt").read_text(
        encoding="utf-8"
    )
)
current = ast.parse(
    (repo / "tests/test_hygon_native_fp32.py").read_text(encoding="utf-8")
)
for tree in [captured, current]:
    tree.body.sort(
        key=lambda n: (
            (0, ast.dump(n)) if isinstance(n, (ast.Import, ast.ImportFrom)) else (1, "")
        )
    )
assert ast.dump(captured) == ast.dump(current)
print(
    "Verified: 197 workbook rows, 179 preserved records, 410 numerical tests, 225 fresh ROIs."
)
print(
    "Candidate acceptance passes for reconstructed metadata; original operands were not replayed."
)
