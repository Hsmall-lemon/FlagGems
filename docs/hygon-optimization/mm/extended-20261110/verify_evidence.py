# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Recompute MM device evidence and route preservation without a GPU."""

import ast
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path

root = Path(__file__).resolve().parent
repo = root.parents[3]
data = json.loads(gzip.decompress((root / "device-observations.json.gz").read_bytes()))
assert data["scope"]["total"] == len(data["rows"]) == 197


def canonical_bytes(path):
    return path.read_bytes().replace(b"\r\n", b"\n")


for name, sha in data["source_sha256"].items():
    assert hashlib.sha256(canonical_bytes(repo / name)).hexdigest() == sha, name
proof = json.loads((root / "source-route-proof.json").read_text(encoding="utf-8"))
prefix = "src/flag_gems/runtime/backend/_hygon/ops/"
for name, sha in proof["canonical_baseline_sha256"].items():
    if name in ["mm.py", "mm_config.py"]:
        assert (
            hashlib.sha256(
                canonical_bytes(root / "baseline" / (name + ".txt"))
            ).hexdigest()
            == sha
        )
    else:
        assert (
            hashlib.sha256(canonical_bytes(repo / prefix / name)).hexdigest() == sha
        ), name
old = {}
new = {}
exec((root / "baseline/mm_config.py.txt").read_text(encoding="utf-8"), old)
exec((repo / prefix / "mm_config.py").read_text(encoding="utf-8"), new)
assert old["CONFIGS"].keys() == new["CONFIGS"].keys() and len(new["CONFIGS"]) == 175
changed = {k: v for k, v in new["CONFIGS"].items() if old["CONFIGS"][k] != v}
expected_changes = {
    ("torch.float32", 905, 1159, 1159, 1159, 1, 1159, 1): dict(
        kind="native_fp32", group=0
    )
}
if "v16_config_change" in proof:
    v16 = proof["v16_config_change"]
    assert v16["old_174_configurations_unchanged"]
    assert v16["dispatcher_byte_identical"] and v16["all_kernel_sources_byte_identical"]
    expected_changes[tuple(v16["changed_key"])] = v16["config"]
    for result in v16["measurements"]:
        assert result["das_over_new"] >= 0.9
        assert result["das_aa"] <= 0.05 and result["new_aa"] <= 0.05
assert changed == expected_changes


class RemoveNative(ast.NodeTransformer):
    def visit_If(self, node):
        if ast.dump(node.test) == ast.dump(
            ast.parse('kind == "native_fp32"', mode="eval").body
        ):
            assert len(node.orelse) == 1
            return self.visit(node.orelse[0])
        return self.generic_visit(node)


dispatcher = ast.parse((repo / prefix / "mm.py").read_text(encoding="utf-8"))
captured = proof["captured_dispatcher_utf8"]
assert ast.dump(dispatcher) == ast.dump(ast.parse(captured))
assert (
    hashlib.sha256(captured.encode("utf-8")).hexdigest()
    == proof["captured_files_sha256"]["mm.py"]
)
assert ast.dump(RemoveNative().visit(dispatcher)) == ast.dump(
    ast.parse((root / "baseline/mm.py.txt").read_text(encoding="utf-8"))
)
for capture in data["captures"].values():
    assert (
        capture["preservation"]["completed"]
        and capture["preservation"]["sqlite_equals_official_json"]
    )
    for record in capture["providers"].values():
        samples = record["samples"]
        assert len(samples) == 15 and len({(s["aa"], s["roi"]) for s in samples}) == 15
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
        assert medians == record["aa_medians_us"] and drift == record["aa_max_drift"]
        assert statistics.median(medians) == record["device_us"]
        assert record["stable"] == (drift <= 0.05)
for row in data["rows"]:
    capture = data["captures"][row["latest_capture"]]
    new = capture["providers"][f"{row['case']}:new_public"]
    native = capture["providers"][f"{row['case']}:das"]
    assert all(r["correctness"] and r["stable"] for r in [native, new])
    assert native["device_us"] / new["device_us"] == row["das_over_new"] >= 0.9
    assert row["status"] == "passes_reconstructed_input"
assert (
    data["submission_allowed"] is False
    and data["original_model_operands_replayed"] is False
)
print(
    f"Verified all {len(data['rows'])} MM workbook metadata rows; earlier failures retained."
)
