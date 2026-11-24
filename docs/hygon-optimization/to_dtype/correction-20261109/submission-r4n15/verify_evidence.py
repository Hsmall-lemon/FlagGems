# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Independently validate compact device evidence without torch or a GPU."""

import argparse
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-submission-pass", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        payload = (root / item["file"]).read_bytes()
        if item.get("normalize_lf"):
            payload = payload.replace(b"\r\n", b"\n")
        assert hashlib.sha256(payload).hexdigest() == item["sha256"], item["file"]
    with gzip.open(
        root / "device-observations.json.gz", "rt", encoding="utf-8"
    ) as source:
        observations = json.load(source)
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    samples = 0
    for dataset in observations.values():
        for group in dataset["results"]:
            rows = group["samples"]
            assert len(rows) == 15
            assert {(r["aa"], r["roi"]) for r in rows} == {
                (a, b) for a in range(3) for b in range(5)
            }
            assert all(
                math.isclose(
                    r["device_us"],
                    r["kernel_us"] + r["copy_us"],
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                for r in rows
            )
            medians = [
                statistics.median(r["device_us"] for r in rows if r["aa"] == a)
                for a in range(3)
            ]
            drift = 0 if max(medians) == 0 else max(medians) / min(medians) - 1
            correct = all(
                r.get("accuracy", {}).get("ok", True)
                and all(
                    r.get("correctness", {}).get(k, True)
                    for k in [
                        "value_ok",
                        "dtype_ok",
                        "stride_ok",
                        "device_ok",
                        "alias_ok",
                    ]
                )
                for r in rows
            )
            assert math.isclose(
                statistics.median(medians),
                group["device_us"],
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            assert medians == group["aa_medians_us"]
            assert math.isclose(
                drift, group["aa_max_drift"], rel_tol=1e-12, abs_tol=1e-12
            )
            assert (
                group["stable"] == (drift <= 0.05) and group["correctness"] == correct
            )
            samples += len(rows)

    def same(a, b):
        return math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-11)

    for row in status["rows"]:
        if row["status"] == "original_review_failure_unmatched":
            assert row["case"] in [2, 4] and not status["submission_allowed"]
            assert row["generated_input_performance_status"] == "passes"
        if row["status"] == "no_device_work":
            assert (
                row["das_us"] == row["new_us"] == 0
                and not row["new_kernels"]
                and not row["new_copies"]
            )
        else:
            assert row["das_correct"] and row["new_correct"]
            assert row["das_aa_drift"] <= 0.05 and row["new_aa_drift"] <= 0.05
            assert same(row["das_us"] / row["new_us"], row["das_over_new"])
            assert row["das_over_new"] >= 0.9
        matches = []
        for name, dataset in observations.items():
            groups = {
                r["provider"]: r for r in dataset["results"] if r["case"] == row["case"]
            }
            if not groups:
                continue
            dp = "das_native_hip_public" if status["operator"] == "to_dtype" else "das"
            for np in ["new_public", "new_export"]:
                if (
                    dp in groups
                    and np in groups
                    and same(groups[dp]["device_us"], row["das_us"])
                    and same(groups[np]["device_us"], row["new_us"])
                ):
                    matches.append(name)
        assert matches, (row["case"], "no matching raw paired observations")
    blocked = [
        r for r in status["rows"] if r["status"] not in ["passes", "no_device_work"]
    ]
    assert status["operator_submission_allowed"] == (
        not blocked and status.get("scope_reconciliation_complete", True)
    )
    print(
        "Evidence verified:",
        status["operator"],
        len(status["rows"]),
        "records;",
        samples,
        "raw ROI samples.",
    )
    print("Overall submission allowed:", status["submission_allowed"])
    if args.require_submission_pass and not status["submission_allowed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
