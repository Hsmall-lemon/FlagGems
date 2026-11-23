"""Summarize a complete rerun of the frozen FP32 mm acceptance suite."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    args = parser.parse_args()
    manifest = Path(__file__).with_name("fp32-acceptance-frozen.csv")
    with manifest.open(encoding="utf-8-sig", newline="") as file:
        frozen = {int(row["case_id"]): row for row in csv.DictReader(file)}
    measurements = {}
    rebuilt = set()
    for batch in sorted(args.capture.glob("gpu*-batch*")):
        summary = json.loads((batch / "full-summary.json").read_text())
        validations = json.loads((batch / "full-validation.json").read_text())
        for line in (batch / "run.log").read_text(errors="replace").splitlines():
            if line.startswith("REBUILT_INPUT_DIAGNOSTIC "):
                rebuilt.add(int(line.split()[1]))
        for result in summary["results"]:
            row, variant = result["row"], result["variant"]
            key = row, variant
            assert key not in measurements and not result.get("not_for_acceptance"), key
            checks = [v for v in validations if (v["row"], v["variant"]) == key]
            case = frozen[row]
            assert sorted(c["seed"] for c in checks) == list(
                range(20261109, 20261514, 101)
            ), key
            assert all(
                c["atol"] == float(case["atol"])
                and c["rtol"] == float(case["rtol"])
                and c["primary_reference"] == "cpu_fp32"
                for c in checks
            ), key
            measurements[key] = (sum(c["correct"] for c in checks), result["device_us"])
    assert set(measurements) == {
        (row, variant) for row in frozen for variant in ("das", "gems")
    }

    rows = []
    for row, case in frozen.items():
        fg_passes, fg_time = measurements[row, "gems"]
        das_passes, das_time = measurements[row, "das"]
        ratio = das_time / fg_time
        performance = ratio >= 0.90
        rows.append(
            {
                **case,
                "rerun_fg_passes": fg_passes,
                "rerun_das_passes": das_passes,
                "rerun_fg_device_us": fg_time,
                "rerun_das_device_us": das_time,
                "rerun_das_over_fg": ratio,
                "rerun_fg_joint_pass": fg_passes == 5 and performance,
                "fg_precision_changed": bool(case["fg_passes"])
                and fg_passes != int(case["fg_passes"]),
                "das_precision_changed": bool(case["das_passes"])
                and das_passes != int(case["das_passes"]),
                "performance_gate_changed": performance
                != (float(case["das_over_fg"]) >= 0.90),
                "table_backing_mismatch": row in rebuilt,
            }
        )
    with (args.capture / "fp32-acceptance-rerun.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter()
    for row in rows:
        atol = row["atol"]
        counts[atol, "total"] += 1
        counts[atol, "fg_precision"] += row["rerun_fg_passes"] == 5
        counts[atol, "das_precision"] += row["rerun_das_passes"] == 5
        counts[atol, "performance"] += row["rerun_das_over_fg"] >= 0.90
        counts[atol, "fg_joint"] += row["rerun_fg_joint_pass"]
    print(
        json.dumps(
            {
                atol: {
                    metric: count
                    for (group, metric), count in counts.items()
                    if group == atol
                }
                for atol in dict.fromkeys(row["atol"] for row in rows)
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
