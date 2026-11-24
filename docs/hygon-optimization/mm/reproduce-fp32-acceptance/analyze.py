"""Sum every device kernel/copy inside our explicit ROIs from native hipprof."""

import argparse
import bisect
import collections
import json
import re
import sqlite3
import statistics
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="ablation")
args = ap.parse_args()
root = Path(__file__).resolve().parent
records = json.loads((root / (args.name + "-records.json")).read_text())
validations = json.loads((root / (args.name + "-validation.json")).read_text())
lookup = {(r["row"], r["variant"], r["block"], r["roi"]): r for r in records}
db = sqlite3.connect(str(root / "profiles" / (args.name + ".db")))
tables = [r[0] for r in db.execute("select name from sqlite_master where type='table'")]


def table(prefix):
    found = [name for name in tables if name.startswith(prefix)]
    assert len(found) == 1, (prefix, found)
    return found[0]


ops, copies, tx = table("HIPOPS_"), table("HIPCOPY_"), table("HIPTX_")
strings = {
    str(i): name
    for i, name in db.execute(
        "select STR_ID,STR_NAME from STR_TABLE where CONFIG_KEY=? and TYPE=6",
        (ops[len("HIPOPS_") :],),
    )
}
rois = db.execute(
    "select message,begin_Index,end_Index from " + tx + " order by begin_Index"
).fetchall()
rois = [r for r in rois if r[0].startswith("fp64audit:")]
assert len(rois) == len(records)
starts = [r[1] for r in rois]
stats = [
    {
        "kernel_ns": 0,
        "copy_ns": 0,
        "kernels": 0,
        "copies": 0,
        "queues": set(),
        "names": set(),
    }
    for _ in rois
]
for index, duration, name, queue, begin, end in db.execute(
    "select _Index,DurationNs,Name,queue_id,BeginNs,EndNs from "
    + ops
    + " order by _Index"
):
    assert end - begin == duration
    i = bisect.bisect_right(starts, index) - 1
    if i >= 0 and index <= rois[i][2]:
        item = stats[i]
        item["kernel_ns"] += duration
        item["kernels"] += 1
        item["queues"].add(queue)
        item["names"].add(strings[str(name)])
for index, duration, queue in db.execute(
    "select _Index,DurationNs,queue_id from " + copies
):
    i = bisect.bisect_right(starts, index) - 1
    if i >= 0 and index <= rois[i][2]:
        item = stats[i]
        item["copy_ns"] += duration
        item["copies"] += 1
        item["queues"].add(queue)
groups = collections.defaultdict(list)
for roi, item in zip(rois, stats):
    match = re.fullmatch(
        r"fp64audit:row=(\d+):variant=(\w+):block=(\d+):roi=(\d+)", roi[0]
    )
    assert match
    key = (int(match[1]), match[2], int(match[3]), int(match[4]))
    record = lookup[key]
    assert item["kernels"] > 0 and item["kernels"] % record["calls"] == 0
    assert "queue_gate" not in item["names"]
    sample = {
        **record,
        **{k: v for k, v in item.items() if k not in ("queues", "names")},
        "queues": sorted(item["queues"]),
        "kernel_names": sorted(item["names"]),
        "device_us": (item["kernel_ns"] + item["copy_ns"]) / (record["calls"] * 1000),
        "kernel_us": item["kernel_ns"] / (record["calls"] * 1000),
        "copy_us": item["copy_ns"] / (record["calls"] * 1000),
    }
    groups[key[:2]].append(sample)
results = []
for (row, variant), samples in sorted(groups.items()):
    numeric = [v for v in validations if v["row"] == row and v["variant"] == variant]
    assert numeric
    blocks = [
        statistics.median(s["device_us"] for s in samples if s["block"] == b)
        for b in sorted({s["block"] for s in samples})
    ]
    result = {
        "row": row,
        "variant": variant,
        "correct": all(v["correct"] for v in numeric),
        "numeric_seeds": [v["seed"] for v in numeric],
        "correct_against_cpu_fp32": (
            all(v.get("against_cpu_fp32", {}).get("correct", False) for v in numeric)
            if any("against_cpu_fp32" in v for v in numeric)
            else None
        ),
        "device_us": statistics.median(blocks),
        "block_medians_us": blocks,
        "block_drift": max(blocks) / min(blocks) - 1,
        "kernel_names": samples[0]["kernel_names"],
        "samples": samples,
    }
    if any(v.get("diagnostic_incomplete_output") for v in numeric):
        result["diagnostic_incomplete_output"] = True
        result["not_for_acceptance"] = True
    results.append(result)
    print(
        "RESULT",
        row,
        variant,
        "correct",
        result["correct"],
        "device_us",
        result["device_us"],
        "block_drift",
        result["block_drift"],
        "kernels",
        result["kernel_names"],
        flush=True,
    )
(root / (args.name + "-summary.json")).write_text(
    json.dumps(
        {
            "metric": "all ROI kernel+copy duration / calls; host and gate excluded",
            "results": results,
        },
        indent=2,
    )
)
