"""Recover device kernel time from explicit, synchronized HIPTX ROIs."""

import bisect
import collections
import json
import re
import statistics
import sys
from pathlib import Path


def extract(path):
    trace_path = Path(path)
    paths = (
        [trace_path]
        if trace_path.exists()
        else sorted(trace_path.parent.glob(trace_path.stem + "_*.json"))
    )
    if not paths:
        raise FileNotFoundError(path)
    events = []
    for part in paths:
        events.extend(json.loads(part.read_text())["traceEvents"])
    kernels = sorted(
        (e for e in events if e.get("cat") == "HIPOPS"),
        key=lambda e: int(e["args"]["index"]),
    )
    indices = [int(e["args"]["index"]) for e in kernels]
    results = collections.defaultdict(list)
    names = collections.defaultdict(set)
    for roi in events:
        match = re.fullmatch(r"mm:row=(\d+):cfg=(\w+):roi=(\d+)", roi.get("name", ""))
        if not match:
            continue
        lo, hi = int(roi["args"]["BeginIndex"]), int(roi["args"]["EndIndex"])
        included = kernels[
            bisect.bisect_left(indices, lo) : bisect.bisect_right(indices, hi)
        ]
        if len(included) < 10 or len(included) % 10:
            raise ValueError(f'Invalid kernel count {len(included)} in {roi["name"]}')
        key = (int(match[1]), match[2])
        results[key].append(sum(k["args"]["DurationNs"] for k in included) / 1000 / 10)
        names[key].update(k["name"] for k in included)
    return {
        key: dict(
            native_us=statistics.median(rois),
            native_rois_us=rois,
            kernels=sorted(names[key]),
        )
        for key, rois in results.items()
    }


def analyze(trace, records):
    timings = extract(trace)
    out = []
    for line in Path(records).read_text().splitlines():
        r = json.loads(line)
        if "error" in r:
            out.append(r)
            continue
        configs = r.get("all_configs", [r])
        successful = []
        for cfg in configs:
            key = (r["row"], str(cfg.get("config_id", "public")))
            if key not in timings:
                raise ValueError(f"Missing ROI for {key}")
            successful.append({**cfg, **timings[key]})
        successful.sort(key=lambda x: x["native_us"])
        best = successful[0]
        out.append(
            {
                **r,
                **best,
                "native_top": successful,
                "sheet_native_ratio": r["sheet_das_us"] / best["native_us"],
            }
        )
    return out


if __name__ == "__main__":
    rows = analyze(sys.argv[1], sys.argv[2])
    Path(sys.argv[3]).write_text(json.dumps(rows, indent=2))
    print("ROWS", len(rows), "ERRORS", sum("error" in r for r in rows))
    for r in rows:
        print(
            r["row"],
            r["description"],
            round(r.get("native_us", 0), 3),
            "us",
            r.get("config", ""),
            "sheet_ratio",
            round(r.get("sheet_native_ratio", 0), 3),
        )
