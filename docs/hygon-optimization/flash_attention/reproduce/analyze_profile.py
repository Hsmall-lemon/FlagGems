import argparse
import collections
import json
import statistics
from pathlib import Path

from accuracy_gate import require_eligible_baselines

parser = argparse.ArgumentParser()
parser.add_argument("profile", type=Path)
args = parser.parse_args()
root = Path.cwd()
v = json.loads((root / "repo_final_profile_validation.json").read_text())
require_eligible_baselines(v)
if v.get("acceptance_timing_started") is not True:
    raise ValueError("No timing run has passed the common full precision gate")
prefix = "flash:" + v["run_id"] + ":"
ev = json.loads(args.profile.read_text())["traceEvents"]
rois = [
    e
    for e in ev
    if e.get("cat") == "HIPTX"
    and e.get("name", "").startswith("flash:")
    and ":roi=" in e["name"]
]
if not rois or any(not e["name"].startswith(prefix) for e in rois):
    raise ValueError("Trace is missing, historical, or belongs to another gate run")
dev = [e for e in ev if e.get("cat") == "HIPOPS" and e.get("ph") == "X"]
arms = {}
for r in rois:
    name = r["name"][len(prefix) :].split(":")[0]
    k = [
        e
        for e in dev
        if r["ts"] <= e["ts"] and e["ts"] + e["dur"] <= r["ts"] + r["dur"]
    ]
    assert len(k) == (30 if name.startswith(("repo", "hip")) else 10), (name, len(k))
    arms.setdefault(name, []).append(
        {
            "roi": r["name"],
            "calls": 10,
            "kernels": dict(collections.Counter(e["name"] for e in k)),
            "us": sum(e["dur"] for e in k) / 10,
        }
    )
assert len(arms) == 6, len(arms)
for name in ("wheel", "aten", "repo", "wheel_repeat", "aten_repeat", "repo_repeat"):
    if len(arms.get(name, [])) != 5:
        raise ValueError(f"Missing five timing ranges for {name}")
timing = {
    key: statistics.median(
        statistics.median(x["us"] for x in arms[name])
        for name in [key, key + "_repeat"]
    )
    for key in ["wheel", "aten", "repo"]
}
result = {
    "status": "passed_common_captured_input_contract",
    "acceptance_scope": (
        "Recorded fixture and common full-output/LSE contract; "
        "no claim of replaying external review inputs"
    ),
    "run_id": v["run_id"],
    "contract_sha256": v["contract_sha256"],
    "input_sha256": v["input_sha256"],
    "device": v["device"],
    "native": v.get("native"),
    "method": "hipprof HIPOPS duration; all helper kernels included; 5 ROIs x 10 calls x 2 repeats",
    "timing_us": timing,
    "diagnostic_wheel_over_repo_under_original_PR_contract": timing["wheel"]
    / timing["repo"],
    "common_contract_meets_90": min(timing["wheel"], timing["aten"]) / timing["repo"]
    >= 0.9,
    "wheel_over_repo": timing["wheel"] / timing["repo"],
    "qualified_native_over_repo": min(timing["wheel"], timing["aten"]) / timing["repo"],
    "meets_90": min(timing["wheel"], timing["aten"]) / timing["repo"] >= 0.9,
    "external_frozen_gate_verified": False,
    "source_sha256": v["source_sha256"],
    "arms": arms,
}
(root / "repo_final_kernel_summary.json").write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != "arms"}, indent=2))
