"""Archive a labeled device-event subset; retain the complete trace's identity."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("profile", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
if args.profile.is_dir():
    whole = args.profile / "profile.json"
    parts = (
        [whole]
        if whole.exists()
        else sorted(
            args.profile.glob("profile_*.json"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
    )
    if not whole.exists() and [p.name for p in parts] != [
        f"profile_{i}.json" for i in range(1, len(parts) + 1)
    ]:
        raise ValueError("Noncontiguous split trace parts")
else:
    parts = [args.profile]
if not parts:
    raise ValueError("No complete hipprof trace files")
identities, retained = [], []
for path in parts:
    original = path.read_bytes()
    trace = json.loads(original)
    events = trace["traceEvents"]
    identities.append(
        {
            "name": path.name,
            "sha256": hashlib.sha256(original).hexdigest(),
            "bytes": len(original),
            "event_count": len(events),
        }
    )
    retained.extend(e for e in events if e.get("cat") in ("HIPOPS", "HIPTX"))
    del events, trace, original
subset = {
    "archive_kind": "filtered_device_trace",
    "original_trace_parts": identities,
    "transformation": (
        "Concatenate complete numbered trace parts; retain HIPOPS and HIPTX "
        "in original order; omit host API events"
    ),
    "traceEvents": retained,
}
args.output.write_bytes(
    gzip.compress(json.dumps(subset, separators=(",", ":")).encode(), mtime=0)
)
print("ARCHIVED_FILTERED_DEVICE_TRACE", args.output, args.output.stat().st_size)
