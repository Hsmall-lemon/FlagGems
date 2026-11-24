"""Replay both stored device traces and verify their measured source identities."""

import gzip
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

doc = Path(__file__).resolve().parents[1]
repo = doc.parents[2]
evidence = doc / "node15-verification"
summary = json.loads((evidence / "summary.json").read_text())
for card in ("3", "4"):
    saved = summary["cards"][card]
    directory = evidence / ("card" + card)
    for path, expected in saved["source_sha256"].items():
        actual = hashlib.sha256((repo / path).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Measured source differs: {path}")
    raw = gzip.decompress((directory / "profile.json.gz").read_bytes())
    if hashlib.sha256(raw).hexdigest() != saved["trace_sha256"]:
        raise ValueError(f"Raw trace differs for card {card}")
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary)
        (output / "profile.json").write_bytes(raw)
        (output / "repo_final_profile_validation.json").write_bytes(
            (directory / "repo_final_profile_validation.json").read_bytes()
        )
        subprocess.run(
            [sys.executable, str(doc / "reproduce/analyze_profile.py"), "profile.json"],
            cwd=output,
            check=True,
            capture_output=True,
        )
        replayed = json.loads((output / "repo_final_kernel_summary.json").read_text())
        for key in ("run_id", "timing_us", "source_sha256", "native"):
            if replayed[key] != saved[key]:
                raise ValueError(f"Replayed {key} differs for card {card}")
        if (
            replayed["meets_90"] is not True
            or replayed["external_frozen_gate_verified"] is not False
        ):
            raise ValueError(f"Acceptance scope differs for card {card}")
        print(
            f"Card {card}: replay passed; "
            f"pinned native / FlagGems = {replayed['qualified_native_over_repo']:.4%}"
        )
