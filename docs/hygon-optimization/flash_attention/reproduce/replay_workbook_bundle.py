"""Verify the archived alternative-baseline experiment, not DAS185 acceptance."""

import gzip
import hashlib
import json
import math
from pathlib import Path

from analyze_workbook_profile import analyze
from workbook_inputs import MANIFEST


def require_full_accuracy(stats, shape):
    for kind, dims in (("output", shape), ("lse", shape[:3])):
        stat = stats[kind]
        if (
            stat["shape"] != dims
            or stat["elements"] != math.prod(dims)
            or stat["failed"] != 0
            or stat["nonfinite"] != 0
        ):
            raise ValueError(f"Incomplete or failed {kind} stability check")


def require_stability(folder, card, source_hashes):
    receipt = json.loads((folder / "workbook-stability.json").read_text())
    if receipt["device"] != {"node": "a01r4n15", "physical_card": card}:
        raise ValueError("Stability checked a different device")
    if receipt["source_sha256"] != source_hashes:
        raise ValueError("Stability checked different source bytes")
    if set(receipt["cases"]) != {c["id"] for c in MANIFEST["cases"]}:
        raise ValueError("Missing stability cases")
    boundaries = {
        "large_qk",
        "zero_q_large_k",
        "large_q_zero_k",
        "large_v",
        "large_scores_large_v",
        "signed_v8",
        "tiny_v",
        "large_finite_v",
        "negative_scores",
        "changing_scores",
    }
    for case in MANIFEST["cases"]:
        records = receipt["cases"][case["id"]]
        expected = {
            ("normal", seed, fp32)
            for seed in (case["seed"], 0, 65537)
            for fp32 in (False, True)
        }
        if case["id"] == "expanded_large":
            expected |= {(pattern, case["seed"], False) for pattern in boundaries}
        observed = {
            (r["pattern"], r["identity"]["seed"], r["identity"]["fp32_generation"])
            for r in records
        }
        if observed != expected or len(records) != len(expected):
            raise ValueError("Incomplete or duplicated stability coverage")
        shape = case["specification"][0][0]["shape"]
        for record in records:
            if record["bitwise_stable"] is not True or len(record["checks"]) != 3:
                raise ValueError("Missing three stable repeats")
            for stats in record["checks"]:
                require_full_accuracy(stats, shape)
    edges = json.loads((folder / "edge-cases.json").read_text())
    if edges["card"] != card or any(
        source_hashes[p] != digest for p, digest in edges["source_sha256"].items()
    ):
        raise ValueError("Edge checks used another source/device")
    if set(edges["cases"]) != {"zero_q_large_k", "large_q_zero_k", "large_v"}:
        raise ValueError("Missing model boundary cases")
    for record in edges["cases"].values():
        if len(record["repo"]) != 3:
            raise ValueError("Missing model boundary repeats")
        for repeat in record["repo"]:
            if not repeat["passed"] or not repeat["bitwise_equal_to_first"]:
                raise ValueError("Model boundary failed")
            require_full_accuracy(repeat["full_accuracy"], [8, 16, 2049, 128])
    if "33 passed" not in (folder / "pytest.log").read_text():
        raise ValueError("GPU regression suite did not pass")


repo = Path(__file__).resolve().parents[4]
root = Path(__file__).resolve().parents[1] / "workbook-verification"
inventory = json.loads((root / "inventory.json").read_text())
for name, digest in inventory["evidence_sha256"].items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
        raise ValueError(f"Evidence identity mismatch: {name}")
for name, digest in inventory["source_sha256"].items():
    if hashlib.sha256((repo / name).read_bytes()).hexdigest() != digest:
        raise ValueError(f"Measured source mismatch: {name}")
for name, digest in inventory["reproduction_sha256"].items():
    if hashlib.sha256((repo / name).read_bytes()).hexdigest() != digest:
        raise ValueError(f"Reproduction tooling mismatch: {name}")
for card in ("3", "4"):
    folder = root / f"gpu{card}"
    gate_path = folder / "workbook-gate.json"
    gate = json.loads(gate_path.read_text())
    summary = json.loads((folder / "workbook-summary.json").read_text())
    trace = json.loads(gzip.decompress((folder / "device-trace.json.gz").read_bytes()))
    if trace.get("archive_kind") != "filtered_device_trace":
        raise ValueError("Unlabeled trace transformation")
    if (
        hashlib.sha256((folder / "device-trace.json.gz").read_bytes()).hexdigest()
        != summary["trace_sha256"]
    ):
        raise ValueError("Device subset belongs to another timing result")
    if not trace.get("original_trace_parts"):
        raise ValueError("Missing identities of complete original trace parts")
    if hashlib.sha256(gate_path.read_bytes()).hexdigest() != summary["gate_sha256"]:
        raise ValueError("Gate receipt identity mismatch")
    actual = analyze(gate, trace)
    expected = {
        k: v for k, v in summary.items() if k not in ("gate_sha256", "trace_sha256")
    }
    if actual != expected or not actual["all_reconstructed_cases_meet_90"]:
        raise ValueError(f"Card {card} does not reproduce all four acceptance results")
    if actual["source_sha256"] != inventory["source_sha256"]:
        raise ValueError("Cards measured different sources")
    if actual["device"] != {"node": "a01r4n15", "physical_card": card}:
        raise ValueError("Results belong to another physical card")
    require_stability(folder, card, inventory["source_sha256"])
    print("ALTERNATIVE_DAS_2_8_4_EXPERIMENT_VERIFIED", card)
print("Original model payload and external frozen gate remain unverified.")

print(
    "REQUESTED_DAS185_ACCEPTANCE_NOT_ESTABLISHED; alternative baseline was not authorized."
)
