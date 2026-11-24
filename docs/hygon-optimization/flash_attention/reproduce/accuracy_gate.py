"""Captured-input full-tensor gate; preserve original PR tolerances for every arm."""

import hashlib
import json

import numpy as np

CONTRACT = {
    "reference": "CPU FP32 SDPA output; CPU FP32 matmul/logsumexp LSE",
    "output_atol": 0.002,
    "output_rtol": 0.02,
    "lse_atol": 2e-5,
    "lse_rtol": 2e-5,
    "output_shape": [8, 16, 2049, 128],
    "lse_shape": [8, 16, 2049],
    "output_elements": 33570816,
    "lse_elements": 262272,
    "seed": 20261105,
    "stride": [12589056, 128, 6144, 1],
    "offsets": [0, 2048, 4096],
    "scale": 0.08838834764831843,
}
CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(CONTRACT, sort_keys=True).encode()
).hexdigest()
ARMS = ("wheel", "aten", "repo")


def compare_arrays(actual, expected, atol, rtol):
    """Report every mismatch, including nonfinite values; forbid broadcasting."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        raise ValueError(f"Output shape mismatch: {actual.shape} vs {expected.shape}")
    finite = np.isfinite(actual) & np.isfinite(expected)
    with np.errstate(invalid="ignore", over="ignore"):
        close = finite & (np.abs(actual - expected) <= atol + rtol * np.abs(expected))
    failed = int(np.count_nonzero(~close))
    delta = np.abs(actual[finite] - expected[finite])
    return {
        "shape": list(expected.shape),
        "elements": int(expected.size),
        "failed": failed,
        "failed_percent": 100 * failed / expected.size,
        "nonfinite": int(np.count_nonzero(~finite)),
        "max_abs_finite": float(delta.max()) if delta.size else None,
    }


def cpu_reference(cpu_inputs):
    import torch

    q, k, v = [x.float() for x in cpu_inputs]
    output = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, scale=CONTRACT["scale"]
    )
    lse = torch.empty(q.shape[:3], dtype=torch.float32)
    for start in range(0, q.shape[2], 128):
        score = q[:, :, start : start + 128] @ k.transpose(-1, -2) * CONTRACT["scale"]
        lse[:, :, start : start + 128] = torch.logsumexp(score, dim=-1)
    return output.numpy(), lse.numpy()


def check_full(output, lse, reference):
    return {
        name: compare_arrays(
            actual.cpu().float().numpy(),
            expected,
            CONTRACT[name + "_atol"],
            CONTRACT[name + "_rtol"],
        )
        for name, actual, expected in zip(("output", "lse"), (output, lse), reference)
    }


def require_eligible_baselines(receipt):
    """Reject historical, missing, partial, asymmetric or failing gate receipts."""
    if (
        receipt.get("schema") != 2
        or receipt.get("contract") != CONTRACT
        or receipt.get("contract_sha256") != CONTRACT_SHA256
    ):
        raise ValueError("No matching common full-output/LSE accuracy contract")
    if not receipt.get("run_id") or not receipt.get("input_sha256"):
        raise ValueError("Missing current-run input identity")
    if receipt.get("device", {}).get("physical_card") not in ("3", "4"):
        raise ValueError("Missing authorized single-card identity")
    for arm in ARMS:
        result = receipt.get("full_accuracy", {}).get(arm, {})
        for kind in ("output", "lse"):
            stat = result.get(kind, {})
            if (
                stat.get("elements") != CONTRACT[kind + "_elements"]
                or stat.get("shape") != CONTRACT[kind + "_shape"]
                or stat.get("failed") != 0
                or stat.get("nonfinite") != 0
            ):
                raise ValueError(f"{arm} {kind} fails the full accuracy gate: {stat}")
