"""One full-tensor precision contract for every worksheet case and backend."""

import math

from workbook_inputs import MANIFEST

TOLERANCES = {
    "output_atol": 0.002,
    "output_rtol": 0.02,
    "lse_atol": 2e-5,
    "lse_rtol": 2e-5,
}
NATIVE_VERSION = "2.8.4+dtk2604.torch2100.2609292239.g74de14"
NATIVE_HASHES = {
    "4dbb514cd0d6b30b720bbc739b13f3655ec5b345765b303e463184a1cd243f25",
    "42b820080757241f048e7ed6673a91703e5563353364ea279fa03cd0ca1a422e",
}


def require_case(receipt, case):
    if (
        receipt.get("schema") != 5
        or receipt.get("workbook_sha256") != MANIFEST["workbook_sha256"]
        or receipt.get("numerical_tolerances") != TOLERANCES
        or not receipt.get("run_id")
        or receipt.get("device", {}).get("physical_card") not in ("3", "4")
        or receipt.get("device", {}).get("node") != "a01r4n15"
    ):
        raise ValueError("Missing matching current workbook precision contract")
    native = receipt.get("native", {})
    if (
        native.get("version") != NATIVE_VERSION
        or set(native.get("loaded_libraries_sha256", {}).values()) != NATIVE_HASHES
        or len(native.get("loaded_libraries_sha256", {})) != 2
    ):
        raise ValueError(
            "Native implementation does not match the pinned qualified package"
        )
    record = receipt.get("cases", {}).get(case["id"], {})
    identity = record.get("input_identity", {})
    if (
        record.get("case") != case
        or len(identity.get("backing_sha256", "")) != 64
        or identity.get("seed") != case["seed"]
        or identity.get("fp32_generation") is not False
        or identity.get("storage_policy") != case["storage"]
    ):
        raise ValueError("Missing worksheet row or reconstructed input identity")
    shape = case["specification"][0][0]["shape"]
    for arm in ("wheel", "aten", "repo"):
        for kind, expected in (("output", shape), ("lse", shape[:3])):
            stat = record.get("full_accuracy", {}).get(arm, {}).get(kind, {})
            if (
                stat.get("shape") != expected
                or stat.get("elements") != math.prod(expected)
                or stat.get("failed") != 0
                or stat.get("nonfinite") != 0
            ):
                raise ValueError(
                    f"{case['id']} {arm} {kind} fails the full precision gate: {stat}"
                )
