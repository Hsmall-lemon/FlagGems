"""Regression checks against accepting an invalid native denominator or trace."""

import copy
import math
import unittest

from analyze_workbook_profile import analyze
from workbook_gate import NATIVE_HASHES, NATIVE_VERSION, TOLERANCES, require_case
from workbook_inputs import MANIFEST


def qualified_receipt():
    receipt = {
        "schema": 5,
        "run_id": "current-run",
        "workbook_sha256": MANIFEST["workbook_sha256"],
        "numerical_tolerances": TOLERANCES,
        "device": {"node": "a01r4n15", "physical_card": "3"},
        "native": {
            "version": NATIVE_VERSION,
            "loaded_libraries_sha256": dict(enumerate(sorted(NATIVE_HASHES))),
        },
        "sampling": {"calls": 1000, "rounds": 5, "warmup": 100},
        "source_sha256": {},
        "cases": {},
    }
    for case in MANIFEST["cases"]:
        shape = case["specification"][0][0]["shape"]
        accuracy = {
            kind: {
                "shape": dims,
                "elements": math.prod(dims),
                "failed": 0,
                "nonfinite": 0,
            }
            for kind, dims in (("output", shape), ("lse", shape[:3]))
        }
        receipt["cases"][case["id"]] = {
            "case": case,
            "input_identity": {
                "backing_sha256": "a" * 64,
                "seed": case["seed"],
                "fp32_generation": False,
                "storage_policy": case["storage"],
            },
            "full_accuracy": {
                arm: copy.deepcopy(accuracy) for arm in ("wheel", "aten", "repo")
            },
            "timing_started": True,
            "timing_completed": True,
        }
    return receipt


def complete_trace(receipt):
    events = []
    timestamp = 0
    for case in MANIFEST["cases"]:
        for arm in ("wheel", "aten", "repo"):
            for i in range(5):
                per_call = {"model_state": 3}.get(case["id"], 1)
                count = 1000 * (per_call if arm == "repo" else 1)
                duration = 3 / (count / 1000)
                events.append(
                    {
                        "cat": "HIPTX",
                        "ph": "X",
                        "ts": timestamp,
                        "dur": 6000,
                        "name": f"workbook:{receipt['run_id']}:{case['id']}:{arm}:round={i}",
                    }
                )
                events.extend(
                    {
                        "cat": "HIPOPS",
                        "ph": "X",
                        "ts": timestamp + k * duration,
                        "dur": duration,
                        "name": "kernel",
                    }
                    for k in range(count)
                )
                timestamp += 10000
    return {"traceEvents": events}


class WorkbookAcceptanceTests(unittest.TestCase):
    def test_full_gate_passes(self):
        receipt = qualified_receipt()
        for case in MANIFEST["cases"]:
            require_case(receipt, case)

    def test_native_lse_mismatch_cannot_be_denominator(self):
        for arm in ("wheel", "aten"):
            receipt = qualified_receipt()
            receipt["cases"]["model_state"]["full_accuracy"][arm]["lse"]["failed"] = 1
            with self.assertRaises(ValueError):
                analyze(receipt, {"traceEvents": []})

    def test_partial_or_nonfinite_results_are_rejected(self):
        for field, value in (("elements", 1), ("nonfinite", 1), ("shape", [1])):
            receipt = qualified_receipt()
            receipt["cases"]["expanded_large"]["full_accuracy"]["repo"]["output"][
                field
            ] = value
            with self.assertRaises(ValueError):
                require_case(receipt, MANIFEST["cases"][3])

    def test_185_and_substituted_library_are_rejected(self):
        for field, value in (
            ("version", "2.8.3+das185"),
            ("loaded_libraries_sha256", {"one": "wrong"}),
        ):
            receipt = qualified_receipt()
            receipt["native"][field] = value
            with self.assertRaises(ValueError):
                require_case(receipt, MANIFEST["cases"][0])

    def test_incomplete_sampling_and_missing_case_are_rejected(self):
        receipt = qualified_receipt()
        receipt["sampling"]["calls"] = 10
        with self.assertRaises(ValueError):
            analyze(receipt, {"traceEvents": []})
        receipt = qualified_receipt()
        del receipt["cases"]["expanded_small"]
        with self.assertRaises(ValueError):
            analyze(receipt, {"traceEvents": []})

    def test_trace_run_identity_and_helper_accounting(self):
        receipt = qualified_receipt()
        trace = complete_trace(receipt)
        self.assertTrue(analyze(receipt, trace)["all_reconstructed_cases_meet_90"])
        del trace["traceEvents"][1]
        with self.assertRaises(ValueError):
            analyze(receipt, trace)
        trace["traceEvents"][0]["name"] = "workbook:old-run:model_state:wheel:round=0"
        with self.assertRaises(ValueError):
            analyze(receipt, trace)


if __name__ == "__main__":
    unittest.main()
