"""CPU regressions for rejecting the invalid baseline acceptance path."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from accuracy_gate import (
    ARMS,
    CONTRACT,
    CONTRACT_SHA256,
    compare_arrays,
    require_eligible_baselines,
)


def valid_receipt():
    return {
        "schema": 2,
        "run_id": "current-test-run",
        "input_sha256": "a" * 64,
        "device": {"physical_card": "4"},
        "contract": copy.deepcopy(CONTRACT),
        "contract_sha256": CONTRACT_SHA256,
        "acceptance_timing_started": True,
        "source_sha256": {},
        "full_accuracy": {
            arm: {
                kind: {
                    "shape": CONTRACT[kind + "_shape"],
                    "elements": CONTRACT[kind + "_elements"],
                    "failed": 0,
                    "nonfinite": 0,
                }
                for kind in ("output", "lse")
            }
            for arm in ARMS
        },
    }


class AcceptanceGateTests(unittest.TestCase):
    def test_legacy_native_tolerance_hides_lse_failures(self):
        expected = np.full(1000, 8.0)
        native = expected.copy()
        native[:236] += 0.00069
        loose = compare_arrays(native, expected, 1e-3, 1e-4)
        common = compare_arrays(
            native, expected, CONTRACT["lse_atol"], CONTRACT["lse_rtol"]
        )
        self.assertEqual(loose["failed"], 0)
        self.assertEqual(common["failed"], 236)
        self.assertAlmostEqual(common["failed_percent"], 23.6)

    def test_nonfinite_values_cannot_pass_even_when_both_nan(self):
        stat = compare_arrays(
            np.array([np.nan, np.inf]), np.array([np.nan, np.inf]), 0, 0
        )
        self.assertEqual(stat["failed"], 2)
        self.assertEqual(stat["nonfinite"], 2)

    def test_shape_broadcasting_is_rejected(self):
        with self.assertRaises(ValueError):
            compare_arrays(np.zeros((2, 1)), np.zeros((2, 2)), 0, 0)

    def test_historical_native_timing_is_not_an_eligible_gate(self):
        with self.assertRaises(ValueError):
            require_eligible_baselines(
                {"timing_us": {"wheel": 1620.6}, "meets_90": True}
            )

    def test_failed_partial_missing_and_relaxed_gates_are_rejected(self):
        for arm in ARMS:
            for kind in ("output", "lse"):
                for field, value in (("failed", 1), ("elements", 12), ("nonfinite", 1)):
                    receipt = valid_receipt()
                    receipt["full_accuracy"][arm][kind][field] = value
                    with self.subTest(arm=arm, kind=kind, field=field):
                        with self.assertRaises(ValueError):
                            require_eligible_baselines(receipt)
        receipt = valid_receipt()
        del receipt["full_accuracy"]["wheel"]
        with self.assertRaises(ValueError):
            require_eligible_baselines(receipt)
        receipt = valid_receipt()
        receipt["contract"]["lse_atol"] = 1e-3
        with self.assertRaises(ValueError):
            require_eligible_baselines(receipt)

    def test_common_full_gate_is_eligible(self):
        require_eligible_baselines(valid_receipt())

    def test_analyzer_rejects_stale_trace_without_producing_ratio(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "repo_final_profile_validation.json").write_text(
                json.dumps(valid_receipt())
            )
            trace = {
                "traceEvents": [{"cat": "HIPTX", "name": "flash:old-run:wheel:roi=0"}]
            }
            (out / "profile.json").write_text(json.dumps(trace))
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("analyze_profile.py")),
                    "profile.json",
                ],
                cwd=tmp,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("historical", result.stderr)
            self.assertFalse((out / "repo_final_kernel_summary.json").exists())

    def test_matching_trace_produces_scoped_ratio_without_external_replay_claim(self):
        events = []
        start = 0
        for suffix in ("", "_repeat"):
            for arm in ARMS:
                for roi in range(5):
                    events.append(
                        {
                            "cat": "HIPTX",
                            "name": f"flash:current-test-run:{arm}{suffix}:roi={roi}",
                            "ts": start,
                            "dur": 50,
                        }
                    )
                    for index in range(30 if arm == "repo" else 10):
                        events.append(
                            {
                                "cat": "HIPOPS",
                                "ph": "X",
                                "name": "synthetic_kernel",
                                "ts": start + 1 + index,
                                "dur": 1,
                            }
                        )
                    start += 100
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "repo_final_profile_validation.json").write_text(
                json.dumps(valid_receipt())
            )
            (out / "profile.json").write_text(json.dumps({"traceEvents": events}))
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("analyze_profile.py")),
                    "profile.json",
                ],
                cwd=tmp,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((out / "repo_final_kernel_summary.json").read_text())
            self.assertAlmostEqual(
                summary["diagnostic_wheel_over_repo_under_original_PR_contract"], 1 / 3
            )
            self.assertAlmostEqual(summary["wheel_over_repo"], 1 / 3)
            self.assertFalse(summary["meets_90"])
            self.assertFalse(summary["external_frozen_gate_verified"])
            self.assertIn("Recorded fixture", summary["acceptance_scope"])


if __name__ == "__main__":
    unittest.main()
