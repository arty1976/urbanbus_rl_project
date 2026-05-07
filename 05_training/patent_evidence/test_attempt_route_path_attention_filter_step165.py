from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from attempt_route_path_attention_filter_step165 import make_sample_inputs, run_filter
from validate_attempt_route_path_attention_filter_step165 import validate_manifest


class Step165AttemptRoutePathAttentionFilterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="step165_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_sample_filter_and_validator_pass(self) -> None:
        inputs = make_sample_inputs(self.tmp / "inputs")
        manifest = run_filter(
            pickup_attempt_events=inputs["pickup_attempt_events"],
            eta_counterfactual=inputs["eta_counterfactual"],
            real_attention=inputs["real_attention"],
            route_stop_sequence=inputs["route_stop_sequence"],
            run_manifest=inputs["run_manifest"],
            output_root=self.tmp / "out",
            top_k_per_attempt=6,
        )
        m = validate_manifest(
            Path(manifest["output_files"]["manifest"]),
            require_real_attention=True,
            require_no_fallback=True,
        )
        self.assertEqual(m["row_counts"]["pickup_attempt_count"], 5)
        self.assertEqual(m["row_counts"]["attempt_count_with_filtered_attention"], 5)
        self.assertEqual(m["row_counts"]["fallback_attempt_count"], 0)
        self.assertGreater(m["row_counts"]["exact_edge_overlap_count"], 0)

    def test_validator_blocks_mutated_claim_flag(self) -> None:
        inputs = make_sample_inputs(self.tmp / "inputs_mutated")
        manifest = run_filter(
            pickup_attempt_events=inputs["pickup_attempt_events"],
            eta_counterfactual=inputs["eta_counterfactual"],
            real_attention=inputs["real_attention"],
            route_stop_sequence=inputs["route_stop_sequence"],
            run_manifest=inputs["run_manifest"],
            output_root=self.tmp / "mutated",
            top_k_per_attempt=6,
        )
        manifest_path = Path(manifest["output_files"]["manifest"])
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["paper_level_claim_allowed"] = True
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        with self.assertRaises(RuntimeError):
            validate_manifest(manifest_path, require_real_attention=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
