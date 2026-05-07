from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from actual_evidence_input_readiness_checklist_step169 import run_check
from validate_actual_evidence_input_readiness_checklist_step169 import validate_manifest


class Step169ActualEvidenceInputReadinessChecklistTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="step169_test_"))
        self.project_root = Path.cwd()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_sample_readiness_and_validator_pass(self) -> None:
        out = self.tmp / "sample"
        manifest = run_check(
            project_root=self.project_root,
            output_root=out,
            mode="sample",
            raw_events=None,
            window_rollup=None,
            route_stop_sequence=None,
            run_manifest=None,
            gatv2_checkpoint=None,
            require_pipeline_scripts=False,
        )
        self.assertEqual(manifest["audit_status"], "PASS")
        self.assertTrue(manifest["ready_for_actual_like_execution"])
        self.assertFalse(manifest["paper_level_claim_allowed"])
        m = validate_manifest(Path(manifest["output_files"]["manifest"]), require_ready=True)
        self.assertEqual(m["bundle_status"], "ACTUAL_EVIDENCE_INPUTS_READY_FOR_ZERO_LOSS_PIPELINE_NONCLAIM")

    def test_validator_blocks_mutated_claim_flag(self) -> None:
        out = self.tmp / "mutated"
        manifest = run_check(
            project_root=self.project_root,
            output_root=out,
            mode="sample",
            raw_events=None,
            window_rollup=None,
            route_stop_sequence=None,
            run_manifest=None,
            gatv2_checkpoint=None,
            require_pipeline_scripts=False,
        )
        p = Path(manifest["output_files"]["manifest"])
        payload = json.loads(p.read_text(encoding="utf-8"))
        payload["paper_level_claim_allowed"] = True
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        with self.assertRaises(RuntimeError):
            validate_manifest(p, require_ready=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
