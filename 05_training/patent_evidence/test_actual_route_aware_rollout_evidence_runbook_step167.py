from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from actual_route_aware_rollout_evidence_runbook_step167 import generate_runbook, parse_args
from validate_actual_route_aware_rollout_evidence_runbook_step167 import validate_manifest


def make_args(tmp: Path):
    args = parse_args()
    args.project_root = str(tmp / "project")
    args.output_root = str(tmp / "out")
    args.actual_output_root = "artifacts/patent_evidence/test_actual_evidence"
    args.raw_events = ""
    args.window_rollup = ""
    args.route_stop_sequence = ""
    args.require_step_files = False
    return args


class Step167ActualRouteAwareRolloutEvidenceRunbookTest(unittest.TestCase):
    def test_template_runbook_and_validator_pass(self):
        with tempfile.TemporaryDirectory(prefix="step167_test_") as d:
            tmp = Path(d)
            (tmp / "project").mkdir(parents=True, exist_ok=True)
            args = make_args(tmp)
            m = generate_runbook(args)
            manifest = Path(m["manifest_path"])
            out = validate_manifest(manifest)
            self.assertEqual(out["audit_status"], "PASS")
            self.assertTrue(out["template_only"])
            self.assertFalse(out["paper_level_claim_allowed"])

    def test_validator_blocks_mutated_claim_flag(self):
        with tempfile.TemporaryDirectory(prefix="step167_test_") as d:
            tmp = Path(d)
            (tmp / "project").mkdir(parents=True, exist_ok=True)
            args = make_args(tmp)
            m = generate_runbook(args)
            manifest = Path(m["manifest_path"])
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["paper_level_claim_allowed"] = True
            manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                validate_manifest(manifest)


if __name__ == "__main__":
    unittest.main(verbosity=2)
