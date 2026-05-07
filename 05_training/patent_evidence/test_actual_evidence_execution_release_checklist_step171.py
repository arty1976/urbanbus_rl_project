from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


THIS = Path(__file__).resolve()
SCRIPT = THIS.with_name("actual_evidence_execution_release_checklist_step171.py")
VALIDATOR = THIS.with_name("validate_actual_evidence_execution_release_checklist_step171.py")


class Step171ActualEvidenceExecutionReleaseChecklistTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="step171_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_py(self, args, expect_ok=True):
        proc = subprocess.run([sys.executable, *map(str, args)], text=True, capture_output=True)
        if expect_ok and proc.returncode != 0:
            raise AssertionError(f"command failed\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
        if not expect_ok and proc.returncode == 0:
            raise AssertionError(f"command unexpectedly passed\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
        return proc

    def test_sample_locked_checklist_passes_validator(self):
        out = self.tmp / "locked"
        self.run_py([
            SCRIPT,
            "--mode", "sample",
            "--output-root", out,
            "--operator-approval-granted", "false",
            "--release-manifest-committed", "false",
        ])
        manifest = out / "actual_evidence_execution_release_checklist_manifest.json"
        self.run_py([VALIDATOR, "--manifest", manifest, "--expect-locked"])

        payload = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(payload["audit_status"], "BLOCKED")
        self.assertFalse(payload["actual_evidence_execution_allowed"])
        self.assertGreater(payload["hard_failures"], 0)

    def test_sample_released_checklist_passes_but_claims_remain_locked(self):
        out = self.tmp / "released"
        self.run_py([
            SCRIPT,
            "--mode", "sample",
            "--output-root", out,
            "--operator-approval-granted", "true",
            "--release-manifest-committed", "true",
            "--operator-name", "sample_operator",
            "--decision-note", "sample approval for unit test only",
        ])
        manifest = out / "actual_evidence_execution_release_checklist_manifest.json"
        self.run_py([VALIDATOR, "--manifest", manifest, "--expect-released"])

        payload = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(payload["audit_status"], "PASS")
        self.assertTrue(payload["actual_evidence_execution_allowed"])
        self.assertFalse(payload["paper_level_claim_allowed"])
        self.assertFalse(payload["causal_performance_claim_allowed"])
        self.assertFalse(payload["actual_operational_claim_allowed"])

    def test_validator_blocks_mutated_claim_flag(self):
        out = self.tmp / "mutated"
        self.run_py([
            SCRIPT,
            "--mode", "sample",
            "--output-root", out,
            "--operator-approval-granted", "false",
            "--release-manifest-committed", "false",
        ])
        manifest = out / "actual_evidence_execution_release_checklist_manifest.json"
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["paper_level_claim_allowed"] = True
        manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        self.run_py([VALIDATOR, "--manifest", manifest], expect_ok=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
