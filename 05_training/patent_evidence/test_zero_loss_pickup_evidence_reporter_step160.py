from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class Step160ZeroLossPickupEvidenceReporterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.reporter = self.repo_root / "05_training" / "patent_evidence" / "zero_loss_pickup_evidence_reporter_step160.py"
        self.validator = self.repo_root / "05_training" / "patent_evidence" / "validate_zero_loss_pickup_evidence_bundle_step160.py"
        self.tmp = Path(tempfile.mkdtemp(prefix="step160_zero_loss_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_cmd(self, cmd):
        proc = subprocess.run(cmd, cwd=self.repo_root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise AssertionError(
                "command failed\n"
                f"cmd={cmd}\n"
                f"stdout={proc.stdout}\n"
                f"stderr={proc.stderr}"
            )
        return proc

    def test_sample_reporter_and_validator_pass(self) -> None:
        out_root = self.tmp / "bundle"
        self.run_cmd([
            sys.executable,
            str(self.reporter),
            "--output-root",
            str(out_root),
            "--make-sample-inputs",
        ])

        manifest = out_root / "zero_loss_evidence_manifest.json"
        self.assertTrue(manifest.exists())

        self.run_cmd([
            sys.executable,
            str(self.validator),
            "--manifest",
            str(manifest),
        ])

        payload = json.loads(manifest.read_text(encoding="utf-8"))
        summary = payload["summary"]
        self.assertEqual(summary["total_pickup_attempts"], 3)
        self.assertEqual(summary["zero_loss_success_count"], 2)
        self.assertFalse(summary["paper_level_claim_allowed"])
        self.assertFalse(summary["causal_performance_claim_allowed"])
        self.assertTrue((out_root / "patent_evidence_report.md").exists())
        self.assertTrue((out_root / "eta_delta_by_attempt.csv").exists())
        self.assertTrue((out_root / "top_attention_edges.csv").exists())

    def test_validator_blocks_mutated_claim_flag(self) -> None:
        out_root = self.tmp / "bundle_mutated"
        self.run_cmd([
            sys.executable,
            str(self.reporter),
            "--output-root",
            str(out_root),
            "--make-sample-inputs",
        ])
        manifest = out_root / "zero_loss_evidence_manifest.json"
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["paper_level_claim_allowed"] = True
        manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        proc = subprocess.run(
            [sys.executable, str(self.validator), "--manifest", str(manifest)],
            cwd=self.repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("paper_level_claim_allowed", proc.stderr + proc.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
