from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
WRITER = THIS_DIR / "route_aware_pickup_attempt_event_writer_step161.py"
VALIDATOR = THIS_DIR / "validate_route_aware_pickup_attempt_events_step161.py"


class Step161RouteAwarePickupAttemptEventWriterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="step161_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_cmd(self, args):
        return subprocess.run(
            [sys.executable, *[str(a) for a in args]],
            cwd=str(THIS_DIR.parents[2]),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

    def test_sample_writer_validator_and_step160_compat_paths_pass(self):
        out = self.tmp / "bundle"
        self.run_cmd([
            WRITER,
            "--mode", "sample",
            "--output-root", out,
            "--condition-id", "A",
            "--seed", "1",
            "--file-format", "csv",
            "--zero-loss-epsilon-sec", "0",
        ])
        manifest_path = out / "route_aware_pickup_attempt_event_writer_manifest.json"
        self.assertTrue(manifest_path.exists())
        self.run_cmd([VALIDATOR, "--manifest", manifest_path, "--require-step160-ready"])

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["audit_status"], "PASS")
        self.assertFalse(manifest["paper_level_claim_allowed"])
        self.assertFalse(manifest["causal_performance_claim_allowed"])
        self.assertGreaterEqual(manifest["summary"]["pickup_attempt_count"], 1)
        for key in ["pickup_attempt_events", "eta_counterfactual", "gatv2_attention", "run_manifest"]:
            self.assertTrue(Path(manifest["output_files"][key]).exists())

    def test_validator_blocks_mutated_claim_flag(self):
        out = self.tmp / "bundle_mutated"
        self.run_cmd([
            WRITER,
            "--mode", "sample",
            "--output-root", out,
            "--file-format", "csv",
        ])
        manifest_path = out / "route_aware_pickup_attempt_event_writer_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["paper_level_claim_allowed"] = True
        mutated = out / "mutated_manifest.json"
        mutated.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(VALIDATOR), "--manifest", str(mutated)],
            cwd=str(THIS_DIR.parents[2]),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("paper_level_claim_allowed", proc.stderr + proc.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
