from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import actual_evidence_command_packet_step170 as runner
import validate_actual_evidence_command_packet_step170 as validator


class Step170ActualEvidenceCommandPacketTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="step170_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_sample_packet_and_validator_pass(self) -> None:
        out = self.tmp / "packet"
        args = type("Args", (), {
            "project_root": str(Path.cwd()),
            "mode": "sample",
            "readiness_manifest": "",
            "output_root": str(out),
            "run_id": "unit_test_run",
            "zero_loss_epsilon_sec": 0.0,
            "top_k_per_attempt": 8,
            "top_k_attention": 20,
            "file_format": "csv",
            "attention_mode": "sample_real_attention_smoke",
        })()
        manifest = runner.generate_packet(args)
        self.assertEqual(manifest["audit_status"], "PASS")
        self.assertFalse(manifest["command_execution_allowed"])
        validator.validate_manifest(Path(manifest["output_files"]["manifest"]), require_pass=True, require_dry_run=True)

    def test_validator_blocks_mutated_execution_flag(self) -> None:
        out = self.tmp / "mutated"
        args = type("Args", (), {
            "project_root": str(Path.cwd()),
            "mode": "sample",
            "readiness_manifest": "",
            "output_root": str(out),
            "run_id": "mutated_run",
            "zero_loss_epsilon_sec": 0.0,
            "top_k_per_attempt": 8,
            "top_k_attention": 20,
            "file_format": "csv",
            "attention_mode": "sample_real_attention_smoke",
        })()
        manifest = runner.generate_packet(args)
        path = Path(manifest["output_files"]["manifest"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["command_execution_allowed"] = True
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        with self.assertRaises(RuntimeError):
            validator.validate_manifest(path, require_pass=True, require_dry_run=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
