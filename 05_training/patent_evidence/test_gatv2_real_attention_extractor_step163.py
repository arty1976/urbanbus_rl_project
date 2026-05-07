from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class Step163GATv2RealAttentionExtractorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[2]
        self.tool = self.root / "05_training" / "patent_evidence" / "gatv2_real_attention_extractor_step163.py"
        self.validator = self.root / "05_training" / "patent_evidence" / "validate_gatv2_real_attention_extractor_step163.py"
        self.tmpdir = Path(tempfile.mkdtemp(prefix="step163_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_sample_extractor_and_validator_pass(self) -> None:
        out = self.tmpdir / "sample"
        cmd = [
            sys.executable,
            str(self.tool),
            "--mode",
            "sample",
            "--output-root",
            str(out),
            "--condition-id",
            "A",
            "--seed",
            "1",
            "--file-format",
            "csv",
            "--top-k-per-attempt",
            "8",
            "--allow-fallback",
        ]
        subprocess.run(cmd, check=True)
        manifest = out / "gatv2_real_attention_extractor_manifest.json"
        self.assertTrue(manifest.exists())
        subprocess.run([sys.executable, str(self.validator), "--manifest", str(manifest)], check=True)
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertFalse(bool(payload["paper_level_claim_allowed"]))
        self.assertFalse(bool(payload["causal_performance_claim_allowed"]))
        self.assertGreater(payload["summary"]["snapshot_attention_row_count"], 0)
        self.assertGreater(payload["summary"]["attempt_attention_row_count"], 0)

    def test_validator_blocks_mutated_claim_flag(self) -> None:
        out = self.tmpdir / "mutated"
        subprocess.run(
            [
                sys.executable,
                str(self.tool),
                "--mode",
                "sample",
                "--output-root",
                str(out),
                "--file-format",
                "csv",
                "--allow-fallback",
            ],
            check=True,
        )
        manifest = out / "gatv2_real_attention_extractor_manifest.json"
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["paper_level_claim_allowed"] = True
        bad = out / "bad_manifest.json"
        bad.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        proc = subprocess.run([sys.executable, str(self.validator), "--manifest", str(bad)], text=True, capture_output=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("paper_level_claim_allowed", proc.stderr + proc.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
