from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "05_training" / "patent_evidence" / "patent_attorney_review_packet_index_step174.py"
VALIDATOR = ROOT / "05_training" / "patent_evidence" / "validate_patent_attorney_review_packet_index_step174.py"


class Step174PatentAttorneyReviewPacketIndexTest(unittest.TestCase):
    def test_sample_packet_and_validator_pass(self):
        with tempfile.TemporaryDirectory(prefix="step174_test_") as td:
            out = Path(td) / "out"
            subprocess.run([sys.executable, str(SCRIPT), "--project-root", str(ROOT), "--output-root", str(out)], check=True)
            manifest = out / "patent_attorney_review_manifest_step174.json"
            subprocess.run([sys.executable, str(VALIDATOR), "--manifest", str(manifest)], check=True)
            m = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(m["audit_status"], "PASS")
            self.assertFalse(m["filing_ready_without_attorney_review"])
            self.assertTrue(m["attorney_review_required"])

    def test_validator_blocks_mutated_claim_flag(self):
        with tempfile.TemporaryDirectory(prefix="step174_test_") as td:
            out = Path(td) / "mutated"
            subprocess.run([sys.executable, str(SCRIPT), "--project-root", str(ROOT), "--output-root", str(out)], check=True)
            manifest = out / "patent_attorney_review_manifest_step174.json"
            m = json.loads(manifest.read_text(encoding="utf-8"))
            m["filing_ready_without_attorney_review"] = True
            manifest.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
            proc = subprocess.run([sys.executable, str(VALIDATOR), "--manifest", str(manifest)], text=True, capture_output=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("guard flag mismatch", proc.stderr + proc.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
