from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
GENERATOR = SCRIPT_DIR / "zero_loss_patent_claim_drafting_packet_step173.py"
VALIDATOR = SCRIPT_DIR / "validate_zero_loss_patent_claim_drafting_packet_step173.py"


class Step173ZeroLossPatentClaimDraftingPacketTest(unittest.TestCase):
    def run_cmd(self, args, expect_ok=True):
        proc = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if expect_ok and proc.returncode != 0:
            self.fail(proc.stdout)
        if not expect_ok and proc.returncode == 0:
            self.fail("command unexpectedly passed:\n" + proc.stdout)
        return proc

    def test_sample_packet_and_validator_pass(self):
        with tempfile.TemporaryDirectory(prefix="step173_test_") as tmp:
            out = Path(tmp) / "out"
            proc = self.run_cmd([sys.executable, str(GENERATOR), "--output-root", str(out)])
            self.assertIn("[OK] Step 173 zero-loss patent claim drafting packet generated", proc.stdout)
            manifest = out / "zero_loss_claim_drafting_manifest_step173.json"
            self.assertTrue(manifest.exists())
            proc2 = self.run_cmd([sys.executable, str(VALIDATOR), "--manifest", str(manifest)])
            self.assertIn("validation PASS", proc2.stdout)

    def test_validator_blocks_mutated_claim_flag(self):
        with tempfile.TemporaryDirectory(prefix="step173_test_") as tmp:
            out = Path(tmp) / "mutated"
            self.run_cmd([sys.executable, str(GENERATOR), "--output-root", str(out)])
            manifest = out / "zero_loss_claim_drafting_manifest_step173.json"
            m = json.loads(manifest.read_text(encoding="utf-8"))
            m["filing_ready_without_attorney_review"] = True
            manifest.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
            proc = self.run_cmd([sys.executable, str(VALIDATOR), "--manifest", str(manifest)], expect_ok=False)
            self.assertIn("guard flag mismatch", proc.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
