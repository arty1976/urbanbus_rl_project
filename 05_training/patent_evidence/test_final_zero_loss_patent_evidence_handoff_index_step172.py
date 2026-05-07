from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from final_zero_loss_patent_evidence_handoff_index_step172 import STEP_FILES, write_outputs
from validate_final_zero_loss_patent_evidence_handoff_index_step172 import validate_manifest


class Step172FinalHandoffIndexTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="step172_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_index_and_validator_pass_with_mock_files(self) -> None:
        for files in STEP_FILES.values():
            for rel in files:
                p = self.tmp / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("mock", encoding="utf-8")
        out = self.tmp / "artifacts" / "step172"
        m = write_outputs(self.tmp, out)
        self.assertEqual(m["audit_status"], "PASS")
        self.assertEqual(m["file_scan"]["missing_file_count"], 0)
        validated = validate_manifest(Path(m["output_files"]["manifest"]), strict_file_coverage=True)
        self.assertTrue(validated["handoff_index_ready"])
        self.assertFalse(validated["paper_level_claim_allowed"])

    def test_validator_blocks_mutated_claim_flag(self) -> None:
        out = self.tmp / "artifacts" / "step172_mutated"
        m = write_outputs(self.tmp, out)
        manifest_path = Path(m["output_files"]["manifest"])
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["paper_level_claim_allowed"] = True
        payload["guards"]["paper_level_claim_allowed"] = True
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        with self.assertRaises(RuntimeError):
            validate_manifest(manifest_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
