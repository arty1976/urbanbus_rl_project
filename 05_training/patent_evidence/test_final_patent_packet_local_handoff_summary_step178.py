from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from final_patent_packet_local_handoff_summary_step178 import build_summary
from validate_final_patent_packet_local_handoff_summary_step178 import validate_manifest


class Step178FinalPatentPacketLocalHandoffSummaryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="step178_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_summary_and_validator_pass(self) -> None:
        out = self.tmp / "out"
        manifest = build_summary(out)
        manifest_path = Path(manifest["output_files"]["manifest_json"])
        validated = validate_manifest(manifest_path)
        self.assertEqual(validated["audit_status"], "PASS")
        self.assertEqual(validated["pipeline_step_count"], 19)
        self.assertFalse(validated["zip_creation_allowed"])
        self.assertFalse(validated["export_zip_created"])
        self.assertTrue(Path(validated["output_files"]["summary_md"]).exists())

    def test_validator_blocks_mutated_claim_or_zip_flag(self) -> None:
        out = self.tmp / "mutated"
        manifest = build_summary(out)
        manifest_path = Path(manifest["output_files"]["manifest_json"])
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["zip_creation_allowed"] = True
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        with self.assertRaises(RuntimeError):
            validate_manifest(manifest_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
