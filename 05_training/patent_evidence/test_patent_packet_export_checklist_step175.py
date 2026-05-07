from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from patent_packet_export_checklist_step175 import run
from validate_patent_packet_export_checklist_step175 import validate_manifest


class Step175PatentPacketExportChecklistTest(unittest.TestCase):
    def test_sample_checklist_and_validator_pass(self) -> None:
        with tempfile.TemporaryDirectory(prefix="step175_test_") as td:
            out = Path(td) / "out"
            run(out, mode="sample")
            manifest = out / "patent_packet_export_manifest_step175.json"
            m = validate_manifest(manifest)
            self.assertEqual(m["audit_status"], "PASS")
            self.assertFalse(m["export_zip_created"])
            self.assertTrue(m["attorney_review_required"])
            self.assertGreaterEqual(m["include_candidate_count"], 10)

    def test_validator_blocks_mutated_export_flag(self) -> None:
        with tempfile.TemporaryDirectory(prefix="step175_test_") as td:
            out = Path(td) / "mutated"
            run(out, mode="sample")
            manifest = out / "patent_packet_export_manifest_step175.json"
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["export_zip_created"] = True
            manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                validate_manifest(manifest)


if __name__ == "__main__":
    unittest.main(verbosity=2)
