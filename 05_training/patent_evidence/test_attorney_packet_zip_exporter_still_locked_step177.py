from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from attorney_packet_zip_exporter_still_locked_step177 import run_exporter
from validate_attorney_packet_zip_exporter_still_locked_step177 import validate_manifest


class Step177AttorneyPacketZipExporterStillLockedTest(unittest.TestCase):
    def test_sample_locked_zip_exporter_and_validator_pass(self) -> None:
        with tempfile.TemporaryDirectory(prefix="step177_test_") as td:
            out = Path(td) / "out"
            manifest = run_exporter(project_root=Path(td), output_root=out, mode="sample")
            manifest_path = Path(manifest["output_files"]["zip_exporter_manifest_json"])
            validated = validate_manifest(manifest_path)
            self.assertEqual(validated["audit_status"], "PASS")
            self.assertTrue(validated["dry_run_only"])
            self.assertTrue(validated["zip_exporter_still_locked"])
            self.assertFalse(validated["zip_creation_allowed"])
            self.assertFalse(validated["export_zip_created"])
            self.assertGreater(validated["would_include_count"], 0)

    def test_validator_blocks_mutated_zip_creation_flag(self) -> None:
        with tempfile.TemporaryDirectory(prefix="step177_test_") as td:
            out = Path(td) / "mutated"
            manifest = run_exporter(project_root=Path(td), output_root=out, mode="sample")
            manifest_path = Path(manifest["output_files"]["zip_exporter_manifest_json"])
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["zip_creation_allowed"] = True
            manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                validate_manifest(manifest_path)

    def test_create_zip_request_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory(prefix="step177_test_") as td:
            out = Path(td) / "blocked"
            manifest = run_exporter(project_root=Path(td), output_root=out, mode="sample", create_zip=True)
            manifest_path = Path(manifest["output_files"]["zip_exporter_manifest_json"])
            with self.assertRaises(RuntimeError):
                validate_manifest(manifest_path)
            # But blocked manifests can be inspected explicitly.
            blocked = validate_manifest(manifest_path, require_pass=False)
            self.assertEqual(blocked["audit_status"], "BLOCKED")
            self.assertFalse(Path(blocked["planned_zip_path"]).exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
