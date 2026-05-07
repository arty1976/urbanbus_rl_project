from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from attorney_packet_dry_run_exporter_step176 import DEFAULT_INCLUDE_CANDIDATES, run_exporter
from validate_attorney_packet_dry_run_exporter_step176 import validate_manifest


class Step176AttorneyPacketDryRunExporterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="step176_test_"))
        self.project = self.tmp / "project"
        self.project.mkdir(parents=True, exist_ok=True)
        for rel, required, category, desc in DEFAULT_INCLUDE_CANDIDATES:
            # Create all default candidates so the sample can PASS.
            path = self.project / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"sample file for {rel}\n", encoding="utf-8")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_sample_exporter_and_validator_pass(self) -> None:
        out = self.tmp / "out"
        m = run_exporter(self.project, out)
        self.assertEqual(m["audit_status"], "PASS")
        self.assertFalse(m["export_zip_created"])
        validate_manifest(out / "attorney_packet_dry_run_manifest_step176.json")

    def test_validator_blocks_mutated_guard_flag(self) -> None:
        out = self.tmp / "mutated"
        run_exporter(self.project, out)
        manifest = out / "attorney_packet_dry_run_manifest_step176.json"
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["export_zip_created"] = True
        manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        with self.assertRaises(RuntimeError):
            validate_manifest(manifest)


if __name__ == "__main__":
    unittest.main(verbosity=2)
