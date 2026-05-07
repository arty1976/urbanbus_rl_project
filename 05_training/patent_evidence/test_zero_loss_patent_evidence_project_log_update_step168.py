from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from zero_loss_patent_evidence_project_log_update_step168 import run_update
from validate_zero_loss_patent_evidence_project_log_update_step168 import validate_manifest


class Step168ZeroLossPatentEvidenceProjectLogUpdateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="step168_test_"))
        self.project_root = self.tmp / "project"
        self.project_root.mkdir(parents=True)
        (self.project_root / "project_log.md").write_text(
            "# Project Log\n\nintro\n\n---\n## old section\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_project_log_update_and_validator_pass(self) -> None:
        manifest = run_update(
            project_root=self.project_root,
            output_root=Path("artifacts/patent_evidence/step168_test"),
            dry_run=False,
        )
        manifest_path = Path(manifest["output_files"]["manifest"])
        checked = validate_manifest(manifest_path)
        self.assertEqual(checked["audit_status"], "PASS")
        text = (self.project_root / "project_log.md").read_text(encoding="utf-8")
        self.assertIn("Step 160~167", text)
        self.assertIn("paper_level_claim_allowed = false", text)

    def test_validator_blocks_mutated_claim_flag(self) -> None:
        manifest = run_update(
            project_root=self.project_root,
            output_root=Path("artifacts/patent_evidence/step168_mutated"),
            dry_run=False,
        )
        manifest_path = Path(manifest["output_files"]["manifest"])
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        data["paper_level_claim_allowed"] = True
        manifest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        with self.assertRaises(RuntimeError):
            validate_manifest(manifest_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
