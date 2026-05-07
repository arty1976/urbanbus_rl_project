from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTER = PROJECT_ROOT / "05_training" / "patent_evidence" / "patent_evidence_report_v2_step166.py"
VALIDATOR = PROJECT_ROOT / "05_training" / "patent_evidence" / "validate_patent_evidence_report_v2_step166.py"


class Step166PatentEvidenceReportV2Test(unittest.TestCase):
    def run_cmd(self, args: list[str], expect_ok: bool = True) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.setdefault("MPLBACKEND", "Agg")
        result = subprocess.run(args, cwd=PROJECT_ROOT, text=True, capture_output=True, timeout=180, env=env)
        if expect_ok and result.returncode != 0:
            self.fail(f"command failed: {args}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
        if not expect_ok and result.returncode == 0:
            self.fail(f"command unexpectedly passed: {args}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
        return result

    def test_sample_report_v2_and_validator_pass(self) -> None:
        with tempfile.TemporaryDirectory(prefix="step166_test_") as td:
            out = Path(td) / "sample"
            self.run_cmd([
                sys.executable,
                str(REPORTER),
                "--mode", "sample",
                "--output-root", str(out),
                "--top-k-edges", "8",
            ])
            manifest = out / "report_v2" / "patent_evidence_report_v2_manifest.json"
            self.assertTrue(manifest.exists())
            self.run_cmd([sys.executable, str(VALIDATOR), "--manifest", str(manifest)])
            with open(manifest, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload["audit_status"], "PASS")
            self.assertFalse(payload["paper_level_claim_allowed"])
            self.assertEqual(payload["summary"]["total_pickup_attempts"], 5)
            self.assertEqual(payload["summary"]["zero_loss_success_count"], 3)
            self.assertGreater(payload["summary"]["filtered_attention_row_count"], 0)

    def test_validator_blocks_mutated_claim_flag(self) -> None:
        with tempfile.TemporaryDirectory(prefix="step166_test_") as td:
            root = Path(td) / "mutated"
            root.mkdir(parents=True, exist_ok=True)
            output_files = {}
            for name, filename in {
                "patent_evidence_report_v2": "patent_evidence_report_v2.md",
                "patent_evidence_summary_v2": "patent_evidence_summary_v2.json",
                "zero_loss_attempt_attention_summary_v2": "zero_loss_attempt_attention_summary_v2.csv",
                "attention_mass_by_attempt_path_v2": "attention_mass_by_attempt_path_v2.csv",
                "attention_mass_by_success_path_v2": "attention_mass_by_success_path_v2.csv",
                "top_filtered_attention_edges_v2": "top_filtered_attention_edges_v2.csv",
            }.items():
                path = root / filename
                path.write_text("placeholder\n", encoding="utf-8")
                output_files[name] = str(path)

            manifest = root / "patent_evidence_report_v2_manifest.json"
            payload = {
                "audit_status": "PASS",
                "bundle_status": "PATENT_EVIDENCE_REPORT_V2_READY_NONCLAIM",
                "output_files": output_files,
                "summary": {
                    "total_pickup_attempts": 5,
                    "zero_loss_success_count": 3,
                    "zero_loss_success_rate": 0.6,
                    "filtered_attention_row_count": 30,
                    "exact_edge_overlap_count": 14,
                    "node_overlap_count": 16,
                    "fallback_attempt_count": 0,
                    "real_gatv2conv_attention_extracted_all": True,
                    "attempt_specific_route_path_filter_applied": True,
                },
                "row_counts": {"attempt_summary_rows": 5},
                "simulation_evidence_only": True,
                "actual_operational_claim_allowed": False,
                "paper_level_claim_allowed": True,
                "causal_performance_claim_allowed": False,
                "train_allowed": False,
            }
            with open(manifest, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            self.run_cmd([sys.executable, str(VALIDATOR), "--manifest", str(manifest)], expect_ok=False)

if __name__ == "__main__":
    unittest.main(verbosity=2)
