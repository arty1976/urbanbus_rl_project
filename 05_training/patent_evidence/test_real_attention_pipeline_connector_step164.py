from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().with_name("real_attention_pipeline_connector_step164.py")
VALIDATOR = Path(__file__).resolve().with_name("validate_real_attention_pipeline_connector_step164.py")


class Step164RealAttentionPipelineConnectorTest(unittest.TestCase):
    def test_sample_connector_and_validator_pass(self) -> None:
        with tempfile.TemporaryDirectory(prefix="step164_test_") as td:
            out = Path(td) / "sample"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--make-sample-inputs",
                    "--output-root",
                    str(out),
                    "--file-format",
                    "csv",
                    "--top-k-per-attempt",
                    "12",
                    "--require-real-attention",
                ],
                check=True,
            )
            manifest = out / "real_attention_pipeline_connector_manifest.json"
            subprocess.run(
                [
                    sys.executable,
                    str(VALIDATOR),
                    "--manifest",
                    str(manifest),
                    "--require-real-attention",
                ],
                check=True,
            )
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["audit_status"], "PASS")
            self.assertTrue(payload["real_attention_connected"])
            self.assertTrue(payload["proxy_attention_replaced"])
            self.assertEqual(payload["summary"]["pickup_attempt_count"], 3)
            self.assertEqual(payload["summary"]["real_attention_row_count"], 36)
            self.assertTrue(payload["summary"]["real_gatv2conv_attention_extracted_all"])

    def test_validator_blocks_mutated_claim_flag(self) -> None:
        with tempfile.TemporaryDirectory(prefix="step164_test_") as td:
            out = Path(td) / "mutated"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--make-sample-inputs",
                    "--output-root",
                    str(out),
                    "--file-format",
                    "csv",
                    "--require-real-attention",
                ],
                check=True,
            )
            manifest = out / "real_attention_pipeline_connector_manifest.json"
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["paper_level_claim_allowed"] = True
            manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            with self.assertRaises(subprocess.CalledProcessError):
                subprocess.run(
                    [sys.executable, str(VALIDATOR), "--manifest", str(manifest), "--require-real-attention"],
                    check=True,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
