from __future__ import annotations

import json
import shutil
import uuid
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    script = root / "05_training" / "adapters" / "inspect_getrealtime02_eta_sampling.py"
    if not script.exists():
        raise SystemExit(f"[FAIL] inspector not found: {script}")

    out_dir = root / "artifacts" / "daegu_bis_api_audit" / f"getrealtime02_eta_sampling_selftest_{uuid.uuid4().hex[:8]}"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    cmd = [
        sys.executable,
        str(script),
        "--self-test",
        "--output-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)

    report_path = out_dir / "getrealtime02_eta_sampling_report.json"
    normalized_path = out_dir / "getrealtime02_eta_normalized.csv"
    headway_path = out_dir / "getrealtime02_headway_candidate.csv"

    if not report_path.exists():
        raise SystemExit("[FAIL] report json was not created")
    if not normalized_path.exists():
        raise SystemExit("[FAIL] normalized csv was not created")
    if not headway_path.exists():
        raise SystemExit("[FAIL] headway candidate csv was not created")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("audit_status") != "PASS":
        raise SystemExit(f"[FAIL] expected PASS, got {report.get('audit_status')}")
    if int(report.get("total_normalized_rows", 0)) < 2:
        raise SystemExit("[FAIL] expected at least two normalized rows")
    if int(report.get("headway_candidate_count", 0)) < 1:
        raise SystemExit("[FAIL] expected at least one ETA headway candidate")
    if report.get("classification_update_candidates", {}).get("actual_headway") != "not_observed":
        raise SystemExit("[FAIL] actual_headway must remain not_observed")

    print("[OK] Step 99-D getRealtime02 ETA sampling self-test PASS")


if __name__ == "__main__":
    main()

