from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    script = root / "05_training" / "adapters" / "inspect_getbasic02_master_snapshot.py"
    out_dir = root / "artifacts" / "daegu_bis_api_audit" / "getbasic02_master_snapshot_selftest"

    if not script.exists():
        raise SystemExit(f"script not found: {script}")

    cmd = [
        sys.executable,
        str(script),
        "--self-test",
        "--output-dir",
        str(out_dir),
    ]
    p = subprocess.run(cmd, cwd=root, text=True, capture_output=True)
    if p.returncode != 0:
        print(p.stdout)
        print(p.stderr)
        raise SystemExit(p.returncode)

    report_path = out_dir / "getbasic02_master_snapshot_audit_report.json"
    if not report_path.exists():
        raise SystemExit("report json missing")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["artifact_version"] == "daegu_bis_api_audit_step99b_getbasic02_v1"
    assert report["mode"] == "self_test_or_schema_only"
    assert report["total_normalized_rows"] >= 2
    assert report["classification_update_candidates"]["route_id"]["candidate_status"] == "observed_candidate"
    assert report["classification_update_candidates"]["route_no"]["candidate_status"] == "observed_candidate"
    assert report["guardrails"]["db_write_forbidden"] is True
    assert report["guardrails"]["tensor_db_overwrite_forbidden"] is True

    print("[OK] Step 99-B /getBasic02 self-test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
