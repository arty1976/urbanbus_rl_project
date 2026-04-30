#!/usr/bin/env python3
"""Self-test for Step 99-A /getBs02 route-stop sequence audit package."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    script = root / "05_training" / "adapters" / "inspect_getbs02_route_stop_sequence.py"
    out_dir = root / "artifacts" / "daegu_bis_api_audit_selftest"

    if out_dir.exists():
        shutil.rmtree(out_dir)

    cmd = f'"{sys.executable}" -S "{script}" --self-test --output-dir "{out_dir}"'
    rc = os.system(cmd)
    if rc != 0:
        raise SystemExit("[FAIL] Step 99-A self-test command failed")

    report_path = out_dir / "getbs02_route_stop_sequence_audit_report.json"
    md_path = out_dir / "getbs02_route_stop_sequence_audit_report.md"
    sample_path = out_dir / "getbs02_sample_response.json"

    for path in [report_path, md_path, sample_path]:
        if not path.exists():
            raise SystemExit(f"[FAIL] expected artifact missing: {path}")

    text = report_path.read_text(encoding="utf-8")
    required_fragments = [
        '"api_called": false',
        '"paper_level_claim_allowed": false',
        '"causal_performance_claim_allowed": false',
        '"fleet_reduction_claim_allowed": false',
        '"db_write_forbidden": true',
        '"bulk_api_collection_forbidden": true',
        '"ordered_stop_sequence_possible_any": true',
        '"route_id"',
        '"direction_id"',
        '"ordered_stop_sequence"',
        '"STOP_A"',
    ]
    missing = [x for x in required_fragments if x not in text]
    if missing:
        raise SystemExit(f"[FAIL] report missing expected fragments: {missing}")

    print("[OK] Step 99-A /getBs02 route-stop sequence audit self-test PASS")
    print(f"[OK] report: {report_path}")
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(int(code))
