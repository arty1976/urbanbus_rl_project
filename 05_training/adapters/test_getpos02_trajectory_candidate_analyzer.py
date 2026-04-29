from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    script = root / "05_training" / "adapters" / "analyze_getpos02_trajectory_candidates.py"

    if not script.exists():
        raise SystemExit(f"[FAIL] analyzer not found: {script}")

    run_dir = root / "artifacts" / "step99c3_getpos02_trajectory_analyzer_selftest"
    if run_dir.exists():
        shutil.rmtree(run_dir)

    (run_dir / "sample_001").mkdir(parents=True)
    (run_dir / "sample_002").mkdir(parents=True)

    manifest = (
        "sample_index,sample_tag,started_at,ended_at,audit_status,candidate_rows,normalized_rows,live_position_possible,sample_dir,report_json\n"
        "1,001,2026-04-29T09:00:00+09:00,2026-04-29T09:00:01+09:00,PASS,1,1,true,sample_001,report.json\n"
        "2,002,2026-04-29T09:05:00+09:00,2026-04-29T09:05:01+09:00,PASS,1,1,true,sample_002,report.json\n"
    )
    (run_dir / "step99c2_repeated_sampling_manifest.csv").write_text(manifest, encoding="utf-8")

    raw1 = {
        "header": {"success": True, "resultCode": "0000"},
        "body": {
            "items": [
                {
                    "routeId": "R1",
                    "routeNo": "T1",
                    "moveDir": "1",
                    "arTime": "090000",
                    "seq": 10,
                    "bsId": "S10",
                    "xPos": 128.0,
                    "yPos": 35.0,
                    "vhcNo2": "BUS1",
                    "busTCd2": "N",
                    "busTCd3": "N",
                }
            ]
        },
    }
    raw2 = {
        "header": {"success": True, "resultCode": "0000"},
        "body": {
            "items": [
                {
                    "routeId": "R1",
                    "routeNo": "T1",
                    "moveDir": "1",
                    "arTime": "090500",
                    "seq": 12,
                    "bsId": "S12",
                    "xPos": 128.01,
                    "yPos": 35.01,
                    "vhcNo2": "BUS1",
                    "busTCd2": "N",
                    "busTCd3": "N",
                }
            ]
        },
    }

    (run_dir / "sample_001" / "getpos02_sample_response_route_R1.raw").write_text(
        json.dumps(raw1, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "sample_002" / "getpos02_sample_response_route_R1.raw").write_text(
        json.dumps(raw2, ensure_ascii=False),
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        str(script),
        "--run-dir",
        str(run_dir),
    ]
    subprocess.run(cmd, check=True)

    report_path = run_dir / "getpos02_trajectory_candidate_report.json"
    if not report_path.exists():
        raise SystemExit("[FAIL] report json not created")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["total_timeseries_rows"] != 2:
        raise SystemExit(f"[FAIL] expected 2 rows, got {report['total_timeseries_rows']}")
    if report["trajectory_candidate_count"] != 1:
        raise SystemExit(f"[FAIL] expected 1 trajectory candidate, got {report['trajectory_candidate_count']}")

    print("[OK] Step 99-C3 trajectory candidate analyzer self-test PASS")


if __name__ == "__main__":
    main()
