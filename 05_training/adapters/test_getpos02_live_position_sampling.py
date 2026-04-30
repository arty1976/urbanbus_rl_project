from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve().parent
    project_root = here.parents[1]

    script = here / "inspect_getpos02_live_position_sampling.py"
    output_dir = project_root / "artifacts" / "daegu_bis_api_audit" / "getpos02_live_position_sampling_selftest"

    if not script.exists():
        raise SystemExit(f"[FAIL] audit script not found: {script}")

    cmd = [
        sys.executable,
        str(script),
        "--output-dir",
        str(output_dir),
        "--max-routes",
        "1",
    ]

    subprocess.run(cmd, check=True)

    report_json = output_dir / "getpos02_live_position_sampling_report.json"
    report_md = output_dir / "getpos02_live_position_sampling_report.md"

    if not report_json.exists():
        raise SystemExit(f"[FAIL] report json not found: {report_json}")
    if not report_md.exists():
        raise SystemExit(f"[FAIL] report md not found: {report_md}")

    print("[OK] Step 99-C /getPos02 self-test PASS")
    print(f"[OK] report json: {report_json}")
    print(f"[OK] report md  : {report_md}")


if __name__ == "__main__":
    main()
