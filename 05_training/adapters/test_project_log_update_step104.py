from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=True)


def main() -> None:
    src = Path(__file__).resolve().with_name("update_project_log_step104.py")
    if not src.exists():
        # Support execution after files are copied into project adapters folder.
        src = Path("05_training/adapters/update_project_log_step104.py").resolve()
    if not src.exists():
        raise SystemExit("update_project_log_step104.py not found")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        adapters = root / "05_training" / "adapters"
        adapters.mkdir(parents=True)
        target = adapters / "update_project_log_step104.py"
        target.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

        project_log = root / "project_log.md"
        project_log.write_text(
            "# 🚌 UrbanBus RL Project - 작업 일지 (Project Journal)\n\n---\n\n## Old entry\n",
            encoding="utf-8",
        )

        cmd = [sys.executable, "-S", str(target)]
        first = run(cmd, cwd=root)
        second = run(cmd, cwd=root)

        text = project_log.read_text(encoding="utf-8")
        if text.count("STEP104_ROUTE_AWARE_V2_LOG_START") != 1:
            raise AssertionError("Step 104 marker duplicated or missing")
        if "Step 99-D~103 BIS API 통합 감사" not in text:
            raise AssertionError("Step 104 section title missing")
        if "causal_performance_claim_allowed = false" not in text:
            raise AssertionError("claim guard missing from project log")

        runbook = root / "05_training" / "adapters" / "route_aware_v2_pipeline_runbook_step104.md"
        if not runbook.exists():
            raise AssertionError("runbook not written")
        runbook_text = runbook.read_text(encoding="utf-8")
        if "kpi_by_window = 24" not in runbook_text:
            raise AssertionError("Step 103 expected count missing from runbook")
        if "causal_allowed = False" not in runbook_text:
            raise AssertionError("causal guard missing from runbook")

        manifest = root / "artifacts" / "daegu_bis_api_audit" / "project_log_runbook_step104" / "project_log_runbook_step104_manifest.json"
        if not manifest.exists():
            raise AssertionError("manifest not written")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("audit_status") != "PASS":
            raise AssertionError("manifest audit_status is not PASS")
        if payload["guardrails"]["causal_performance_claim_allowed"] is not False:
            raise AssertionError("manifest guardrail should be false")

        if "[OK] Step 104" not in first.stdout:
            raise AssertionError("first run did not print OK")
        if "[OK] Step 104" not in second.stdout:
            raise AssertionError("second run did not print OK")

    print("[OK] Step 104 project log/runbook update self-test PASS")


if __name__ == "__main__":
    main()
