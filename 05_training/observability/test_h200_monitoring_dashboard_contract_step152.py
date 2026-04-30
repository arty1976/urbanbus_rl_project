from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd, cwd: Path, expect_ok: bool = True):
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if expect_ok and proc.returncode != 0:
        print(proc.stdout)
        raise RuntimeError(f"command failed: {' '.join(map(str, cmd))}")
    if not expect_ok and proc.returncode == 0:
        print(proc.stdout)
        raise RuntimeError(f"command unexpectedly passed: {' '.join(map(str, cmd))}")
    return proc


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    generator = project_root / "05_training" / "observability" / "h200_experiment_observability_logger_step152.py"
    validator = project_root / "05_training" / "observability" / "validate_h200_monitoring_dashboard_contract_step152.py"

    if not generator.exists():
        raise RuntimeError(f"generator not found: {generator}")
    if not validator.exists():
        raise RuntimeError(f"validator not found: {validator}")

    with tempfile.TemporaryDirectory(prefix="step152_obs_test_") as td:
        temp_root = Path(td)
        output_root = temp_root / "observability_out"
        run([
            sys.executable,
            str(generator),
            "--project-root",
            str(project_root),
            "--output-root",
            str(output_root),
            "--run-id",
            "step152_test_run",
            "--condition-id",
            "A",
            "--reward-id",
            "R0",
            "--seed",
            "1",
            "--write-sample-events",
        ], cwd=project_root)

        manifest = output_root / "h200_monitoring_dashboard_contract_step152_manifest.json"
        run([sys.executable, str(validator), "--manifest", str(manifest)], cwd=project_root)

        bad_root = temp_root / "bad_train_allowed"
        shutil.copytree(output_root, bad_root)
        bad_manifest = bad_root / "h200_monitoring_dashboard_contract_step152_manifest.json"
        payload = load_json(bad_manifest)
        payload["train_allowed"] = True
        dump_json(bad_manifest, payload)
        run([sys.executable, str(validator), "--manifest", str(bad_manifest)], cwd=project_root, expect_ok=False)

        bad_root2 = temp_root / "bad_control_policy"
        shutil.copytree(output_root, bad_root2)
        bad_manifest2 = bad_root2 / "h200_monitoring_dashboard_contract_step152_manifest.json"
        payload2 = load_json(bad_manifest2)
        payload2["control_policy"] = "LIVE_MUTATION_ALLOWED_UNSAFE"
        dump_json(bad_manifest2, payload2)
        run([sys.executable, str(validator), "--manifest", str(bad_manifest2)], cwd=project_root, expect_ok=False)

        bad_root3 = temp_root / "bad_missing_kpi_event"
        shutil.copytree(output_root, bad_root3)
        bad_manifest3 = bad_root3 / "h200_monitoring_dashboard_contract_step152_manifest.json"
        payload3 = load_json(bad_manifest3)
        metrics_path = Path(payload3["outputs"]["metrics_events_jsonl"])
        replacement = []
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if '"event_type": "traffic_kpis"' not in line:
                replacement.append(line)
        metrics_path.write_text("\n".join(replacement) + "\n", encoding="utf-8")
        run([sys.executable, str(validator), "--manifest", str(bad_manifest3)], cwd=project_root, expect_ok=False)

    print("[OK] Step 152 H200 monitoring dashboard contract self-test PASS")


if __name__ == "__main__":
    main()
