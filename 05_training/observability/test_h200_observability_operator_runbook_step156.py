from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd, cwd: Path, expect_ok: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    if expect_ok and proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"[FAIL] command failed: {' '.join(map(str, cmd))}")
    if (not expect_ok) and proc.returncode == 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"[FAIL] command unexpectedly passed: {' '.join(map(str, cmd))}")
    return proc


def copy_sources(src_root: Path, dst_root: Path) -> None:
    obs_src = src_root / "05_training" / "observability"
    obs_dst = dst_root / "05_training" / "observability"
    obs_dst.mkdir(parents=True, exist_ok=True)
    for name in (
        "h200_observability_operator_runbook_step156.py",
        "validate_h200_observability_operator_runbook_step156.py",
    ):
        shutil.copy2(obs_src / name, obs_dst / name)


def write_fake_component_manifests(root: Path) -> None:
    specs = [
        ("h200_monitoring_dashboard_contract_step152", "h200_monitoring_dashboard_contract_step152_manifest.json", "contract_status"),
        ("h200_tensorboard_jsonl_logger_step153", "h200_tensorboard_jsonl_logger_step153_manifest.json", "integration_status"),
        ("h200_nvidia_smi_gpu_monitor_step154", "h200_nvidia_smi_gpu_monitor_step154_manifest.json", "monitor_status"),
        ("h200_observability_dashboard_index_step155", "h200_observability_dashboard_index_step155_manifest.json", "index_status"),
    ]
    for folder, filename, status_key in specs:
        out_dir = root / "artifacts" / "observability" / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            status_key: f"FAKE_{status_key.upper()}_READY_MONITORING_ONLY_STILL_LOCKED",
            "dashboard_mode": "MONITORING_ONLY",
            "control_policy": "NO_LIVE_MUTATION_CONFIG_BASED_NEXT_RUN_ONLY",
            "train_allowed": False,
            "live_mutation_allowed": False,
        }
        (out_dir / filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    source_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="step156_selftest_") as td:
        root = Path(td)
        copy_sources(source_root, root)
        write_fake_component_manifests(root)

        gen = root / "05_training" / "observability" / "h200_observability_operator_runbook_step156.py"
        val = root / "05_training" / "observability" / "validate_h200_observability_operator_runbook_step156.py"
        manifest = root / "artifacts" / "observability" / "h200_observability_operator_runbook_step156" / "h200_observability_operator_runbook_step156_manifest.json"

        run([sys.executable, str(gen), "--project-root", str(root), "--output-root", "artifacts/observability/h200_observability_operator_runbook_step156"], cwd=root)
        run([sys.executable, str(val), "--manifest", str(manifest)], cwd=root)

        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload["component_status"]["present_count"] != 4:
            raise SystemExit("[FAIL] expected all fake components to be present")

        bad = json.loads(manifest.read_text(encoding="utf-8"))
        bad["train_allowed"] = True
        bad_manifest = manifest.parent / "bad_train_allowed_true.json"
        bad_manifest.write_text(json.dumps(bad, indent=2), encoding="utf-8")
        run([sys.executable, str(val), "--manifest", str(bad_manifest)], cwd=root, expect_ok=False)
        print("[OK] expected failure: train_allowed=true rejected")

        bad = json.loads(manifest.read_text(encoding="utf-8"))
        bad["live_mutation_allowed"] = True
        bad_manifest = manifest.parent / "bad_live_mutation_true.json"
        bad_manifest.write_text(json.dumps(bad, indent=2), encoding="utf-8")
        run([sys.executable, str(val), "--manifest", str(bad_manifest)], cwd=root, expect_ok=False)
        print("[OK] expected failure: live_mutation_allowed=true rejected")

        bad = json.loads(manifest.read_text(encoding="utf-8"))
        bad["h200_observation_commands"][0]["command"] = "python 05_training/train_mappo_actual.py --actual-execution-allowed true"
        bad_manifest = manifest.parent / "bad_forbidden_command.json"
        bad_manifest.write_text(json.dumps(bad, indent=2), encoding="utf-8")
        run([sys.executable, str(val), "--manifest", str(bad_manifest)], cwd=root, expect_ok=False)
        print("[OK] expected failure: forbidden command rejected")

        print("[OK] Step 156 H200 observability operator runbook self-test PASS")
        print(f"[OK] manifest: {manifest}")
        print(f"[OK] status  : {payload['runbook_status']}")
        print(f"[OK] control : {payload['control_policy']}")


if __name__ == "__main__":
    main()
