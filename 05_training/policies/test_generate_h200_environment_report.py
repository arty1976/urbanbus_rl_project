from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "05_training" / "policies" / "generate_h200_environment_report.py"


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def run_generator(output_path: Path, *extra_args: str, expect_ok: bool = True) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(GENERATOR),
        "--project-root",
        str(ROOT),
        "--output-path",
        str(output_path),
        *extra_args,
    ]

    completed = subprocess.run(cmd, text=True, capture_output=True)

    if expect_ok and completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr)
        raise AssertionError(f"expected PASS but got returncode={completed.returncode}")

    if (not expect_ok) and completed.returncode == 0:
        print(completed.stdout)
        print(completed.stderr)
        raise AssertionError("expected BLOCKED but generator passed")

    return completed


def test_report_generation(tmp: Path) -> None:
    out = tmp / "environment_report.json"
    completed = run_generator(out, expect_ok=True)

    assert "[OK] Step 67" in completed.stdout
    assert out.exists()

    report = read_json(out)

    assert report["artifact_version"] == "h200_environment_report_v1_step67"
    assert report["report_status"] == "PASS"
    assert "created_at_utc" in report
    assert "hostname" in report
    assert "platform" in report
    assert "python" in report
    assert "torch" in report
    assert "nvidia_smi" in report
    assert "git" in report
    assert "environment_variables" in report
    assert "training_relevance" in report

    assert "python_version" in report["python"]
    assert "python_executable" in report["python"]

    assert "torch_importable" in report["torch"]
    assert "cuda_available" in report["torch"]
    assert "gpu_count" in report["torch"]
    assert "gpu_names" in report["torch"]

    assert "git_commit" in report["git"]
    assert "repo_dirty" in report["git"]
    assert "git_status_short" in report["git"]

    relevance = report["training_relevance"]
    assert relevance["qwen_train_expected"] is False
    assert relevance["qwen_inference_expected"] is False
    assert relevance["qwen_trigger_rate_expected"] == 0.0
    assert relevance["reward_version_expected"] == "mappo_reward_v1"
    assert relevance["energy_proxy_model_version_expected"] == "daegu_energy_proxy_v1"
    assert relevance["k_dist_kwh_per_m_expected"] == 0.0012
    assert relevance["k_acc_kwh_per_event_expected"] == 0.1800
    assert relevance["k_idle_kwh_per_sec_expected"] == 0.0080


def test_require_h200_name_blocks_on_non_h200_environment(tmp: Path) -> None:
    out = tmp / "h200_required_report.json"
    completed = run_generator(out, "--require-h200-name", expect_ok=False)

    report = read_json(out)
    assert report["report_status"] == "BLOCKED"
    assert "H200 GPU name required" in report["error"]


def test_missing_project_root_blocks(tmp: Path) -> None:
    out = tmp / "missing_project_report.json"
    missing = tmp / "missing_project_root"
    cmd = [
        sys.executable,
        str(GENERATOR),
        "--project-root",
        str(missing),
        "--output-path",
        str(out),
    ]

    completed = subprocess.run(cmd, text=True, capture_output=True)
    assert completed.returncode != 0

    report = read_json(out)
    assert report["report_status"] == "BLOCKED"
    assert "project root not found" in report["error"]


def main() -> None:
    if not GENERATOR.exists():
        raise AssertionError(f"generator not found: {GENERATOR}")

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        test_report_generation(tmp)
        test_require_h200_name_blocks_on_non_h200_environment(tmp)
        test_missing_project_root_blocks(tmp)

    print("[OK] Step 67 H200 environment report generator self-test PASS")


if __name__ == "__main__":
    main()
