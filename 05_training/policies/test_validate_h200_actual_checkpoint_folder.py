from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = ROOT / "05_training" / "policies" / "validate_h200_actual_checkpoint_folder.py"
DOC = ROOT / "05_training" / "policies" / "H200_actual_checkpoint_folder_contract.md"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def base_metadata() -> dict:
    return {
        "condition_id": "A",
        "qwen_train": False,
        "qwen_inference": False,
        "qwen_trigger_rate": 0.0,
        "reward_version": "mappo_reward_v1",
        "energy_proxy_model_version": "daegu_energy_proxy_v1",
        "k_dist_kwh_per_m": 0.0012,
        "k_acc_kwh_per_event": 0.1800,
        "k_idle_kwh_per_sec": 0.0080,
        "trained_model": True,
        "performance_claim_allowed": True,
    }


def make_valid_folder(run_root: Path) -> None:
    meta = base_metadata()
    best = run_root / "checkpoints" / "best_mappo.pt"

    write_json(
        run_root / "environment_report.json",
        {
            "artifact_version": "h200_environment_report_v1_step67",
            "report_status": "PASS",
            "git": {"git_commit": "abcdef1234567890", "repo_dirty": False},
            "torch": {"cuda_available": True, "gpu_count": 8, "gpu_names": ["NVIDIA H200"]},
        },
    )

    write_json(
        run_root / "training_config.json",
        {
            **meta,
            "seeds": [1, 2, 3],
            "device": "cuda",
            "training_budget": {"env_steps": 1000},
        },
    )

    write_text(
        run_root / "training_command.txt",
        "python 05_training/train_mappo_actual.py "
        "--condition-id A "
        "--qwen-train false "
        "--qwen-inference false "
        "--qwen-trigger-rate 0.0 "
        "--reward-version mappo_reward_v1 "
        "--energy-proxy-model-version daegu_energy_proxy_v1",
    )

    write_text(run_root / "git_commit.txt", "abcdef1234567890\n")

    write_text(run_root / "logs" / "train_stdout.log", "[OK] training stdout\n")
    write_text(run_root / "logs" / "train_stderr.log", "[OK] training stderr\n")
    write_text(run_root / "logs" / "train_metrics.jsonl", '{"step": 1, "reward": 0.0}\n')

    best.parent.mkdir(parents=True, exist_ok=True)
    best.write_bytes(b"fake checkpoint bytes for folder contract self-test")

    write_json(
        run_root / "checkpoints" / "checkpoint_manifest.json",
        {
            **meta,
            "best_checkpoint_path": str(best),
            "checkpoint_path": str(best),
            "last_checkpoint_path": str(run_root / "checkpoints" / "last_mappo.pt"),
        },
    )

    write_json(
        run_root / "validation" / "validate_mappo_checkpoint_actual_report.json",
        {
            **meta,
            "report_status": "PASS",
            "validation_status": "PASS",
            "mode": "actual",
            "checkpoint_path": str(best),
        },
    )

    write_json(
        run_root / "validation" / "h200_actual_checkpoint_preflight_report.json",
        {
            **meta,
            "preflight_status": "PASS",
            "status": "PASS",
            "checkpoint_path": str(best),
        },
    )

    write_text(
        run_root / "README_run_summary.md",
        "# H200 MAPPO run summary\n\n"
        "best_mappo.pt was produced.\n"
        "trained_model = true\n"
        "performance_claim_allowed = true\n"
        "This does not prove causal superiority.\n",
    )


def run_validator(run_root: Path, report_path: Path, expect_ok: bool) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(VALIDATOR),
        "--run-root",
        str(run_root),
        "--report-path",
        str(report_path),
    ]

    completed = subprocess.run(cmd, text=True, capture_output=True)

    if expect_ok and completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr)
        raise AssertionError(f"expected PASS but got returncode={completed.returncode}")

    if (not expect_ok) and completed.returncode == 0:
        print(completed.stdout)
        print(completed.stderr)
        raise AssertionError("expected BLOCKED but validator passed")

    return completed


def test_valid_folder_passes(tmp: Path) -> None:
    run_root = tmp / "valid_run"
    make_valid_folder(run_root)

    report_path = run_root / "validation" / "folder_report.json"
    completed = run_validator(run_root, report_path, expect_ok=True)

    assert "[OK] Step 68" in completed.stdout

    report = read_json(report_path)
    assert report["folder_contract_status"] == "PASS"
    assert report["claim_boundary"]["ready_for_step62_arrival_workflow"] is True
    assert report["claim_boundary"]["causal_performance_claim_allowed"] is False
    assert report["metadata_summary"]["condition_id"] == "A"


def test_missing_best_checkpoint_blocks(tmp: Path) -> None:
    run_root = tmp / "missing_best"
    make_valid_folder(run_root)
    (run_root / "checkpoints" / "best_mappo.pt").unlink()

    report_path = run_root / "validation" / "folder_report.json"
    completed = run_validator(run_root, report_path, expect_ok=False)

    assert "missing_files" in completed.stderr
    report = read_json(report_path)
    assert report["folder_contract_status"] == "BLOCKED"


def test_qwen_enabled_blocks(tmp: Path) -> None:
    run_root = tmp / "qwen_bad"
    make_valid_folder(run_root)

    cfg_path = run_root / "training_config.json"
    cfg = read_json(cfg_path)
    cfg["qwen_train"] = True
    write_json(cfg_path, cfg)

    report_path = run_root / "validation" / "folder_report.json"
    completed = run_validator(run_root, report_path, expect_ok=False)

    assert "qwen_train must be false" in completed.stderr


def test_preflight_blocked_blocks(tmp: Path) -> None:
    run_root = tmp / "preflight_bad"
    make_valid_folder(run_root)

    path = run_root / "validation" / "h200_actual_checkpoint_preflight_report.json"
    payload = read_json(path)
    payload["preflight_status"] = "BLOCKED"
    payload["status"] = "BLOCKED"
    write_json(path, payload)

    report_path = run_root / "validation" / "folder_report.json"
    completed = run_validator(run_root, report_path, expect_ok=False)

    assert "must be PASS-like" in completed.stderr


def test_bad_checkpoint_path_blocks(tmp: Path) -> None:
    run_root = tmp / "bad_path"
    make_valid_folder(run_root)

    path = run_root / "validation" / "validate_mappo_checkpoint_actual_report.json"
    payload = read_json(path)
    payload["checkpoint_path"] = str(run_root / "checkpoints" / "wrong.pt")
    write_json(path, payload)

    report_path = run_root / "validation" / "folder_report.json"
    completed = run_validator(run_root, report_path, expect_ok=False)

    assert "does not reference best_mappo.pt" in completed.stderr


def test_doc_contains_contract() -> None:
    if not DOC.exists():
        raise AssertionError(f"doc not found: {DOC}")

    text = DOC.read_text(encoding="utf-8-sig")
    required = [
        "Step 68",
        "H200 actual checkpoint folder contract",
        "best_mappo.pt",
        "environment_report.json",
        "training_config.json",
        "checkpoint_manifest.json",
        "validate_mappo_checkpoint_actual_report.json",
        "h200_actual_checkpoint_preflight_report.json",
        "trained_model = true",
        "performance_claim_allowed = true",
        "Causal performance claims require Phase 2 causal simulator evaluation.",
    ]
    missing = [x for x in required if x not in text]
    if missing:
        raise AssertionError(f"doc missing required phrases: {missing}")


def main() -> None:
    if not VALIDATOR.exists():
        raise AssertionError(f"validator not found: {VALIDATOR}")

    test_doc_contains_contract()

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        test_valid_folder_passes(tmp)
        test_missing_best_checkpoint_blocks(tmp)
        test_qwen_enabled_blocks(tmp)
        test_preflight_blocked_blocks(tmp)
        test_bad_checkpoint_path_blocks(tmp)

    print("[OK] Step 68 H200 actual checkpoint folder contract self-test PASS")


if __name__ == "__main__":
    main()
