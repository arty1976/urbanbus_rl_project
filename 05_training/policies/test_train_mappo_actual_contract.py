from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "05_training" / "train_mappo_actual.py"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def make_contracts(tmp: Path) -> tuple[Path, Path]:
    baseline = tmp / "baseline_contract.json"
    experiment = tmp / "experiment_A_contract.json"

    write_json(
        baseline,
        {
            "artifact_version": "baseline_contract_v1",
            "evaluation_horizon_minutes": 30,
            "seeds": [1, 2, 3],
            "time_bands": ["peak", "offpeak", "night"],
            "shared_kpis": [
                "cv_headway",
                "avg_wait_seconds",
                "bunching_rate",
                "on_time_rate",
                "intervention_rate",
                "energy_proxy",
            ],
            "fairness_constraints": {
                "same_initial_state": True,
                "same_exogenous_events": True,
                "same_eval_window": True,
            },
        },
    )

    write_json(
        experiment,
        {
            "artifact_version": "experiment_A_v1",
            "condition_id": "A",
            "condition_name": "pure_mappo_baseline",
            "qwen_train": False,
            "qwen_inference": False,
            "linked_baseline_contract": str(baseline),
        },
    )

    return experiment, baseline


def base_cmd(tmp: Path, output_root: Path) -> list[str]:
    experiment, baseline = make_contracts(tmp)

    return [
        sys.executable,
        str(TRAIN),
        "--condition-id",
        "A",
        "--experiment-contract",
        str(experiment),
        "--baseline-contract",
        str(baseline),
        "--output-root",
        str(output_root),
        "--seeds",
        "1",
        "2",
        "3",
        "--device",
        "cuda",
        "--qwen-train",
        "false",
        "--qwen-inference",
        "false",
        "--qwen-trigger-rate",
        "0.0",
        "--reward-version",
        "mappo_reward_v1",
        "--energy-proxy-model-version",
        "daegu_energy_proxy_v1",
        "--k-dist-kwh-per-m",
        "0.0012",
        "--k-acc-kwh-per-event",
        "0.1800",
        "--k-idle-kwh-per-sec",
        "0.0080",
    ]


def run_cmd(cmd: list[str], expect_ok: bool) -> subprocess.CompletedProcess:
    completed = subprocess.run(cmd, text=True, capture_output=True)

    if expect_ok and completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr)
        raise AssertionError(f"expected PASS but got returncode={completed.returncode}")

    if (not expect_ok) and completed.returncode == 0:
        print(completed.stdout)
        print(completed.stderr)
        raise AssertionError("expected BLOCKED but command passed")

    return completed


def replace_arg(cmd: list[str], flag: str, value: str) -> list[str]:
    out = list(cmd)
    idx = out.index(flag)
    out[idx + 1] = value
    return out


def test_dry_run_contract_writes_scaffold(tmp: Path) -> None:
    output_root = tmp / "run_scaffold"
    cmd = base_cmd(tmp, output_root)
    cmd.append("--dry-run-contract")

    completed = run_cmd(cmd, expect_ok=True)
    assert "[OK] Step 69" in completed.stdout

    expected_files = [
        output_root / "training_config.json",
        output_root / "training_command.txt",
        output_root / "git_commit.txt",
        output_root / "logs" / "train_stdout.log",
        output_root / "logs" / "train_stderr.log",
        output_root / "logs" / "train_metrics.jsonl",
        output_root / "checkpoints" / "checkpoint_manifest.json",
        output_root / "validation" / "train_mappo_actual_contract_report.json",
        output_root / "README_run_summary.md",
    ]

    for path in expected_files:
        assert path.exists(), f"missing expected scaffold file: {path}"

    cfg = read_json(output_root / "training_config.json")
    assert cfg["condition_id"] == "A"
    assert cfg["qwen_train"] is False
    assert cfg["qwen_inference"] is False
    assert cfg["qwen_trigger_rate"] == 0.0
    assert cfg["reward_version"] == "mappo_reward_v1"
    assert cfg["energy_proxy_model_version"] == "daegu_energy_proxy_v1"
    assert cfg["trained_model"] is False
    assert cfg["performance_claim_allowed"] is False

    manifest = read_json(output_root / "checkpoints" / "checkpoint_manifest.json")
    assert manifest["checkpoint_status"] == "not_trained"
    assert manifest["best_checkpoint_exists"] is False
    assert manifest["trained_model"] is False
    assert manifest["performance_claim_allowed"] is False

    report = read_json(output_root / "validation" / "train_mappo_actual_contract_report.json")
    assert report["contract_status"] == "PASS"
    assert report["ready_for_step58_actual_preflight"] is False


def test_without_dry_run_contract_blocks(tmp: Path) -> None:
    output_root = tmp / "no_dry_run"
    cmd = base_cmd(tmp, output_root)

    completed = run_cmd(cmd, expect_ok=False)
    assert "not implemented in Step 69" in completed.stderr


def test_non_a_condition_blocks(tmp: Path) -> None:
    output_root = tmp / "bad_condition"
    cmd = base_cmd(tmp, output_root)
    cmd = replace_arg(cmd, "--condition-id", "B2")
    cmd.append("--dry-run-contract")

    completed = run_cmd(cmd, expect_ok=False)
    assert "condition_id must be A" in completed.stderr


def test_qwen_train_true_blocks(tmp: Path) -> None:
    output_root = tmp / "bad_qwen"
    cmd = base_cmd(tmp, output_root)
    cmd = replace_arg(cmd, "--qwen-train", "true")
    cmd.append("--dry-run-contract")

    completed = run_cmd(cmd, expect_ok=False)
    assert "qwen_train must be false" in completed.stderr


def test_energy_constant_mismatch_blocks(tmp: Path) -> None:
    output_root = tmp / "bad_energy"
    cmd = base_cmd(tmp, output_root)
    cmd = replace_arg(cmd, "--k-dist-kwh-per-m", "0.999")
    cmd.append("--dry-run-contract")

    completed = run_cmd(cmd, expect_ok=False)
    assert "k_dist_kwh_per_m mismatch" in completed.stderr


def test_seed_mismatch_blocks(tmp: Path) -> None:
    output_root = tmp / "bad_seed"
    cmd = base_cmd(tmp, output_root)
    idx = cmd.index("--seeds")
    # Replace the three values after --seeds.
    cmd[idx + 1: idx + 4] = ["1", "2", "4"]
    cmd.append("--dry-run-contract")

    completed = run_cmd(cmd, expect_ok=False)
    assert "seeds must be exactly 1,2,3" in completed.stderr


def main() -> None:
    if not TRAIN.exists():
        raise AssertionError(f"train scaffold not found: {TRAIN}")

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        test_dry_run_contract_writes_scaffold(tmp)
        test_without_dry_run_contract_blocks(tmp)
        test_non_a_condition_blocks(tmp)
        test_qwen_train_true_blocks(tmp)
        test_energy_constant_mismatch_blocks(tmp)
        test_seed_mismatch_blocks(tmp)

    print("[OK] Step 69 train_mappo_actual.py CLI contract self-test PASS")


if __name__ == "__main__":
    main()
