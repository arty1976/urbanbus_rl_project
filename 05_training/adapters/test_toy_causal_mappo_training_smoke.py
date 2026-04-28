from __future__ import annotations

import csv
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = ROOT / "05_training"

PHASE2_12_KPIS = {
    "cv_headway",
    "avg_wait_seconds",
    "bunching_rate",
    "on_time_rate",
    "intervention_rate",
    "energy_proxy",
    "passenger_demand_generated",
    "passenger_served_count",
    "passenger_service_rate",
    "passenger_wait_p95_seconds",
    "energy_proxy_per_passenger",
    "fleet_reduction_ratio",
}

REWARD_VERSION = "mappo_reward_v1"
REWARD_CLAIM_BOUNDARY = "toy_causal_training_reward_contract_not_paper_performance_claim"


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def safe_rmtree(path: Path) -> None:
    if not path.exists():
        return

    def on_error(func, target, exc_info):
        try:
            import os
            os.chmod(target, 0o700)
            func(target)
        except Exception:
            pass

    for attempt in range(5):
        try:
            try:
                shutil.rmtree(path, onexc=on_error)
            except TypeError:
                shutil.rmtree(path, onerror=on_error)
            return
        except PermissionError:
            time.sleep(0.5 + 0.5 * attempt)

    raise PermissionError(f"could not remove locked directory: {path}")


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def run_cmd(cmd: list[str], cwd: Path) -> str:
    completed = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    if completed.returncode != 0:
        print("[STDOUT]")
        print(completed.stdout)
        print("[STDERR]")
        print(completed.stderr)
        raise RuntimeError(f"command failed: {' '.join(cmd)}")
    print(completed.stdout)
    return completed.stdout


def read_reward_sequence(path: Path) -> list[float]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [float(row["reward_total"]) for row in reader]


def test_dry_run_contract(py: Path) -> None:
    output_root = ROOT / "artifacts" / "phase2_toy_causal_mappo_smoke_selftest_dryrun"
    safe_rmtree(output_root)

    run_cmd(
        [
            str(py),
            str(TRAINING_DIR / "train_toy_causal_mappo_smoke.py"),
            "--output-root",
            str(output_root),
            "--dry-run-contract",
            "--clean",
        ],
        cwd=ROOT,
    )

    manifest_path = output_root / "training_manifest_dry_run.json"
    assert_true(manifest_path.exists(), "dry-run manifest missing")
    manifest = load_json(manifest_path)

    assert_true(manifest["trained_model"] is False, "dry-run trained_model must be false")
    assert_true(manifest["performance_claim_allowed"] is False, "dry-run performance claim must be false")
    assert_true(manifest["reward_version"] == REWARD_VERSION, "dry-run reward version mismatch")
    assert_true(manifest["reward_claim_boundary"] == REWARD_CLAIM_BOUNDARY, "dry-run claim boundary mismatch")


def test_smoke_training_outputs(py: Path) -> None:
    if not (has_module("torch") and has_module("numpy")):
        print("[SKIP] torch/numpy not installed; smoke training skipped after dry-run contract validation")
        return

    import torch

    output_root = ROOT / "artifacts" / "phase2_toy_causal_mappo_smoke_selftest"
    safe_rmtree(output_root)

    run_cmd(
        [
            str(py),
            str(TRAINING_DIR / "train_toy_causal_mappo_smoke.py"),
            "--output-root",
            str(output_root),
            "--condition-id",
            "A",
            "--seed",
            "88",
            "--num-agents",
            "8",
            "--steps",
            "4",
            "--epochs",
            "1",
            "--device",
            "cpu",
            "--clean",
        ],
        cwd=ROOT,
    )

    manifest_path = output_root / "training_manifest.json"
    checkpoint_path = output_root / "checkpoints" / "toy_mappo_smoke_checkpoint.pt"
    trace_path = output_root / "training_trace.csv"

    assert_true(manifest_path.exists(), "training_manifest.json missing")
    assert_true(checkpoint_path.exists(), "checkpoint missing")
    assert_true(trace_path.exists(), "training_trace.csv missing")

    manifest = load_json(manifest_path)
    assert_true(manifest["trained_model"] is False, "manifest trained_model must be false")
    assert_true(manifest["performance_claim_allowed"] is False, "manifest performance claim must be false")
    assert_true(manifest["smoke_training_only"] is True, "manifest smoke_training_only must be true")
    assert_true(manifest["causal_comparison_allowed"] is True, "manifest causal flag must be true")
    assert_true(manifest["reward_version"] == REWARD_VERSION, "manifest reward version mismatch")
    assert_true(manifest["reward_claim_boundary"] == REWARD_CLAIM_BOUNDARY, "manifest claim boundary mismatch")
    assert_true(set(manifest["reward_metric_keys"]) == PHASE2_12_KPIS, "manifest 12-KPI keys mismatch")
    assert_true(int(manifest["total_transitions"]) > 0, "total_transitions must be positive")
    assert_true(math.isfinite(float(manifest["loss_summary"]["total_loss"])), "total_loss must be finite")
    assert_true(math.isfinite(float(manifest["reward_summary"]["reward_total_mean"])), "reward mean must be finite")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    for key in [
        "artifact_version",
        "condition_id",
        "seed",
        "num_agents",
        "steps",
        "epochs",
        "hidden_dim",
        "actor_obs_dim",
        "critic_obs_dim",
        "action_dim",
        "reward_version",
        "reward_claim_boundary",
        "trained_model",
        "performance_claim_allowed",
        "causal_comparison_allowed",
        "smoke_training_only",
        "model_state_dict",
        "optimizer_state_dict",
        "training_summary",
        "created_at_utc",
    ]:
        assert_true(key in checkpoint, f"checkpoint missing key: {key}")

    assert_true(checkpoint["trained_model"] is False, "checkpoint trained_model must be false")
    assert_true(checkpoint["performance_claim_allowed"] is False, "checkpoint performance claim must be false")
    assert_true(checkpoint["smoke_training_only"] is True, "checkpoint smoke_training_only must be true")
    assert_true(checkpoint["reward_version"] == REWARD_VERSION, "checkpoint reward version mismatch")
    assert_true(checkpoint["reward_claim_boundary"] == REWARD_CLAIM_BOUNDARY, "checkpoint claim boundary mismatch")
    assert_true(int(checkpoint["action_dim"]) == 3, "checkpoint action_dim must be 3")
    assert_true(int(checkpoint["actor_obs_dim"]) > 0, "actor_obs_dim must be positive")
    assert_true(int(checkpoint["critic_obs_dim"]) > 0, "critic_obs_dim must be positive")
    assert_true(isinstance(checkpoint["model_state_dict"], dict), "model_state_dict must be dict")

    output_root_2 = ROOT / "artifacts" / "phase2_toy_causal_mappo_smoke_selftest_repeat"
    safe_rmtree(output_root_2)

    run_cmd(
        [
            str(py),
            str(TRAINING_DIR / "train_toy_causal_mappo_smoke.py"),
            "--output-root",
            str(output_root_2),
            "--condition-id",
            "A",
            "--seed",
            "88",
            "--num-agents",
            "8",
            "--steps",
            "4",
            "--epochs",
            "1",
            "--device",
            "cpu",
            "--clean",
        ],
        cwd=ROOT,
    )

    rewards_1 = read_reward_sequence(trace_path)
    rewards_2 = read_reward_sequence(output_root_2 / "training_trace.csv")
    assert_true(rewards_1 == rewards_2, "reward_total sequence must be deterministic for same seed")

    print("[OK] Step 88 toy causal MAPPO smoke training self-test PASS")


def main() -> None:
    py = Path(sys.executable)
    test_dry_run_contract(py)
    test_smoke_training_outputs(py)


if __name__ == "__main__":
    main()
