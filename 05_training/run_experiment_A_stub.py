from pathlib import Path
import json

from mappo_runner import (
    DistributedConfig,
    FreezePhaseConfig,
    MAPPOExperimentRunner,
    RewardNormConfig,
    RunnerConfig,
    dump_json,
    load_json_any_encoding,
)


def build_or_load_contract(root: Path) -> Path:
    exp_path = root / "artifacts" / "experiment_A_v1" / "experiment_A_contract.json"
    baseline_path = root / "artifacts" / "baseline_v1" / "baseline_contract.json"

    if exp_path.exists():
        return exp_path

    if not baseline_path.exists():
        raise SystemExit(f"baseline contract not found: {baseline_path}")

    baseline = load_json_any_encoding(baseline_path)
    payload = {
        "artifact_version": "experiment_A_v1",
        "phase": "Phase-1",
        "condition_id": "A",
        "condition_name": "pure_mappo_baseline",
        "qwen_train": False,
        "qwen_inference": False,
        "linked_baseline_contract": str(baseline_path),
        "dataset_artifact": baseline.get("dataset_artifact"),
        "evaluation_horizon_minutes": baseline.get("evaluation_horizon_minutes"),
        "seeds": baseline.get("seeds", [1, 2, 3]),
        "time_bands": baseline.get("time_bands", ["peak", "offpeak", "night"]),
        "shared_kpis": baseline.get("shared_kpis", []),
        "fairness_constraints": baseline.get("fairness_constraints", {}),
        "baseline_status_snapshot": baseline.get("baselines", {}),
        "status": "contract_linked_not_trained",
    }
    dump_json(exp_path, payload)
    return exp_path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    exp_contract_path = build_or_load_contract(root)
    contract = load_json_any_encoding(exp_contract_path)

    runner = MAPPOExperimentRunner(
        RunnerConfig(
            root_dir=str(root),
            experiment_contract_path=str(exp_contract_path),
            run_root_dir=str(root / "artifacts" / "experiment_A_v1" / "runs"),
            simulator_adapter_path="",
            shared_policy=True,
            use_ctde=True,
            use_active_bus_mask=True,
            store_edge_index_in_rollout_buffer=False,
            grad_norm_clip=0.5,
            checkpoint_save_rng_state=True,
            qwen_trigger_rate_expected=0.0,
            reward_norm=RewardNormConfig(enabled=True, window_size=1000, clip_value=10.0),
            freeze_schedule=FreezePhaseConfig(
                phase_a_end_env_steps=20000,
                phase_b_end_env_steps=50000,
                phase_b_lr_scale=0.1,
            ),
            distributed=DistributedConfig(
                executor_backend="single_process",
                distributed_enabled=False,
            ),
        )
    )

    runner.validate_contract(contract)

    for seed in contract.get("seeds", [1, 2, 3]):
        runner.run_seed(contract, int(seed))

    print("[OK] experiment A stub prepared")
    print(f"[OK] contract : {exp_contract_path}")
    print(f"[OK] run root : {root / 'artifacts' / 'experiment_A_v1' / 'runs'}")


if __name__ == "__main__":
    main()
