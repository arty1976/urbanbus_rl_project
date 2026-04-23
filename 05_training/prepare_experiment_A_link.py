import json
from pathlib import Path

import yaml


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "05_training" / "configs" / "gatv2_h200_A_baseline.yaml"

    if not cfg_path.exists():
        raise SystemExit(f"config not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    baseline_contract_path = root / cfg["paths"]["baseline_contract_path"].replace("./", "")
    out_dir = root / cfg["paths"]["experiment_contract_output_dir"].replace("./", "")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not baseline_contract_path.exists():
        raise SystemExit(f"baseline contract not found: {baseline_contract_path}")

    with open(baseline_contract_path, "r", encoding="utf-8-sig") as f:
        baseline = json.load(f)

    experiment_contract = {
        "artifact_version": "experiment_A_v1",
        "phase": "Phase-1",
        "condition_id": "A",
        "condition_name": "pure_mappo_baseline",
        "qwen_train": False,
        "qwen_inference": False,
        "linked_baseline_contract": str(baseline_contract_path),
        "dataset_artifact": baseline.get("dataset_artifact"),
        "evaluation_horizon_minutes": baseline.get("evaluation_horizon_minutes"),
        "seeds": baseline.get("seeds"),
        "time_bands": baseline.get("time_bands"),
        "shared_kpis": baseline.get("shared_kpis"),
        "fairness_constraints": baseline.get("fairness_constraints"),
        "baseline_status_snapshot": baseline.get("baselines"),
        "train_manifest": cfg["paths"]["train_manifest"],
        "val_manifest": cfg["paths"]["val_manifest"],
        "test_manifest": cfg["paths"]["test_manifest"],
        "status": "contract_linked_not_trained",
        "note": "A is linked to baseline contract but actual MAPPO training/evaluation runner is not yet executed in this step."
    }

    out_path = out_dir / "experiment_A_contract.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(experiment_contract, f, ensure_ascii=False, indent=2)

    print("[OK] wrote config:", cfg_path)
    print("[OK] wrote experiment contract:", out_path)


if __name__ == "__main__":
    main()
