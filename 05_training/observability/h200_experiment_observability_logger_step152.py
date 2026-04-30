from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

ARTIFACT_VERSION = "h200_monitoring_dashboard_contract_step152_v1"
STEP_ID = "Step 152"
CONTRACT_STATUS = "OBSERVABILITY_CONTRACT_READY_MONITORING_ONLY_STILL_LOCKED"
DASHBOARD_MODE = "MONITORING_ONLY"
CONTROL_POLICY = "NO_LIVE_MUTATION_CONFIG_BASED_NEXT_RUN_ONLY"
NEXT_GATE = "WAITING_FOR_REAL_H200_STEP149_EXPECT_H200_RESULT"

REQUIRED_METRIC_GROUPS = [
    "run_identity",
    "training_metrics",
    "traffic_kpis",
    "safety_gates",
    "system_metrics",
    "claim_guards",
]

TENSORBOARD_SCALARS = {
    "training_metrics": [
        "train/policy_loss",
        "train/value_loss",
        "train/entropy",
        "train/kl_divergence",
        "train/grad_norm",
        "train/learning_rate",
        "train/reward_mean",
        "train/reward_std",
        "train/episode_length_mean",
    ],
    "traffic_kpis": [
        "kpi/avg_wait_seconds",
        "kpi/passenger_service_rate",
        "kpi/passenger_wait_p95_seconds",
        "kpi/bunching_rate",
        "kpi/on_time_rate",
        "kpi/energy_proxy_per_passenger",
        "kpi/fleet_reduction_ratio",
        "kpi/intervention_rate",
        "kpi/cv_headway",
    ],
    "safety_gates": [
        "safety/nan_detected",
        "safety/inf_detected",
        "safety/grad_norm_exceeded",
        "safety/reward_explosion_detected",
        "safety/loss_explosion_detected",
        "safety/abort_recommended",
    ],
    "system_metrics": [
        "system/gpu_utilization_pct",
        "system/gpu_memory_used_mb",
        "system/gpu_power_draw_w",
        "system/gpu_temperature_c",
        "system/ecc_error_count",
    ],
}

LOCK_VALUES = {
    "actual_execution_allowed": False,
    "actual_execution_released": False,
    "train_allowed": False,
    "operator_approval_recorded": False,
    "operator_approval_granted": False,
    "actual_results": False,
    "winner_selected": False,
    "trainable_reward_promoted": False,
    "paper_level_claim_allowed": False,
    "causal_performance_claim_allowed": False,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


def finite_or_none(value: Any) -> Any:
    if value is None:
        return None
    try:
        x = float(value)
    except Exception:
        return value
    if not math.isfinite(x):
        return None
    return x


def sample_metric_events(run_id: str, condition_id: str, reward_id: str, seed: int) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for step in range(3):
        events.append({
            "artifact_version": ARTIFACT_VERSION,
            "event_type": "training_metrics",
            "run_id": run_id,
            "condition_id": condition_id,
            "reward_id": reward_id,
            "seed": int(seed),
            "global_step": int(step),
            "wall_time_utc": utc_now(),
            "metrics": {
                "policy_loss": finite_or_none(0.20 - step * 0.01),
                "value_loss": finite_or_none(0.40 - step * 0.02),
                "entropy": finite_or_none(0.98 - step * 0.01),
                "kl_divergence": finite_or_none(0.001 + step * 0.0001),
                "grad_norm": finite_or_none(0.35 + step * 0.01),
                "learning_rate": finite_or_none(3e-4),
                "reward_mean": finite_or_none(-0.10 + step * 0.01),
                "reward_std": finite_or_none(0.02 + step * 0.001),
                "episode_length_mean": finite_or_none(30),
            },
        })
        events.append({
            "artifact_version": ARTIFACT_VERSION,
            "event_type": "traffic_kpis",
            "run_id": run_id,
            "condition_id": condition_id,
            "reward_id": reward_id,
            "seed": int(seed),
            "global_step": int(step),
            "wall_time_utc": utc_now(),
            "metrics": {
                "avg_wait_seconds": finite_or_none(240.0 - step),
                "passenger_service_rate": finite_or_none(0.95),
                "passenger_wait_p95_seconds": finite_or_none(540.0 - step * 2),
                "bunching_rate": finite_or_none(0.08),
                "on_time_rate": finite_or_none(0.82),
                "energy_proxy_per_passenger": finite_or_none(1.25),
                "fleet_reduction_ratio": finite_or_none(0.0),
                "intervention_rate": finite_or_none(0.0),
                "cv_headway": finite_or_none(0.21),
            },
        })
    return events


def write_jsonl(path: Path, events: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_gpu_csv(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp_utc",
        "gpu_index",
        "gpu_name",
        "gpu_utilization_pct",
        "gpu_memory_used_mb",
        "gpu_memory_total_mb",
        "gpu_power_draw_w",
        "gpu_temperature_c",
        "ecc_error_count",
        "source",
    ]
    rows = [
        {
            "timestamp_utc": utc_now(),
            "gpu_index": 0,
            "gpu_name": "H200_SAMPLE_PLACEHOLDER_NOT_ASSERTED",
            "gpu_utilization_pct": 0.0,
            "gpu_memory_used_mb": 0.0,
            "gpu_memory_total_mb": 0.0,
            "gpu_power_draw_w": 0.0,
            "gpu_temperature_c": 0.0,
            "ecc_error_count": 0,
            "source": "step152_schema_sample_not_h200_measurement",
        }
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def build_run_status(
    project_root: Path,
    run_id: str,
    condition_id: str,
    reward_id: str,
    seed: int,
) -> Dict[str, Any]:
    return {
        "artifact_version": ARTIFACT_VERSION,
        "step_id": STEP_ID,
        "run_id": run_id,
        "condition_id": condition_id,
        "reward_id": reward_id,
        "seed": int(seed),
        "project_root": str(project_root),
        "git_commit": "UNKNOWN_LOCAL_STEP152_SAMPLE",
        "started_at_utc": utc_now(),
        "dashboard_mode": DASHBOARD_MODE,
        "control_policy": CONTROL_POLICY,
        "next_gate": NEXT_GATE,
        **LOCK_VALUES,
        "trained_model": False,
        "checkpoint_validation_passed": False,
        "safety_gates": {
            "nan_detected": False,
            "inf_detected": False,
            "grad_norm_exceeded": False,
            "reward_explosion_detected": False,
            "loss_explosion_detected": False,
            "abort_recommended": False,
        },
    }


def build_tensorboard_contract() -> Dict[str, Any]:
    return {
        "artifact_version": ARTIFACT_VERSION,
        "step_id": STEP_ID,
        "writer": "torch.utils.tensorboard.SummaryWriter",
        "tensorboard_required": False,
        "tensorboard_recommended": True,
        "logdir_pattern": "artifacts/tensorboard/{condition_id}/{reward_id}/seed_{seed:03d}/{run_id}",
        "scalars": TENSORBOARD_SCALARS,
        "notes": [
            "Step 152 does not require TensorBoard to be installed for the contract self-test.",
            "Actual H200 training should write these scalars when TensorBoard is available.",
        ],
    }


def validate_manifest_payload(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if payload.get("contract_status") != CONTRACT_STATUS:
        errors.append("contract_status mismatch")
    if payload.get("dashboard_mode") != DASHBOARD_MODE:
        errors.append("dashboard_mode mismatch")
    if payload.get("control_policy") != CONTROL_POLICY:
        errors.append("control_policy mismatch")
    groups = payload.get("required_metric_groups", [])
    if groups != REQUIRED_METRIC_GROUPS:
        errors.append("required_metric_groups mismatch")
    for key, expected in LOCK_VALUES.items():
        if payload.get(key) is not expected:
            errors.append(f"lock value mismatch: {key}")
    if payload.get("next_gate") != NEXT_GATE:
        errors.append("next_gate mismatch")
    return errors


def generate_outputs(args: argparse.Namespace) -> Dict[str, Any]:
    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    run_status_path = output_root / "run_status_step152_sample.json"
    metrics_jsonl_path = output_root / "metrics_events_step152_sample.jsonl"
    gpu_csv_path = output_root / "gpu_metrics_step152_sample.csv"
    tensorboard_contract_path = output_root / "tensorboard_scalar_contract_step152.json"
    manifest_path = output_root / "h200_monitoring_dashboard_contract_step152_manifest.json"

    run_status = build_run_status(
        project_root=project_root,
        run_id=args.run_id,
        condition_id=args.condition_id,
        reward_id=args.reward_id,
        seed=args.seed,
    )
    dump_json(run_status_path, run_status)

    event_count = 0
    if args.write_sample_events:
        event_count = write_jsonl(
            metrics_jsonl_path,
            sample_metric_events(
                run_id=args.run_id,
                condition_id=args.condition_id,
                reward_id=args.reward_id,
                seed=args.seed,
            ),
        )
    else:
        metrics_jsonl_path.write_text("", encoding="utf-8")

    gpu_row_count = write_gpu_csv(gpu_csv_path)
    dump_json(tensorboard_contract_path, build_tensorboard_contract())

    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "step_id": STEP_ID,
        "contract_status": CONTRACT_STATUS,
        "dashboard_mode": DASHBOARD_MODE,
        "control_policy": CONTROL_POLICY,
        "next_gate": NEXT_GATE,
        "project_root": str(project_root),
        "output_root": str(output_root),
        "run_id": args.run_id,
        "condition_id": args.condition_id,
        "reward_id": args.reward_id,
        "seed": int(args.seed),
        "required_metric_groups": REQUIRED_METRIC_GROUPS,
        "tensorboard_scalar_groups": TENSORBOARD_SCALARS,
        **LOCK_VALUES,
        "outputs": {
            "run_status": str(run_status_path),
            "metrics_events_jsonl": str(metrics_jsonl_path),
            "gpu_metrics_csv": str(gpu_csv_path),
            "tensorboard_scalar_contract": str(tensorboard_contract_path),
            "manifest": str(manifest_path),
        },
        "row_counts": {
            "metrics_events_jsonl": int(event_count),
            "gpu_metrics_csv": int(gpu_row_count),
        },
        "hard_failures": 0,
        "warnings": [],
        "created_at_utc": utc_now(),
        "notes": [
            "This is a monitoring-only observability contract.",
            "Live mutation of experiment variables is forbidden.",
            "Parameter changes must be applied to next-run config only.",
        ],
    }

    errors = validate_manifest_payload(manifest)
    if errors:
        manifest["hard_failures"] = len(errors)
        manifest["warnings"] = errors

    dump_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 152 H200 monitoring dashboard contract generator")
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--output-root",
        default="artifacts/observability/h200_monitoring_dashboard_contract_step152",
    )
    parser.add_argument("--run-id", default="step152_selftest_run")
    parser.add_argument("--condition-id", default="A")
    parser.add_argument("--reward-id", default="R0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--write-sample-events", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = generate_outputs(args)
    if int(manifest.get("hard_failures", 0)) != 0:
        print("[FAIL] Step 152 manifest generated with hard failures")
        print(json.dumps(manifest.get("warnings", []), ensure_ascii=False, indent=2))
        raise SystemExit(1)

    print("[OK] Step 152 H200 monitoring dashboard contract generated")
    print(f"[OK] contract_status : {manifest['contract_status']}")
    print(f"[OK] dashboard_mode  : {manifest['dashboard_mode']}")
    print(f"[OK] control_policy  : {manifest['control_policy']}")
    print(f"[OK] train_allowed   : {manifest['train_allowed']}")
    print(f"[OK] manifest        : {manifest['outputs']['manifest']}")


if __name__ == "__main__":
    main()
