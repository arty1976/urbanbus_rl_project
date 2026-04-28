from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List


TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))


PHASE2_12_KPIS = [
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
]

REWARD_VERSION = "mappo_reward_v1"
REWARD_CLAIM_BOUNDARY = "toy_causal_training_reward_contract_not_paper_performance_claim"
CHECKPOINT_ARTIFACT_VERSION = "toy_causal_mappo_smoke_checkpoint_v1_step88"
MANIFEST_ARTIFACT_VERSION = "toy_causal_mappo_smoke_v1_step88"

REQUIRED_CHECKPOINT_KEYS = [
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
]

REQUIRED_MANIFEST_KEYS = [
    "artifact_version",
    "output_root",
    "checkpoint_path",
    "condition_id",
    "seed",
    "num_agents",
    "steps",
    "epochs",
    "device",
    "reward_version",
    "reward_claim_boundary",
    "trained_model",
    "performance_claim_allowed",
    "causal_comparison_allowed",
    "smoke_training_only",
    "total_transitions",
    "loss_summary",
    "reward_summary",
    "reward_metric_keys",
    "training_trace",
    "created_at_utc",
    "note",
]


class ValidationError(RuntimeError):
    pass


def load_json(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise ValidationError(f"failed to read json: {path}")


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise ValidationError(f"{label} missing: {path}")
    if not path.is_file():
        raise ValidationError(f"{label} is not a file: {path}")


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def assert_required_keys(payload: Dict[str, Any], required: List[str], label: str) -> None:
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValidationError(f"{label} missing keys: {missing}")


def assert_false(value: Any, label: str) -> None:
    assert_true(value is False, f"{label} must be false, got {value!r}")


def assert_true_value(value: Any, label: str) -> None:
    assert_true(value is True, f"{label} must be true, got {value!r}")


def assert_finite(value: Any, label: str) -> float:
    try:
        numeric = float(value)
    except Exception as exc:
        raise ValidationError(f"{label} is not numeric: {value!r}") from exc
    assert_true(math.isfinite(numeric), f"{label} must be finite, got {numeric!r}")
    return numeric


def load_checkpoint(path: Path) -> Dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ValidationError(
            "torch is required to validate the smoke checkpoint. "
            "Install torch in 05_training\\.venv or run this validator on the training host."
        ) from exc

    # Step 88 checkpoint currently contains optimizer_state_dict and metadata dicts,
    # so weights_only=False is intentional here. The file is local project-generated.
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise ValidationError("checkpoint must be a dict")
    return obj


def validate_trace(trace_path: Path, expected_steps: int) -> Dict[str, Any]:
    require_file(trace_path, "training trace")

    rows: List[Dict[str, str]] = []
    with open(trace_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))

    assert_true(len(rows) > 0, "training_trace.csv must have at least one row")
    assert_true(len(rows) <= expected_steps, "training_trace.csv row count cannot exceed steps for one epoch smoke")

    required_columns = [
        "epoch",
        "step",
        "reward_total",
        "policy_loss",
        "value_loss",
        "entropy",
        "total_loss",
        "grad_norm",
        "passenger_service_rate",
        "passenger_wait_p95_seconds",
        "energy_proxy_per_passenger",
        "fleet_reduction_ratio",
    ]
    missing_columns = [c for c in required_columns if c not in rows[0]]
    if missing_columns:
        raise ValidationError(f"training_trace.csv missing columns: {missing_columns}")

    reward_values = []
    for idx, row in enumerate(rows):
        reward_values.append(assert_finite(row["reward_total"], f"trace[{idx}].reward_total"))
        assert_finite(row["total_loss"], f"trace[{idx}].total_loss")
        assert_finite(row["grad_norm"], f"trace[{idx}].grad_norm")

        for bounded in ["passenger_service_rate", "fleet_reduction_ratio"]:
            value = assert_finite(row[bounded], f"trace[{idx}].{bounded}")
            assert_true(0.0 <= value <= 1.0, f"trace[{idx}].{bounded} must be in [0, 1]")

        assert_true(
            assert_finite(row["passenger_wait_p95_seconds"], f"trace[{idx}].passenger_wait_p95_seconds") >= 0,
            "p95 wait must be non-negative",
        )
        assert_true(
            assert_finite(row["energy_proxy_per_passenger"], f"trace[{idx}].energy_proxy_per_passenger") >= 0,
            "energy per passenger must be non-negative",
        )

    return {
        "row_count": len(rows),
        "reward_total_min": float(min(reward_values)),
        "reward_total_max": float(max(reward_values)),
        "reward_total_mean": float(sum(reward_values) / len(reward_values)),
    }


def validate_checkpoint_and_manifest(
    checkpoint_path: Path,
    manifest_path: Path,
    trace_path: Path | None = None,
) -> Dict[str, Any]:
    require_file(checkpoint_path, "checkpoint")
    require_file(manifest_path, "manifest")

    checkpoint = load_checkpoint(checkpoint_path)
    manifest = load_json(manifest_path)

    assert_required_keys(checkpoint, REQUIRED_CHECKPOINT_KEYS, "checkpoint")
    assert_required_keys(manifest, REQUIRED_MANIFEST_KEYS, "manifest")

    assert_true(checkpoint["artifact_version"] == CHECKPOINT_ARTIFACT_VERSION, "checkpoint artifact_version mismatch")
    assert_true(manifest["artifact_version"] == MANIFEST_ARTIFACT_VERSION, "manifest artifact_version mismatch")

    for label, payload in [("checkpoint", checkpoint), ("manifest", manifest)]:
        assert_true(payload["reward_version"] == REWARD_VERSION, f"{label} reward_version mismatch")
        assert_true(payload["reward_claim_boundary"] == REWARD_CLAIM_BOUNDARY, f"{label} reward_claim_boundary mismatch")
        assert_false(payload["trained_model"], f"{label}.trained_model")
        assert_false(payload["performance_claim_allowed"], f"{label}.performance_claim_allowed")
        assert_true_value(payload["causal_comparison_allowed"], f"{label}.causal_comparison_allowed")
        assert_true_value(payload["smoke_training_only"], f"{label}.smoke_training_only")

    assert_true(
        "not a paper-level performance claim" in str(manifest.get("note", "")),
        "manifest note must prohibit paper-level performance claim",
    )

    for key in ["condition_id", "seed", "num_agents", "steps", "epochs"]:
        assert_true(checkpoint[key] == manifest[key], f"checkpoint/manifest mismatch for {key}")

    assert_true(int(checkpoint["action_dim"]) == 3, "checkpoint action_dim must be 3")
    assert_true(int(checkpoint["actor_obs_dim"]) > 0, "actor_obs_dim must be positive")
    assert_true(int(checkpoint["critic_obs_dim"]) > 0, "critic_obs_dim must be positive")
    assert_true(int(checkpoint["num_agents"]) >= 5, "num_agents lower bound failed")
    assert_true(int(checkpoint["num_agents"]) <= 10, "num_agents upper bound failed")

    assert_true(isinstance(checkpoint["model_state_dict"], dict), "model_state_dict must be dict")
    assert_true(len(checkpoint["model_state_dict"]) > 0, "model_state_dict must not be empty")
    assert_true(isinstance(checkpoint["optimizer_state_dict"], dict), "optimizer_state_dict must be dict")

    metric_keys = set(manifest["reward_metric_keys"])
    assert_true(metric_keys == set(PHASE2_12_KPIS), f"manifest reward_metric_keys mismatch: {sorted(metric_keys)}")

    total_transitions = int(manifest["total_transitions"])
    assert_true(total_transitions > 0, "manifest total_transitions must be positive")
    assert_true(
        int(checkpoint["training_summary"]["total_transitions"]) == total_transitions,
        "checkpoint training_summary total_transitions mismatch",
    )

    loss_summary = manifest["loss_summary"]
    reward_summary = manifest["reward_summary"]
    for key in ["policy_loss", "value_loss", "entropy", "total_loss", "grad_norm"]:
        assert_finite(loss_summary[key], f"manifest.loss_summary.{key}")
    for key in ["reward_total_mean", "reward_total_min", "reward_total_max", "reward_total_count"]:
        assert_finite(reward_summary[key], f"manifest.reward_summary.{key}")

    if trace_path is None:
        trace_path = Path(manifest["training_trace"])
        if not trace_path.is_absolute():
            trace_path = manifest_path.parent / trace_path

    trace_summary = validate_trace(trace_path, expected_steps=int(manifest["steps"]))

    return {
        "valid": True,
        "artifact_version": "toy_mappo_smoke_checkpoint_validation_v1_step89",
        "checkpoint_path": str(checkpoint_path),
        "manifest_path": str(manifest_path),
        "trace_path": str(trace_path),
        "condition_id": checkpoint["condition_id"],
        "seed": int(checkpoint["seed"]),
        "num_agents": int(checkpoint["num_agents"]),
        "steps": int(checkpoint["steps"]),
        "epochs": int(checkpoint["epochs"]),
        "reward_version": checkpoint["reward_version"],
        "reward_claim_boundary": checkpoint["reward_claim_boundary"],
        "trained_model": bool(checkpoint["trained_model"]),
        "performance_claim_allowed": bool(checkpoint["performance_claim_allowed"]),
        "causal_comparison_allowed": bool(checkpoint["causal_comparison_allowed"]),
        "smoke_training_only": bool(checkpoint["smoke_training_only"]),
        "total_transitions": total_transitions,
        "loss_summary": loss_summary,
        "reward_summary": reward_summary,
        "trace_summary": trace_summary,
        "reward_metric_keys": PHASE2_12_KPIS,
        "note": "validated toy causal MAPPO smoke checkpoint; not a paper-level performance claim",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--trace", default=None)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    manifest_path = Path(args.manifest)
    trace_path = Path(args.trace) if args.trace else None

    report = validate_checkpoint_and_manifest(
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
        trace_path=trace_path,
    )

    output_json = Path(args.output_json) if args.output_json else manifest_path.parent / "checkpoint_validation_report.json"
    dump_json(output_json, report)

    print("[OK] Step 89 toy MAPPO smoke checkpoint validation PASS")
    print(f"[OK] checkpoint     : {checkpoint_path}")
    print(f"[OK] manifest       : {manifest_path}")
    print(f"[OK] trace          : {report['trace_path']}")
    print(f"[OK] output_json    : {output_json}")
    print(f"[OK] reward_version : {report['reward_version']}")
    print(f"[OK] performance_claim_allowed: {report['performance_claim_allowed']}")
    print(f"[OK] smoke_training_only      : {report['smoke_training_only']}")


if __name__ == "__main__":
    main()
