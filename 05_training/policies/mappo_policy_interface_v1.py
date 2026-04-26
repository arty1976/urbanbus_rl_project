"""
mappo_policy_interface_v1.py
============================

Actual MAPPO inference interface contract for UrbanBus RL.

MAPPO means Multi-Agent Proximal Policy Optimization.
GAT means Graph Attention Network.
KPI means Key Performance Indicator.
Qwen remains disabled for A/A90/A80/A70 in the current phase.

Purpose
-------
This module defines the input/output contract that a future trained MAPPO
checkpoint must satisfy.

This file does NOT implement neural network inference yet.
It creates a safe boundary for:

checkpoint metadata
-> observation dict
-> action dict
-> rollout row provenance

Why this exists
---------------
Step 34 proved that fake-checkpoint MAPPO boundary mode can pass the pipeline.
Step 35 defines what a real MAPPO checkpoint must expose when H200 training
produces an actual model.

Core rule
---------
No placeholder fallback is allowed.
If actual inference is requested but the checkpoint contract is invalid,
execution must fail clearly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional
import json


INTERFACE_VERSION = "mappo_policy_interface_v1"

A_FAMILY_CONDITIONS = {"A", "A90", "A80", "A70"}

REQUIRED_CHECKPOINT_METADATA_KEYS = [
    "artifact_type",
    "interface_version",
    "policy_kind",
    "trained_model",
    "action_space_version",
    "observation_space_version",
]

EXPECTED_CHECKPOINT_METADATA = {
    "artifact_type": "urbanbus_mappo_checkpoint",
    "interface_version": INTERFACE_VERSION,
    "policy_kind": "mappo",
    "action_space_version": "bus_control_action_v1",
    "observation_space_version": "urbanbus_observation_v1",
}

REQUIRED_OBSERVATION_KEYS = [
    "condition_id",
    "window_id",
    "state_ts",
    "time_band",
    "seed",
    "baseline_bus_count",
    "active_bus_count",
    "passenger_demand_generated",
    "headway_mean_seconds",
    "headway_std_seconds",
    "bunching_rate",
    "on_time_rate",
    "avg_wait_seconds",
    "passenger_wait_p95_seconds",
    "energy_proxy_per_passenger",
]

REQUIRED_ACTION_KEYS = [
    "action_version",
    "condition_id",
    "policy_source",
    "source_mode",
    "qwen_trigger_rate",
    "dispatch_delta",
    "hold_seconds",
    "skip_stop_flag",
    "target_headway_ratio",
    "active_bus_count",
    "policy_debug",
]


@dataclass(frozen=True)
class MappoCheckpointMetadata:
    artifact_type: str
    interface_version: str
    policy_kind: str
    trained_model: bool
    action_space_version: str
    observation_space_version: str
    checkpoint_path: str
    train_run_id: Optional[str] = None
    git_commit: Optional[str] = None
    reward_version: Optional[str] = None
    rollout_schema_version: Optional[str] = None
    notes: Optional[str] = None


@dataclass(frozen=True)
class UrbanBusObservation:
    condition_id: str
    window_id: str
    state_ts: str
    time_band: str
    seed: int
    baseline_bus_count: float
    active_bus_count: float
    passenger_demand_generated: float
    headway_mean_seconds: float
    headway_std_seconds: float
    bunching_rate: float
    on_time_rate: float
    avg_wait_seconds: float
    passenger_wait_p95_seconds: float
    energy_proxy_per_passenger: float


@dataclass(frozen=True)
class BusControlAction:
    action_version: str
    condition_id: str
    policy_source: str
    source_mode: str
    qwen_trigger_rate: float
    dispatch_delta: float
    hold_seconds: float
    skip_stop_flag: bool
    target_headway_ratio: float
    active_bus_count: float
    policy_debug: Dict[str, Any]


def normalize_condition_id(condition_id: Any) -> str:
    return str(condition_id).strip().upper()


def as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except Exception:
        return int(default)


def clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def validate_checkpoint_metadata_dict(payload: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []

    for key in REQUIRED_CHECKPOINT_METADATA_KEYS:
        if key not in payload:
            errors.append(f"missing checkpoint metadata key: {key}")

    for key, expected in EXPECTED_CHECKPOINT_METADATA.items():
        if payload.get(key) != expected:
            errors.append(
                f"checkpoint metadata mismatch for {key}: expected={expected}, got={payload.get(key)}"
            )

    trained_model = payload.get("trained_model")
    if trained_model is not True:
        errors.append("trained_model must be True for actual MAPPO inference")

    return errors


def load_checkpoint_metadata(checkpoint_path: str | Path) -> MappoCheckpointMetadata:
    """
    Load a lightweight JSON metadata sidecar or JSON checkpoint.

    Supported forms:
    1. checkpoint_path itself is JSON text.
    2. checkpoint_path + '.metadata.json' exists.

    This does not load torch weights.
    Actual torch loading should be implemented later in the real inference adapter.
    """
    path = Path(checkpoint_path)

    if not path.exists():
        raise FileNotFoundError(f"MAPPO checkpoint does not exist: {path}")

    sidecar = Path(str(path) + ".metadata.json")

    if sidecar.exists():
        metadata_path = sidecar
    else:
        metadata_path = path

    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(
            f"checkpoint metadata is not readable JSON: {metadata_path}. "
            "Provide a .metadata.json sidecar for binary torch checkpoints."
        ) from exc

    errors = validate_checkpoint_metadata_dict(payload)

    if errors:
        raise ValueError("invalid MAPPO checkpoint metadata: " + "; ".join(errors))

    return MappoCheckpointMetadata(
        artifact_type=str(payload["artifact_type"]),
        interface_version=str(payload["interface_version"]),
        policy_kind=str(payload["policy_kind"]),
        trained_model=bool(payload["trained_model"]),
        action_space_version=str(payload["action_space_version"]),
        observation_space_version=str(payload["observation_space_version"]),
        checkpoint_path=str(path),
        train_run_id=payload.get("train_run_id"),
        git_commit=payload.get("git_commit"),
        reward_version=payload.get("reward_version"),
        rollout_schema_version=payload.get("rollout_schema_version"),
        notes=payload.get("notes"),
    )


def validate_observation_dict(row: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []

    for key in REQUIRED_OBSERVATION_KEYS:
        if key not in row:
            errors.append(f"missing observation key: {key}")

    condition_id = normalize_condition_id(row.get("condition_id"))

    if condition_id not in A_FAMILY_CONDITIONS:
        errors.append(f"condition_id must be one of {sorted(A_FAMILY_CONDITIONS)}, got {condition_id}")

    qwen_trigger_rate = as_float(row.get("qwen_trigger_rate", 0.0), default=0.0)
    if abs(qwen_trigger_rate) > 1e-12:
        errors.append("qwen_trigger_rate must remain 0.0 for A-family pure MAPPO")

    baseline_bus_count = as_float(row.get("baseline_bus_count"), default=0.0)
    active_bus_count = as_float(row.get("active_bus_count"), default=0.0)

    if baseline_bus_count <= 0:
        errors.append("baseline_bus_count must be positive")

    if active_bus_count < 0:
        errors.append("active_bus_count must be non-negative")

    passenger_demand_generated = as_float(row.get("passenger_demand_generated"), default=0.0)

    if passenger_demand_generated < 0:
        errors.append("passenger_demand_generated must be non-negative")

    return errors


def observation_from_rollout_row(row: Mapping[str, Any]) -> UrbanBusObservation:
    errors = validate_observation_dict(row)

    if errors:
        raise ValueError("invalid MAPPO observation: " + "; ".join(errors))

    return UrbanBusObservation(
        condition_id=normalize_condition_id(row["condition_id"]),
        window_id=str(row["window_id"]),
        state_ts=str(row["state_ts"]),
        time_band=str(row["time_band"]),
        seed=as_int(row["seed"], default=0),
        baseline_bus_count=as_float(row["baseline_bus_count"]),
        active_bus_count=as_float(row["active_bus_count"]),
        passenger_demand_generated=as_float(row["passenger_demand_generated"]),
        headway_mean_seconds=as_float(row["headway_mean_seconds"]),
        headway_std_seconds=as_float(row["headway_std_seconds"]),
        bunching_rate=as_float(row["bunching_rate"]),
        on_time_rate=as_float(row["on_time_rate"]),
        avg_wait_seconds=as_float(row["avg_wait_seconds"]),
        passenger_wait_p95_seconds=as_float(row["passenger_wait_p95_seconds"]),
        energy_proxy_per_passenger=as_float(row["energy_proxy_per_passenger"]),
    )


def conservative_mock_action(
    observation: UrbanBusObservation,
    metadata: MappoCheckpointMetadata,
) -> BusControlAction:
    """
    Deterministic contract-only action.

    This is not a learned neural MAPPO action.
    It exists to validate the interface shape before actual inference is wired.
    """
    condition_id = normalize_condition_id(observation.condition_id)

    if condition_id == "A":
        fleet_ratio = 1.0
    elif condition_id == "A90":
        fleet_ratio = 0.9
    elif condition_id == "A80":
        fleet_ratio = 0.8
    elif condition_id == "A70":
        fleet_ratio = 0.7
    else:
        raise ValueError(f"unsupported condition_id for A-family action: {condition_id}")

    active_bus_count = round(observation.baseline_bus_count * fleet_ratio, 6)

    # Simple safe defaults. Actual policy logits/actions will replace this later.
    dispatch_delta = 0.0
    hold_seconds = 0.0
    skip_stop_flag = False
    target_headway_ratio = 1.0

    return BusControlAction(
        action_version="bus_control_action_v1",
        condition_id=condition_id,
        policy_source="mappo_policy",
        source_mode=f"causal_{condition_id}_mappo_policy_v1",
        qwen_trigger_rate=0.0,
        dispatch_delta=dispatch_delta,
        hold_seconds=hold_seconds,
        skip_stop_flag=skip_stop_flag,
        target_headway_ratio=target_headway_ratio,
        active_bus_count=float(active_bus_count),
        policy_debug={
            "interface_version": INTERFACE_VERSION,
            "checkpoint_path": metadata.checkpoint_path,
            "trained_model": metadata.trained_model,
            "mock_action": True,
            "performance_claim_allowed": False,
            "note": "Contract-only action; replace with neural MAPPO inference later.",
        },
    )


def validate_action_dict(action: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []

    for key in REQUIRED_ACTION_KEYS:
        if key not in action:
            errors.append(f"missing action key: {key}")

    condition_id = normalize_condition_id(action.get("condition_id"))

    if condition_id not in A_FAMILY_CONDITIONS:
        errors.append(f"invalid action condition_id: {condition_id}")

    if action.get("policy_source") != "mappo_policy":
        errors.append("policy_source must be mappo_policy")

    source_mode = str(action.get("source_mode", ""))

    if not source_mode.startswith("causal_"):
        errors.append("source_mode must start with causal_")

    if "_mappo_policy_v1" not in source_mode:
        errors.append("source_mode must contain _mappo_policy_v1")

    if "placeholder" in source_mode.lower() or "stub" in source_mode.lower() or "smoke" in source_mode.lower():
        errors.append("source_mode must not contain placeholder/stub/smoke markers")

    qwen_trigger_rate = as_float(action.get("qwen_trigger_rate"), default=0.0)

    if abs(qwen_trigger_rate) > 1e-12:
        errors.append("qwen_trigger_rate must be 0.0")

    active_bus_count = as_float(action.get("active_bus_count"), default=-1.0)

    if active_bus_count < 0:
        errors.append("active_bus_count must be non-negative")

    return errors


def action_to_dict(action: BusControlAction) -> Dict[str, Any]:
    return asdict(action)


def run_policy_interface_once(
    checkpoint_path: str | Path,
    rollout_row: Mapping[str, Any],
) -> Dict[str, Any]:
    metadata = load_checkpoint_metadata(checkpoint_path)
    observation = observation_from_rollout_row(rollout_row)
    action = conservative_mock_action(observation, metadata)
    payload = action_to_dict(action)

    errors = validate_action_dict(payload)

    if errors:
        raise ValueError("invalid MAPPO action: " + "; ".join(errors))

    return {
        "interface_version": INTERFACE_VERSION,
        "checkpoint_metadata": asdict(metadata),
        "observation": asdict(observation),
        "action": payload,
        "passed": True,
    }


def interface_contract() -> Dict[str, Any]:
    return {
        "interface_version": INTERFACE_VERSION,
        "required_checkpoint_metadata_keys": REQUIRED_CHECKPOINT_METADATA_KEYS,
        "expected_checkpoint_metadata": EXPECTED_CHECKPOINT_METADATA,
        "required_observation_keys": REQUIRED_OBSERVATION_KEYS,
        "required_action_keys": REQUIRED_ACTION_KEYS,
        "rules": {
            "no_placeholder_fallback": True,
            "qwen_disabled_for_a_family": True,
            "source_mode_must_be_causal_mappo_policy_v1": True,
            "fake_or_mock_action_not_performance_claim": True,
            "binary_torch_checkpoint_requires_metadata_sidecar": True,
        },
    }


def write_sample_metadata(path: str | Path, trained_model: bool = True) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_type": "urbanbus_mappo_checkpoint",
        "interface_version": INTERFACE_VERSION,
        "policy_kind": "mappo",
        "trained_model": bool(trained_model),
        "action_space_version": "bus_control_action_v1",
        "observation_space_version": "urbanbus_observation_v1",
        "train_run_id": "sample_contract_only",
        "git_commit": "unknown",
        "reward_version": "mappo_reward_v1",
        "rollout_schema_version": "rollout_schema_v1",
        "notes": "Contract-only sample metadata. Not a trained model.",
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def sample_rollout_row(condition_id: str = "A80") -> Dict[str, Any]:
    return {
        "condition_id": condition_id,
        "window_id": "sample_window_001",
        "state_ts": "2023-01-01T08:00:00+09:00",
        "time_band": "peak",
        "seed": 1,
        "baseline_bus_count": 10.0,
        "active_bus_count": 8.0,
        "passenger_demand_generated": 40.0,
        "headway_mean_seconds": 600.0,
        "headway_std_seconds": 90.0,
        "bunching_rate": 0.05,
        "on_time_rate": 0.85,
        "avg_wait_seconds": 300.0,
        "passenger_wait_p95_seconds": 540.0,
        "energy_proxy_per_passenger": 10.0,
        "qwen_trigger_rate": 0.0,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--print-contract", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--checkpoint-path",
        default="artifacts/step35_mappo_policy_interface/sample_checkpoint_metadata.json",
    )

    args = parser.parse_args()

    if args.print_contract:
        print(json.dumps(interface_contract(), ensure_ascii=False, indent=2))
        return 0

    if args.self_test:
        ckpt = write_sample_metadata(args.checkpoint_path, trained_model=True)
        result = run_policy_interface_once(ckpt, sample_rollout_row("A80"))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("[OK] mappo_policy_interface_v1 self-test passed")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
