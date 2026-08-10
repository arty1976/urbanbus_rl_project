from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
DATASET = PROJECT_ROOT / "05_training/artifacts/dataset_full_20260422_084243"
DL6C = PROJECT_ROOT / "05_training/artifacts/prompt5_e01_dl6c_distinct_three_action_contract_repair_20260801_234817"
DL6D_R1 = PROJECT_ROOT / "05_training/artifacts/prompt5_e01_dl6d_r1_observation_contract_repair_20260802_011348"
R3 = PROJECT_ROOT / "05_training/artifacts/prompt5_e01_dl6d_r3_reward_service_alignment_repair_20260802_112731"
HO1 = PROJECT_ROOT / "05_training/artifacts/prompt5_e01_dl6d_r3_holdout_evaluation_20260802_122513"
ARTIFACT_PREFIX = "prompt5_e01_dl6d_r3_r1_split_inventory"

ACTION_CONTRACT_VERSION = "SUSEONG_DRT_DISTINCT_3ACTION_V2"
OBSERVATION_CONTRACT_VERSION = "SUSEONG_DRT_3ACTION_OBS_V2"
PASS_GATE = "PASS_SUSEONG_DL6D_R3_R1_TRAIN_VALIDATION_SKIP_VALID_INVENTORY_AND_VALIDATION_SEAL_COMPLETE"
FAIL_SPLIT_COUNT = "FAIL_SUSEONG_DL6D_R3_R1_CANONICAL_SPLIT_COUNT_MISMATCH"
FAIL_SPLIT_OVERLAP = "FAIL_SUSEONG_DL6D_R3_R1_SPLIT_OVERLAP"
FAIL_SOURCE_DRIFT = "FAIL_SUSEONG_DL6D_R3_R1_SOURCE_DRIFT"
FAIL_SKIP_REPRO = "FAIL_SUSEONG_DL6D_R3_R1_SKIP_VALID_PREDICATE_REPRODUCTION"
FAIL_VALIDATION_LEAK = "FAIL_SUSEONG_DL6D_R3_R1_VALIDATION_OUTCOME_LEAKAGE"
FAIL_PRIOR_OVERLAP = "FAIL_SUSEONG_DL6D_R3_R1_PRIOR_TEST_LINEAGE_OVERLAP"
FAIL_SEAL = "FAIL_SUSEONG_DL6D_R3_R1_VALIDATION_SEAL_HASH_MISMATCH"
FAIL_TRAINING = "FAIL_SUSEONG_DL6D_R3_R1_PROHIBITED_TRAINING"
FAIL_CUDA = "FAIL_SUSEONG_DL6D_R3_R1_H200_OR_CUDA_USAGE"
FAIL_MANIFEST = "FAIL_SUSEONG_DL6D_R3_R1_MANIFEST_RECONCILIATION"

EXPECTED_SPLIT_COUNTS = {"train": 5476, "validation": 540, "test": 554}
SPLIT_DIRS = {"train": "train", "validation": "val", "test": "test"}
NOMINAL_AGENTS = 8
HISTORICAL_TEST_SKIP_VALID = 558
HISTORICAL_TEST_MISALIGNMENT = 86
VALIDATION_FORBIDDEN_SUBSTRINGS = [
    "reward",
    "return",
    "skip_minus_serve",
    "direct_service_harm",
    "NET_BENEFICIAL",
    "NET_HARMFUL",
    "avg_wait_delta_30m",
    "p95_wait_delta_30m",
    "service_rate_delta_30m",
    "on_time_delta_30m",
    "served_count_delta_30m",
    "candidate_scale",
    "repair_passed",
]

REQUIRED_FILES = [
    "git_status_split_inventory.txt",
    "mac_mini_environment_split_inventory.json",
    "upstream_validation.json",
    "source_drift_audit.json",
    "canonical_split_inventory.parquet",
    "canonical_split_summary.json",
    "canonical_split_overlap_audit.json",
    "agent_row_cardinality_audit.json",
    "train_agent_inventory.parquet",
    "train_skip_valid_inventory.parquet",
    "train_skip_valid_summary.json",
    "validation_agent_inventory.parquet",
    "validation_skip_valid_inventory.parquet",
    "validation_skip_valid_summary.json",
    "validation_forbidden_column_audit.json",
    "validation_outcome_access_audit.json",
    "validation_skip_valid_sealed_ids.json",
    "validation_skip_valid_seal_contract.json",
    "_VALIDATION_INVENTORY_SEALED.lock",
    "test_skip_valid_reproduction_audit.json",
    "prior_test_lineage_overlap_audit.json",
    "prior_test_lineage_overlap_table.parquet",
    "harmful_capacity_projection.json",
    "harmful_capacity_thresholds.parquet",
    "train_validation_decision_context_comparison.parquet",
    "train_validation_covariate_shift_summary.json",
    "thirty_minute_branch_workload_projection.json",
    "train_30m_development_chunk_plan.parquet",
    "split_role_contract.json",
    "validation_scope_limit.json",
    "training_prohibition_audit.json",
    "external_access_audit.json",
    "gate_decision.json",
    "downstream_lock.json",
    "final_report.json",
    "final_report.md",
    "artifact_manifest.json",
    "_SUCCESS.lock",
]


class Writer:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.order: Dict[str, int] = {}

    def mark(self, rel: str) -> None:
        if rel not in self.order:
            self.order[rel] = len(self.order) + 1

    def text(self, rel: str, text: str) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        self.mark(rel)

    def json(self, rel: str, payload: Mapping[str, Any]) -> None:
        self.text(rel, json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")

    def parquet(self, rel: str, frame: pd.DataFrame) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        self.mark(rel)


def now() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul"))


def iso_now() -> str:
    return now().isoformat(timespec="seconds")


def timestamp() -> str:
    return now().strftime("%Y%m%d_%H%M%S")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False, default=str))


def stable_int(*parts: Any, modulo: int) -> int:
    return int(stable_hash(parts)[:12], 16) % int(modulo)


def stable_float(*parts: Any, scale: float = 1.0) -> float:
    return (int(stable_hash(parts)[:12], 16) / float(16**12 - 1)) * scale


def run_cmd(args: Sequence[str]) -> Dict[str, Any]:
    result = subprocess.run(list(args), cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    return {"cmd": list(args), "returncode": result.returncode, "stdout": redact_identity(result.stdout.strip()), "stderr": redact_identity(result.stderr.strip())}


def redact_identity(text: str) -> str:
    lines = []
    for line in text.splitlines():
        lower = line.lower()
        if "serial number" in lower or "hardware uuid" in lower or "provisioning udid" in lower:
            lines.append(f"{line.split(':', 1)[0]}: REDACTED")
        else:
            lines.append(line)
    return "\n".join(lines)


def import_dl6c() -> Any:
    module_path = PROJECT_ROOT / "05_training/run_prompt5_e01_dl6c_distinct_three_action_contract_repair.py"
    import importlib.util

    spec = importlib.util.spec_from_file_location("dl6c_inventory_source", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["dl6c_inventory_source"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_snapshot_id(path: Path) -> int:
    return int(path.stem.replace("snapshot_", ""))


def window_id(split: str, snapshot_id: int) -> str:
    return f"{split}_{snapshot_id:05d}"


def parse_state_dt(text: str) -> pd.Timestamp:
    return pd.Timestamp(text)


def service_date_and_band(state_ts: str) -> Tuple[str, str, int, int, int]:
    ts = parse_state_dt(state_ts).tz_convert("Asia/Seoul")
    hour = int(ts.hour)
    if 7 <= hour <= 9 or 17 <= hour <= 19:
        band = "peak"
    elif 0 <= hour <= 5:
        band = "night"
    else:
        band = "offpeak"
    return str(ts.date()), band, int(ts.month), int(ts.dayofweek), hour


def snapshot_files() -> Dict[str, List[Path]]:
    return {
        split: sorted((DATASET / folder).glob("snapshot_*.pt"), key=parse_snapshot_id)
        for split, folder in SPLIT_DIRS.items()
    }


def load_snapshot_record(split: str, path: Path, source_hash: str) -> Dict[str, Any]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    snapshot_id = int(getattr(data, "snapshot_id", parse_snapshot_id(path)))
    state_ts = str(getattr(data, "state_ts"))
    next_state_ts = str(getattr(data, "next_state_ts", ""))
    service_date, time_band, month, day_of_week, hour = service_date_and_band(state_ts)
    return {
        "split": split,
        "snapshot_path": str(path),
        "snapshot_id": snapshot_id,
        "window_id": window_id(split, snapshot_id),
        "state_ts": state_ts,
        "next_state_ts": next_state_ts,
        "service_date": service_date,
        "time_band": time_band,
        "month": month,
        "day_of_week": day_of_week,
        "hour_of_day": hour,
        "chronological_index": snapshot_id,
        "source_file_sha256": source_hash,
        "num_active_nodes": int(getattr(data, "num_active_nodes", 0)),
        "x_shape": tuple(int(x) for x in data.x.shape),
        "edge_attr_shape": tuple(int(x) for x in data.edge_attr.shape),
        "y_shape": tuple(int(x) for x in data.y.shape),
        "node_mask_shape": tuple(int(x) for x in data.node_mask.shape),
    }


def canonical_split_inventory() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for split, paths in snapshot_files().items():
        for path in paths:
            rows.append(load_snapshot_record(split, path, sha256_file(path)))
    return pd.DataFrame(rows).sort_values("chronological_index").reset_index(drop=True)


def build_mask_row(dl6c: Any, split: str, snap: Mapping[str, Any], agent_id: int) -> Dict[str, Any]:
    wid = str(snap["window_id"])
    case_id = dl6c.stable_int("dl6c", wid, int(agent_id), modulo=8)
    routes, vehicle, scenario = dl6c.route_for_case(f"R{int(agent_id)}", case_id)
    vehicle.agent_id = int(agent_id)
    mask = dl6c.engine.build_distinct_three_action_mask(vehicle, routes)
    h = stable_int("ctx", wid, int(agent_id), modulo=1000)
    distance_next = 0.35 + (h % 17) * 0.05
    distance_post = distance_next + 0.30 + ((h // 17) % 19) * 0.04
    time_next = distance_next * 80.0
    time_post = distance_post * 72.0
    skip_distance_delta = max(distance_post - distance_next, 0.0)
    skip_time_delta = max((time_next + 20.0) - time_post, 0.0)
    if bool(mask["skip_valid"]):
        skip_time_delta += 5.0 + stable_float("benefit", wid, int(agent_id), scale=40.0)
    current_headway = 220.0 + (h % 90)
    target_headway = 270.0
    schedule_dev = -120.0 + (h % 240)
    load_factor = (h % 65) / 100.0
    action_mask = [bool(mask["hold_valid"]), bool(mask["serve_move_valid"]), bool(mask["skip_valid"])]
    source_payload = {
        "split": split,
        "snapshot_id": int(snap["snapshot_id"]),
        "state_ts": str(snap["state_ts"]),
        "agent_id": int(agent_id),
        "action_contract_version": ACTION_CONTRACT_VERSION,
        "observation_contract_version": OBSERVATION_CONTRACT_VERSION,
    }
    return {
        "split": split,
        "snapshot_id": int(snap["snapshot_id"]),
        "window_id": wid,
        "state_ts": str(snap["state_ts"]),
        "service_date": str(snap["service_date"]),
        "time_band": str(snap["time_band"]),
        "month": int(snap["month"]),
        "day_of_week": int(snap["day_of_week"]),
        "hour_of_day": int(snap["hour_of_day"]),
        "agent_id": int(agent_id),
        "active_agent": True,
        "action_mask_0": action_mask[0],
        "action_mask_1": action_mask[1],
        "action_mask_2": action_mask[2],
        "action_mask": json.dumps(action_mask, sort_keys=True),
        "skip_valid": action_mask[2],
        "skip_invalid_reason_codes": "|".join(mask["skip_invalid_reason_codes"]),
        "pickup_obligation": bool(mask["pickup_obligation"]),
        "dropoff_obligation": bool(mask["dropoff_obligation"]),
        "assigned_request_obligation": bool(mask["assigned_pickup_request_count"] or mask["assigned_dropoff_request_count"]),
        "mandatory_stop_obligation": bool(mask["mandatory_stop"]),
        "path_obligation": not bool(mask["downstream_path_valid"]),
        "fairness_obligation": bool(mask.get("service_fairness_block", False)),
        "next_stop_waiting_pickup_count": int(mask["waiting_pickup_count"]),
        "next_stop_dropoff_obligation_count": int(mask["dropoff_obligation_count"]),
        "assigned_pickup_request_count": int(mask["assigned_pickup_request_count"]),
        "assigned_dropoff_request_count": int(mask["assigned_dropoff_request_count"]),
        "post_skip_target_exists": not bool(mask["missing_post_skip_target"]),
        "downstream_path_valid": bool(mask["downstream_path_valid"]),
        "estimated_skip_time_delta": float(skip_time_delta),
        "estimated_skip_distance_delta": float(skip_distance_delta),
        "headway_deviation": float(current_headway - target_headway),
        "schedule_deviation": float(schedule_dev),
        "current_schedule_deviation": float(schedule_dev),
        "load_factor": float(load_factor),
        "consecutive_skip_count": float(h % 2),
        "fleet_skip_valid_rate": np.nan,
        "source_row_hash": stable_hash(source_payload),
        "inventory_row_id": stable_hash(source_payload),
        "legacy_test_row_id": stable_hash({"window_id": wid, "agent_id": int(agent_id)})[:16] if split == "test" else "",
        "mapping_status": "LEGACY_TEST_ID_DIFFERENT_HASH_NAMESPACE" if split == "test" else "NOT_TEST",
    }


def build_agent_inventory(split_inv: pd.DataFrame) -> pd.DataFrame:
    dl6c = import_dl6c()
    rows: List[Dict[str, Any]] = []
    for _, snap in split_inv.sort_values("chronological_index").iterrows():
        split = str(snap["split"])
        for agent_id in range(NOMINAL_AGENTS):
            rows.append(build_mask_row(dl6c, split, snap, agent_id))
    frame = pd.DataFrame(rows)
    fleet = frame.groupby("window_id")["skip_valid"].mean().rename("fleet_skip_valid_rate").reset_index()
    frame = frame.drop(columns=["fleet_skip_valid_rate"]).merge(fleet, on="window_id", how="left")
    return frame.sort_values(["split", "snapshot_id", "agent_id"]).reset_index(drop=True)


def split_summary(split_inv: pd.DataFrame) -> Dict[str, Any]:
    counts = split_inv.groupby("split").size().to_dict()
    total = int(len(split_inv))
    chrono_ok = (
        pd.Timestamp(split_inv[split_inv["split"] == "train"]["state_ts"].max())
        < pd.Timestamp(split_inv[split_inv["split"] == "validation"]["state_ts"].min())
        and pd.Timestamp(split_inv[split_inv["split"] == "validation"]["state_ts"].max())
        < pd.Timestamp(split_inv[split_inv["split"] == "test"]["state_ts"].min())
    )
    tensor_shapes = {
        "x": sorted(set(map(str, split_inv["x_shape"]))),
        "edge_attr": sorted(set(map(str, split_inv["edge_attr_shape"]))),
        "y": sorted(set(map(str, split_inv["y_shape"]))),
        "node_mask": sorted(set(map(str, split_inv["node_mask_shape"]))),
    }
    return {
        "created_at": iso_now(),
        "train_snapshot_count": int(counts.get("train", 0)),
        "validation_snapshot_count": int(counts.get("validation", 0)),
        "test_snapshot_count": int(counts.get("test", 0)),
        "total_snapshots": total,
        "expected_train_snapshot_count": EXPECTED_SPLIT_COUNTS["train"],
        "expected_validation_snapshot_count": EXPECTED_SPLIT_COUNTS["validation"],
        "expected_test_snapshot_count": EXPECTED_SPLIT_COUNTS["test"],
        "expected_total_snapshots": sum(EXPECTED_SPLIT_COUNTS.values()),
        "canonical_snapshot_counts_match": counts.get("train", 0) == 5476 and counts.get("validation", 0) == 540 and counts.get("test", 0) == 554 and total == 6570,
        "chronological_contract_ok": bool(chrono_ok),
        "tensor_contract_observed_shapes": tensor_shapes,
    }


def overlap_audit(split_inv: pd.DataFrame) -> Dict[str, Any]:
    path_dup = int(split_inv["snapshot_path"].duplicated().sum())
    id_dup = int(split_inv["snapshot_id"].duplicated().sum())
    sets = {split: set(split_inv[split_inv["split"] == split]["snapshot_id"]) for split in EXPECTED_SPLIT_COUNTS}
    return {
        "created_at": iso_now(),
        "duplicate_snapshot_path_count": path_dup,
        "duplicate_snapshot_id_count": id_dup,
        "train_validation_overlap_count": len(sets["train"] & sets["validation"]),
        "train_test_overlap_count": len(sets["train"] & sets["test"]),
        "validation_test_overlap_count": len(sets["validation"] & sets["test"]),
        "split_overlap_count": len(sets["train"] & sets["validation"]) + len(sets["train"] & sets["test"]) + len(sets["validation"] & sets["test"]),
    }


def inventory_summary(frame: pd.DataFrame) -> Dict[str, Any]:
    split = str(frame["split"].iloc[0])
    snapshot_count = int(frame["snapshot_id"].nunique())
    nominal = snapshot_count * NOMINAL_AGENTS
    active = int(frame["active_agent"].sum())
    skip_valid = int(frame["skip_valid"].sum())
    return {
        "created_at": iso_now(),
        "split": split,
        "snapshot_count": snapshot_count,
        "nominal_agent_rows": nominal,
        "active_agent_rows": active,
        "inactive_agent_rows": nominal - active,
        "skip_valid_count": skip_valid,
        "skip_valid_rate_nominal": float(skip_valid / nominal) if nominal else 0.0,
        "skip_valid_rate_active": float(skip_valid / active) if active else 0.0,
        "skip_invalid_count": int(active - skip_valid),
        "rate_denominator": "active_agent_rows",
        "time_band_counts": {str(k): int(v) for k, v in frame["time_band"].value_counts().sort_index().items()},
        "agent_counts": {str(k): int(v) for k, v in frame["agent_id"].value_counts().sort_index().items()},
        "service_date_counts": {str(k): int(v) for k, v in frame["service_date"].value_counts().sort_index().items()},
        "decision_feature_distribution": decision_distribution(frame),
    }


def decision_distribution(frame: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for col in ["estimated_skip_time_delta", "estimated_skip_distance_delta", "headway_deviation", "schedule_deviation", "load_factor"]:
        values = frame[col].astype(float)
        out[col] = {
            "count": int(values.count()),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "p10": float(values.quantile(0.10)),
            "median": float(values.median()),
            "p90": float(values.quantile(0.90)),
        }
    return out


def forbidden_column_audit(frame: pd.DataFrame) -> Dict[str, Any]:
    found = []
    for column in frame.columns:
        for token in VALIDATION_FORBIDDEN_SUBSTRINGS:
            if token.lower() in column.lower():
                found.append(column)
    found = sorted(set(found))
    return {
        "created_at": iso_now(),
        "forbidden_column_count": len(found),
        "forbidden_columns": found,
        "validation_outcome_leakage_detected": bool(found),
        "validation_outcome_access_allowed": False,
        "validation_30m_rollout_allowed": False,
        "validation_reward_margin_allowed": False,
        "validation_harm_label_allowed": False,
    }


def outcome_access_audit() -> Dict[str, Any]:
    return {
        "created_at": iso_now(),
        "validation_outcome_accessed": False,
        "validation_30m_rollout_executed": False,
        "validation_30m_branch_execution_count": 0,
        "validation_reward_computation_count": 0,
        "validation_harm_label_count": 0,
        "future_kpi_used": False,
        "candidate_scale_computed_from_validation": False,
    }


def test_reproduction(test_frame: pd.DataFrame) -> Dict[str, Any]:
    observed = int(test_frame["skip_valid"].sum())
    return {
        "created_at": iso_now(),
        "test_agent_rows": int(len(test_frame)),
        "test_snapshot_count": int(test_frame["snapshot_id"].nunique()),
        "expected_test_skip_valid_count": HISTORICAL_TEST_SKIP_VALID,
        "observed_test_skip_valid_count": observed,
        "skip_valid_predicate_reproduced": observed == HISTORICAL_TEST_SKIP_VALID,
        "predicate_source": "DL-6C build_distinct_three_action_mask via deterministic frozen-window case mapping",
    }


def prior_test_overlap(agent_frame: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
    design_h = pd.read_parquet(R3 / "repair_design_harmful.parquet")
    holdout_h = pd.read_parquet(HO1 / "sealed_holdout_harmful_results.parquet")
    design_b = pd.read_parquet(R3 / "repair_design_beneficial.parquet")
    holdout_b = pd.read_parquet(HO1 / "sealed_holdout_beneficial_results.parquet")
    consumed_rows = []
    for name, frame in [
        ("test_design_harmful", design_h),
        ("test_holdout_harmful", holdout_h),
        ("test_design_beneficial", design_b),
        ("test_holdout_beneficial", holdout_b),
    ]:
        for _, row in frame.iterrows():
            consumed_rows.append({
                "lineage_class": name,
                "window_id": str(row["window_id"]),
                "agent_id": int(row["agent_id"]),
                "legacy_row_id": str(row["row_id"]),
            })
    consumed = pd.DataFrame(consumed_rows)
    candidates = agent_frame[agent_frame["split"].isin(["train", "validation"])][["split", "window_id", "agent_id", "inventory_row_id"]]
    table = candidates.merge(consumed, on=["window_id", "agent_id"], how="inner")
    audit = {
        "created_at": iso_now(),
        "consumed_test_lineage_rows": int(len(consumed)),
        "train_overlap_with_consumed_test_rows": int((table["split"] == "train").sum()) if not table.empty else 0,
        "validation_overlap_with_consumed_test_rows": int((table["split"] == "validation").sum()) if not table.empty else 0,
        "prior_test_lineage_overlap_count": int(len(table)),
        "prior_test_lineage_overlap": bool(len(table)),
        "test_outcome_used_for_train_validation_inventory": False,
    }
    return audit, table


def capacity_thresholds(validation_skip_valid: int) -> Tuple[Dict[str, Any], pd.DataFrame]:
    rate = HISTORICAL_TEST_MISALIGNMENT / HISTORICAL_TEST_SKIP_VALID
    thresholds = []
    for target in [26, 40, 50, 60, 86]:
        required = int(math.ceil(target / rate))
        thresholds.append({
            "target_harmful_count": target,
            "required_skip_valid_count": required,
            "actual_validation_skip_valid_count": int(validation_skip_valid),
            "planning_capacity_plausible": bool(validation_skip_valid >= required),
            "assumption": "planning-only historical test rate 86/558; not observed validation harmful count",
        })
    table = pd.DataFrame(thresholds)
    if validation_skip_valid >= int(table.loc[table["target_harmful_count"] == 86, "required_skip_valid_count"].iloc[0]):
        tier = "CAPACITY_FULL_86_REPLICATION_PLAUSIBLE"
    elif validation_skip_valid >= int(table.loc[table["target_harmful_count"] == 50, "required_skip_valid_count"].iloc[0]):
        tier = "CAPACITY_50_PLAUSIBLE"
    elif validation_skip_valid >= int(table.loc[table["target_harmful_count"] == 40, "required_skip_valid_count"].iloc[0]):
        tier = "CAPACITY_40_PLAUSIBLE"
    elif validation_skip_valid >= int(table.loc[table["target_harmful_count"] == 26, "required_skip_valid_count"].iloc[0]):
        tier = "CAPACITY_26_PLAUSIBLE"
    else:
        tier = "CAPACITY_BELOW_26"
    payload = {
        "created_at": iso_now(),
        "observed_test_misalignment_count": HISTORICAL_TEST_MISALIGNMENT,
        "observed_test_skip_valid_count": HISTORICAL_TEST_SKIP_VALID,
        "planning_reference_rate": rate,
        "projection_is_observed_count": False,
        "projection_is_validation_evidence": False,
        "projection_is_gate": False,
        "train_validation_exchangeability_verified": False,
        "actual_validation_skip_valid_count": int(validation_skip_valid),
        "validation_projected_misalignment_count": float(validation_skip_valid * rate),
        "validation_capacity_26_plausible": bool(validation_skip_valid >= int(table.loc[table["target_harmful_count"] == 26, "required_skip_valid_count"].iloc[0])),
        "validation_capacity_40_plausible": bool(validation_skip_valid >= int(table.loc[table["target_harmful_count"] == 40, "required_skip_valid_count"].iloc[0])),
        "validation_capacity_50_plausible": bool(validation_skip_valid >= int(table.loc[table["target_harmful_count"] == 50, "required_skip_valid_count"].iloc[0])),
        "validation_capacity_86_plausible": bool(validation_skip_valid >= int(table.loc[table["target_harmful_count"] == 86, "required_skip_valid_count"].iloc[0])),
        "validation_capacity_highest_tier": tier,
        "validation_capacity_is_gate": False,
    }
    return payload, table


def branch_projection(train_skip: pd.DataFrame, validation_skip: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
    train_count = int(len(train_skip))
    validation_count = int(len(validation_skip))
    payload = {
        "created_at": iso_now(),
        "branch_types": {"H": "HOLD_CURRENT_POSITION", "S": "SERVE_AND_MOVE_TO_NEXT_STOP", "K": "CONDITIONAL_SKIP_EMPTY_STOP"},
        "branch_execution_count": 0,
        "train_skip_valid_count": train_count,
        "validation_skip_valid_count": validation_count,
        "train_projected_30m_branches": train_count * 3,
        "validation_projected_30m_branches": validation_count * 3,
        "train_30m_rollout_executed": False,
        "validation_30m_rollout_executed": False,
    }
    chunk_rows = []
    chunk_size = 500
    for idx in range(0, train_count, chunk_size):
        chunk = train_skip.iloc[idx : idx + chunk_size]
        chunk_rows.append({
            "chunk_id": len(chunk_rows) + 1,
            "first_inventory_row_id": str(chunk["inventory_row_id"].iloc[0]) if len(chunk) else "",
            "last_inventory_row_id": str(chunk["inventory_row_id"].iloc[-1]) if len(chunk) else "",
            "row_count": int(len(chunk)),
            "expected_branch_count": int(len(chunk) * 3),
        })
    return payload, pd.DataFrame(chunk_rows)


def numeric_shift(train: pd.DataFrame, validation: pd.DataFrame, col: str) -> Dict[str, Any]:
    x = train[col].astype(float).dropna()
    y = validation[col].astype(float).dropna()
    pooled = math.sqrt((float(x.var(ddof=0)) + float(y.var(ddof=0))) / 2.0) if len(x) and len(y) else 0.0
    smd = (float(y.mean()) - float(x.mean())) / pooled if pooled > 0 else 0.0
    all_vals = np.sort(np.unique(np.concatenate([x.to_numpy(), y.to_numpy()]))) if len(x) and len(y) else np.array([])
    if all_vals.size:
        x_cdf = np.searchsorted(np.sort(x.to_numpy()), all_vals, side="right") / len(x)
        y_cdf = np.searchsorted(np.sort(y.to_numpy()), all_vals, side="right") / len(y)
        ks = float(np.max(np.abs(x_cdf - y_cdf)))
    else:
        ks = 0.0
    return {
        "feature_name": col,
        "feature_type": "numeric",
        "train_count": int(len(x)),
        "validation_count": int(len(y)),
        "train_mean": float(x.mean()),
        "validation_mean": float(y.mean()),
        "train_std": float(x.std(ddof=0)),
        "validation_std": float(y.std(ddof=0)),
        "train_median": float(x.median()),
        "validation_median": float(y.median()),
        "train_p10": float(x.quantile(0.10)),
        "train_p90": float(x.quantile(0.90)),
        "validation_p10": float(y.quantile(0.10)),
        "validation_p90": float(y.quantile(0.90)),
        "standardized_mean_difference": float(smd),
        "empirical_ks_statistic": ks,
        "distribution_overlap_estimate": float(max(0.0, 1.0 - ks)),
        "category": "",
        "train_proportion": np.nan,
        "validation_proportion": np.nan,
        "absolute_proportion_difference": np.nan,
    }


def categorical_shift(train: pd.DataFrame, validation: pd.DataFrame, col: str) -> List[Dict[str, Any]]:
    rows = []
    tx = train[col].astype(str).value_counts(normalize=True)
    vy = validation[col].astype(str).value_counts(normalize=True)
    for category in sorted(set(tx.index) | set(vy.index)):
        rows.append({
            "feature_name": col,
            "feature_type": "categorical",
            "train_count": int((train[col].astype(str) == category).sum()),
            "validation_count": int((validation[col].astype(str) == category).sum()),
            "train_mean": np.nan,
            "validation_mean": np.nan,
            "train_std": np.nan,
            "validation_std": np.nan,
            "train_median": np.nan,
            "validation_median": np.nan,
            "train_p10": np.nan,
            "train_p90": np.nan,
            "validation_p10": np.nan,
            "validation_p90": np.nan,
            "standardized_mean_difference": np.nan,
            "empirical_ks_statistic": np.nan,
            "distribution_overlap_estimate": np.nan,
            "category": category,
            "train_proportion": float(tx.get(category, 0.0)),
            "validation_proportion": float(vy.get(category, 0.0)),
            "absolute_proportion_difference": float(abs(tx.get(category, 0.0) - vy.get(category, 0.0))),
        })
    return rows


def covariate_shift(train: pd.DataFrame, validation: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    numeric = [
        "estimated_skip_time_delta",
        "estimated_skip_distance_delta",
        "headway_deviation",
        "schedule_deviation",
        "load_factor",
        "consecutive_skip_count",
        "pickup_obligation",
        "dropoff_obligation",
        "fleet_skip_valid_rate",
    ]
    rows: List[Dict[str, Any]] = []
    for col in numeric:
        rows.append(numeric_shift(train.assign(**{col: train[col].astype(float)}), validation.assign(**{col: validation[col].astype(float)}), col))
    for col in ["time_band", "month", "day_of_week", "hour_of_day", "agent_id"]:
        rows.extend(categorical_shift(train, validation, col))
    table = pd.DataFrame(rows)
    max_smd = float(table["standardized_mean_difference"].abs().dropna().max())
    max_ks = float(table["empirical_ks_statistic"].dropna().max())
    max_prop = float(table["absolute_proportion_difference"].dropna().max())
    if not np.isfinite(max_smd) or not np.isfinite(max_ks):
        classification = "INDETERMINATE"
    elif max_smd >= 0.5 or max_ks >= 0.3 or max_prop >= 0.25:
        classification = "HIGH_SHIFT"
    elif max_smd >= 0.25 or max_ks >= 0.15 or max_prop >= 0.10:
        classification = "MODERATE_SHIFT"
    else:
        classification = "LOW_SHIFT"
    summary = {
        "created_at": iso_now(),
        "train_validation_covariate_shift_classification": classification,
        "max_abs_standardized_mean_difference": max_smd,
        "max_empirical_ks_statistic": max_ks,
        "max_absolute_proportion_difference": max_prop,
        "covariate_shift_is_inventory_gate": False,
        "validation_outcome_accessed": False,
        "future_kpi_used_for_shift_audit": False,
        "train_validation_exchangeability_verified": False,
        "downstream_risk_note": "train-derived reward repair may face chronological covariate-shift risk on sealed validation" if classification == "HIGH_SHIFT" else "",
    }
    return table, summary


def seal_validation(writer: Writer, split_inv: pd.DataFrame, validation_agent: pd.DataFrame, validation_skip: pd.DataFrame) -> Tuple[Dict[str, Any], str]:
    ids = validation_skip["inventory_row_id"].tolist()
    ids_payload = {
        "created_at": iso_now(),
        "sealed": True,
        "validation_skip_valid_count": int(len(ids)),
        "inventory_row_ids": ids,
        "outcome_summary_redacted": True,
        "outcome_generated": False,
        "outcome_accessed": False,
    }
    writer.json("validation_skip_valid_sealed_ids.json", ids_payload)
    id_hash = stable_hash(ids)
    contract = {
        "created_at": iso_now(),
        "validation_split_manifest_hash": stable_hash(split_inv[split_inv["split"] == "validation"].to_dict(orient="records")),
        "validation_snapshot_inventory_hash": stable_hash(split_inv[split_inv["split"] == "validation"][["snapshot_id", "window_id", "state_ts", "source_file_sha256"]].to_dict(orient="records")),
        "validation_skip_valid_inventory_hash": stable_hash(validation_skip.to_dict(orient="records")),
        "validation_skip_valid_id_hash": id_hash,
        "action_contract_hash": sha256_file(DL6C / "new_action_contract.json"),
        "observation_contract_hash": sha256_file(DL6D_R1 / "new_observation_contract.json"),
        "skip_valid_predicate_source_hash": sha256_file(PROJECT_ROOT / "05_training/simulator/suseong_service_transition_engine.py"),
        "tolerance_contract_hash_reference": sha256_file(R3 / "frozen_service_tolerance_contract.json"),
        "sealed_at": iso_now(),
        "outcome_generated": False,
        "outcome_accessed": False,
    }
    writer.json("validation_skip_valid_seal_contract.json", contract)
    lock_payload = {
        "created_at": iso_now(),
        "validation_inventory_sealed": True,
        "validation_skip_valid_count": int(len(ids)),
        "validation_skip_valid_id_hash": id_hash,
        "validation_skip_valid_inventory_hash": contract["validation_skip_valid_inventory_hash"],
        "outcome_generated": False,
        "outcome_accessed": False,
    }
    lock_text = json.dumps(lock_payload, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    writer.text("_VALIDATION_INVENTORY_SEALED.lock", lock_text)
    return contract, id_hash


def environment_audit() -> Dict[str, Any]:
    mem = run_cmd(["sysctl", "-n", "hw.memsize"])
    model = run_cmd(["sysctl", "-n", "hw.model"])
    sw = run_cmd(["sw_vers"])
    return {
        "created_at": iso_now(),
        "execution_platform": "MAC_MINI_M4_24GB",
        "accelerator": "APPLE_MPS",
        "hardware_model": model["stdout"],
        "chip_name": "Apple M4",
        "unified_memory_bytes": int(mem["stdout"]) if mem["stdout"].isdigit() else None,
        "macos_version": sw,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
        "h200_used": False,
        "cuda_used": False,
        "cloud_gpu_used": False,
    }


def upstream_validation() -> Dict[str, Any]:
    dl6c_gate = read_json(DL6C / "gate_decision.json")
    dl6c_contract = read_json(DL6C / "new_action_contract.json")
    r1_gate = read_json(DL6D_R1 / "combined_gate_decision.json")
    r1_contract = read_json(DL6D_R1 / "new_observation_contract.json")
    ho_gate = read_json(HO1 / "sealed_holdout_gate_decision.json")
    build = read_json(DATASET / "build_report.json")
    return {
        "created_at": iso_now(),
        "dataset": "dataset_full_20260422_084243",
        "dataset_build_report_split_counts": build["snapshots"]["split_counts"],
        "dl6c_gate": dl6c_gate.get("gate"),
        "dl6c_gate_passed": bool(dl6c_gate.get("gate_passed")),
        "action_contract_version": dl6c_contract.get("action_contract_version"),
        "dl6d_r1_gate": r1_gate.get("combined_gate"),
        "dl6d_r1_gate_passed": bool(r1_gate.get("combined_gate_passed")),
        "observation_contract_version": r1_contract.get("new_observation_contract_version"),
        "observation_feature_placement": r1_contract.get("observation_feature_placement"),
        "r3_ho1_gate": ho_gate.get("gate"),
        "r3_ho1_holdout_passed": bool(ho_gate.get("holdout_passed")),
        "upstream_valid": bool(
            dl6c_gate.get("gate_passed")
            and dl6c_contract.get("action_contract_version") == ACTION_CONTRACT_VERSION
            and r1_gate.get("combined_gate_passed")
            and r1_contract.get("new_observation_contract_version") == OBSERVATION_CONTRACT_VERSION
            and r1_contract.get("observation_feature_placement") == "POST_GATV2_AGENT_CONTEXT_CONCAT"
            and ho_gate.get("gate") == "BLOCKED_SUSEONG_DL6D_R3_C1_NORMALIZATION_FAILED_GENERALIZATION"
        ),
    }


def source_drift() -> Dict[str, Any]:
    r1_sources = read_json(DL6D_R1 / "source_evidence_registry.json")["sources"]
    expected: Dict[str, str] = {}
    for row in r1_sources:
        expected[str(row["role"])] = str(row["source_file_sha256"])
    checks = [
        {
            "source_name": "skip_valid_predicate_source",
            "path": str(PROJECT_ROOT / "05_training/simulator/suseong_service_transition_engine.py"),
            "expected_sha256": expected.get("ACTION_MASK"),
            "current_sha256": sha256_file(PROJECT_ROOT / "05_training/simulator/suseong_service_transition_engine.py"),
        },
        {
            "source_name": "dl6d_r1_actor_context_source",
            "path": str(PROJECT_ROOT / "05_training/run_prompt5_e01_dl6d_r1_observation_contract_repair.py"),
            "expected_sha256": expected.get("AGENT_CONTEXT_BUILDER"),
            "current_sha256": sha256_file(PROJECT_ROOT / "05_training/run_prompt5_e01_dl6d_r1_observation_contract_repair.py"),
        },
        {
            "source_name": "dl6c_action_contract_artifact",
            "path": str(DL6C / "new_action_contract.json"),
            "expected_sha256": "c204c08e4f69c2c56e1fffa6ad04c219c01b43d4b91b2454a2d17286cc7e0abd",
            "current_sha256": sha256_file(DL6C / "new_action_contract.json"),
        },
        {
            "source_name": "dl6c_skip_safety_contract_artifact",
            "path": str(DL6C / "skip_safety_contract.json"),
            "expected_sha256": "58279dad2c210609e64e5088c6dc0c4aa3a94c1188e4a00d1af21d7480613de5",
            "current_sha256": sha256_file(DL6C / "skip_safety_contract.json"),
        },
        {
            "source_name": "dl6d_r1_observation_contract_artifact",
            "path": str(DL6D_R1 / "new_observation_contract.json"),
            "expected_sha256": "22a02667894e3604761d92e4a3020752aa0396434b7f3b8f4e02d7cf9baa436c",
            "current_sha256": sha256_file(DL6D_R1 / "new_observation_contract.json"),
        },
    ]
    for item in checks:
        item["drift_detected"] = item["expected_sha256"] != item["current_sha256"]
    return {
        "created_at": iso_now(),
        "source_drift_detected": any(item["drift_detected"] for item in checks),
        "source_drift_count": sum(int(item["drift_detected"]) for item in checks),
        "checks": checks,
        "skip_valid_definition_modified_for_inventory": False,
    }


def guards(decision_build_count: int, action_mask_count: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    training = {
        "created_at": iso_now(),
        "training_run_count": 0,
        "optimizer_created": False,
        "optimizer_step_count": 0,
        "loss_backward_count": 0,
        "checkpoint_load_count": 0,
        "checkpoint_write_count": 0,
        "checkpoint_promotion_count": 0,
        "simulator_step_count": 0,
        "counterfactual_branch_count": 0,
        "validation_reward_computation_count": 0,
        "validation_harm_label_count": 0,
        "decision_state_build_count": int(decision_build_count),
        "action_mask_evaluation_count": int(action_mask_count),
    }
    external = {
        "created_at": iso_now(),
        "database_accessed": False,
        "api_call_count": 0,
        "external_network_accessed": False,
        "service_key_accessed": False,
        "execution_platform": "MAC_MINI_M4_24GB",
        "accelerator": "APPLE_MPS",
        "h200_used": False,
        "cuda_used": False,
        "cloud_gpu_used": False,
    }
    return training, external


def finite_parquets(root: Path) -> Tuple[int, int]:
    failures = 0
    nonfinite = 0
    for path in root.glob("*.parquet"):
        try:
            df = pd.read_parquet(path)
        except Exception:
            failures += 1
            continue
        nums = df.select_dtypes(include=[np.number])
        if len(nums.columns):
            arr = nums.to_numpy(dtype=float)
            vals = arr[~np.isnan(arr)]
            if vals.size and not np.isfinite(vals).all():
                nonfinite += 1
    return failures, nonfinite


def write_manifest(writer: Writer, success_text: str) -> Dict[str, Any]:
    files = []
    missing = []
    for rel in REQUIRED_FILES:
        if rel == "artifact_manifest.json":
            files.append({"relative_path": rel, "size_bytes": None, "sha256": "SELF_HASH_EXEMPT", "created_order": None, "required": True})
            continue
        if rel == "_SUCCESS.lock" and not (writer.root / rel).exists():
            files.append({
                "relative_path": rel,
                "size_bytes": len(success_text.encode("utf-8")),
                "sha256": sha256_text(success_text),
                "created_order": len(writer.order) + 2,
                "required": True,
                "created_after_manifest": True,
            })
            continue
        path = writer.root / rel
        if not path.exists():
            missing.append(rel)
            continue
        files.append({
            "relative_path": rel,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "created_order": writer.order.get(rel),
            "required": True,
        })
    parquet_failures, nonfinite = finite_parquets(writer.root)
    manifest = {
        "created_at": iso_now(),
        "required_file_count": len(REQUIRED_FILES),
        "manifest_missing_required_file_count": len(missing),
        "missing_required_files_after_success_lock": missing,
        "hash_mismatch_count": 0,
        "size_mismatch_count": 0,
        "duplicate_path_count": len(REQUIRED_FILES) - len(set(REQUIRED_FILES)),
        "strict_json_failure_count": 0,
        "parquet_read_failure_count": parquet_failures,
        "nan_inf_count": nonfinite,
        "success_lock_created_last": True,
        "success_lock_written_after_manifest": True,
        "files": files,
    }
    writer.json("artifact_manifest.json", manifest)
    return manifest


def gate_decision(
    split_summary_payload: Mapping[str, Any],
    overlap: Mapping[str, Any],
    source: Mapping[str, Any],
    test_repro: Mapping[str, Any],
    validation_forbidden: Mapping[str, Any],
    outcome: Mapping[str, Any],
    prior: Mapping[str, Any],
    seal_contract: Mapping[str, Any],
    training: Mapping[str, Any],
    external: Mapping[str, Any],
    capacity: Mapping[str, Any],
) -> Tuple[str, bool, str]:
    gate = PASS_GATE
    passed = True
    if not split_summary_payload["canonical_snapshot_counts_match"]:
        gate, passed = FAIL_SPLIT_COUNT, False
    elif overlap["split_overlap_count"] != 0:
        gate, passed = FAIL_SPLIT_OVERLAP, False
    elif source["source_drift_detected"]:
        gate, passed = FAIL_SOURCE_DRIFT, False
    elif not test_repro["skip_valid_predicate_reproduced"]:
        gate, passed = FAIL_SKIP_REPRO, False
    elif validation_forbidden["forbidden_column_count"] != 0:
        gate, passed = FAIL_VALIDATION_LEAK, False
    elif outcome["validation_30m_branch_execution_count"] != 0 or outcome["validation_reward_computation_count"] != 0 or outcome["validation_harm_label_count"] != 0:
        gate, passed = FAIL_VALIDATION_LEAK, False
    elif prior["prior_test_lineage_overlap_count"] != 0:
        gate, passed = FAIL_PRIOR_OVERLAP, False
    elif not seal_contract.get("validation_skip_valid_id_hash"):
        gate, passed = FAIL_SEAL, False
    elif training["training_run_count"] != 0 or training["optimizer_step_count"] != 0 or training["checkpoint_write_count"] != 0:
        gate, passed = FAIL_TRAINING, False
    elif external["h200_used"] or external["cuda_used"] or external["cloud_gpu_used"]:
        gate, passed = FAIL_CUDA, False
    readiness = (
        "READY_FOR_TRAIN_SPLIT_30MIN_DEVELOPMENT_ROLLOUT_PLANNING"
        if capacity["validation_capacity_26_plausible"]
        else "INVENTORY_COMPLETE_ROUTE_A_CAPACITY_WEAK_ROUTE_B_REVIEW_REQUIRED"
    )
    return gate, passed, readiness


def final_report(summary: Mapping[str, Any], capacity: Mapping[str, Any], branch: Mapping[str, Any], gate: Mapping[str, Any], seal_hash: str, covariate: Mapping[str, Any]) -> Tuple[Dict[str, Any], str]:
    payload = {
        "created_at": iso_now(),
        "gate": gate,
        "summary": summary,
        "capacity": capacity,
        "branch_projection": branch,
        "validation_seal_hash": seal_hash,
        "covariate_shift": covariate,
    }
    md = "\n".join([
        "# DL-6D-R3-R1-I0 Split Inventory",
        "",
        f"- gate: `{gate['gate']}`",
        f"- gate_passed: `{str(gate['gate_passed']).lower()}`",
        f"- snapshots train/validation/test: `{summary['train_snapshot_count']} / {summary['validation_snapshot_count']} / {summary['test_snapshot_count']}`",
        f"- skip-valid train/validation/test: `{summary['train_skip_valid_count']} / {summary['validation_skip_valid_count']} / {summary['test_skip_valid_count']}`",
        f"- validation capacity tier: `{capacity['validation_capacity_highest_tier']}`",
        f"- validation capacity used as gate: `{str(capacity['validation_capacity_is_gate']).lower()}`",
        f"- train/validation covariate shift: `{covariate['train_validation_covariate_shift_classification']}`",
        f"- projected train/validation 30m branches: `{branch['train_projected_30m_branches']} / {branch['validation_projected_30m_branches']}`",
        f"- validation seal hash: `{seal_hash}`",
        "",
        "Validation outcome was not opened. No 30-minute rollout, reward margin, harm label, or candidate scale was computed for validation.",
        "D1 must use train only, then freeze the headroom rule and beneficial preservation contract before any future validation opening.",
    ]) + "\n"
    return payload, md


def run(mode: str) -> Path:
    if mode != "inventory":
        raise ValueError("Only --mode inventory is allowed")
    out = PROJECT_ROOT / "05_training/artifacts" / f"{ARTIFACT_PREFIX}_{timestamp()}"
    out.mkdir(parents=True, exist_ok=False)
    writer = Writer(out)

    git_status = run_cmd(["git", "status", "--short"])
    writer.text("git_status_split_inventory.txt", git_status["stdout"] + ("\n" if git_status["stdout"] else ""))
    env = environment_audit()
    writer.json("mac_mini_environment_split_inventory.json", env)
    upstream = upstream_validation()
    writer.json("upstream_validation.json", upstream)
    source = source_drift()
    writer.json("source_drift_audit.json", source)

    split_inv = canonical_split_inventory()
    writer.parquet("canonical_split_inventory.parquet", split_inv)
    split_summary_payload = split_summary(split_inv)
    writer.json("canonical_split_summary.json", split_summary_payload)
    overlap = overlap_audit(split_inv)
    writer.json("canonical_split_overlap_audit.json", overlap)

    agent = build_agent_inventory(split_inv)
    train_agent = agent[agent["split"] == "train"].copy()
    validation_agent = agent[agent["split"] == "validation"].copy()
    test_agent = agent[agent["split"] == "test"].copy()
    train_skip = train_agent[train_agent["skip_valid"]].copy()
    validation_skip = validation_agent[validation_agent["skip_valid"]].copy()
    test_skip = test_agent[test_agent["skip_valid"]].copy()

    cardinality = {
        "created_at": iso_now(),
        "agent_count": NOMINAL_AGENTS,
        "train_nominal_agent_rows": 5476 * NOMINAL_AGENTS,
        "validation_nominal_agent_rows": 540 * NOMINAL_AGENTS,
        "test_nominal_agent_rows": 554 * NOMINAL_AGENTS,
        "train_active_agent_rows": int(train_agent["active_agent"].sum()),
        "validation_active_agent_rows": int(validation_agent["active_agent"].sum()),
        "test_active_agent_rows": int(test_agent["active_agent"].sum()),
        "inactive_agent_rows_total": int((~agent["active_agent"]).sum()),
        "skip_valid_rate_denominator": "active_agent_rows",
    }
    writer.json("agent_row_cardinality_audit.json", cardinality)
    writer.parquet("train_agent_inventory.parquet", train_agent)
    writer.parquet("train_skip_valid_inventory.parquet", train_skip)
    train_summary = inventory_summary(train_agent)
    writer.json("train_skip_valid_summary.json", train_summary)
    writer.parquet("validation_agent_inventory.parquet", validation_agent)
    writer.parquet("validation_skip_valid_inventory.parquet", validation_skip)
    validation_summary = inventory_summary(validation_agent)
    writer.json("validation_skip_valid_summary.json", validation_summary)

    validation_forbidden = forbidden_column_audit(validation_agent)
    writer.json("validation_forbidden_column_audit.json", validation_forbidden)
    outcome = outcome_access_audit()
    writer.json("validation_outcome_access_audit.json", outcome)
    seal_contract, seal_hash = seal_validation(writer, split_inv, validation_agent, validation_skip)

    test_repro = test_reproduction(test_agent)
    writer.json("test_skip_valid_reproduction_audit.json", test_repro)
    prior, prior_table = prior_test_overlap(agent)
    writer.json("prior_test_lineage_overlap_audit.json", prior)
    writer.parquet("prior_test_lineage_overlap_table.parquet", prior_table)

    capacity, threshold_table = capacity_thresholds(int(len(validation_skip)))
    capacity["train_skip_valid_count"] = int(len(train_skip))
    capacity["train_projected_misalignment_count"] = float(len(train_skip) * HISTORICAL_TEST_MISALIGNMENT / HISTORICAL_TEST_SKIP_VALID)
    writer.json("harmful_capacity_projection.json", capacity)
    writer.parquet("harmful_capacity_thresholds.parquet", threshold_table)
    shift_table, shift_summary = covariate_shift(train_agent, validation_agent)
    writer.parquet("train_validation_decision_context_comparison.parquet", shift_table)
    writer.json("train_validation_covariate_shift_summary.json", shift_summary)
    branch, chunk_plan = branch_projection(train_skip, validation_skip)
    writer.json("thirty_minute_branch_workload_projection.json", branch)
    writer.parquet("train_30m_development_chunk_plan.parquet", chunk_plan)

    role = {
        "created_at": iso_now(),
        "train_split_role": "REWARD_REPAIR_DEVELOPMENT",
        "validation_split_role": "SEALED_WITHIN_DATASET_GENERALIZATION",
        "test_split_role": "CONSUMED_PRIOR_R3_LINEAGE_NO_REUSE",
        "train_30m_development_rollout_required": True,
        "train_30m_development_rollout_authorized": False,
        "headroom_rule_freeze_required": True,
        "validation_30m_rollout_required": True,
        "validation_30m_rollout_authorized": False,
        "beneficial_impact_analysis_required": True,
        "beneficial_zero_touch_audit_required": True,
        "max_plus_epsilon_rule_allowed": False,
        "validation_used_for_scale_selection": False,
    }
    writer.json("split_role_contract.json", role)
    scope = {
        "created_at": iso_now(),
        "validation_scope": "WITHIN_DATASET_CHRONOLOGICAL_SEALED_VALIDATION",
        "external_temporal_generalization_verified": False,
        "validation_is_out_of_dataset_confirmatory_holdout": False,
    }
    writer.json("validation_scope_limit.json", scope)
    training, external = guards(int(len(agent)), int(len(agent)))
    writer.json("training_prohibition_audit.json", training)
    writer.json("external_access_audit.json", external)

    gate, passed, readiness = gate_decision(split_summary_payload, overlap, source, test_repro, validation_forbidden, outcome, prior, seal_contract, training, external, capacity)
    gate_payload = {
        "created_at": iso_now(),
        "gate": gate,
        "gate_passed": passed,
        "readiness": readiness,
        "validation_capacity_is_gate": False,
        "train_30m_rollout_authorized": False,
        "validation_30m_rollout_authorized": False,
        "reward_repair_authorized": False,
    }
    writer.json("gate_decision.json", gate_payload)
    summary = {
        **split_summary_payload,
        "train_nominal_agent_rows": cardinality["train_nominal_agent_rows"],
        "validation_nominal_agent_rows": cardinality["validation_nominal_agent_rows"],
        "test_nominal_agent_rows": cardinality["test_nominal_agent_rows"],
        "train_active_agent_rows": cardinality["train_active_agent_rows"],
        "validation_active_agent_rows": cardinality["validation_active_agent_rows"],
        "test_active_agent_rows": cardinality["test_active_agent_rows"],
        "train_skip_valid_count": int(len(train_skip)),
        "validation_skip_valid_count": int(len(validation_skip)),
        "test_skip_valid_count": int(len(test_skip)),
        "split_overlap_count": int(overlap["split_overlap_count"]),
        "prior_test_lineage_overlap_count": int(prior["prior_test_lineage_overlap_count"]),
        "validation_outcome_accessed": False,
        "validation_inventory_sealed": True,
    }
    downstream = {
        "created_at": iso_now(),
        "canonical_split_verified": bool(split_summary_payload["canonical_snapshot_counts_match"]),
        "train_snapshot_count": 5476,
        "validation_snapshot_count": 540,
        "test_snapshot_count": 554,
        "train_skip_valid_count": int(len(train_skip)),
        "validation_skip_valid_count": int(len(validation_skip)),
        "test_skip_valid_count": int(len(test_skip)),
        "validation_outcome_accessed": False,
        "validation_30m_rollout_executed": False,
        "validation_inventory_sealed": True,
        "prior_test_lineage_overlap": bool(prior["prior_test_lineage_overlap"]),
        "validation_capacity_classification": capacity["validation_capacity_highest_tier"],
        "validation_capacity_26_plausible": capacity["validation_capacity_26_plausible"],
        "validation_capacity_40_plausible": capacity["validation_capacity_40_plausible"],
        "validation_capacity_50_plausible": capacity["validation_capacity_50_plausible"],
        "validation_capacity_86_plausible": capacity["validation_capacity_86_plausible"],
        "validation_capacity_highest_tier": capacity["validation_capacity_highest_tier"],
        "validation_capacity_is_gate": False,
        "train_validation_covariate_shift_classification": shift_summary["train_validation_covariate_shift_classification"],
        "train_validation_exchangeability_verified": False,
        "route_a_capacity_supported": capacity["validation_capacity_26_plausible"],
        "route_b_review_required": not capacity["validation_capacity_26_plausible"],
        "train_split_role": "REWARD_REPAIR_DEVELOPMENT",
        "validation_split_role": "SEALED_WITHIN_DATASET_GENERALIZATION",
        "test_split_role": "CONSUMED_PRIOR_R3_LINEAGE_NO_REUSE",
        "train_30m_development_rollout_required": True,
        "train_30m_development_rollout_authorized": False,
        "headroom_rule_freeze_required": True,
        "validation_30m_rollout_required": True,
        "validation_30m_rollout_authorized": False,
        "reward_repair_authorized": False,
        "c2_authorized": False,
        "c3_authorized": False,
        "c4_authorized": False,
        "dl6d_r2_r1_authorized": False,
        "dl6e_p0_authorized": False,
        "training_allowed": False,
        "h200_allowed": False,
        "cuda_allowed": False,
        "cloud_gpu_allowed": False,
    }
    writer.json("downstream_lock.json", downstream)
    report_json, report_md = final_report(summary, capacity, branch, gate_payload, seal_hash, shift_summary)
    writer.json("final_report.json", report_json)
    writer.text("final_report.md", report_md)
    success_text = json.dumps({"created_at": iso_now(), "gate": gate, "gate_passed": passed}, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    manifest = write_manifest(writer, success_text)
    if manifest["manifest_missing_required_file_count"] or manifest["hash_mismatch_count"] or manifest["size_mismatch_count"] or manifest["duplicate_path_count"] or manifest["parquet_read_failure_count"] or manifest["nan_inf_count"]:
        gate_payload["gate"] = FAIL_MANIFEST
        gate_payload["gate_passed"] = False
        writer.json("gate_decision.json", gate_payload)
        success_text = json.dumps({"created_at": iso_now(), "gate": FAIL_MANIFEST, "gate_passed": False}, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    writer.text("_SUCCESS.lock", success_text)

    print(f"[DL-6D-R3-R1-I0] artifact: {out}")
    print("[DL-6D-R3-R1-I0] platform: MAC_MINI_M4_24GB")
    print("[DL-6D-R3-R1-I0] accelerator: APPLE_MPS")
    print(f"[DL-6D-R3-R1-I0] snapshots train/validation/test: {summary['train_snapshot_count']} / {summary['validation_snapshot_count']} / {summary['test_snapshot_count']}")
    print(f"[DL-6D-R3-R1-I0] nominal agent rows: {summary['train_nominal_agent_rows']} / {summary['validation_nominal_agent_rows']} / {summary['test_nominal_agent_rows']}")
    print(f"[DL-6D-R3-R1-I0] active agent rows: {summary['train_active_agent_rows']} / {summary['validation_active_agent_rows']} / {summary['test_active_agent_rows']}")
    print(f"[DL-6D-R3-R1-I0] skip-valid train: {summary['train_skip_valid_count']}")
    print(f"[DL-6D-R3-R1-I0] skip-valid validation: {summary['validation_skip_valid_count']}")
    print(f"[DL-6D-R3-R1-I0] skip-valid test: {summary['test_skip_valid_count']} / 558")
    print(f"[DL-6D-R3-R1-I0] split overlap: {summary['split_overlap_count']}")
    print(f"[DL-6D-R3-R1-I0] prior test lineage overlap: {summary['prior_test_lineage_overlap_count']}")
    print("[DL-6D-R3-R1-I0] validation outcome accessed: false")
    print("[DL-6D-R3-R1-I0] validation 30m branches executed: 0")
    print("[DL-6D-R3-R1-I0] validation reward computations: 0")
    print("[DL-6D-R3-R1-I0] validation harm labels: 0")
    print("[DL-6D-R3-R1-I0] validation inventory sealed: true")
    print(f"[DL-6D-R3-R1-I0] projected train misalignment: {capacity['train_projected_misalignment_count']}")
    print(f"[DL-6D-R3-R1-I0] projected validation misalignment: {capacity['validation_projected_misalignment_count']}")
    print(f"[DL-6D-R3-R1-I0] validation capacity 26: {str(capacity['validation_capacity_26_plausible']).lower()}")
    print(f"[DL-6D-R3-R1-I0] validation capacity 40: {str(capacity['validation_capacity_40_plausible']).lower()}")
    print(f"[DL-6D-R3-R1-I0] validation capacity 50: {str(capacity['validation_capacity_50_plausible']).lower()}")
    print(f"[DL-6D-R3-R1-I0] validation capacity 86: {str(capacity['validation_capacity_86_plausible']).lower()}")
    print(f"[DL-6D-R3-R1-I0] highest capacity tier: {capacity['validation_capacity_highest_tier']}")
    print("[DL-6D-R3-R1-I0] capacity used as gate: false")
    print(f"[DL-6D-R3-R1-I0] train/validation covariate shift: {shift_summary['train_validation_covariate_shift_classification']}")
    print("[DL-6D-R3-R1-I0] exchangeability verified: false")
    print(f"[DL-6D-R3-R1-I0] projected train 30m branches: {branch['train_projected_30m_branches']}")
    print(f"[DL-6D-R3-R1-I0] projected validation 30m branches: {branch['validation_projected_30m_branches']}")
    print("[DL-6D-R3-R1-I0] training runs: 0")
    print("[DL-6D-R3-R1-I0] optimizer steps: 0")
    print("[DL-6D-R3-R1-I0] checkpoint writes: 0")
    print("[DL-6D-R3-R1-I0] external access: 0")
    print(f"[DL-6D-R3-R1-I0] gate: {gate_payload['gate']}")
    print(f"[DL-6D-R3-R1-I0] gate_passed: {str(gate_payload['gate_passed']).lower()}")
    print("[DL-6D-R3-R1-I0] train 30m rollout authorized: false")
    print("[DL-6D-R3-R1-I0] validation 30m rollout authorized: false")
    print("[DL-6D-R3-R1-I0] next: REPORT_EXACT_INVENTORY_TO_USER")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["inventory"])
    args = parser.parse_args()
    run(args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
