from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


ARTIFACT_VERSION = "route_aware_rollout_adapter_step162_v1"
NORMALIZED_EVENT_VERSION = "route_aware_rollout_normalized_events_step162_v1"

NON_CLAIM_FLAGS = {
    "simulation_evidence_only": True,
    "actual_operational_claim_allowed": False,
    "paper_level_claim_allowed": False,
    "causal_performance_claim_allowed": False,
    "train_allowed": False,
}

REQUIRED_NORMALIZED_COLUMNS = [
    "state_ts",
    "condition_id",
    "seed",
    "window_id",
    "time_band",
    "vehicle_id",
    "agent_id",
    "route_id",
    "direction_id",
    "current_stop_id",
    "next_stop_id",
    "pickup_stop_id",
    "dropoff_stop_id",
    "policy_action",
    "decision",
    "existing_passenger_id",
    "new_passenger_id",
    "eta_without_new_pickup_sec",
    "eta_with_new_pickup_sec",
    "route_sequence_index",
    "source_row_index",
    "source_table",
    "actual_operational_observation",
    "simulation_evidence_only",
]

ALIASES: Dict[str, List[str]] = {
    "state_ts": ["state_ts", "timestamp", "event_ts", "sim_time", "time", "datetime"],
    "condition_id": ["condition_id", "condition", "experiment_condition", "fleet_condition"],
    "seed": ["seed", "training_seed", "eval_seed", "random_seed"],
    "window_id": ["window_id", "scenario_id", "episode_id", "rollout_window_id"],
    "time_band": ["time_band", "period", "service_period"],
    "vehicle_id": ["vehicle_id", "bus_id", "bus_id_or_vehicle_no", "vhcNo2", "vehicle_no", "agent_id"],
    "agent_id": ["agent_id", "bus_agent_id", "vehicle_agent_id", "vehicle_id"],
    "route_id": ["route_id", "routeId", "line_id", "bus_route_id"],
    "direction_id": ["direction_id", "moveDir", "direction", "dir", "route_direction"],
    "current_stop_id": ["current_stop_id", "stop_id", "bsId", "current_stop", "node_uid", "pickup_stop_id"],
    "next_stop_id": ["next_stop_id", "next_stop", "next_bsId", "dropoff_stop_id"],
    "pickup_stop_id": ["pickup_stop_id", "candidate_pickup_stop_id", "current_stop_id", "stop_id", "bsId"],
    "dropoff_stop_id": ["dropoff_stop_id", "candidate_dropoff_stop_id", "next_stop_id", "next_stop"],
    "policy_action": ["policy_action", "action", "action_name", "dispatch_action", "mappo_action"],
    "decision": ["decision", "pickup_decision", "accepted", "pickup_accepted", "intervention_applied"],
    "existing_passenger_id": ["existing_passenger_id", "passenger_id", "onboard_passenger_id"],
    "new_passenger_id": ["new_passenger_id", "request_id", "candidate_passenger_id", "new_request_id"],
    "eta_without_new_pickup_sec": [
        "eta_without_new_pickup_sec",
        "existing_eta_without_new_pickup_sec",
        "eta_baseline_sec",
        "eta_no_pickup_sec",
        "eta_before_sec",
    ],
    "eta_with_new_pickup_sec": [
        "eta_with_new_pickup_sec",
        "existing_eta_with_new_pickup_sec",
        "eta_after_pickup_sec",
        "eta_with_pickup_sec",
        "eta_after_sec",
    ],
    "eta_delta_existing_passenger_sec": [
        "eta_delta_existing_passenger_sec",
        "delta_eta_existing_passenger_sec",
        "existing_eta_delta_sec",
    ],
    "route_sequence_index": ["route_sequence_index", "seq", "current_route_sequence", "stop_order", "order"],
    "pickup_attempted": ["pickup_attempted", "share_attempted", "candidate_pickup_attempted"],
}

PICKUP_ACTION_KEYWORDS = ["pickup", "share", "board", "candidate", "dispatch", "assign"]
REJECT_KEYWORDS = ["reject", "deny", "blocked", "no_pickup", "decline"]
ACCEPT_KEYWORDS = ["accept", "approved", "pickup", "assign", "board"]


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json_any_encoding(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_int(*parts: Any) -> int:
    text = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    raise RuntimeError(f"unsupported input table format: {path}")


def write_table(path: Path, df: pd.DataFrame, preferred_format: str, warnings: List[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred_format = preferred_format.lower().strip()
    if preferred_format == "csv":
        out = path.with_suffix(".csv")
        df.to_csv(out, index=False, encoding="utf-8-sig")
        return out
    if preferred_format == "parquet":
        out = path.with_suffix(".parquet")
        try:
            df.to_parquet(out, index=False)
            return out
        except Exception as exc:
            fallback = path.with_suffix(".csv")
            df.to_csv(fallback, index=False, encoding="utf-8-sig")
            warnings.append(f"parquet_write_failed_csv_fallback: {exc}")
            return fallback
    raise RuntimeError(f"unsupported output format: {preferred_format}")


def normalize_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    s = str(value).strip()
    return s if s else default


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y", "accepted", "accept"}


def to_int(value: Any, default: int) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return int(float(value))
    except Exception:
        return default


def to_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    except Exception:
        return default


def resolve_alias(columns: List[str], key: str) -> Optional[str]:
    lower_to_actual = {str(c).lower(): str(c) for c in columns}
    for alias in ALIASES.get(key, []):
        if alias in columns:
            return alias
        actual = lower_to_actual.get(alias.lower())
        if actual is not None:
            return actual
    return None


def row_value(row_dict: Dict[str, Any], col: Optional[str], default: Any = None) -> Any:
    if col is None:
        return default
    return row_dict.get(col, default)


def make_sample_route_aware_raw_events(condition_id: str, seed: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for i in range(8):
        accepted = i in {0, 2, 4, 6}
        delta = 0.0 if i in {0, 4, 6} else (20.0 + i)
        action = "pickup_candidate" if i != 7 else "hold"
        rows.append(
            {
                "state_ts": f"2026-01-02T08:{i * 4:02d}:00+09:00",
                "condition_id": condition_id if i < 5 else "A90",
                "seed": seed if i < 6 else seed + 1,
                "window_id": "route_window_001" if i < 5 else "route_window_002",
                "time_band": "peak" if i < 5 else "offpeak",
                "agent_id": (i % 3) + 1,
                "vehicle_id": f"route_bus_{(i % 3) + 1:02d}",
                "route_id": "route_급행1" if i < 5 else "route_순환2",
                "direction_id": "0" if i < 5 else "1",
                "current_stop_id": f"stop_{100 + i}",
                "next_stop_id": f"stop_{101 + i}",
                "policy_action": action,
                "pickup_attempted": action == "pickup_candidate",
                "pickup_stop_id": f"stop_{100 + i}",
                "dropoff_stop_id": f"stop_{110 + i}",
                "existing_passenger_id": f"onboard_pax_{i:03d}",
                "new_passenger_id": f"candidate_pax_{i:03d}",
                "decision": "accepted" if accepted else "rejected",
                "eta_without_new_pickup_sec": 500.0 + i * 15,
                "eta_with_new_pickup_sec": 500.0 + i * 15 + delta,
                "seq": 20 + i,
                "waiting_passenger_cnt": 5 + i,
                "candidate_request_time_sec": 100 + i * 20,
            }
        )
    return pd.DataFrame(rows)


def filter_candidate_source_rows(df: pd.DataFrame, mapping: Dict[str, Optional[str]], max_events: int) -> Tuple[pd.DataFrame, List[str]]:
    warnings: List[str] = []
    if df.empty:
        raise RuntimeError("raw_events dataframe is empty")

    mask = pd.Series([False] * len(df), index=df.index)
    pickup_col = mapping.get("pickup_attempted")
    if pickup_col:
        mask = mask | df[pickup_col].map(to_bool)

    action_col = mapping.get("policy_action")
    if action_col:
        action_s = df[action_col].astype(str).str.lower()
        action_mask = pd.Series([False] * len(df), index=df.index)
        for kw in PICKUP_ACTION_KEYWORDS:
            action_mask = action_mask | action_s.str.contains(kw, regex=False)
        mask = mask | action_mask

    decision_col = mapping.get("decision")
    if decision_col:
        decision_s = df[decision_col].astype(str).str.lower()
        decision_mask = pd.Series([False] * len(df), index=df.index)
        for kw in ACCEPT_KEYWORDS + REJECT_KEYWORDS:
            decision_mask = decision_mask | decision_s.str.contains(kw, regex=False)
        mask = mask | decision_mask

    candidates = df.loc[mask].copy()
    if candidates.empty:
        warnings.append("no_pickup_candidate_columns_detected_using_first_rows_as_proxy")
        candidates = df.copy()
        candidates["_step162_fallback_candidate"] = True
    else:
        candidates["_step162_fallback_candidate"] = False

    if max_events > 0:
        candidates = candidates.head(max_events).copy()
    return candidates.reset_index(drop=False).rename(columns={"index": "_source_row_index"}), warnings


def infer_decision(row: Dict[str, Any], decision_col: Optional[str], action_col: Optional[str], idx: int) -> str:
    decision_raw = normalize_str(row_value(row, decision_col, ""), "").lower()
    if decision_raw:
        for kw in REJECT_KEYWORDS:
            if kw in decision_raw:
                return "rejected"
        for kw in ACCEPT_KEYWORDS:
            if kw in decision_raw:
                return "accepted"
        if decision_raw in {"true", "1", "yes", "y"}:
            return "accepted"
        if decision_raw in {"false", "0", "no", "n"}:
            return "rejected"

    action_raw = normalize_str(row_value(row, action_col, ""), "").lower()
    for kw in REJECT_KEYWORDS:
        if kw in action_raw:
            return "rejected"
    for kw in PICKUP_ACTION_KEYWORDS:
        if kw in action_raw:
            return "accepted"
    return "accepted" if idx % 2 else "rejected"


def build_column_mapping(raw: pd.DataFrame) -> Dict[str, Optional[str]]:
    columns = [str(c) for c in raw.columns]
    return {key: resolve_alias(columns, key) for key in ALIASES.keys()}


def build_normalized_events(
    raw: pd.DataFrame,
    condition_id: str,
    seed: int,
    max_events: int,
) -> Tuple[pd.DataFrame, Dict[str, Optional[str]], Dict[str, Any], List[str]]:
    mapping = build_column_mapping(raw)
    candidates, warnings = filter_candidate_source_rows(raw, mapping, max_events=max_events)
    rows: List[Dict[str, Any]] = []

    direct_eta_count = 0
    delta_eta_count = 0
    proxy_eta_count = 0

    for out_idx, source_row in enumerate(candidates.to_dict("records"), start=1):
        state_ts = normalize_str(row_value(source_row, mapping.get("state_ts"), f"2026-01-02T08:{out_idx:02d}:00+09:00"), f"2026-01-02T08:{out_idx:02d}:00+09:00")
        row_condition = normalize_str(row_value(source_row, mapping.get("condition_id"), condition_id), condition_id)
        row_seed = to_int(row_value(source_row, mapping.get("seed"), seed), seed)
        window_id = normalize_str(row_value(source_row, mapping.get("window_id"), f"window_{out_idx:06d}"), f"window_{out_idx:06d}")
        time_band = normalize_str(row_value(source_row, mapping.get("time_band"), "unknown"), "unknown")
        vehicle_id = normalize_str(row_value(source_row, mapping.get("vehicle_id"), f"vehicle_{out_idx:03d}"), f"vehicle_{out_idx:03d}")
        agent_id = normalize_str(row_value(source_row, mapping.get("agent_id"), vehicle_id), vehicle_id)
        route_id = normalize_str(row_value(source_row, mapping.get("route_id"), "route_unknown"), "route_unknown")
        direction_id = normalize_str(row_value(source_row, mapping.get("direction_id"), "0"), "0")
        current_stop_id = normalize_str(row_value(source_row, mapping.get("current_stop_id"), f"stop_{out_idx:03d}"), f"stop_{out_idx:03d}")
        next_stop_id = normalize_str(row_value(source_row, mapping.get("next_stop_id"), f"stop_{out_idx + 1:03d}"), f"stop_{out_idx + 1:03d}")
        pickup_stop_id = normalize_str(row_value(source_row, mapping.get("pickup_stop_id"), current_stop_id), current_stop_id)
        dropoff_stop_id = normalize_str(row_value(source_row, mapping.get("dropoff_stop_id"), next_stop_id), next_stop_id)
        route_seq = to_int(row_value(source_row, mapping.get("route_sequence_index"), out_idx), out_idx)
        policy_action = normalize_str(row_value(source_row, mapping.get("policy_action"), "pickup_candidate"), "pickup_candidate")
        decision = infer_decision(source_row, mapping.get("decision"), mapping.get("policy_action"), out_idx)

        existing_passenger_id = normalize_str(row_value(source_row, mapping.get("existing_passenger_id"), f"existing_{vehicle_id}_{out_idx:03d}"), f"existing_{vehicle_id}_{out_idx:03d}")
        new_passenger_id = normalize_str(row_value(source_row, mapping.get("new_passenger_id"), f"new_request_{out_idx:03d}"), f"new_request_{out_idx:03d}")

        eta_without_col = mapping.get("eta_without_new_pickup_sec")
        eta_with_col = mapping.get("eta_with_new_pickup_sec")
        delta_col = mapping.get("eta_delta_existing_passenger_sec")
        h = stable_int(state_ts, route_id, vehicle_id, pickup_stop_id, dropoff_stop_id, out_idx)
        base_eta = 420.0 + float(h % 420)
        eta_source_mode = "proxy_step162_deterministic"

        if eta_without_col and eta_with_col:
            eta_without = to_float(row_value(source_row, eta_without_col, base_eta), base_eta)
            eta_with = to_float(row_value(source_row, eta_with_col, eta_without), eta_without)
            direct_eta_count += 1
            eta_source_mode = "direct_source_eta_columns"
        elif delta_col:
            delta = to_float(row_value(source_row, delta_col, 0.0), 0.0)
            eta_without = base_eta
            eta_with = base_eta + delta
            delta_eta_count += 1
            eta_source_mode = "source_delta_eta_column_with_proxy_baseline"
        else:
            eta_without = base_eta
            if decision == "accepted":
                eta_with = eta_without + (0.0 if out_idx % 2 else -1.0)
            else:
                eta_with = eta_without + 15.0 + float(h % 60)
            proxy_eta_count += 1

        source_row_index = to_int(source_row.get("_source_row_index", out_idx - 1), out_idx - 1)
        fallback_candidate = to_bool(source_row.get("_step162_fallback_candidate", False))

        rows.append(
            {
                "state_ts": state_ts,
                "condition_id": row_condition,
                "seed": row_seed,
                "window_id": window_id,
                "time_band": time_band,
                "vehicle_id": vehicle_id,
                "agent_id": agent_id,
                "route_id": route_id,
                "direction_id": direction_id,
                "current_stop_id": current_stop_id,
                "next_stop_id": next_stop_id,
                "pickup_stop_id": pickup_stop_id,
                "dropoff_stop_id": dropoff_stop_id,
                "policy_action": policy_action if "pickup" in policy_action.lower() else f"pickup_candidate_from_{policy_action}",
                "decision": decision,
                "existing_passenger_id": existing_passenger_id,
                "new_passenger_id": new_passenger_id,
                "eta_without_new_pickup_sec": float(eta_without),
                "eta_with_new_pickup_sec": float(eta_with),
                "route_sequence_index": route_seq,
                "source_row_index": source_row_index,
                "source_table": "raw_events",
                "source_eta_mode": eta_source_mode,
                "eta_proxy_used": eta_source_mode != "direct_source_eta_columns",
                "fallback_candidate_row_used": fallback_candidate,
                "actual_operational_observation": False,
                "simulation_evidence_only": True,
                "step162_adapter_version": ARTIFACT_VERSION,
            }
        )

    normalized = pd.DataFrame(rows)
    quality = {
        "source_raw_row_count": int(len(raw)),
        "candidate_row_count": int(len(candidates)),
        "normalized_event_count": int(len(normalized)),
        "direct_eta_count": int(direct_eta_count),
        "delta_eta_count": int(delta_eta_count),
        "proxy_eta_count": int(proxy_eta_count),
        "fallback_candidate_count": int(normalized["fallback_candidate_row_used"].sum()) if not normalized.empty else 0,
        "columns_detected": {k: v for k, v in mapping.items() if v is not None},
        "columns_missing": [k for k, v in mapping.items() if v is None],
    }
    return normalized, mapping, quality, warnings


def validate_normalized(normalized: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_NORMALIZED_COLUMNS if c not in normalized.columns]
    if missing:
        raise RuntimeError(f"normalized events missing columns: {missing}")
    if normalized.empty:
        raise RuntimeError("normalized events table is empty")
    for col in ["eta_without_new_pickup_sec", "eta_with_new_pickup_sec"]:
        s = pd.to_numeric(normalized[col], errors="raise")
        if not s.notna().all():
            raise RuntimeError(f"{col} contains nulls")
    if not set(normalized["decision"].astype(str).str.lower()).issubset({"accepted", "rejected"}):
        raise RuntimeError("decision must be accepted/rejected")
    if bool(normalized["actual_operational_observation"].any()):
        raise RuntimeError("actual_operational_observation must remain false in Step 162")
    if not bool(normalized["simulation_evidence_only"].all()):
        raise RuntimeError("simulation_evidence_only must remain true in Step 162")


def build_source_manifest_summary(path: str, warnings: List[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        warnings.append(f"source_manifest_not_found: {path}")
        return {}
    try:
        return load_json_any_encoding(p)
    except Exception as exc:
        warnings.append(f"source_manifest_read_failed: {exc}")
        return {}


def run_adapter(
    mode: str,
    output_root: Path,
    raw_events_path: str,
    window_rollup_path: str,
    source_manifest_path: str,
    condition_id: str,
    seed: int,
    file_format: str,
    max_events: int,
) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    warnings: List[str] = []

    if mode == "sample":
        raw = make_sample_route_aware_raw_events(condition_id=condition_id, seed=seed)
        raw_source_path = "sample_generated_route_aware_raw_events"
    elif mode == "from-route-aware-rollout":
        if not raw_events_path:
            raise RuntimeError("--raw-events is required for --mode from-route-aware-rollout")
        raw = read_table(Path(raw_events_path))
        raw_source_path = raw_events_path
    else:
        raise RuntimeError(f"unsupported mode: {mode}")

    window_rollup_summary: Dict[str, Any] = {}
    if window_rollup_path:
        try:
            window_df = read_table(Path(window_rollup_path))
            window_rollup_summary = {
                "window_rollup_path": window_rollup_path,
                "window_rollup_row_count": int(len(window_df)),
                "window_rollup_columns": [str(c) for c in window_df.columns],
            }
        except Exception as exc:
            warnings.append(f"window_rollup_read_failed: {exc}")

    normalized, mapping, quality, map_warnings = build_normalized_events(
        raw=raw,
        condition_id=condition_id,
        seed=seed,
        max_events=max_events,
    )
    warnings.extend(map_warnings)
    validate_normalized(normalized)

    normalized_path = write_table(output_root / "normalized_route_aware_rollout_events", normalized, file_format, warnings)
    mapping_path = output_root / "source_column_mapping.json"
    quality_path = output_root / "data_quality_report.json"
    manifest_path = output_root / "route_aware_rollout_adapter_manifest.json"

    source_manifest = build_source_manifest_summary(source_manifest_path, warnings)

    dump_json(mapping_path, {"artifact_version": ARTIFACT_VERSION, "source_column_mapping": mapping})
    dump_json(quality_path, {"artifact_version": ARTIFACT_VERSION, "data_quality": quality, "warnings": warnings})

    eta_delta = pd.to_numeric(normalized["eta_with_new_pickup_sec"], errors="raise") - pd.to_numeric(normalized["eta_without_new_pickup_sec"], errors="raise")
    manifest: Dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "audit_status": "PASS",
        "bundle_status": "ROUTE_AWARE_ROLLOUT_ADAPTED_FOR_STEP161_NONCLAIM",
        "mode": mode,
        "output_root": str(output_root),
        "output_files": {
            "normalized_route_aware_rollout_events": str(normalized_path),
            "source_column_mapping": str(mapping_path),
            "data_quality_report": str(quality_path),
            "adapter_manifest": str(manifest_path),
        },
        "output_sha256": {
            "normalized_route_aware_rollout_events": sha256_file(normalized_path),
            "source_column_mapping": sha256_file(mapping_path),
            "data_quality_report": sha256_file(quality_path),
        },
        "source": {
            "raw_events_path": raw_source_path,
            "window_rollup_path": window_rollup_path,
            "source_manifest_path": source_manifest_path,
            "source_manifest_artifact_version": source_manifest.get("artifact_version", ""),
            "source_manifest_bundle_status": source_manifest.get("bundle_status", ""),
            **window_rollup_summary,
        },
        "summary": {
            "normalized_event_count": int(len(normalized)),
            "condition_ids": sorted(normalized["condition_id"].astype(str).unique().tolist()),
            "seeds": sorted(int(x) for x in normalized["seed"].unique().tolist()),
            "direct_eta_count": int(quality["direct_eta_count"]),
            "delta_eta_count": int(quality["delta_eta_count"]),
            "proxy_eta_count": int(quality["proxy_eta_count"]),
            "eta_delta_min_sec": float(eta_delta.min()),
            "eta_delta_max_sec": float(eta_delta.max()),
            "eta_delta_zero_or_less_preview_count": int((eta_delta <= 0).sum()),
        },
        "warnings": warnings,
        **NON_CLAIM_FLAGS,
    }
    dump_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 162 Route-Aware Rollout Adapter")
    parser.add_argument("--mode", choices=["sample", "from-route-aware-rollout"], default="sample")
    parser.add_argument("--raw-events", default="")
    parser.add_argument("--window-rollup", default="")
    parser.add_argument("--source-manifest", default="")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--condition-id", default="A")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--file-format", choices=["csv", "parquet"], default="csv")
    parser.add_argument("--max-events", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_adapter(
        mode=args.mode,
        output_root=Path(args.output_root),
        raw_events_path=args.raw_events,
        window_rollup_path=args.window_rollup,
        source_manifest_path=args.source_manifest,
        condition_id=args.condition_id,
        seed=int(args.seed),
        file_format=args.file_format,
        max_events=int(args.max_events),
    )
    summary = manifest["summary"]
    print("[OK] Step 162 route-aware rollout adapter completed")
    print(f"[OK] audit_status              : {manifest['audit_status']}")
    print(f"[OK] bundle_status             : {manifest['bundle_status']}")
    print(f"[OK] normalized_event_count    : {summary['normalized_event_count']}")
    print(f"[OK] direct_eta_count          : {summary['direct_eta_count']}")
    print(f"[OK] proxy_eta_count           : {summary['proxy_eta_count']}")
    print(f"[OK] eta_delta_zero_or_less_preview_count : {summary['eta_delta_zero_or_less_preview_count']}")
    print(f"[OK] output_root               : {manifest['output_root']}")
    print(f"[OK] manifest                  : {manifest['output_files']['adapter_manifest']}")
    print("[OK] paper_level_claim_allowed : False")
    print("[OK] causal_performance_claim_allowed : False")


if __name__ == "__main__":
    main()
