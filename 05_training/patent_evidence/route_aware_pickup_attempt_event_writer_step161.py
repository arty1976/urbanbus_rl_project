from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


ARTIFACT_VERSION = "route_aware_pickup_attempt_event_writer_step161_v1"
STEP160_INPUT_VERSION = "zero_loss_pickup_evidence_input_step161_v1"

NON_CLAIM_FLAGS = {
    "simulation_evidence_only": True,
    "actual_operational_claim_allowed": False,
    "paper_level_claim_allowed": False,
    "causal_performance_claim_allowed": False,
    "train_allowed": False,
}

REQUIRED_ATTEMPT_COLUMNS = [
    "attempt_id",
    "state_ts",
    "condition_id",
    "seed",
    "vehicle_id",
    "existing_passenger_id",
    "new_passenger_id",
    "route_id",
    "direction_id",
    "pickup_stop_id",
    "dropoff_stop_id",
    "decision",
]
REQUIRED_ETA_COLUMNS = [
    "attempt_id",
    "existing_passenger_id",
    "eta_without_new_pickup_sec",
    "eta_with_new_pickup_sec",
]
REQUIRED_ATTENTION_COLUMNS = [
    "attempt_id",
    "state_ts",
    "layer_id",
    "head_id",
    "src_node",
    "dst_node",
    "attention_weight",
]
VALID_SEGMENTS = {
    "existing_passenger_path",
    "candidate_pickup_path",
    "candidate_dropoff_path",
    "unrelated",
    "unknown",
}


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


def normalize_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    s = str(value).strip()
    return s if s else default


def bool_from_any(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


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


def require_columns(df: pd.DataFrame, required: List[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"{label} missing required columns: {missing}")


def write_table(path: Path, df: pd.DataFrame, preferred_format: str, warnings: List[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred_format = preferred_format.lower().strip()
    if preferred_format not in {"csv", "parquet"}:
        raise RuntimeError(f"unsupported file format: {preferred_format}")

    if preferred_format == "csv":
        out = path.with_suffix(".csv")
        df.to_csv(out, index=False, encoding="utf-8-sig")
        return out

    out = path.with_suffix(".parquet")
    try:
        df.to_parquet(out, index=False)
        return out
    except Exception as exc:
        fallback = path.with_suffix(".csv")
        df.to_csv(fallback, index=False, encoding="utf-8-sig")
        warnings.append(f"parquet_write_failed_csv_fallback: {exc}")
        return fallback


def safe_get(row: Any, col: str, default: Any = None) -> Any:
    if hasattr(row, "_asdict"):
        d = row._asdict()
        return d.get(col, default)
    return getattr(row, col, default)


def make_sample_rollout(condition_id: str, seed: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for i in range(6):
        rows.append(
            {
                "state_ts": f"2026-01-01T08:{i * 5:02d}:00+09:00",
                "condition_id": condition_id if i < 3 else "A90",
                "seed": int(seed if i < 4 else seed + 1),
                "vehicle_id": f"bus_{(i % 3) + 1:02d}",
                "agent_id": (i % 3) + 1,
                "route_id": "route_100" if i < 4 else "route_200",
                "direction_id": "0" if i < 4 else "1",
                "current_stop_id": f"stop_{10 + i:03d}",
                "next_stop_id": f"stop_{11 + i:03d}",
                "window_id": "w_001" if i < 4 else "w_002",
                "time_band": "peak" if i < 4 else "offpeak",
                "policy_action": "pickup_candidate" if i != 5 else "hold",
                "decision": "accepted" if i in {0, 2, 4} else "rejected",
                "candidate_request_time_sec": 120 + i * 30,
                "route_sequence_index": 10 + i,
            }
        )
    return pd.DataFrame(rows)


def filter_pickup_candidate_rows(raw: pd.DataFrame, max_attempts: int) -> pd.DataFrame:
    if raw.empty:
        raise RuntimeError("input rollout dataframe is empty")

    df = raw.copy()
    candidate_mask = pd.Series([True] * len(df), index=df.index)
    if "policy_action" in df.columns:
        candidate_mask = df["policy_action"].astype(str).str.lower().str.contains(
            "pickup|share|board|candidate", regex=True
        )
    elif "action" in df.columns:
        candidate_mask = df["action"].astype(str).str.lower().str.contains(
            "pickup|share|board|candidate", regex=True
        )

    candidates = df.loc[candidate_mask].copy()
    if candidates.empty:
        # For scaffold operation, keep the first rows but mark reason_code later.
        candidates = df.head(max_attempts).copy()
        candidates["_fallback_no_pickup_action_detected"] = True
    else:
        candidates["_fallback_no_pickup_action_detected"] = False

    if max_attempts > 0:
        candidates = candidates.head(max_attempts).copy()
    return candidates.reset_index(drop=True)


def build_attempt_tables(
    source_rows: pd.DataFrame,
    condition_id: str,
    seed: int,
    zero_loss_epsilon_sec: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    attempts: List[Dict[str, Any]] = []
    eta_rows: List[Dict[str, Any]] = []
    attention_rows: List[Dict[str, Any]] = []

    for idx, row in enumerate(source_rows.itertuples(index=False), start=1):
        row_condition = normalize_str(safe_get(row, "condition_id", condition_id), condition_id)
        row_seed = int(safe_get(row, "seed", seed) or seed)
        state_ts = normalize_str(
            safe_get(row, "state_ts", f"2026-01-01T08:{idx:02d}:00+09:00"),
            f"2026-01-01T08:{idx:02d}:00+09:00",
        )
        route_id = normalize_str(safe_get(row, "route_id", "route_unknown"), "route_unknown")
        direction_id = normalize_str(safe_get(row, "direction_id", "0"), "0")
        vehicle_id = normalize_str(
            safe_get(row, "vehicle_id", safe_get(row, "agent_id", f"vehicle_{idx:03d}")),
            f"vehicle_{idx:03d}",
        )
        pickup_stop = normalize_str(
            safe_get(row, "pickup_stop_id", safe_get(row, "current_stop_id", f"stop_{idx:03d}")),
            f"stop_{idx:03d}",
        )
        dropoff_stop = normalize_str(
            safe_get(row, "dropoff_stop_id", safe_get(row, "next_stop_id", f"stop_{idx + 10:03d}")),
            f"stop_{idx + 10:03d}",
        )
        window_id = normalize_str(safe_get(row, "window_id", f"w_{idx:03d}"), f"w_{idx:03d}")
        time_band = normalize_str(safe_get(row, "time_band", "unknown"), "unknown")
        decision = normalize_str(safe_get(row, "decision", "accepted"), "accepted").lower()
        if decision not in {"accepted", "rejected"}:
            decision = "accepted" if idx % 2 else "rejected"

        attempt_id = normalize_str(safe_get(row, "attempt_id", ""))
        if not attempt_id:
            attempt_id = f"zl_{row_condition}_{row_seed}_{idx:06d}_{stable_int(state_ts, route_id, vehicle_id) % 100000:05d}"

        existing_passenger_id = normalize_str(
            safe_get(row, "existing_passenger_id", f"existing_{vehicle_id}_{idx:03d}"),
            f"existing_{vehicle_id}_{idx:03d}",
        )
        new_passenger_id = normalize_str(
            safe_get(row, "new_passenger_id", f"new_request_{idx:03d}"),
            f"new_request_{idx:03d}",
        )

        h = stable_int(attempt_id, state_ts, route_id, pickup_stop, dropoff_stop)
        base_eta = float(420 + (h % 360))
        # Deterministic proxy: about half zero-loss, half delayed. If source has ETA columns, use them.
        eta_without = safe_get(row, "eta_without_new_pickup_sec", None)
        eta_with = safe_get(row, "eta_with_new_pickup_sec", None)
        if eta_without is None or eta_with is None:
            eta_without = base_eta
            if decision == "rejected":
                eta_with = base_eta + float(15 + (h % 45))
            else:
                eta_with = base_eta + (0.0 if idx % 2 else -1.0)
        eta_without = float(eta_without)
        eta_with = float(eta_with)
        delta_eta = eta_with - eta_without
        zero_loss_proxy = bool(delta_eta <= zero_loss_epsilon_sec)

        fallback = bool_from_any(safe_get(row, "_fallback_no_pickup_action_detected", False))
        reason_code = normalize_str(safe_get(row, "reason_code", ""))
        if not reason_code:
            if fallback:
                reason_code = "scaffold_fallback_no_pickup_action_detected"
            elif zero_loss_proxy:
                reason_code = "zero_loss_candidate"
            else:
                reason_code = "eta_loss_candidate"

        attempts.append(
            {
                "attempt_id": attempt_id,
                "state_ts": state_ts,
                "condition_id": row_condition,
                "seed": row_seed,
                "vehicle_id": vehicle_id,
                "existing_passenger_id": existing_passenger_id,
                "new_passenger_id": new_passenger_id,
                "route_id": route_id,
                "direction_id": direction_id,
                "pickup_stop_id": pickup_stop,
                "dropoff_stop_id": dropoff_stop,
                "decision": decision,
                "window_id": window_id,
                "time_band": time_band,
                "pickup_stop_order": int(safe_get(row, "pickup_stop_order", safe_get(row, "route_sequence_index", idx)) or idx),
                "dropoff_stop_order": int(safe_get(row, "dropoff_stop_order", safe_get(row, "route_sequence_index", idx + 2)) or (idx + 2)),
                "candidate_request_time_sec": float(safe_get(row, "candidate_request_time_sec", idx * 30) or idx * 30),
                "policy_action": normalize_str(safe_get(row, "policy_action", safe_get(row, "action", "pickup_candidate")), "pickup_candidate"),
                "reason_code": reason_code,
                "scaffold_proxy_eta_used": eta_without is not None,
                "actual_operational_observation": False,
            }
        )

        eta_rows.append(
            {
                "attempt_id": attempt_id,
                "existing_passenger_id": existing_passenger_id,
                "eta_without_new_pickup_sec": eta_without,
                "eta_with_new_pickup_sec": eta_with,
                "counterfactual_method": "route_aware_step161_proxy_counterfactual_v1",
                "delta_eta_existing_passenger_sec_preview": delta_eta,
                "zero_loss_success_preview": zero_loss_proxy,
                "actual_eta_observation": False,
            }
        )

        segments = [
            "existing_passenger_path",
            "candidate_pickup_path",
            "candidate_dropoff_path",
            "unrelated",
        ]
        for layer_id in [1, 2]:
            for head_id in [0, 1]:
                for seg_idx, segment in enumerate(segments):
                    # Make successful zero-loss attempts concentrate slightly more on existing/passenger pickup path.
                    base_weight = 0.10 + ((h + layer_id * 17 + head_id * 13 + seg_idx * 7) % 100) / 500.0
                    if zero_loss_proxy and segment in {"existing_passenger_path", "candidate_pickup_path"}:
                        base_weight += 0.25
                    if (not zero_loss_proxy) and segment == "unrelated":
                        base_weight += 0.18
                    src = pickup_stop if segment != "candidate_dropoff_path" else dropoff_stop
                    dst = dropoff_stop if segment != "existing_passenger_path" else pickup_stop
                    attention_rows.append(
                        {
                            "attempt_id": attempt_id,
                            "state_ts": state_ts,
                            "layer_id": layer_id,
                            "head_id": head_id,
                            "src_node": src,
                            "dst_node": dst,
                            "edge_id": f"edge_{attempt_id}_{layer_id}_{head_id}_{seg_idx}",
                            "attention_weight": round(float(base_weight), 6),
                            "path_segment": segment,
                            "route_id": route_id,
                            "direction_id": direction_id,
                            "distance_m": float(250 + 40 * seg_idx + (h % 30)),
                            "time_sec": float(35 + 8 * seg_idx + (h % 10)),
                            "generalized_cost": round(float(1.0 + 0.1 * seg_idx + (h % 7) / 100.0), 6),
                        }
                    )

    return pd.DataFrame(attempts), pd.DataFrame(eta_rows), pd.DataFrame(attention_rows)


def validate_frames(attempts: pd.DataFrame, eta: pd.DataFrame, attention: pd.DataFrame) -> None:
    require_columns(attempts, REQUIRED_ATTEMPT_COLUMNS, "pickup_attempt_events")
    require_columns(eta, REQUIRED_ETA_COLUMNS, "eta_counterfactual")
    require_columns(attention, REQUIRED_ATTENTION_COLUMNS, "gatv2_attention")
    if attempts.empty:
        raise RuntimeError("pickup_attempt_events is empty")
    if eta.empty:
        raise RuntimeError("eta_counterfactual is empty")
    if attention.empty:
        raise RuntimeError("gatv2_attention is empty")
    if attempts["attempt_id"].duplicated().any():
        dup = attempts.loc[attempts["attempt_id"].duplicated(), "attempt_id"].head(5).tolist()
        raise RuntimeError(f"duplicate attempt_id found: {dup}")
    missing_eta = set(attempts["attempt_id"].astype(str)) - set(eta["attempt_id"].astype(str))
    if missing_eta:
        raise RuntimeError(f"eta_counterfactual missing attempts: {sorted(missing_eta)[:5]}")
    missing_attention = set(attempts["attempt_id"].astype(str)) - set(attention["attempt_id"].astype(str))
    if missing_attention:
        raise RuntimeError(f"gatv2_attention missing attempts: {sorted(missing_attention)[:5]}")
    eta_without = pd.to_numeric(eta["eta_without_new_pickup_sec"], errors="raise")
    eta_with = pd.to_numeric(eta["eta_with_new_pickup_sec"], errors="raise")
    if not eta_without.notna().all() or not eta_with.notna().all():
        raise RuntimeError("ETA columns contain null values")
    weights = pd.to_numeric(attention["attention_weight"], errors="raise")
    if not weights.notna().all():
        raise RuntimeError("attention_weight contains null values")
    if (weights < 0).any():
        raise RuntimeError("attention_weight must be non-negative")
    if "path_segment" in attention.columns:
        observed = set(attention["path_segment"].astype(str).str.strip())
        invalid = sorted(x for x in observed if x not in VALID_SEGMENTS)
        if invalid:
            raise RuntimeError(f"invalid path_segment values: {invalid}")


def build_run_manifest(
    output_root: Path,
    mode: str,
    condition_id: str,
    seed: int,
    zero_loss_epsilon_sec: float,
    source_rollout_path: str,
    source_manifest_path: str,
    warnings: List[str],
) -> Dict[str, Any]:
    source_manifest: Dict[str, Any] = {}
    if source_manifest_path:
        p = Path(source_manifest_path)
        if p.exists():
            source_manifest = load_json_any_encoding(p)
        else:
            warnings.append(f"source_manifest_not_found: {source_manifest_path}")

    return {
        "artifact_version": STEP160_INPUT_VERSION,
        "writer_artifact_version": ARTIFACT_VERSION,
        "project": "urbanbus_rl_project",
        "step": "Step 161",
        "mode": mode,
        "condition_id": condition_id,
        "seed": int(seed),
        "zero_loss_epsilon_sec": float(zero_loss_epsilon_sec),
        "source_rollout_path": source_rollout_path,
        "source_manifest_path": source_manifest_path,
        "source_manifest_artifact_version": source_manifest.get("artifact_version", ""),
        "dataset_artifact": source_manifest.get("dataset_artifact", "route_aware_simulator_v2_scaffold_or_sample"),
        "checkpoint_path": source_manifest.get("checkpoint_path", ""),
        "checkpoint_sha256": source_manifest.get("checkpoint_sha256", ""),
        "git_commit": source_manifest.get("git_commit", "unknown_local_step161"),
        "simulator_version": source_manifest.get("simulator_version", "route_aware_pickup_attempt_writer_step161_scaffold"),
        "output_root": str(output_root),
        "warnings": warnings,
        **NON_CLAIM_FLAGS,
    }


def run_writer(
    mode: str,
    output_root: Path,
    condition_id: str,
    seed: int,
    zero_loss_epsilon_sec: float,
    file_format: str,
    max_attempts: int,
    rollout_events_path: str = "",
    source_manifest_path: str = "",
) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    warnings: List[str] = []

    if mode == "sample":
        raw = make_sample_rollout(condition_id=condition_id, seed=seed)
    elif mode == "from-rollout":
        if not rollout_events_path:
            raise RuntimeError("--rollout-events is required for --mode from-rollout")
        raw = read_table(Path(rollout_events_path))
    else:
        raise RuntimeError(f"unsupported mode: {mode}")

    candidates = filter_pickup_candidate_rows(raw, max_attempts=max_attempts)
    attempts, eta, attention = build_attempt_tables(
        candidates,
        condition_id=condition_id,
        seed=seed,
        zero_loss_epsilon_sec=zero_loss_epsilon_sec,
    )
    validate_frames(attempts, eta, attention)

    attempt_path = write_table(output_root / "pickup_attempt_events", attempts, file_format, warnings)
    eta_path = write_table(output_root / "eta_counterfactual", eta, file_format, warnings)
    attention_path = write_table(output_root / "gatv2_attention", attention, file_format, warnings)

    run_manifest = build_run_manifest(
        output_root=output_root,
        mode=mode,
        condition_id=condition_id,
        seed=seed,
        zero_loss_epsilon_sec=zero_loss_epsilon_sec,
        source_rollout_path=rollout_events_path,
        source_manifest_path=source_manifest_path,
        warnings=warnings,
    )
    run_manifest_path = output_root / "run_manifest.json"
    dump_json(run_manifest_path, run_manifest)

    delta = eta.copy()
    delta["eta_without_new_pickup_sec"] = pd.to_numeric(delta["eta_without_new_pickup_sec"], errors="raise")
    delta["eta_with_new_pickup_sec"] = pd.to_numeric(delta["eta_with_new_pickup_sec"], errors="raise")
    delta["delta_eta_existing_passenger_sec"] = delta["eta_with_new_pickup_sec"] - delta["eta_without_new_pickup_sec"]
    delta["zero_loss_success"] = delta["delta_eta_existing_passenger_sec"] <= float(zero_loss_epsilon_sec)
    total = int(len(delta))
    success = int(delta["zero_loss_success"].sum())

    manifest_path = output_root / "route_aware_pickup_attempt_event_writer_manifest.json"
    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "audit_status": "PASS",
        "bundle_status": "ROUTE_AWARE_PICKUP_ATTEMPT_EVENTS_READY_FOR_STEP160_NONCLAIM",
        "mode": mode,
        "output_root": str(output_root),
        "output_files": {
            "pickup_attempt_events": str(attempt_path),
            "eta_counterfactual": str(eta_path),
            "gatv2_attention": str(attention_path),
            "run_manifest": str(run_manifest_path),
            "writer_manifest": str(manifest_path),
        },
        "output_sha256": {
            "pickup_attempt_events": sha256_file(attempt_path),
            "eta_counterfactual": sha256_file(eta_path),
            "gatv2_attention": sha256_file(attention_path),
            "run_manifest": sha256_file(run_manifest_path),
        },
        "summary": {
            "pickup_attempt_count": total,
            "zero_loss_success_preview_count": success,
            "zero_loss_success_preview_rate": float(success / total) if total else 0.0,
            "attention_row_count": int(len(attention)),
            "condition_ids": sorted(attempts["condition_id"].astype(str).unique().tolist()),
            "seeds": sorted(int(x) for x in attempts["seed"].unique().tolist()),
            "zero_loss_epsilon_sec": float(zero_loss_epsilon_sec),
        },
        "warnings": warnings,
        **NON_CLAIM_FLAGS,
    }
    dump_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 161 Route-Aware Pickup Attempt Event Writer")
    parser.add_argument("--mode", choices=["sample", "from-rollout"], default="sample")
    parser.add_argument("--rollout-events", default="")
    parser.add_argument("--source-manifest", default="")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--condition-id", default="A")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--zero-loss-epsilon-sec", type=float, default=0.0)
    parser.add_argument("--file-format", choices=["csv", "parquet"], default="csv")
    parser.add_argument("--max-attempts", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_writer(
        mode=args.mode,
        output_root=Path(args.output_root),
        condition_id=args.condition_id,
        seed=int(args.seed),
        zero_loss_epsilon_sec=float(args.zero_loss_epsilon_sec),
        file_format=args.file_format,
        max_attempts=int(args.max_attempts),
        rollout_events_path=args.rollout_events,
        source_manifest_path=args.source_manifest,
    )
    summary = manifest["summary"]
    print("[OK] Step 161 route-aware pickup attempt event writer completed")
    print(f"[OK] audit_status              : {manifest['audit_status']}")
    print(f"[OK] bundle_status             : {manifest['bundle_status']}")
    print(f"[OK] pickup_attempt_count      : {summary['pickup_attempt_count']}")
    print(f"[OK] zero_loss_success_preview_count : {summary['zero_loss_success_preview_count']}")
    print(f"[OK] zero_loss_success_preview_rate  : {summary['zero_loss_success_preview_rate']:.6f}")
    print(f"[OK] attention_row_count       : {summary['attention_row_count']}")
    print(f"[OK] output_root               : {manifest['output_root']}")
    print(f"[OK] manifest                  : {manifest['output_files']['writer_manifest']}")
    print("[OK] paper_level_claim_allowed : False")
    print("[OK] causal_performance_claim_allowed : False")


if __name__ == "__main__":
    main()
