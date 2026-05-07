from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


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
NON_CLAIM_FLAGS = {
    "simulation_evidence_only": True,
    "actual_operational_claim_allowed": False,
    "paper_level_claim_allowed": False,
    "causal_performance_claim_allowed": False,
    "train_allowed": False,
}


def load_json_any_encoding(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


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
    raise RuntimeError(f"unsupported table format: {path}")


def require_columns(df: pd.DataFrame, required: List[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"{label} missing required columns: {missing}")


def validate_manifest(manifest: Dict[str, Any]) -> None:
    if manifest.get("audit_status") != "PASS":
        raise RuntimeError(f"audit_status must be PASS, got {manifest.get('audit_status')}")
    expected_status = "ROUTE_AWARE_PICKUP_ATTEMPT_EVENTS_READY_FOR_STEP160_NONCLAIM"
    if manifest.get("bundle_status") != expected_status:
        raise RuntimeError(f"bundle_status must be {expected_status}, got {manifest.get('bundle_status')}")
    for key, expected in NON_CLAIM_FLAGS.items():
        observed = manifest.get(key)
        if bool_from_any(observed) != expected:
            raise RuntimeError(f"manifest guard mismatch: {key} must be {expected}, got {observed}")


def validate_files(manifest: Dict[str, Any], require_step160_ready: bool) -> Dict[str, Any]:
    files = manifest.get("output_files", {})
    required_file_keys = ["pickup_attempt_events", "eta_counterfactual", "gatv2_attention", "run_manifest"]
    if require_step160_ready:
        required_file_keys.append("writer_manifest")

    paths: Dict[str, Path] = {}
    for key in required_file_keys:
        value = files.get(key)
        if not value:
            raise RuntimeError(f"output_files.{key} missing in manifest")
        path = Path(value)
        if not path.exists():
            raise RuntimeError(f"output file does not exist: {key}={path}")
        paths[key] = path

    attempts = read_table(paths["pickup_attempt_events"])
    eta = read_table(paths["eta_counterfactual"])
    attention = read_table(paths["gatv2_attention"])
    run_manifest = load_json_any_encoding(paths["run_manifest"])

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

    attempt_ids = set(attempts["attempt_id"].astype(str).tolist())
    eta_attempt_ids = set(eta["attempt_id"].astype(str).tolist())
    attention_attempt_ids = set(attention["attempt_id"].astype(str).tolist())
    if attempt_ids - eta_attempt_ids:
        raise RuntimeError(f"eta_counterfactual missing attempt ids: {sorted(attempt_ids - eta_attempt_ids)[:5]}")
    if attempt_ids - attention_attempt_ids:
        raise RuntimeError(f"gatv2_attention missing attempt ids: {sorted(attempt_ids - attention_attempt_ids)[:5]}")

    eta_without = pd.to_numeric(eta["eta_without_new_pickup_sec"], errors="raise")
    eta_with = pd.to_numeric(eta["eta_with_new_pickup_sec"], errors="raise")
    if eta_without.isna().any() or eta_with.isna().any():
        raise RuntimeError("ETA columns contain null")

    weights = pd.to_numeric(attention["attention_weight"], errors="raise")
    if weights.isna().any():
        raise RuntimeError("attention_weight contains null")
    if (weights < 0).any():
        raise RuntimeError("attention_weight must be non-negative")

    for key, expected in NON_CLAIM_FLAGS.items():
        observed = run_manifest.get(key)
        if bool_from_any(observed) != expected:
            raise RuntimeError(f"run_manifest guard mismatch: {key} must be {expected}, got {observed}")

    epsilon = float(run_manifest.get("zero_loss_epsilon_sec", 0.0))
    delta = eta_with - eta_without
    success_count = int((delta <= epsilon).sum())

    summary = manifest.get("summary", {})
    if int(summary.get("pickup_attempt_count", -1)) != int(len(attempts)):
        raise RuntimeError("manifest summary pickup_attempt_count mismatch")
    if int(summary.get("zero_loss_success_preview_count", -1)) != success_count:
        raise RuntimeError("manifest summary zero_loss_success_preview_count mismatch")

    return {
        "attempt_count": int(len(attempts)),
        "eta_count": int(len(eta)),
        "attention_row_count": int(len(attention)),
        "zero_loss_success_preview_count": success_count,
        "zero_loss_epsilon_sec": epsilon,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Step 161 route-aware pickup attempt event bundle")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--require-step160-ready", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    manifest = load_json_any_encoding(manifest_path)
    validate_manifest(manifest)
    result = validate_files(manifest, require_step160_ready=bool(args.require_step160_ready))
    print("[OK] Step 161 route-aware pickup attempt event bundle validation PASS")
    print(f"[OK] manifest                  : {manifest_path}")
    print(f"[OK] attempt_count             : {result['attempt_count']}")
    print(f"[OK] eta_count                 : {result['eta_count']}")
    print(f"[OK] attention_row_count       : {result['attention_row_count']}")
    print(f"[OK] zero_loss_success_preview_count : {result['zero_loss_success_preview_count']}")
    print("[OK] paper_level_claim_allowed : False")
    print("[OK] causal_performance_claim_allowed : False")


if __name__ == "__main__":
    main()
