from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


ARTIFACT_VERSION = "route_aware_rollout_adapter_validator_step162_v1"
EXPECTED_ADAPTER_VERSION = "route_aware_rollout_adapter_step162_v1"
EXPECTED_STATUS = "ROUTE_AWARE_ROLLOUT_ADAPTED_FOR_STEP161_NONCLAIM"

REQUIRED_NORMALIZED_COLUMNS = [
    "state_ts",
    "condition_id",
    "seed",
    "window_id",
    "time_band",
    "vehicle_id",
    "route_id",
    "direction_id",
    "pickup_stop_id",
    "dropoff_stop_id",
    "policy_action",
    "decision",
    "existing_passenger_id",
    "new_passenger_id",
    "eta_without_new_pickup_sec",
    "eta_with_new_pickup_sec",
    "actual_operational_observation",
    "simulation_evidence_only",
]


def load_json_any_encoding(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


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


def require_false(payload: Dict[str, Any], key: str, errors: List[str]) -> None:
    if bool(payload.get(key, False)):
        errors.append(f"{key} must be false")


def require_true(payload: Dict[str, Any], key: str, errors: List[str]) -> None:
    if not bool(payload.get(key, False)):
        errors.append(f"{key} must be true")


def validate_manifest(manifest_path: Path) -> Dict[str, Any]:
    errors: List[str] = []
    manifest = load_json_any_encoding(manifest_path)

    if manifest.get("artifact_version") != EXPECTED_ADAPTER_VERSION:
        errors.append(f"artifact_version mismatch: {manifest.get('artifact_version')}")
    if manifest.get("audit_status") != "PASS":
        errors.append(f"audit_status must be PASS, got {manifest.get('audit_status')}")
    if manifest.get("bundle_status") != EXPECTED_STATUS:
        errors.append(f"bundle_status mismatch: {manifest.get('bundle_status')}")

    require_true(manifest, "simulation_evidence_only", errors)
    for key in [
        "actual_operational_claim_allowed",
        "paper_level_claim_allowed",
        "causal_performance_claim_allowed",
        "train_allowed",
    ]:
        require_false(manifest, key, errors)

    output_files = manifest.get("output_files", {})
    normalized_path = Path(str(output_files.get("normalized_route_aware_rollout_events", "")))
    mapping_path = Path(str(output_files.get("source_column_mapping", "")))
    quality_path = Path(str(output_files.get("data_quality_report", "")))
    for label, path in [
        ("normalized_route_aware_rollout_events", normalized_path),
        ("source_column_mapping", mapping_path),
        ("data_quality_report", quality_path),
    ]:
        if not str(path):
            errors.append(f"missing output_files.{label}")
        elif not path.exists():
            errors.append(f"output file does not exist: {label}={path}")

    normalized_event_count = int(manifest.get("summary", {}).get("normalized_event_count", -1))
    if normalized_event_count <= 0:
        errors.append("summary.normalized_event_count must be positive")

    if normalized_path.exists():
        df = read_table(normalized_path)
        missing = [c for c in REQUIRED_NORMALIZED_COLUMNS if c not in df.columns]
        if missing:
            errors.append(f"normalized table missing columns: {missing}")
        if len(df) != normalized_event_count:
            errors.append(f"normalized row count mismatch: file={len(df)} manifest={normalized_event_count}")
        if not df.empty:
            decisions = set(df["decision"].astype(str).str.lower()) if "decision" in df.columns else set()
            if not decisions.issubset({"accepted", "rejected"}):
                errors.append(f"invalid decision values: {sorted(decisions)}")
            if "actual_operational_observation" in df.columns and bool(df["actual_operational_observation"].astype(bool).any()):
                errors.append("actual_operational_observation must be false for all rows")
            if "simulation_evidence_only" in df.columns and not bool(df["simulation_evidence_only"].astype(bool).all()):
                errors.append("simulation_evidence_only must be true for all rows")
            for col in ["eta_without_new_pickup_sec", "eta_with_new_pickup_sec"]:
                if col in df.columns:
                    numeric = pd.to_numeric(df[col], errors="coerce")
                    if not numeric.notna().all():
                        errors.append(f"{col} contains non-numeric/null values")

    if quality_path.exists():
        quality = load_json_any_encoding(quality_path)
        q = quality.get("data_quality", {})
        if int(q.get("normalized_event_count", -1)) != normalized_event_count:
            errors.append("data_quality.normalized_event_count mismatch")

    if errors:
        raise RuntimeError("Step 162 validation failed: " + "; ".join(errors))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Step 162 route-aware rollout adapter output")
    parser.add_argument("--manifest", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = validate_manifest(Path(args.manifest))
    summary = manifest["summary"]
    print("[OK] Step 162 route-aware rollout adapter validation PASS")
    print(f"[OK] manifest               : {args.manifest}")
    print(f"[OK] normalized_event_count : {summary['normalized_event_count']}")
    print(f"[OK] direct_eta_count       : {summary['direct_eta_count']}")
    print(f"[OK] proxy_eta_count        : {summary['proxy_eta_count']}")
    print("[OK] paper_level_claim_allowed : False")
    print("[OK] causal_performance_claim_allowed : False")


if __name__ == "__main__":
    main()
