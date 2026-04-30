from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

INTEGRATION_STATUS = "LOGGER_INTEGRATION_READY_MONITORING_ONLY_STILL_LOCKED"
DASHBOARD_MODE = "MONITORING_ONLY"
CONTROL_POLICY = "NO_LIVE_MUTATION_CONFIG_BASED_NEXT_RUN_ONLY"
REQUIRED_METRIC_GROUPS = {"train", "kpi", "safety", "runtime"}


class ValidationError(RuntimeError):
    pass


def load_json(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with path.open("r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise ValidationError(f"failed to read json: {path}")


def assert_false(payload: Dict[str, Any], key: str) -> None:
    if bool(payload.get(key, False)):
        raise ValidationError(f"{key} must be false")


def resolve_path(raw: str, manifest_path: Path) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    return (manifest_path.parent / p).resolve()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValidationError(f"invalid JSONL at line {idx}: {exc}") from exc
    return rows


def validate_manifest(manifest_path: Path) -> Dict[str, Any]:
    if not manifest_path.exists():
        raise ValidationError(f"manifest not found: {manifest_path}")
    manifest = load_json(manifest_path)

    if manifest.get("integration_status") != INTEGRATION_STATUS:
        raise ValidationError(f"integration_status mismatch: {manifest.get('integration_status')}")
    if manifest.get("dashboard_mode") != DASHBOARD_MODE:
        raise ValidationError(f"dashboard_mode mismatch: {manifest.get('dashboard_mode')}")
    if manifest.get("control_policy") != CONTROL_POLICY:
        raise ValidationError(f"control_policy mismatch: {manifest.get('control_policy')}")

    for key in [
        "actual_execution_allowed",
        "actual_execution_released",
        "train_allowed",
        "paper_level_claim_allowed",
        "causal_performance_claim_allowed",
        "live_mutation_allowed",
    ]:
        assert_false(manifest, key)

    files = manifest.get("output_files", {})
    required_file_keys = [
        "run_status",
        "metrics_events_jsonl",
        "gpu_metrics_csv",
        "tensorboard_scalar_contract",
        "tensorboard_event_dir",
    ]
    missing_keys = [k for k in required_file_keys if k not in files]
    if missing_keys:
        raise ValidationError(f"manifest output_files missing keys: {missing_keys}")

    run_status_path = resolve_path(files["run_status"], manifest_path)
    metrics_path = resolve_path(files["metrics_events_jsonl"], manifest_path)
    gpu_path = resolve_path(files["gpu_metrics_csv"], manifest_path)
    tb_contract_path = resolve_path(files["tensorboard_scalar_contract"], manifest_path)
    tb_event_dir = resolve_path(files["tensorboard_event_dir"], manifest_path)

    for path in [run_status_path, metrics_path, gpu_path, tb_contract_path]:
        if not path.exists():
            raise ValidationError(f"expected output file not found: {path}")

    run_status = load_json(run_status_path)
    if run_status.get("control_policy") != CONTROL_POLICY:
        raise ValidationError("run_status control_policy mismatch")
    for key in [
        "actual_execution_allowed",
        "actual_execution_released",
        "train_allowed",
        "paper_level_claim_allowed",
        "causal_performance_claim_allowed",
        "live_mutation_allowed",
    ]:
        assert_false(run_status, key)

    tb_contract = load_json(tb_contract_path)
    if tb_contract.get("control_policy") != CONTROL_POLICY:
        raise ValidationError("tensorboard scalar contract control_policy mismatch")
    if bool(tb_contract.get("live_mutation_allowed", True)):
        raise ValidationError("tensorboard scalar contract must forbid live mutation")

    events = read_jsonl(metrics_path)
    scalar_events = [e for e in events if e.get("event_type") == "scalar"]
    if len(scalar_events) < 1:
        raise ValidationError("metrics JSONL has no scalar events")
    groups_seen = {str(e.get("metric_group")) for e in scalar_events}
    missing_groups = sorted(REQUIRED_METRIC_GROUPS - groups_seen)
    if missing_groups:
        raise ValidationError(f"metrics JSONL missing required groups: {missing_groups}")
    for e in scalar_events:
        if bool(e.get("live_mutation_allowed", True)):
            raise ValidationError("scalar event must record live_mutation_allowed=false")
        if e.get("control_policy") != CONTROL_POLICY:
            raise ValidationError("scalar event control_policy mismatch")
        try:
            float(e["metric_value"])
        except Exception as exc:
            raise ValidationError(f"invalid scalar metric_value in event: {e}") from exc

    with gpu_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_gpu_cols = {
            "time_utc",
            "run_id",
            "step",
            "gpu_index",
            "gpu_name",
            "utilization_pct",
            "memory_used_mb",
            "memory_total_mb",
            "temperature_c",
            "power_draw_w",
            "ecc_error_count",
        }
        if set(reader.fieldnames or []) != required_gpu_cols:
            raise ValidationError(f"gpu csv columns mismatch: {reader.fieldnames}")
        gpu_rows = list(reader)
        if len(gpu_rows) < 1:
            raise ValidationError("gpu csv has no telemetry rows")

    tb = manifest.get("tensorboard", {})
    if bool(tb.get("available", False)) and bool(tb.get("enabled", False)):
        if not tb_event_dir.exists():
            raise ValidationError(f"TensorBoard event dir missing despite available=true: {tb_event_dir}")

    if int(manifest.get("metric_event_count", 0)) < len(scalar_events):
        raise ValidationError("manifest metric_event_count is smaller than scalar events")
    if int(manifest.get("gpu_row_count", 0)) < 1:
        raise ValidationError("manifest gpu_row_count must be at least 1")

    return {
        "manifest": str(manifest_path),
        "scalar_event_count": len(scalar_events),
        "groups_seen": sorted(groups_seen),
        "gpu_rows": len(gpu_rows),
        "tensorboard_status": tb.get("status", "unknown"),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Step 153 H200 TensorBoard + JSONL logger integration")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args(argv)

    result = validate_manifest(Path(args.manifest).resolve())
    print("[OK] Step 153 H200 TensorBoard + JSONL logger validation PASS")
    print(f"[OK] manifest      : {result['manifest']}")
    print(f"[OK] scalar_events : {result['scalar_event_count']}")
    print(f"[OK] groups_seen   : {result['groups_seen']}")
    print(f"[OK] gpu_rows      : {result['gpu_rows']}")
    print(f"[OK] tensorboard   : {result['tensorboard_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
