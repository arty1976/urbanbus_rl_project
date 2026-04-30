from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List

EXPECTED_STATUS = "OBSERVABILITY_CONTRACT_READY_MONITORING_ONLY_STILL_LOCKED"
EXPECTED_DASHBOARD_MODE = "MONITORING_ONLY"
EXPECTED_CONTROL_POLICY = "NO_LIVE_MUTATION_CONFIG_BASED_NEXT_RUN_ONLY"
EXPECTED_NEXT_GATE = "WAITING_FOR_REAL_H200_STEP149_EXPECT_H200_RESULT"
EXPECTED_GROUPS = [
    "run_identity",
    "training_metrics",
    "traffic_kpis",
    "safety_gates",
    "system_metrics",
    "claim_guards",
]
LOCK_FALSE_KEYS = [
    "actual_execution_allowed",
    "actual_execution_released",
    "train_allowed",
    "operator_approval_recorded",
    "operator_approval_granted",
    "actual_results",
    "winner_selected",
    "trainable_reward_promoted",
    "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
]
REQUIRED_OUTPUT_KEYS = [
    "run_status",
    "metrics_events_jsonl",
    "gpu_metrics_csv",
    "tensorboard_scalar_contract",
    "manifest",
]
REQUIRED_TRAINING_METRICS = {
    "policy_loss",
    "value_loss",
    "entropy",
    "kl_divergence",
    "grad_norm",
    "learning_rate",
    "reward_mean",
    "reward_std",
    "episode_length_mean",
}
REQUIRED_TRAFFIC_KPIS = {
    "avg_wait_seconds",
    "passenger_service_rate",
    "passenger_wait_p95_seconds",
    "bunching_rate",
    "on_time_rate",
    "energy_proxy_per_passenger",
    "fleet_reduction_ratio",
    "intervention_rate",
    "cv_headway",
}
REQUIRED_GPU_COLUMNS = [
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


def load_json(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


def is_finite_number(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def validate_metrics_jsonl(path: Path) -> List[str]:
    errors: List[str] = []
    if not path.exists():
        return [f"metrics jsonl missing: {path}"]

    event_types = set()
    training_seen = False
    kpi_seen = False

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except Exception as exc:
                errors.append(f"invalid jsonl line {line_no}: {exc}")
                continue

            event_type = str(event.get("event_type", ""))
            event_types.add(event_type)
            metrics = event.get("metrics", {})
            if not isinstance(metrics, dict):
                errors.append(f"line {line_no}: metrics must be object")
                continue
            for key, value in metrics.items():
                if not is_finite_number(value):
                    errors.append(f"line {line_no}: non-finite metric {key}")
            if event_type == "training_metrics":
                missing = REQUIRED_TRAINING_METRICS.difference(metrics.keys())
                if missing:
                    errors.append(f"line {line_no}: missing training metrics {sorted(missing)}")
                training_seen = True
            if event_type == "traffic_kpis":
                missing = REQUIRED_TRAFFIC_KPIS.difference(metrics.keys())
                if missing:
                    errors.append(f"line {line_no}: missing traffic kpis {sorted(missing)}")
                kpi_seen = True

    if not training_seen:
        errors.append("no training_metrics event found")
    if not kpi_seen:
        errors.append("no traffic_kpis event found")
    return errors


def validate_gpu_csv(path: Path) -> List[str]:
    errors: List[str] = []
    if not path.exists():
        return [f"gpu csv missing: {path}"]
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != REQUIRED_GPU_COLUMNS:
            errors.append(f"gpu csv header mismatch: {reader.fieldnames}")
        row_count = 0
        for row in reader:
            row_count += 1
        if row_count < 1:
            errors.append("gpu csv has no data rows")
    return errors


def validate_manifest(path: Path) -> List[str]:
    errors: List[str] = []
    manifest = load_json(path)

    if manifest.get("contract_status") != EXPECTED_STATUS:
        errors.append("contract_status mismatch")
    if manifest.get("dashboard_mode") != EXPECTED_DASHBOARD_MODE:
        errors.append("dashboard_mode mismatch")
    if manifest.get("control_policy") != EXPECTED_CONTROL_POLICY:
        errors.append("control_policy mismatch")
    if manifest.get("next_gate") != EXPECTED_NEXT_GATE:
        errors.append("next_gate mismatch")
    if manifest.get("required_metric_groups") != EXPECTED_GROUPS:
        errors.append("required_metric_groups mismatch")

    for key in LOCK_FALSE_KEYS:
        if manifest.get(key) is not False:
            errors.append(f"required false lock is not false: {key}")

    if int(manifest.get("hard_failures", -1)) != 0:
        errors.append("hard_failures must be 0")

    outputs = manifest.get("outputs", {})
    if not isinstance(outputs, dict):
        errors.append("outputs must be object")
        outputs = {}
    for key in REQUIRED_OUTPUT_KEYS:
        raw = outputs.get(key)
        if not raw:
            errors.append(f"missing output path: {key}")
            continue
        out_path = Path(raw)
        if not out_path.exists():
            errors.append(f"output file does not exist: {key}={out_path}")

    run_status_path = Path(outputs.get("run_status", ""))
    if run_status_path.exists():
        run_status = load_json(run_status_path)
        for key in LOCK_FALSE_KEYS:
            if run_status.get(key) is not False:
                errors.append(f"run_status false lock mismatch: {key}")
        if run_status.get("control_policy") != EXPECTED_CONTROL_POLICY:
            errors.append("run_status control_policy mismatch")

    tensorboard_contract_path = Path(outputs.get("tensorboard_scalar_contract", ""))
    if tensorboard_contract_path.exists():
        tb = load_json(tensorboard_contract_path)
        scalars = tb.get("scalars", {})
        if "training_metrics" not in scalars:
            errors.append("tensorboard contract missing training_metrics")
        if "traffic_kpis" not in scalars:
            errors.append("tensorboard contract missing traffic_kpis")
        if "safety_gates" not in scalars:
            errors.append("tensorboard contract missing safety_gates")
        if "system_metrics" not in scalars:
            errors.append("tensorboard contract missing system_metrics")

    metrics_path = Path(outputs.get("metrics_events_jsonl", ""))
    errors.extend(validate_metrics_jsonl(metrics_path))

    gpu_path = Path(outputs.get("gpu_metrics_csv", ""))
    errors.extend(validate_gpu_csv(gpu_path))

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Step 152 H200 monitoring dashboard contract")
    parser.add_argument("--manifest", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = Path(args.manifest)
    errors = validate_manifest(manifest)
    if errors:
        print("[FAIL] Step 152 monitoring dashboard contract validation failed")
        for err in errors:
            print(f"[FAIL] {err}")
        raise SystemExit(1)
    print("[OK] Step 152 monitoring dashboard contract validation PASS")
    print(f"[OK] manifest: {manifest}")


if __name__ == "__main__":
    main()
