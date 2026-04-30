from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


EXPECTED_STATUS = "NVIDIA_SMI_GPU_MONITOR_SCAFFOLD_READY_MONITORING_ONLY_STILL_LOCKED"
EXPECTED_MODE = "MONITORING_ONLY"
EXPECTED_CONTROL = "NO_LIVE_MUTATION_CONFIG_BASED_NEXT_RUN_ONLY"

REQUIRED_COLUMNS = [
    "timestamp_utc",
    "run_id",
    "condition_id",
    "reward_id",
    "seed",
    "global_step",
    "gpu_index",
    "gpu_name",
    "gpu_uuid",
    "driver_version",
    "cuda_version",
    "utilization_gpu_pct",
    "utilization_memory_pct",
    "memory_total_mib",
    "memory_used_mib",
    "memory_free_mib",
    "temperature_gpu_c",
    "power_draw_w",
    "power_limit_w",
    "pcie_link_gen_current",
    "pcie_link_width_current",
    "ecc_volatile_uncorrected",
    "ecc_aggregate_uncorrected",
    "nvidia_smi_available",
    "nvidia_smi_status",
    "source_mode",
    "monitoring_only",
    "actual_execution_allowed",
    "train_allowed",
    "live_mutation_allowed",
]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def validate_manifest(manifest_path: Path) -> Dict[str, Any]:
    if not manifest_path.exists():
        raise RuntimeError(f"manifest not found: {manifest_path}")

    manifest = load_json(manifest_path)

    if manifest.get("monitor_status") != EXPECTED_STATUS:
        raise RuntimeError(f"monitor_status mismatch: {manifest.get('monitor_status')}")
    if manifest.get("monitor_mode") != EXPECTED_MODE:
        raise RuntimeError(f"monitor_mode mismatch: {manifest.get('monitor_mode')}")
    if manifest.get("control_policy") != EXPECTED_CONTROL:
        raise RuntimeError(f"control_policy mismatch: {manifest.get('control_policy')}")

    for key in [
        "actual_execution_allowed",
        "actual_execution_released",
        "train_allowed",
        "live_mutation_allowed",
        "paper_level_claim_allowed",
        "causal_performance_claim_allowed",
    ]:
        if bool(manifest.get(key, False)):
            raise RuntimeError(f"{key} must be false")

    files = manifest.get("output_files", {})
    csv_path = Path(files.get("gpu_metrics_csv", ""))
    jsonl_path = Path(files.get("gpu_metrics_jsonl", ""))
    status_path = Path(files.get("gpu_monitor_status", ""))
    query_path = Path(files.get("nvidia_smi_query_contract", ""))

    for p in [csv_path, jsonl_path, status_path, query_path]:
        if not p.exists():
            raise RuntimeError(f"expected output file not found: {p}")

    rows = read_csv_rows(csv_path)
    if not rows:
        raise RuntimeError("gpu metrics csv has no rows")

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in rows[0]]
    if missing_cols:
        raise RuntimeError(f"gpu metrics csv missing columns: {missing_cols}")

    for row in rows:
        if not as_bool(row["monitoring_only"]):
            raise RuntimeError("monitoring_only must be true for all rows")
        for key in ["actual_execution_allowed", "train_allowed", "live_mutation_allowed"]:
            if as_bool(row[key]):
                raise RuntimeError(f"{key} must be false for all rows")
        if not str(row["nvidia_smi_status"]).strip():
            raise RuntimeError("nvidia_smi_status must not be empty")

    jsonl_lines = [line for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(jsonl_lines) != len(rows):
        raise RuntimeError(f"jsonl row count mismatch: jsonl={len(jsonl_lines)} csv={len(rows)}")

    for line in jsonl_lines:
        payload = json.loads(line)
        for col in REQUIRED_COLUMNS:
            if col not in payload:
                raise RuntimeError(f"jsonl row missing column: {col}")

    status = load_json(status_path)
    if status.get("monitor_status") != EXPECTED_STATUS:
        raise RuntimeError("status json monitor_status mismatch")
    if bool(status.get("train_allowed", True)):
        raise RuntimeError("status json train_allowed must be false")

    query = load_json(query_path)
    for col in REQUIRED_COLUMNS:
        if col not in query.get("csv_columns", []):
            raise RuntimeError(f"query contract missing csv column: {col}")

    return {
        "manifest": str(manifest_path),
        "csv_rows": len(rows),
        "nvidia_smi_status": manifest.get("nvidia_smi_status"),
        "source_modes": sorted(set(row["source_mode"] for row in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    result = validate_manifest(Path(args.manifest))

    print("[OK] Step 154 H200 nvidia-smi GPU monitor validation PASS")
    print(f"[OK] manifest        : {result['manifest']}")
    print(f"[OK] csv_rows        : {result['csv_rows']}")
    print(f"[OK] nvidia-smi      : {result['nvidia_smi_status']}")
    print(f"[OK] source_modes    : {result['source_modes']}")


if __name__ == "__main__":
    main()
