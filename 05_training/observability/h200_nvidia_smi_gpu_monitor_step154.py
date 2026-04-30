from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


MONITOR_STATUS = "NVIDIA_SMI_GPU_MONITOR_SCAFFOLD_READY_MONITORING_ONLY_STILL_LOCKED"
MONITOR_MODE = "MONITORING_ONLY"
CONTROL_POLICY = "NO_LIVE_MUTATION_CONFIG_BASED_NEXT_RUN_ONLY"

GPU_METRIC_COLUMNS = [
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

NVIDIA_SMI_QUERY_FIELDS = [
    "index",
    "name",
    "uuid",
    "driver_version",
    "cuda_version",
    "utilization.gpu",
    "utilization.memory",
    "memory.total",
    "memory.used",
    "memory.free",
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "pcie.link.gen.current",
    "pcie.link.width.current",
    "ecc.errors.uncorrected.volatile.total",
    "ecc.errors.uncorrected.aggregate.total",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=GPU_METRIC_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in GPU_METRIC_COLUMNS})


def parse_numeric(value: str) -> Optional[float]:
    value = str(value).strip()
    if value in {"", "N/A", "[Not Supported]", "Not Supported", "nan"}:
        return None
    cleaned = (
        value.replace("%", "")
        .replace("MiB", "")
        .replace("W", "")
        .replace("C", "")
        .strip()
    )
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_int_like(value: str) -> Optional[int]:
    parsed = parse_numeric(value)
    if parsed is None:
        return None
    return int(parsed)


def nvidia_smi_path() -> Optional[str]:
    return shutil.which("nvidia-smi")


def build_query_command() -> List[str]:
    exe = nvidia_smi_path() or "nvidia-smi"
    return [
        exe,
        "--query-gpu=" + ",".join(NVIDIA_SMI_QUERY_FIELDS),
        "--format=csv,noheader,nounits",
    ]


def probe_nvidia_smi(timeout_seconds: int = 5) -> Dict[str, Any]:
    exe = nvidia_smi_path()
    if exe is None:
        return {
            "available": False,
            "status": "nvidia_smi_unavailable",
            "rows": [],
            "stderr": "nvidia-smi executable not found on PATH",
            "command": build_query_command(),
        }

    cmd = build_query_command()
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "available": True,
            "status": "nvidia_smi_timeout",
            "rows": [],
            "stderr": f"nvidia-smi timed out after {timeout_seconds}s",
            "command": cmd,
        }
    except Exception as exc:
        return {
            "available": True,
            "status": f"nvidia_smi_error:{type(exc).__name__}",
            "rows": [],
            "stderr": str(exc),
            "command": cmd,
        }

    if proc.returncode != 0:
        return {
            "available": True,
            "status": f"nvidia_smi_failed_returncode_{proc.returncode}",
            "rows": [],
            "stderr": proc.stderr.strip(),
            "command": cmd,
        }

    rows = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != len(NVIDIA_SMI_QUERY_FIELDS):
            continue
        rows.append(dict(zip(NVIDIA_SMI_QUERY_FIELDS, parts)))

    return {
        "available": True,
        "status": "nvidia_smi_probe_ok" if rows else "nvidia_smi_no_gpu_rows",
        "rows": rows,
        "stderr": proc.stderr.strip(),
        "command": cmd,
    }


def row_from_probe_row(
    raw: Dict[str, str],
    *,
    run_id: str,
    condition_id: str,
    reward_id: str,
    seed: int,
    global_step: int,
) -> Dict[str, Any]:
    return {
        "timestamp_utc": utc_now_iso(),
        "run_id": run_id,
        "condition_id": condition_id,
        "reward_id": reward_id,
        "seed": int(seed),
        "global_step": int(global_step),
        "gpu_index": parse_int_like(raw.get("index", "")),
        "gpu_name": raw.get("name", ""),
        "gpu_uuid": raw.get("uuid", ""),
        "driver_version": raw.get("driver_version", ""),
        "cuda_version": raw.get("cuda_version", ""),
        "utilization_gpu_pct": parse_numeric(raw.get("utilization.gpu", "")),
        "utilization_memory_pct": parse_numeric(raw.get("utilization.memory", "")),
        "memory_total_mib": parse_numeric(raw.get("memory.total", "")),
        "memory_used_mib": parse_numeric(raw.get("memory.used", "")),
        "memory_free_mib": parse_numeric(raw.get("memory.free", "")),
        "temperature_gpu_c": parse_numeric(raw.get("temperature.gpu", "")),
        "power_draw_w": parse_numeric(raw.get("power.draw", "")),
        "power_limit_w": parse_numeric(raw.get("power.limit", "")),
        "pcie_link_gen_current": parse_numeric(raw.get("pcie.link.gen.current", "")),
        "pcie_link_width_current": parse_numeric(raw.get("pcie.link.width.current", "")),
        "ecc_volatile_uncorrected": parse_numeric(raw.get("ecc.errors.uncorrected.volatile.total", "")),
        "ecc_aggregate_uncorrected": parse_numeric(raw.get("ecc.errors.uncorrected.aggregate.total", "")),
        "nvidia_smi_available": True,
        "nvidia_smi_status": "nvidia_smi_probe_ok",
        "source_mode": "nvidia_smi_probe_once",
        "monitoring_only": True,
        "actual_execution_allowed": False,
        "train_allowed": False,
        "live_mutation_allowed": False,
    }


def fallback_sample_row(
    *,
    run_id: str,
    condition_id: str,
    reward_id: str,
    seed: int,
    global_step: int,
    nvidia_smi_status: str,
) -> Dict[str, Any]:
    return {
        "timestamp_utc": utc_now_iso(),
        "run_id": run_id,
        "condition_id": condition_id,
        "reward_id": reward_id,
        "seed": int(seed),
        "global_step": int(global_step),
        "gpu_index": 0,
        "gpu_name": "SCHEMA_SAMPLE_H200_GPU",
        "gpu_uuid": "SCHEMA_SAMPLE_GPU_UUID",
        "driver_version": "schema_sample",
        "cuda_version": "schema_sample",
        "utilization_gpu_pct": 0.0,
        "utilization_memory_pct": 0.0,
        "memory_total_mib": 0.0,
        "memory_used_mib": 0.0,
        "memory_free_mib": 0.0,
        "temperature_gpu_c": 0.0,
        "power_draw_w": 0.0,
        "power_limit_w": 0.0,
        "pcie_link_gen_current": 0.0,
        "pcie_link_width_current": 0.0,
        "ecc_volatile_uncorrected": 0.0,
        "ecc_aggregate_uncorrected": 0.0,
        "nvidia_smi_available": False,
        "nvidia_smi_status": nvidia_smi_status,
        "source_mode": "schema_sample_fallback",
        "monitoring_only": True,
        "actual_execution_allowed": False,
        "train_allowed": False,
        "live_mutation_allowed": False,
    }


def validate_safety_flags(row: Dict[str, Any]) -> None:
    forbidden_true = [
        "actual_execution_allowed",
        "train_allowed",
        "live_mutation_allowed",
    ]
    for key in forbidden_true:
        if bool(row.get(key, False)):
            raise RuntimeError(f"forbidden safety flag is true: {key}")
    if not bool(row.get("monitoring_only", False)):
        raise RuntimeError("monitoring_only must be true")


def generate(args: argparse.Namespace) -> Dict[str, Any]:
    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    probe = probe_nvidia_smi(timeout_seconds=args.timeout_seconds) if args.probe_once else {
        "available": nvidia_smi_path() is not None,
        "status": "nvidia_smi_probe_skipped_scaffold_mode",
        "rows": [],
        "stderr": "",
        "command": build_query_command(),
    }

    metric_rows: List[Dict[str, Any]] = []
    if args.probe_once and probe["rows"]:
        for raw in probe["rows"]:
            row = row_from_probe_row(
                raw,
                run_id=args.run_id,
                condition_id=args.condition_id,
                reward_id=args.reward_id,
                seed=args.seed,
                global_step=args.global_step,
            )
            validate_safety_flags(row)
            metric_rows.append(row)

    if not metric_rows:
        row = fallback_sample_row(
            run_id=args.run_id,
            condition_id=args.condition_id,
            reward_id=args.reward_id,
            seed=args.seed,
            global_step=args.global_step,
            nvidia_smi_status=str(probe["status"]),
        )
        validate_safety_flags(row)
        metric_rows.append(row)

    csv_path = output_root / "gpu_metrics_step154_sample.csv"
    jsonl_path = output_root / "gpu_metrics_step154_sample.jsonl"
    status_path = output_root / "gpu_monitor_status_step154.json"
    query_contract_path = output_root / "nvidia_smi_query_contract_step154.json"
    manifest_path = output_root / "h200_nvidia_smi_gpu_monitor_step154_manifest.json"

    write_csv(csv_path, metric_rows)
    append_jsonl(jsonl_path, metric_rows)

    nvidia_status = str(probe["status"])
    status_payload = {
        "artifact_version": "h200_nvidia_smi_gpu_monitor_step154",
        "monitor_status": MONITOR_STATUS,
        "monitor_mode": MONITOR_MODE,
        "control_policy": CONTROL_POLICY,
        "timestamp_utc": utc_now_iso(),
        "project_root": str(project_root),
        "nvidia_smi_available": bool(probe["available"]),
        "nvidia_smi_status": nvidia_status,
        "probe_once": bool(args.probe_once),
        "row_count": len(metric_rows),
        "monitoring_only": True,
        "actual_execution_allowed": False,
        "actual_execution_released": False,
        "train_allowed": False,
        "live_mutation_allowed": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
    }
    dump_json(status_path, status_payload)

    query_payload = {
        "artifact_version": "h200_nvidia_smi_query_contract_step154",
        "query_fields": NVIDIA_SMI_QUERY_FIELDS,
        "query_command": build_query_command(),
        "csv_columns": GPU_METRIC_COLUMNS,
        "timeout_seconds": int(args.timeout_seconds),
        "notes": [
            "This contract is monitoring-only.",
            "Missing local nvidia-smi is allowed in scaffold mode.",
            "H200 probe may be run with --probe-once.",
            "This logger cannot mutate training variables.",
        ],
    }
    dump_json(query_contract_path, query_payload)

    manifest = {
        "artifact_version": "h200_nvidia_smi_gpu_monitor_step154",
        "monitor_status": MONITOR_STATUS,
        "monitor_mode": MONITOR_MODE,
        "control_policy": CONTROL_POLICY,
        "timestamp_utc": utc_now_iso(),
        "project_root": str(project_root),
        "output_root": str(output_root),
        "nvidia_smi_available": bool(probe["available"]),
        "nvidia_smi_status": nvidia_status,
        "nvidia_smi_stderr": str(probe.get("stderr", ""))[:2000],
        "probe_once": bool(args.probe_once),
        "metric_row_count": len(metric_rows),
        "gpu_metric_columns": GPU_METRIC_COLUMNS,
        "output_files": {
            "gpu_metrics_csv": str(csv_path),
            "gpu_metrics_jsonl": str(jsonl_path),
            "gpu_monitor_status": str(status_path),
            "nvidia_smi_query_contract": str(query_contract_path),
            "manifest": str(manifest_path),
        },
        "actual_execution_allowed": False,
        "actual_execution_released": False,
        "train_allowed": False,
        "live_mutation_allowed": False,
        "operator_approval_recorded": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "warnings": [] if probe["available"] else ["nvidia_smi_unavailable_allowed_in_local_scaffold"],
    }
    dump_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-root", default="artifacts/observability/h200_nvidia_smi_gpu_monitor_step154")
    parser.add_argument("--run-id", default="step154_schema_sample")
    parser.add_argument("--condition-id", default="A")
    parser.add_argument("--reward-id", default="R0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--global-step", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=5)
    parser.add_argument("--probe-once", action="store_true")
    args = parser.parse_args()

    manifest = generate(args)

    print("[OK] Step 154 H200 nvidia-smi GPU monitor scaffold generated")
    print(f"[OK] monitor_status : {manifest['monitor_status']}")
    print(f"[OK] monitor_mode   : {manifest['monitor_mode']}")
    print(f"[OK] control_policy : {manifest['control_policy']}")
    print(f"[OK] nvidia-smi     : {manifest['nvidia_smi_status']}")
    print(f"[OK] train_allowed  : {manifest['train_allowed']}")
    print(f"[OK] manifest       : {manifest['output_files']['manifest']}")


if __name__ == "__main__":
    main()
