from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

SCHEMA_VERSION = "h200_tensorboard_jsonl_logger_v1_step153"
INTEGRATION_STATUS = "LOGGER_INTEGRATION_READY_MONITORING_ONLY_STILL_LOCKED"
DASHBOARD_MODE = "MONITORING_ONLY"
CONTROL_POLICY = "NO_LIVE_MUTATION_CONFIG_BASED_NEXT_RUN_ONLY"

REQUIRED_METRIC_GROUPS = ["train", "kpi", "safety", "runtime"]

TENSORBOARD_SCALAR_CONTRACT = {
    "schema_version": "tensorboard_scalar_contract_v1_step153",
    "metric_names": {
        "train": [
            "train/policy_loss",
            "train/value_loss",
            "train/reward_mean",
            "train/reward_std",
            "train/entropy",
            "train/kl_divergence",
            "train/learning_rate",
        ],
        "kpi": [
            "kpi/avg_wait_seconds",
            "kpi/passenger_service_rate",
            "kpi/passenger_wait_p95_seconds",
            "kpi/bunching_rate",
            "kpi/on_time_rate",
            "kpi/energy_proxy_per_passenger",
        ],
        "safety": [
            "safety/grad_norm",
            "safety/nan_detected",
            "safety/inf_detected",
            "safety/checkpoint_validation_passed",
        ],
        "runtime": [
            "runtime/env_steps",
            "runtime/update_index",
            "runtime/wall_clock_seconds",
        ],
        "gpu": [
            "gpu/utilization_pct",
            "gpu/memory_used_mb",
            "gpu/temperature_c",
            "gpu/power_draw_w",
        ],
    },
    "control_policy": CONTROL_POLICY,
    "live_mutation_allowed": False,
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def load_json(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with path.open("r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


def get_git_commit(project_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def try_create_summary_writer(log_dir: Path, enabled: bool = True, require_tensorboard: bool = False):
    if not enabled:
        return None, False, "tensorboard_disabled"
    try:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore

        log_dir.mkdir(parents=True, exist_ok=True)
        return SummaryWriter(log_dir=str(log_dir)), True, "tensorboard_available"
    except Exception as exc:
        if require_tensorboard:
            raise RuntimeError(f"TensorBoard writer required but unavailable: {exc}") from exc
        return None, False, f"tensorboard_unavailable: {exc.__class__.__name__}"


@dataclass
class ObservabilityRunContext:
    run_id: str
    condition_id: str
    reward_id: str
    seed: int
    project_root: str
    output_root: str
    git_commit: str = "UNKNOWN"
    dashboard_mode: str = DASHBOARD_MODE
    control_policy: str = CONTROL_POLICY
    actual_execution_allowed: bool = False
    actual_execution_released: bool = False
    train_allowed: bool = False
    paper_level_claim_allowed: bool = False
    causal_performance_claim_allowed: bool = False
    live_mutation_allowed: bool = False
    created_at_utc: str = ""

    def normalized(self) -> "ObservabilityRunContext":
        if not self.created_at_utc:
            self.created_at_utc = utc_now_iso()
        return self

    def validate_locked(self) -> None:
        if self.dashboard_mode != DASHBOARD_MODE:
            raise ValueError(f"dashboard_mode must be {DASHBOARD_MODE}")
        if self.control_policy != CONTROL_POLICY:
            raise ValueError(f"control_policy must be {CONTROL_POLICY}")
        forbidden_true = {
            "actual_execution_allowed": self.actual_execution_allowed,
            "actual_execution_released": self.actual_execution_released,
            "train_allowed": self.train_allowed,
            "paper_level_claim_allowed": self.paper_level_claim_allowed,
            "causal_performance_claim_allowed": self.causal_performance_claim_allowed,
            "live_mutation_allowed": self.live_mutation_allowed,
        }
        bad = [k for k, v in forbidden_true.items() if bool(v)]
        if bad:
            raise ValueError(f"locked Step 153 context has forbidden true flags: {bad}")


class H200ExperimentMetricLogger:
    def __init__(
        self,
        context: ObservabilityRunContext,
        enable_tensorboard: bool = True,
        require_tensorboard: bool = False,
    ) -> None:
        self.context = context.normalized()
        self.context.validate_locked()
        self.output_root = Path(self.context.output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

        self.metrics_jsonl_path = self.output_root / "metrics_events_step153_sample.jsonl"
        self.gpu_csv_path = self.output_root / "gpu_metrics_step153_sample.csv"
        self.run_status_path = self.output_root / "run_status_step153_sample.json"
        self.tb_contract_path = self.output_root / "tensorboard_scalar_contract_step153.json"
        self.tb_event_dir = self.output_root / "tensorboard_events" / self.context.run_id

        self.writer, self.tensorboard_available, self.tensorboard_status = try_create_summary_writer(
            self.tb_event_dir,
            enabled=enable_tensorboard,
            require_tensorboard=require_tensorboard,
        )
        self.metric_event_count = 0
        self.metric_groups_seen: set[str] = set()
        self.gpu_row_count = 0
        self.closed = False

        dump_json(self.tb_contract_path, TENSORBOARD_SCALAR_CONTRACT)
        self.write_run_status(status="RUN_STATUS_INITIALIZED_MONITORING_ONLY")
        self._ensure_gpu_csv_header()

    def write_run_status(self, status: str, extra: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "integration_status": INTEGRATION_STATUS,
            "run_context": asdict(self.context),
            "dashboard_mode": DASHBOARD_MODE,
            "control_policy": CONTROL_POLICY,
            "actual_execution_allowed": False,
            "actual_execution_released": False,
            "train_allowed": False,
            "paper_level_claim_allowed": False,
            "causal_performance_claim_allowed": False,
            "live_mutation_allowed": False,
            "updated_at_utc": utc_now_iso(),
        }
        if extra:
            payload["extra"] = extra
        dump_json(self.run_status_path, payload)

    def _ensure_gpu_csv_header(self) -> None:
        if self.gpu_csv_path.exists():
            return
        with self.gpu_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
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
                ],
            )
            writer.writeheader()

    def log_scalar(
        self,
        metric_name: str,
        metric_value: float,
        step: int,
        metric_group: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.closed:
            raise RuntimeError("logger is already closed")
        if not isinstance(step, int) or step < 0:
            raise ValueError("step must be a non-negative integer")
        value = float(metric_value)
        if not math.isfinite(value):
            raise ValueError(f"metric_value must be finite: {metric_name}={metric_value}")
        group = metric_group or metric_name.split("/", 1)[0]
        event = {
            "schema_version": SCHEMA_VERSION,
            "event_type": "scalar",
            "time_utc": utc_now_iso(),
            "run_id": self.context.run_id,
            "condition_id": self.context.condition_id,
            "reward_id": self.context.reward_id,
            "seed": self.context.seed,
            "git_commit": self.context.git_commit,
            "step": int(step),
            "metric_group": group,
            "metric_name": metric_name,
            "metric_value": value,
            "dashboard_mode": DASHBOARD_MODE,
            "control_policy": CONTROL_POLICY,
            "live_mutation_allowed": False,
        }
        if tags:
            event["tags"] = tags
        with self.metrics_jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        self.metric_event_count += 1
        self.metric_groups_seen.add(group)
        if self.writer is not None:
            self.writer.add_scalar(metric_name, value, int(step))

    def log_many_scalars(self, scalars: Dict[str, float], step: int) -> None:
        for name, value in scalars.items():
            self.log_scalar(metric_name=name, metric_value=value, step=step)

    def log_gpu_snapshot(
        self,
        step: int,
        gpu_index: int = 0,
        gpu_name: str = "UNKNOWN",
        utilization_pct: float = 0.0,
        memory_used_mb: float = 0.0,
        memory_total_mb: float = 0.0,
        temperature_c: float = 0.0,
        power_draw_w: float = 0.0,
        ecc_error_count: int = 0,
    ) -> None:
        if self.closed:
            raise RuntimeError("logger is already closed")
        row = {
            "time_utc": utc_now_iso(),
            "run_id": self.context.run_id,
            "step": int(step),
            "gpu_index": int(gpu_index),
            "gpu_name": str(gpu_name),
            "utilization_pct": float(utilization_pct),
            "memory_used_mb": float(memory_used_mb),
            "memory_total_mb": float(memory_total_mb),
            "temperature_c": float(temperature_c),
            "power_draw_w": float(power_draw_w),
            "ecc_error_count": int(ecc_error_count),
        }
        with self.gpu_csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writerow(row)
        self.gpu_row_count += 1
        for key in ("utilization_pct", "memory_used_mb", "temperature_c", "power_draw_w"):
            self.log_scalar(metric_name=f"gpu/{key}", metric_value=float(row[key]), step=step, metric_group="gpu")

    def record_control_mutation_attempt(self, parameter_name: str, proposed_value: Any, reason: str = "") -> None:
        event = {
            "schema_version": SCHEMA_VERSION,
            "event_type": "control_mutation_blocked",
            "time_utc": utc_now_iso(),
            "run_id": self.context.run_id,
            "parameter_name": str(parameter_name),
            "proposed_value": proposed_value,
            "reason": reason,
            "allowed": False,
            "control_policy": CONTROL_POLICY,
            "message": "Live mutation is forbidden. Put changes into the next run config only.",
        }
        with self.metrics_jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        raise RuntimeError("live mutation blocked by Step 153 control policy")

    def finalize(self, final_status: str = "LOGGER_FINALIZED_MONITORING_ONLY") -> Dict[str, Any]:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
        self.closed = True
        self.write_run_status(
            status=final_status,
            extra={
                "metric_event_count": self.metric_event_count,
                "metric_groups_seen": sorted(self.metric_groups_seen),
                "gpu_row_count": self.gpu_row_count,
            },
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifact_version": "step153_h200_tensorboard_jsonl_logger_integration_v1",
            "integration_status": INTEGRATION_STATUS,
            "dashboard_mode": DASHBOARD_MODE,
            "control_policy": CONTROL_POLICY,
            "actual_execution_allowed": False,
            "actual_execution_released": False,
            "train_allowed": False,
            "paper_level_claim_allowed": False,
            "causal_performance_claim_allowed": False,
            "live_mutation_allowed": False,
            "run_context": asdict(self.context),
            "output_root": str(self.output_root),
            "output_files": {
                "run_status": str(self.run_status_path),
                "metrics_events_jsonl": str(self.metrics_jsonl_path),
                "gpu_metrics_csv": str(self.gpu_csv_path),
                "tensorboard_scalar_contract": str(self.tb_contract_path),
                "tensorboard_event_dir": str(self.tb_event_dir),
            },
            "tensorboard": {
                "enabled": True,
                "available": bool(self.tensorboard_available),
                "status": self.tensorboard_status,
            },
            "required_metric_groups": REQUIRED_METRIC_GROUPS,
            "metric_event_count": int(self.metric_event_count),
            "metric_groups_seen": sorted(self.metric_groups_seen),
            "gpu_row_count": int(self.gpu_row_count),
            "created_at_utc": self.context.created_at_utc,
            "finalized_at_utc": utc_now_iso(),
            "next_gate": "WAITING_FOR_H200_STEP149_EXPECT_H200_RESULT_AND_FUTURE_LOGGER_WIRING",
        }
        manifest_path = self.output_root / "h200_tensorboard_jsonl_logger_step153_manifest.json"
        dump_json(manifest_path, manifest)
        return manifest


def emit_dry_run_sample(logger: H200ExperimentMetricLogger) -> None:
    sample_steps = [0, 1, 2]
    for step in sample_steps:
        logger.log_many_scalars(
            {
                "train/policy_loss": 0.30 - step * 0.02,
                "train/value_loss": 0.80 - step * 0.04,
                "train/reward_mean": -1.20 + step * 0.10,
                "train/reward_std": 0.15 + step * 0.01,
                "train/entropy": 0.60 - step * 0.02,
                "train/kl_divergence": 0.010 + step * 0.001,
                "train/learning_rate": 0.0003,
                "kpi/avg_wait_seconds": 300.0 - step * 5.0,
                "kpi/passenger_service_rate": 0.95 + step * 0.002,
                "kpi/passenger_wait_p95_seconds": 720.0 - step * 8.0,
                "kpi/bunching_rate": 0.08 - step * 0.005,
                "kpi/on_time_rate": 0.80 + step * 0.003,
                "kpi/energy_proxy_per_passenger": 1.25 - step * 0.02,
                "safety/grad_norm": 0.45 + step * 0.01,
                "safety/nan_detected": 0.0,
                "safety/inf_detected": 0.0,
                "safety/checkpoint_validation_passed": 1.0,
                "runtime/env_steps": float(step * 1024),
                "runtime/update_index": float(step),
                "runtime/wall_clock_seconds": float(step * 12),
            },
            step=step,
        )
    logger.log_gpu_snapshot(
        step=2,
        gpu_index=0,
        gpu_name="H200_EXPECTED_PLACEHOLDER",
        utilization_pct=0.0,
        memory_used_mb=0.0,
        memory_total_mb=0.0,
        temperature_c=0.0,
        power_draw_w=0.0,
        ecc_error_count=0,
    )


def build_context_from_args(args: argparse.Namespace) -> ObservabilityRunContext:
    project_root = Path(args.project_root).resolve()
    git_commit = args.git_commit or get_git_commit(project_root)
    return ObservabilityRunContext(
        run_id=args.run_id,
        condition_id=args.condition_id,
        reward_id=args.reward_id,
        seed=int(args.seed),
        project_root=str(project_root),
        output_root=str(Path(args.output_root).resolve()),
        git_commit=git_commit,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Step 153 H200 TensorBoard + JSONL logger integration scaffold")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-root", default="artifacts/observability/h200_tensorboard_jsonl_logger_step153")
    parser.add_argument("--run-id", default="step153_sample_A_R0_seed001")
    parser.add_argument("--condition-id", default="A")
    parser.add_argument("--reward-id", default="R0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--git-commit", default="")
    parser.add_argument("--dry-run-sample", action="store_true")
    parser.add_argument("--disable-tensorboard", action="store_true")
    parser.add_argument("--require-tensorboard", action="store_true")
    args = parser.parse_args(argv)

    context = build_context_from_args(args)
    logger = H200ExperimentMetricLogger(
        context=context,
        enable_tensorboard=not args.disable_tensorboard,
        require_tensorboard=bool(args.require_tensorboard),
    )
    if args.dry_run_sample:
        emit_dry_run_sample(logger)
    manifest = logger.finalize()

    print("[OK] Step 153 H200 TensorBoard + JSONL logger integration generated")
    print(f"[OK] integration_status: {manifest['integration_status']}")
    print(f"[OK] dashboard_mode    : {manifest['dashboard_mode']}")
    print(f"[OK] control_policy    : {manifest['control_policy']}")
    print(f"[OK] tensorboard       : {manifest['tensorboard']['status']}")
    print(f"[OK] train_allowed     : {manifest['train_allowed']}")
    print(f"[OK] manifest          : {Path(manifest['output_root']) / 'h200_tensorboard_jsonl_logger_step153_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
