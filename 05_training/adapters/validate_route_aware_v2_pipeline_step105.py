#!/usr/bin/env python3
"""
Step 105 — Route-aware v2 pipeline readiness gate

Purpose
-------
Validate that Step 100~103 outputs form a coherent route-aware causal simulator
v2 scaffold pipeline before moving to the next implementation stage.

This gate does NOT make performance claims. It only checks that:
- Step 100 contract exists and keeps observed/candidate/not_observed boundaries.
- Step 101 simulator scaffold exists and is ready for Step 102.
- Step 102 rollout writer exists and produced raw_events/window_rollup outputs.
- Step 103 canonical KPI integration exists and kept causal_allowed=false.
- Paper/casual performance claim guards remain disabled.

No DB writes.
No tensor DB overwrite.
No API calls.
Only local artifacts are read.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_AUDIT_ROOT = Path("artifacts") / "daegu_bis_api_audit"

STEP100_DIR = DEFAULT_AUDIT_ROOT / "causal_simulator_v2_contract_step100"
STEP101_DIR = DEFAULT_AUDIT_ROOT / "route_aware_minimal_simulator_step101"
STEP102_DIR = DEFAULT_AUDIT_ROOT / "route_aware_rollout_writer_step102"
STEP103_DIR = DEFAULT_AUDIT_ROOT / "route_aware_canonical_kpi_step103"
STEP104_DIR = DEFAULT_AUDIT_ROOT / "project_log_runbook_step104"
DEFAULT_OUTPUT_DIR = DEFAULT_AUDIT_ROOT / "route_aware_v2_pipeline_readiness_step105"

BLOCKED_CLAIM_FIELDS = [
    "actual_headway",
    "actual_arrival_departure_time",
    "actual_dwell",
]

EXPECTED_FALSE_GUARDS = [
    "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
]

READY_STATUSES = {
    "step100_contract_status": "READY_FOR_ROUTE_AWARE_MINIMAL_SCAFFOLD",
    "step101_scaffold_status": "READY_FOR_STEP102_ROLLOUT_WRITER_SCAFFOLD",
    "step102_rollout_status": "READY_FOR_STEP103_CANONICAL_KPI_SCAFFOLD",
    "step103_integration_status": "READY_FOR_STEP104_PROJECT_LOG_RUNBOOK_UPDATE",
}


@dataclass
class CheckResult:
    check_id: str
    status: str
    severity: str
    message: str
    value: Any = None


def read_json(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "check_id",
        "status",
        "severity",
        "message",
        "value",
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            out = dict(row)
            if not isinstance(out.get("value"), str):
                out["value"] = json.dumps(out.get("value"), ensure_ascii=False)
            w.writerow(out)


def nested_get(obj: Any, dotted: str, default: Any = None) -> Any:
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def find_first_existing(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def file_rows_csv(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return 0
    return max(0, len(rows) - 1)


def parquet_or_csv_row_count(parquet_path: Path, csv_path: Optional[Path] = None) -> Optional[int]:
    if parquet_path.exists():
        try:
            import pandas as pd  # type: ignore
            return int(len(pd.read_parquet(parquet_path)))
        except Exception:
            # pyarrow may be missing in a stripped environment; fall through to csv.
            pass
    if csv_path and csv_path.exists():
        return file_rows_csv(csv_path)
    return None


def append_check(
    checks: List[CheckResult],
    check_id: str,
    ok: bool,
    message_ok: str,
    message_fail: str,
    value: Any = None,
    severity: str = "ERROR",
) -> None:
    checks.append(
        CheckResult(
            check_id=check_id,
            status="PASS" if ok else "FAIL",
            severity="INFO" if ok else severity,
            message=message_ok if ok else message_fail,
            value=value,
        )
    )


def extract_contract_guard(contract: Dict[str, Any], field: str) -> Optional[str]:
    """
    Tolerant extraction for field classification. Supports both a flat
    "field_classification" dict and a list/matrix style contract.
    """
    fc = contract.get("field_classification")
    if isinstance(fc, dict) and field in fc:
        val = fc[field]
        if isinstance(val, dict):
            return str(val.get("classification") or val.get("status") or val.get("value"))
        return str(val)

    # Step 100 contract stores classifications under field_groups.
    fg = contract.get("field_groups")
    if isinstance(fg, dict):
        for group_name in (
            "required_route_aware_inputs",
            "optional_api_candidate_inputs",
            "proxy_or_estimated_only_inputs",
            "blocked_actual_observed_inputs",
        ):
            group = fg.get(group_name)
            if isinstance(group, dict) and field in group:
                val = group[field]
                if isinstance(val, dict):
                    return str(val.get("classification") or val.get("status") or val.get("value"))
                return str(val)

    matrix = contract.get("field_readiness_matrix") or contract.get("fields")
    if isinstance(matrix, list):
        for item in matrix:
            if not isinstance(item, dict):
                continue
            name = item.get("field") or item.get("field_name") or item.get("name")
            if name == field:
                return str(item.get("classification") or item.get("status") or item.get("readiness"))

    # Fallback: search a few common top-level values.
    val = contract.get(field)
    if isinstance(val, dict):
        return str(val.get("classification") or val.get("status") or val.get("value"))
    if val is not None:
        return str(val)
    return None


def extract_guard_bool(payload: Dict[str, Any], key: str) -> Optional[bool]:
    val = payload.get(key)
    if isinstance(val, bool):
        return val

    # Common locations used by prior step manifests/reports.
    for dotted in (
        f"claim_guards.{key}",
        f"guards.{key}",
        f"contract_guards.{key}",
        f"non_claim_guards.{key}",
    ):
        v = nested_get(payload, dotted)
        if isinstance(v, bool):
            return v

    return None


def load_optional_json(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not path.exists():
        return None, f"missing: {path}"
    try:
        return read_json(path), None
    except Exception as e:
        return None, f"failed to read {path}: {e}"


def validate_pipeline(project_root: Path, output_dir: Path, require_step104: bool = False) -> Dict[str, Any]:
    audit_root = project_root / DEFAULT_AUDIT_ROOT
    step100_dir = project_root / STEP100_DIR
    step101_dir = project_root / STEP101_DIR
    step102_dir = project_root / STEP102_DIR
    step103_dir = project_root / STEP103_DIR
    step104_dir = project_root / STEP104_DIR
    output_dir = project_root / output_dir if not output_dir.is_absolute() else output_dir

    checks: List[CheckResult] = []

    # ------------------------------------------------------------------ #
    # Step 100 contract
    # ------------------------------------------------------------------ #
    contract_json_path = step100_dir / "causal_simulator_v2_contract.json"
    contract, err = load_optional_json(contract_json_path)
    append_check(
        checks,
        "S100_contract_json_exists",
        contract is not None,
        "Step 100 causal simulator v2 contract JSON exists.",
        err or "Step 100 contract JSON is missing or unreadable.",
        str(contract_json_path),
    )

    if contract:
        status = str(contract.get("contract_status", contract.get("status", "")))
        append_check(
            checks,
            "S100_contract_status_ready",
            status == READY_STATUSES["step100_contract_status"],
            "Step 100 contract status is ready for route-aware minimal scaffold.",
            "Step 100 contract status is not READY_FOR_ROUTE_AWARE_MINIMAL_SCAFFOLD.",
            status,
        )

        for field in BLOCKED_CLAIM_FIELDS:
            classification = extract_contract_guard(contract, field)
            append_check(
                checks,
                f"S100_{field}_not_observed",
                classification == "not_observed",
                f"{field} remains not_observed in Step 100 contract.",
                f"{field} must remain not_observed, got {classification!r}.",
                classification,
            )

        for guard in EXPECTED_FALSE_GUARDS:
            guard_value = extract_guard_bool(contract, guard)
            append_check(
                checks,
                f"S100_{guard}_false",
                guard_value is False,
                f"{guard}=false in Step 100 contract.",
                f"{guard} must be false in Step 100 contract.",
                guard_value,
            )

    # ------------------------------------------------------------------ #
    # Step 101 manifest
    # ------------------------------------------------------------------ #
    s101_manifest_path = step101_dir / "route_aware_minimal_simulator_manifest.json"
    s101, err = load_optional_json(s101_manifest_path)
    append_check(
        checks,
        "S101_manifest_exists",
        s101 is not None,
        "Step 101 simulator scaffold manifest exists.",
        err or "Step 101 manifest is missing or unreadable.",
        str(s101_manifest_path),
    )
    if s101:
        scaffold_status = str(s101.get("scaffold_status", ""))
        append_check(
            checks,
            "S101_scaffold_status_ready",
            scaffold_status == READY_STATUSES["step101_scaffold_status"],
            "Step 101 scaffold is ready for Step 102 rollout writer.",
            "Step 101 scaffold status is not ready for Step 102.",
            scaffold_status,
        )
        trace_rows = int(
            s101.get(
                "trace_rows",
                s101.get(
                    "smoke_trace_rows",
                    nested_get(s101, "smoke_rollout.trace_rows", 0),
                ),
            )
            or 0
        )
        append_check(
            checks,
            "S101_trace_rows_positive",
            trace_rows > 0,
            "Step 101 produced positive smoke rollout trace rows.",
            "Step 101 trace row count must be positive.",
            trace_rows,
        )

    # ------------------------------------------------------------------ #
    # Step 102 rollout writer
    # ------------------------------------------------------------------ #
    s102_manifest_path = step102_dir / "route_aware_rollout_writer_manifest.json"
    s102, err = load_optional_json(s102_manifest_path)
    append_check(
        checks,
        "S102_manifest_exists",
        s102 is not None,
        "Step 102 rollout writer manifest exists.",
        err or "Step 102 manifest is missing or unreadable.",
        str(s102_manifest_path),
    )

    raw_events_parquet = step102_dir / "raw_events.parquet"
    raw_events_csv = step102_dir / "raw_events.csv"
    window_rollup_parquet = step102_dir / "window_rollup.parquet"
    window_rollup_csv = step102_dir / "window_rollup.csv"

    raw_event_rows = parquet_or_csv_row_count(raw_events_parquet, raw_events_csv)
    window_rollup_rows = parquet_or_csv_row_count(window_rollup_parquet, window_rollup_csv)

    append_check(
        checks,
        "S102_raw_events_exist",
        raw_events_parquet.exists() or raw_events_csv.exists(),
        "Step 102 raw_events output exists.",
        "Step 102 raw_events output is missing.",
        {"parquet": str(raw_events_parquet), "csv": str(raw_events_csv)},
    )
    append_check(
        checks,
        "S102_window_rollup_exists",
        window_rollup_parquet.exists() or window_rollup_csv.exists(),
        "Step 102 window_rollup output exists.",
        "Step 102 window_rollup output is missing.",
        {"parquet": str(window_rollup_parquet), "csv": str(window_rollup_csv)},
    )
    append_check(
        checks,
        "S102_raw_event_rows_positive",
        raw_event_rows is not None and raw_event_rows > 0,
        "Step 102 raw_events has positive row count.",
        "Step 102 raw_events must have positive row count.",
        raw_event_rows,
    )
    append_check(
        checks,
        "S102_window_rollup_rows_positive",
        window_rollup_rows is not None and window_rollup_rows > 0,
        "Step 102 window_rollup has positive row count.",
        "Step 102 window_rollup must have positive row count.",
        window_rollup_rows,
    )

    if s102:
        rollout_status = str(s102.get("rollout_status", ""))
        append_check(
            checks,
            "S102_rollout_status_ready",
            rollout_status == READY_STATUSES["step102_rollout_status"],
            "Step 102 rollout status is ready for Step 103 canonical KPI scaffold.",
            "Step 102 rollout status is not ready for Step 103.",
            rollout_status,
        )
        parquet_ready = bool(
            s102.get(
                "parquet_ready",
                bool(nested_get(s102, "parquet_status.raw_events.written", False))
                and bool(nested_get(s102, "parquet_status.window_rollup.written", False)),
            )
        )
        append_check(
            checks,
            "S102_parquet_ready",
            parquet_ready is True,
            "Step 102 parquet_ready=true.",
            "Step 102 parquet_ready must be true.",
            parquet_ready,
        )

    # ------------------------------------------------------------------ #
    # Step 103 canonical KPI integration
    # ------------------------------------------------------------------ #
    s103_manifest_path = step103_dir / "route_aware_canonical_kpi_step103_manifest.json"
    s103, err = load_optional_json(s103_manifest_path)
    append_check(
        checks,
        "S103_manifest_exists",
        s103 is not None,
        "Step 103 canonical KPI integration manifest exists.",
        err or "Step 103 manifest is missing or unreadable.",
        str(s103_manifest_path),
    )

    canonical_eval = step103_dir / "canonical_eval"
    kpi_by_window = canonical_eval / "kpi_by_window.parquet"
    kpi_by_seed = canonical_eval / "kpi_by_seed.parquet"
    kpi_by_time_band = canonical_eval / "kpi_by_time_band.parquet"
    kpi_overall_path = canonical_eval / "kpi_overall.json"

    append_check(
        checks,
        "S103_kpi_by_window_exists",
        kpi_by_window.exists(),
        "Step 103 canonical kpi_by_window.parquet exists.",
        "Step 103 canonical kpi_by_window.parquet is missing.",
        str(kpi_by_window),
    )
    append_check(
        checks,
        "S103_kpi_by_seed_exists",
        kpi_by_seed.exists(),
        "Step 103 canonical kpi_by_seed.parquet exists.",
        "Step 103 canonical kpi_by_seed.parquet is missing.",
        str(kpi_by_seed),
    )
    append_check(
        checks,
        "S103_kpi_by_time_band_exists",
        kpi_by_time_band.exists(),
        "Step 103 canonical kpi_by_time_band.parquet exists.",
        "Step 103 canonical kpi_by_time_band.parquet is missing.",
        str(kpi_by_time_band),
    )

    kpi_overall, err = load_optional_json(kpi_overall_path)
    append_check(
        checks,
        "S103_kpi_overall_exists",
        kpi_overall is not None,
        "Step 103 canonical kpi_overall.json exists.",
        err or "Step 103 canonical kpi_overall.json is missing or unreadable.",
        str(kpi_overall_path),
    )

    if s103:
        integration_status = str(s103.get("integration_status", ""))
        append_check(
            checks,
            "S103_integration_status_ready",
            integration_status == READY_STATUSES["step103_integration_status"],
            "Step 103 integration status is ready for Step 104 project log/runbook update.",
            "Step 103 integration status is not ready for Step 104.",
            integration_status,
        )

        causal_allowed = s103.get(
            "causal_allowed",
            nested_get(s103, "canonical_summary.causal_comparison_allowed"),
        )
        append_check(
            checks,
            "S103_causal_allowed_false",
            causal_allowed is False,
            "Step 103 causal_allowed=false.",
            "Step 103 causal_allowed must remain false.",
            causal_allowed,
        )

        s103_row_counts = nested_get(s103, "canonical_summary.row_counts", {}) or {}
        for key in ("kpi_by_window", "kpi_by_seed", "kpi_by_time_band"):
            value = s103.get(key)
            if not isinstance(value, int) and isinstance(s103_row_counts, dict):
                value = s103_row_counts.get(key)
            if isinstance(value, int):
                append_check(
                    checks,
                    f"S103_{key}_rows_positive",
                    value > 0,
                    f"Step 103 {key} row count is positive.",
                    f"Step 103 {key} row count must be positive.",
                    value,
                )

    if kpi_overall:
        causal_comparison_allowed = kpi_overall.get("causal_comparison_allowed")
        append_check(
            checks,
            "S103_overall_causal_comparison_allowed_false",
            causal_comparison_allowed is False,
            "canonical kpi_overall keeps causal_comparison_allowed=false.",
            "canonical kpi_overall must keep causal_comparison_allowed=false.",
            causal_comparison_allowed,
        )

    # ------------------------------------------------------------------ #
    # Step 104 docs/runbook — warning by default, error if required.
    # ------------------------------------------------------------------ #
    project_log_path = project_root / "project_log.md"
    runbook_path = project_root / "05_training" / "adapters" / "route_aware_v2_pipeline_runbook_step104.md"
    step104_manifest_path = step104_dir / "project_log_runbook_step104_manifest.json"

    append_check(
        checks,
        "S104_project_log_exists",
        project_log_path.exists(),
        "project_log.md exists.",
        "project_log.md is missing.",
        str(project_log_path),
        severity="ERROR" if require_step104 else "WARN",
    )
    append_check(
        checks,
        "S104_runbook_exists",
        runbook_path.exists(),
        "Step 104 route-aware v2 runbook exists.",
        "Step 104 runbook is missing. Run Step 104 before treating documentation as closed.",
        str(runbook_path),
        severity="ERROR" if require_step104 else "WARN",
    )
    append_check(
        checks,
        "S104_manifest_exists",
        step104_manifest_path.exists(),
        "Step 104 manifest exists.",
        "Step 104 manifest is missing. This is a warning unless --require-step104 is used.",
        str(step104_manifest_path),
        severity="ERROR" if require_step104 else "WARN",
    )

    # ------------------------------------------------------------------ #
    # Aggregate status.
    # ------------------------------------------------------------------ #
    hard_failures = [c for c in checks if c.status == "FAIL" and c.severity == "ERROR"]
    warnings = [c for c in checks if c.status == "FAIL" and c.severity == "WARN"]

    audit_status = "PASS" if not hard_failures else "FAIL"
    readiness_status = (
        "READY_FOR_STEP106_MINIMAL_QUEUE_DEMAND_SCAFFOLD"
        if audit_status == "PASS"
        else "BLOCKED"
    )

    payload: Dict[str, Any] = {
        "artifact_version": "route_aware_v2_pipeline_readiness_step105_v1",
        "audit_status": audit_status,
        "readiness_status": readiness_status,
        "project_root": str(project_root),
        "audit_root": str(audit_root),
        "output_root": str(output_dir),
        "step_inputs": {
            "step100_contract_json": str(contract_json_path),
            "step101_manifest": str(s101_manifest_path),
            "step102_manifest": str(s102_manifest_path),
            "step102_raw_events_parquet": str(raw_events_parquet),
            "step102_window_rollup_parquet": str(window_rollup_parquet),
            "step103_manifest": str(s103_manifest_path),
            "step103_canonical_eval": str(canonical_eval),
            "step104_runbook": str(runbook_path),
        },
        "row_counts": {
            "step101_trace_rows": int(
                s101.get(
                    "trace_rows",
                    s101.get(
                        "smoke_trace_rows",
                        nested_get(s101, "smoke_rollout.trace_rows", 0),
                    ),
                )
                or 0
            ) if s101 else None,
            "step102_raw_event_rows": raw_event_rows,
            "step102_window_rollup_rows": window_rollup_rows,
            "step103_kpi_by_window_rows": (
                s103.get("kpi_by_window")
                or nested_get(s103, "canonical_summary.row_counts.kpi_by_window")
            ) if s103 else None,
            "step103_kpi_by_seed_rows": (
                s103.get("kpi_by_seed")
                or nested_get(s103, "canonical_summary.row_counts.kpi_by_seed")
            ) if s103 else None,
            "step103_kpi_by_time_band_rows": (
                s103.get("kpi_by_time_band")
                or nested_get(s103, "canonical_summary.row_counts.kpi_by_time_band")
            ) if s103 else None,
        },
        "claim_guards": {
            "actual_headway": "not_observed",
            "actual_arrival_departure_time": "not_observed",
            "actual_dwell": "not_observed",
            "paper_level_claim_allowed": False,
            "causal_performance_claim_allowed": False,
            "canonical_causal_comparison_allowed": False,
        },
        "checks": [asdict(c) for c in checks],
        "hard_failure_count": len(hard_failures),
        "warning_count": len(warnings),
        "warnings": [asdict(c) for c in warnings],
        "hard_failures": [asdict(c) for c in hard_failures],
        "next_recommended_step": "Step 106 — minimal queue/demand scaffold on route-aware simulator",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_json = output_dir / "route_aware_v2_pipeline_readiness_step105.json"
    report_md = output_dir / "route_aware_v2_pipeline_readiness_step105.md"
    matrix_csv = output_dir / "route_aware_v2_pipeline_readiness_matrix_step105.csv"

    write_json(report_json, payload)
    write_csv(matrix_csv, [asdict(c) for c in checks])
    write_markdown_report(report_md, payload)

    payload["output_files"] = {
        "readiness_json": str(report_json),
        "readiness_md": str(report_md),
        "readiness_matrix_csv": str(matrix_csv),
    }
    write_json(report_json, payload)

    return payload


def write_markdown_report(path: Path, payload: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Step 105 — Route-aware v2 Pipeline Readiness Gate")
    lines.append("")
    lines.append(f"- audit_status: `{payload['audit_status']}`")
    lines.append(f"- readiness_status: `{payload['readiness_status']}`")
    lines.append(f"- hard_failure_count: `{payload['hard_failure_count']}`")
    lines.append(f"- warning_count: `{payload['warning_count']}`")
    lines.append("")
    lines.append("## Row counts")
    lines.append("")
    for k, v in payload.get("row_counts", {}).items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append("## Claim guards")
    lines.append("")
    for k, v in payload.get("claim_guards", {}).items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| check_id | status | severity | value |")
    lines.append("|---|---:|---:|---|")
    for c in payload["checks"]:
        value = c.get("value")
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False)
        value = value.replace("|", "\\|")
        lines.append(f"| `{c['check_id']}` | `{c['status']}` | `{c['severity']}` | `{value}` |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "This readiness gate validates software wiring and claim guards only. "
        "It does not validate real-world performance, actual headway, actual arrival/departure time, or actual dwell time."
    )
    lines.append("")
    lines.append("## Next recommended step")
    lines.append("")
    lines.append(f"`{payload['next_recommended_step']}`")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".", help="urbanbus_rl_project root")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="output directory for Step 105 readiness reports",
    )
    parser.add_argument(
        "--require-step104",
        action="store_true",
        help="treat missing Step 104 project log/runbook outputs as hard failures",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    payload = validate_pipeline(
        project_root=project_root,
        output_dir=Path(args.output_dir),
        require_step104=bool(args.require_step104),
    )

    print("[OK] Step 105 route-aware v2 pipeline readiness gate completed" if payload["audit_status"] == "PASS" else "[FAIL] Step 105 route-aware v2 pipeline readiness gate failed")
    print(f"[OK] audit_status    : {payload['audit_status']}")
    print(f"[OK] readiness_status: {payload['readiness_status']}")
    print(f"[OK] output_root     : {payload['output_root']}")
    print(f"[OK] hard_failures   : {payload['hard_failure_count']}")
    print(f"[OK] warnings        : {payload['warning_count']}")
    if payload["hard_failure_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
