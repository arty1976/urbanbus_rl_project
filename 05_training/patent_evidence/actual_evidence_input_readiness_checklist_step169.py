from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

ARTIFACT_VERSION = "actual_evidence_input_readiness_checklist_step169_v1"
BUNDLE_READY = "ACTUAL_EVIDENCE_INPUTS_READY_FOR_ZERO_LOSS_PIPELINE_NONCLAIM"
BUNDLE_TEMPLATE = "ACTUAL_EVIDENCE_INPUT_READINESS_TEMPLATE_STILL_LOCKED"
BUNDLE_BLOCKED = "ACTUAL_EVIDENCE_INPUTS_BLOCKED_NONCLAIM"

REQUIRED_RAW_EVENT_COLUMNS = [
    "attempt_id",
    "state_ts",
    "condition_id",
    "seed",
    "route_id",
    "direction_id",
    "vehicle_id",
    "current_stop_id",
    "pickup_stop_id",
    "dropoff_stop_id",
    "existing_passenger_id",
    "new_passenger_id",
    "eta_without_new_pickup_sec",
    "eta_with_new_pickup_sec",
]

REQUIRED_WINDOW_ROLLUP_COLUMNS = [
    "condition_id",
    "seed",
    "window_id",
    "state_ts",
    "time_band",
]

REQUIRED_ROUTE_SEQUENCE_COLUMNS = [
    "route_id",
    "direction_id",
    "stop_id",
    "stop_order",
]

REQUIRED_RUN_MANIFEST_KEYS = [
    "run_id",
    "condition_id",
    "seed",
]

PIPELINE_SCRIPTS = {
    "step160_reporter": "05_training/patent_evidence/zero_loss_pickup_evidence_reporter_step160.py",
    "step161_writer": "05_training/patent_evidence/route_aware_pickup_attempt_event_writer_step161.py",
    "step162_adapter": "05_training/patent_evidence/route_aware_rollout_adapter_step162.py",
    "step163_attention_extractor": "05_training/patent_evidence/gatv2_real_attention_extractor_step163.py",
    "step165_path_filter": "05_training/patent_evidence/attempt_route_path_attention_filter_step165.py",
    "step166_report_v2": "05_training/patent_evidence/patent_evidence_report_v2_step166.py",
}

LOCKED_FALSE_FLAGS = [
    "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
    "actual_operational_claim_allowed",
    "train_allowed",
    "winner_selected",
]


def sha256_file(path: Path) -> Optional[str]:
    if not path or not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with path.open("r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise RuntimeError(f"unsupported table format for {path}; expected .csv or .parquet")


def check_table(path: Optional[Path], required_cols: List[str], label: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "label": label,
        "path": str(path) if path else "",
        "exists": False,
        "row_count": 0,
        "column_count": 0,
        "missing_columns": list(required_cols),
        "sha256": None,
        "status": "MISSING",
    }
    if path is None:
        return out
    if not path.exists():
        return out
    out["exists"] = True
    out["sha256"] = sha256_file(path)
    try:
        df = read_table(path)
    except Exception as exc:
        out["status"] = "READ_ERROR"
        out["error"] = str(exc)
        return out
    out["row_count"] = int(len(df))
    out["column_count"] = int(len(df.columns))
    out["columns"] = [str(c) for c in df.columns]
    missing = [c for c in required_cols if c not in df.columns]
    out["missing_columns"] = missing
    if len(df) <= 0:
        out["status"] = "EMPTY"
    elif missing:
        out["status"] = "MISSING_COLUMNS"
    else:
        out["status"] = "PASS"
    return out


def check_json_manifest(path: Optional[Path], required_keys: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "label": "run_manifest",
        "path": str(path) if path else "",
        "exists": False,
        "missing_keys": list(required_keys),
        "sha256": None,
        "status": "MISSING",
    }
    if path is None:
        return out
    if not path.exists():
        return out
    out["exists"] = True
    out["sha256"] = sha256_file(path)
    try:
        payload = load_json(path)
    except Exception as exc:
        out["status"] = "READ_ERROR"
        out["error"] = str(exc)
        return out
    missing = [k for k in required_keys if k not in payload]
    out["missing_keys"] = missing
    out["manifest_keys"] = sorted(str(k) for k in payload.keys())
    for flag in LOCKED_FALSE_FLAGS:
        if flag in payload:
            out[flag] = bool(payload.get(flag))
    out["status"] = "PASS" if not missing else "MISSING_KEYS"
    return out


def make_sample_inputs(output_root: Path) -> Dict[str, str]:
    sample_root = output_root / "sample_inputs"
    sample_root.mkdir(parents=True, exist_ok=True)

    raw_rows = []
    deltas = [0.0, 14.0, 0.0, 22.0, 0.0, 35.0]
    stops = ["S001", "S002", "S003", "S004", "S005", "S006", "S007", "S008"]
    for i, delta in enumerate(deltas):
        raw_rows.append({
            "attempt_id": f"sample_attempt_{i:03d}",
            "state_ts": f"2026-01-02T08:{i * 5:02d}:00+09:00",
            "condition_id": "A",
            "seed": 1,
            "route_id": "R_SAMPLE_1",
            "direction_id": "0",
            "vehicle_id": f"BUS_{i % 2}",
            "current_stop_id": stops[i],
            "pickup_stop_id": stops[i + 1],
            "dropoff_stop_id": stops[i + 2],
            "existing_passenger_id": f"existing_{i:03d}",
            "new_passenger_id": f"new_{i:03d}",
            "eta_without_new_pickup_sec": 600.0 + i * 20.0,
            "eta_with_new_pickup_sec": 600.0 + i * 20.0 + delta,
            "decision": "accept" if delta <= 0 else "reject_eta_loss",
        })
    raw_path = sample_root / "raw_events.csv"
    pd.DataFrame(raw_rows).to_csv(raw_path, index=False)

    win_rows = []
    for i in range(3):
        win_rows.append({
            "condition_id": "A",
            "seed": 1,
            "window_id": f"sample_window_{i:03d}",
            "state_ts": f"2026-01-02T08:{i * 10:02d}:00+09:00",
            "time_band": "peak",
            "evaluation_horizon_minutes": 30,
        })
    window_path = sample_root / "window_rollup.csv"
    pd.DataFrame(win_rows).to_csv(window_path, index=False)

    route_rows = []
    for idx, stop_id in enumerate(stops):
        route_rows.append({
            "route_id": "R_SAMPLE_1",
            "direction_id": "0",
            "stop_id": stop_id,
            "stop_order": idx + 1,
            "node_id": stop_id,
            "node_index": idx,
        })
    route_path = sample_root / "route_stop_sequence.csv"
    pd.DataFrame(route_rows).to_csv(route_path, index=False)

    run_manifest = {
        "run_id": "step169_sample_run",
        "condition_id": "A",
        "seed": 1,
        "simulation_evidence_only": True,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "actual_operational_claim_allowed": False,
        "train_allowed": False,
        "winner_selected": False,
    }
    run_manifest_path = sample_root / "run_manifest.json"
    dump_json(run_manifest_path, run_manifest)

    checkpoint_path = sample_root / "gatv2_checkpoint_stub.pt"
    checkpoint_path.write_text("step169 sample checkpoint placeholder; not trained", encoding="utf-8")

    return {
        "raw_events": str(raw_path),
        "window_rollup": str(window_path),
        "route_stop_sequence": str(route_path),
        "run_manifest": str(run_manifest_path),
        "gatv2_checkpoint": str(checkpoint_path),
    }


def resolve_path(value: Optional[str], project_root: Path) -> Optional[Path]:
    if not value:
        return None
    p = Path(value)
    if not p.is_absolute():
        p = project_root / p
    return p


def check_pipeline_scripts(project_root: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, rel in PIPELINE_SCRIPTS.items():
        p = project_root / rel
        result[key] = {
            "path": str(p),
            "exists": bool(p.exists()),
            "sha256": sha256_file(p),
        }
    return result


def run_check(
    *,
    project_root: Path,
    output_root: Path,
    mode: str,
    raw_events: Optional[Path],
    window_rollup: Optional[Path],
    route_stop_sequence: Optional[Path],
    run_manifest: Optional[Path],
    gatv2_checkpoint: Optional[Path],
    require_pipeline_scripts: bool,
) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    sample_inputs: Dict[str, str] = {}
    template_only = mode == "template"
    sample_inputs_generated = False

    if mode == "sample":
        sample_inputs = make_sample_inputs(output_root)
        sample_inputs_generated = True
        raw_events = Path(sample_inputs["raw_events"])
        window_rollup = Path(sample_inputs["window_rollup"])
        route_stop_sequence = Path(sample_inputs["route_stop_sequence"])
        run_manifest = Path(sample_inputs["run_manifest"])
        gatv2_checkpoint = Path(sample_inputs["gatv2_checkpoint"])

    checks = {
        "raw_events": check_table(raw_events, REQUIRED_RAW_EVENT_COLUMNS, "raw_events"),
        "window_rollup": check_table(window_rollup, REQUIRED_WINDOW_ROLLUP_COLUMNS, "window_rollup"),
        "route_stop_sequence": check_table(route_stop_sequence, REQUIRED_ROUTE_SEQUENCE_COLUMNS, "route_stop_sequence"),
        "run_manifest": check_json_manifest(run_manifest, REQUIRED_RUN_MANIFEST_KEYS),
    }

    checkpoint_check = {
        "label": "gatv2_checkpoint",
        "path": str(gatv2_checkpoint) if gatv2_checkpoint else "",
        "exists": bool(gatv2_checkpoint and gatv2_checkpoint.exists()),
        "sha256": sha256_file(gatv2_checkpoint) if gatv2_checkpoint else None,
        "status": "PASS" if gatv2_checkpoint and gatv2_checkpoint.exists() else "MISSING",
        "trained_model_claim_allowed": False,
    }

    pipeline_scripts = check_pipeline_scripts(project_root)
    hard_failures: List[str] = []
    warnings: List[str] = []

    if not template_only:
        for name, check in checks.items():
            if check.get("status") != "PASS":
                hard_failures.append(f"{name}:{check.get('status')}")
        if checkpoint_check.get("status") != "PASS":
            warnings.append("gatv2_checkpoint_missing_or_not_supplied_step163_may_need_checkpoint_before_actual_run")
        if require_pipeline_scripts:
            for name, entry in pipeline_scripts.items():
                if not entry.get("exists"):
                    hard_failures.append(f"pipeline_script_missing:{name}")

    if template_only:
        hard_failures.append("template_only_no_actual_inputs_supplied")

    ready_for_actual_like_execution = (not hard_failures) and (not template_only)
    audit_status = "PASS" if not hard_failures else "BLOCKED"
    if template_only:
        bundle_status = BUNDLE_TEMPLATE
    elif ready_for_actual_like_execution:
        bundle_status = BUNDLE_READY
    else:
        bundle_status = BUNDLE_BLOCKED

    manifest_path = output_root / "actual_evidence_input_readiness_checklist_manifest.json"
    report_path = output_root / "actual_evidence_input_readiness_checklist_report.md"
    data_quality_path = output_root / "actual_evidence_input_data_quality_report.json"
    command_plan_path = output_root / "actual_evidence_command_plan_step169.md"

    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "audit_status": audit_status,
        "bundle_status": bundle_status,
        "mode": mode,
        "template_only": bool(template_only),
        "sample_inputs_generated": bool(sample_inputs_generated),
        "ready_for_actual_like_execution": bool(ready_for_actual_like_execution),
        "ready_for_actual_evidence_pipeline": bool(ready_for_actual_like_execution),
        "actual_evidence_execution_allowed": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "actual_operational_claim_allowed": False,
        "train_allowed": False,
        "winner_selected": False,
        "project_root": str(project_root),
        "output_root": str(output_root),
        "input_checks": checks,
        "checkpoint_check": checkpoint_check,
        "pipeline_scripts": pipeline_scripts,
        "hard_failures": hard_failures,
        "warnings": warnings,
        "sample_inputs": sample_inputs,
        "output_files": {
            "manifest": str(manifest_path),
            "report": str(report_path),
            "data_quality_report": str(data_quality_path),
            "command_plan": str(command_plan_path),
        },
    }

    dump_json(manifest_path, manifest)
    dump_json(data_quality_path, {
        "input_checks": checks,
        "checkpoint_check": checkpoint_check,
        "pipeline_scripts": pipeline_scripts,
        "hard_failures": hard_failures,
        "warnings": warnings,
    })
    write_report(report_path, manifest)
    write_command_plan(command_plan_path, manifest)
    return manifest


def write_report(path: Path, manifest: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Step 169 Actual Evidence Input Readiness Checklist Report")
    lines.append("")
    lines.append(f"- audit_status: `{manifest['audit_status']}`")
    lines.append(f"- bundle_status: `{manifest['bundle_status']}`")
    lines.append(f"- mode: `{manifest['mode']}`")
    lines.append(f"- ready_for_actual_like_execution: `{manifest['ready_for_actual_like_execution']}`")
    lines.append(f"- paper_level_claim_allowed: `{manifest['paper_level_claim_allowed']}`")
    lines.append(f"- causal_performance_claim_allowed: `{manifest['causal_performance_claim_allowed']}`")
    lines.append(f"- actual_operational_claim_allowed: `{manifest['actual_operational_claim_allowed']}`")
    lines.append("")
    lines.append("## Input checks")
    lines.append("")
    lines.append("| input | status | rows | missing | path |")
    lines.append("|---|---:|---:|---|---|")
    for name, check in manifest["input_checks"].items():
        lines.append(
            f"| {name} | {check.get('status')} | {check.get('row_count', '')} | "
            f"{', '.join(check.get('missing_columns', check.get('missing_keys', [])))} | {check.get('path', '')} |"
        )
    lines.append("")
    lines.append("## Hard failures")
    lines.append("")
    if manifest["hard_failures"]:
        for item in manifest["hard_failures"]:
            lines.append(f"- {item}")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Non-claim interpretation")
    lines.append("")
    lines.append("This readiness report only verifies input availability and structural compatibility. It does not authorize paper-level, causal-performance, operational, or training claims.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_command_plan(path: Path, manifest: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Step 169 Actual Evidence Command Plan")
    lines.append("")
    lines.append("Run this plan only after replacing sample paths with actual run artifacts.")
    lines.append("")
    lines.append("```powershell")
    lines.append("# 1. Step 162 route-aware rollout adapter")
    lines.append("python 05_training/patent_evidence/route_aware_rollout_adapter_step162.py `")
    lines.append("  --mode from-route-aware-rollout `")
    lines.append("  --raw-events <RAW_EVENTS_PARQUET_OR_CSV> `")
    lines.append("  --window-rollup <WINDOW_ROLLUP_PARQUET_OR_CSV> `")
    lines.append("  --output-root artifacts/patent_evidence/step169_actual/step162")
    lines.append("")
    lines.append("# 2. Step 163 real GATv2 attention extractor")
    lines.append("python 05_training/patent_evidence/gatv2_real_attention_extractor_step163.py `")
    lines.append("  --mode from-checkpoint `")
    lines.append("  --checkpoint <GATV2_CHECKPOINT> `")
    lines.append("  --output-root artifacts/patent_evidence/step169_actual/step163")
    lines.append("")
    lines.append("# 3. Step 165 attempt-specific route/path attention filter")
    lines.append("python 05_training/patent_evidence/attempt_route_path_attention_filter_step165.py `")
    lines.append("  --pickup-attempt-events artifacts/patent_evidence/step169_actual/step162/pickup_attempt_events.csv `")
    lines.append("  --eta-counterfactual artifacts/patent_evidence/step169_actual/step162/eta_counterfactual.csv `")
    lines.append("  --real-attention artifacts/patent_evidence/step169_actual/step163/gatv2_attention.csv `")
    lines.append("  --route-stop-sequence <ROUTE_STOP_SEQUENCE_CSV_OR_PARQUET> `")
    lines.append("  --output-root artifacts/patent_evidence/step169_actual/step165")
    lines.append("")
    lines.append("# 4. Step 166 patent evidence report v2")
    lines.append("python 05_training/patent_evidence/patent_evidence_report_v2_step166.py `")
    lines.append("  --step160-manifest artifacts/patent_evidence/step169_actual/step160/zero_loss_evidence_manifest.json `")
    lines.append("  --step165-manifest artifacts/patent_evidence/step169_actual/step165/attempt_route_path_attention_filter_manifest.json `")
    lines.append("  --output-root artifacts/patent_evidence/step169_actual/step166")
    lines.append("```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 169 actual evidence input readiness checklist")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mode", choices=["template", "sample", "check"], default="sample")
    parser.add_argument("--raw-events", default="")
    parser.add_argument("--window-rollup", default="")
    parser.add_argument("--route-stop-sequence", default="")
    parser.add_argument("--run-manifest", default="")
    parser.add_argument("--gatv2-checkpoint", default="")
    parser.add_argument("--require-pipeline-scripts", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = project_root / output_root

    manifest = run_check(
        project_root=project_root,
        output_root=output_root,
        mode=args.mode,
        raw_events=resolve_path(args.raw_events, project_root),
        window_rollup=resolve_path(args.window_rollup, project_root),
        route_stop_sequence=resolve_path(args.route_stop_sequence, project_root),
        run_manifest=resolve_path(args.run_manifest, project_root),
        gatv2_checkpoint=resolve_path(args.gatv2_checkpoint, project_root),
        require_pipeline_scripts=bool(args.require_pipeline_scripts),
    )

    print("[OK] Step 169 actual evidence input readiness checklist generated")
    print(f"[OK] audit_status              : {manifest['audit_status']}")
    print(f"[OK] bundle_status             : {manifest['bundle_status']}")
    print(f"[OK] mode                      : {manifest['mode']}")
    print(f"[OK] template_only             : {manifest['template_only']}")
    print(f"[OK] ready_for_actual_like_execution : {manifest['ready_for_actual_like_execution']}")
    print(f"[OK] actual_evidence_execution_allowed : {manifest['actual_evidence_execution_allowed']}")
    print(f"[OK] hard_failures             : {len(manifest['hard_failures'])}")
    print(f"[OK] output_root               : {manifest['output_root']}")
    print(f"[OK] manifest                  : {manifest['output_files']['manifest']}")
    print(f"[OK] paper_level_claim_allowed : {manifest['paper_level_claim_allowed']}")
    print(f"[OK] causal_performance_claim_allowed : {manifest['causal_performance_claim_allowed']}")


if __name__ == "__main__":
    main()
