from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

ARTIFACT_VERSION = "actual_route_aware_rollout_evidence_runbook_step167_v1"
BUNDLE_STATUS = "ACTUAL_ROUTE_AWARE_ROLLOUT_EVIDENCE_RUNBOOK_READY_NONCLAIM"

REQUIRED_STEP_FILES = [
    "05_training/patent_evidence/zero_loss_pickup_evidence_reporter_step160.py",
    "05_training/patent_evidence/route_aware_pickup_attempt_event_writer_step161.py",
    "05_training/patent_evidence/route_aware_rollout_adapter_step162.py",
    "05_training/patent_evidence/gatv2_real_attention_extractor_step163.py",
    "05_training/patent_evidence/attempt_route_path_attention_filter_step165.py",
    "05_training/patent_evidence/patent_evidence_report_v2_step166.py",
]

GUARD_FLAGS = {
    "actual_operational_claim_allowed": False,
    "paper_level_claim_allowed": False,
    "causal_performance_claim_allowed": False,
    "train_allowed": False,
    "actual_results_claimed": False,
}


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def rel_exists(project_root: Path, rel: str) -> bool:
    return (project_root / rel).exists()


def build_command_plan(args: argparse.Namespace) -> List[Dict[str, Any]]:
    raw_events = args.raw_events or "<RAW_EVENTS_PATH>"
    window_rollup = args.window_rollup or "<WINDOW_ROLLUP_PATH>"
    route_stop_sequence = args.route_stop_sequence or "<ROUTE_STOP_SEQUENCE_PATH>"
    base = args.actual_output_root.replace("\\", "/")

    return [
        {
            "stage": "A_step162_adapt_rollout",
            "status": "template_only" if not args.raw_events else "ready_when_inputs_exist",
            "command": (
                "python 05_training/patent_evidence/route_aware_rollout_adapter_step162.py "
                f"--mode from-route-aware-rollout --raw-events {raw_events} --window-rollup {window_rollup} "
                f"--output-root {base}/route_aware_rollout_adapter --file-format csv"
            ),
        },
        {
            "stage": "B_step161_pickup_attempt_writer",
            "status": "template_only",
            "command": (
                "python 05_training/patent_evidence/route_aware_pickup_attempt_event_writer_step161.py "
                f"--mode from-normalized-rollout --normalized-events {base}/route_aware_rollout_adapter/normalized_route_aware_rollout_events.csv "
                f"--output-root {base}/pickup_attempt_events --file-format csv"
            ),
        },
        {
            "stage": "C_step163_real_attention_extractor",
            "status": "template_only",
            "command": (
                "python 05_training/patent_evidence/gatv2_real_attention_extractor_step163.py "
                f"--mode from-gatv2-snapshot --output-root {base}/gatv2_real_attention --require-real-gatv2conv"
            ),
        },
        {
            "stage": "D_step165_attempt_path_filter",
            "status": "template_only" if not args.route_stop_sequence else "ready_when_inputs_exist",
            "command": (
                "python 05_training/patent_evidence/attempt_route_path_attention_filter_step165.py "
                f"--pickup-attempt-events {base}/pickup_attempt_events/pickup_attempt_events.csv "
                f"--eta-counterfactual {base}/pickup_attempt_events/eta_counterfactual.csv "
                f"--real-attention {base}/gatv2_real_attention/gatv2_attention.csv "
                f"--route-stop-sequence {route_stop_sequence} "
                f"--output-root {base}/attempt_path_attention --file-format csv"
            ),
        },
        {
            "stage": "E_step160_zero_loss_reporter",
            "status": "template_only",
            "command": (
                "python 05_training/patent_evidence/zero_loss_pickup_evidence_reporter_step160.py "
                f"--pickup-attempt-events {base}/attempt_path_attention/pickup_attempt_events.csv "
                f"--eta-counterfactual {base}/attempt_path_attention/eta_counterfactual.csv "
                f"--gatv2-attention {base}/attempt_path_attention/gatv2_attention.csv "
                f"--run-manifest {base}/attempt_path_attention/run_manifest.json "
                f"--output-root {base}/zero_loss_evidence"
            ),
        },
        {
            "stage": "F_step166_patent_report_v2",
            "status": "template_only",
            "command": (
                "python 05_training/patent_evidence/patent_evidence_report_v2_step166.py "
                f"--zero-loss-evidence-root {base}/zero_loss_evidence "
                f"--attempt-path-attention-root {base}/attempt_path_attention "
                f"--output-root {base}/patent_evidence_report_v2"
            ),
        },
    ]


def write_command_md(path: Path, commands: List[Dict[str, Any]]) -> None:
    lines = [
        "# Step 167 Actual Route-Aware Rollout Evidence Command Plan",
        "",
        "Status: template / non-claim until actual inputs are supplied and validators pass.",
        "",
    ]
    for item in commands:
        lines.append(f"## {item['stage']}")
        lines.append("")
        lines.append(f"status: `{item['status']}`")
        lines.append("")
        lines.append("```powershell")
        lines.append(item["command"])
        lines.append("```")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_runbook(args: argparse.Namespace) -> Dict[str, Any]:
    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    missing_step_files = [rel for rel in REQUIRED_STEP_FILES if not rel_exists(project_root, rel)]
    raw_events_exists = bool(args.raw_events and Path(args.raw_events).exists())
    window_rollup_exists = bool(args.window_rollup and Path(args.window_rollup).exists())
    route_stop_sequence_exists = bool(args.route_stop_sequence and Path(args.route_stop_sequence).exists())

    commands = build_command_plan(args)
    command_plan_json = output_root / "actual_route_aware_rollout_evidence_command_plan_step167.json"
    command_plan_md = output_root / "actual_route_aware_rollout_evidence_command_plan_step167.md"
    dump_json(command_plan_json, {"commands": commands})
    write_command_md(command_plan_md, commands)

    hard_failures: List[str] = []
    warnings: List[str] = []
    if missing_step_files and args.require_step_files:
        hard_failures.append("missing_required_step_files")
    elif missing_step_files:
        warnings.append("missing_step_files_template_only")

    if args.raw_events and not raw_events_exists:
        hard_failures.append("raw_events_path_not_found")
    if args.window_rollup and not window_rollup_exists:
        warnings.append("window_rollup_path_not_found_or_not_supplied")
    if args.route_stop_sequence and not route_stop_sequence_exists:
        hard_failures.append("route_stop_sequence_path_not_found")

    audit_status = "PASS" if not hard_failures else "BLOCKED"
    manifest_path = output_root / "actual_route_aware_rollout_evidence_runbook_manifest_step167.json"

    manifest: Dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "audit_status": audit_status,
        "bundle_status": BUNDLE_STATUS,
        "project_root": str(project_root),
        "output_root": str(output_root),
        "actual_output_root": args.actual_output_root,
        "required_step_files": REQUIRED_STEP_FILES,
        "missing_step_files": missing_step_files,
        "input_paths": {
            "raw_events": args.raw_events,
            "window_rollup": args.window_rollup,
            "route_stop_sequence": args.route_stop_sequence,
        },
        "input_path_status": {
            "raw_events_exists": raw_events_exists,
            "window_rollup_exists": window_rollup_exists,
            "route_stop_sequence_exists": route_stop_sequence_exists,
        },
        "command_plan_json": str(command_plan_json),
        "command_plan_md": str(command_plan_md),
        "execution_chain": [c["stage"] for c in commands],
        "ready_for_actual_like_execution": bool(
            audit_status == "PASS" and raw_events_exists and route_stop_sequence_exists and not missing_step_files
        ),
        "template_only": bool(not args.raw_events or not args.route_stop_sequence),
        "hard_failures": hard_failures,
        "warnings": warnings,
        **GUARD_FLAGS,
        "manifest_path": str(manifest_path),
    }
    dump_json(manifest_path, manifest)

    print("[OK] Step 167 actual route-aware rollout evidence runbook generated")
    print(f"[OK] audit_status              : {manifest['audit_status']}")
    print(f"[OK] bundle_status             : {manifest['bundle_status']}")
    print(f"[OK] template_only             : {manifest['template_only']}")
    print(f"[OK] ready_for_actual_like_execution : {manifest['ready_for_actual_like_execution']}")
    print(f"[OK] missing_step_files        : {len(missing_step_files)}")
    print(f"[OK] command_plan_md           : {command_plan_md}")
    print(f"[OK] manifest                  : {manifest_path}")
    print(f"[OK] paper_level_claim_allowed : {manifest['paper_level_claim_allowed']}")
    print(f"[OK] causal_performance_claim_allowed : {manifest['causal_performance_claim_allowed']}")
    return manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 167 actual route-aware rollout evidence runbook generator")
    p.add_argument("--project-root", default=".")
    p.add_argument("--output-root", default="artifacts/patent_evidence/actual_route_aware_rollout_evidence_runbook_step167_selftest")
    p.add_argument("--actual-output-root", default="artifacts/patent_evidence/actual_route_aware_rollout_evidence_step167")
    p.add_argument("--raw-events", default="")
    p.add_argument("--window-rollup", default="")
    p.add_argument("--route-stop-sequence", default="")
    p.add_argument("--require-step-files", action="store_true")
    return p.parse_args()


def main() -> None:
    generate_runbook(parse_args())


if __name__ == "__main__":
    main()
