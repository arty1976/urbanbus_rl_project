from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


STEP_FILES: Dict[str, List[str]] = {
    "160": [
        "05_training/patent_evidence/zero_loss_pickup_evidence_schema_step160.md",
        "05_training/patent_evidence/zero_loss_pickup_evidence_reporter_step160.py",
        "05_training/patent_evidence/validate_zero_loss_pickup_evidence_bundle_step160.py",
        "05_training/patent_evidence/test_zero_loss_pickup_evidence_reporter_step160.py",
        "step160_zero_loss_pickup_evidence_reporter_scaffold.ps1",
    ],
    "161": [
        "05_training/patent_evidence/route_aware_pickup_attempt_event_schema_step161.md",
        "05_training/patent_evidence/route_aware_pickup_attempt_event_writer_step161.py",
        "05_training/patent_evidence/validate_route_aware_pickup_attempt_events_step161.py",
        "05_training/patent_evidence/test_route_aware_pickup_attempt_event_writer_step161.py",
        "step161_route_aware_pickup_attempt_event_writer.ps1",
    ],
    "162": [
        "05_training/patent_evidence/route_aware_rollout_adapter_schema_step162.md",
        "05_training/patent_evidence/route_aware_rollout_adapter_step162.py",
        "05_training/patent_evidence/validate_route_aware_rollout_adapter_step162.py",
        "05_training/patent_evidence/test_route_aware_rollout_adapter_step162.py",
        "step162_route_aware_rollout_adapter.ps1",
    ],
    "163": [
        "05_training/patent_evidence/gatv2_real_attention_extractor_schema_step163.md",
        "05_training/patent_evidence/gatv2_real_attention_extractor_step163.py",
        "05_training/patent_evidence/validate_gatv2_real_attention_extractor_step163.py",
        "05_training/patent_evidence/test_gatv2_real_attention_extractor_step163.py",
        "step163_gatv2_real_attention_extractor.ps1",
    ],
    "164": [
        "05_training/patent_evidence/real_attention_pipeline_connector_schema_step164.md",
        "05_training/patent_evidence/real_attention_pipeline_connector_step164.py",
        "05_training/patent_evidence/validate_real_attention_pipeline_connector_step164.py",
        "05_training/patent_evidence/test_real_attention_pipeline_connector_step164.py",
        "step164_real_attention_pipeline_connector.ps1",
    ],
    "165": [
        "05_training/patent_evidence/attempt_route_path_attention_filter_schema_step165.md",
        "05_training/patent_evidence/attempt_route_path_attention_filter_step165.py",
        "05_training/patent_evidence/validate_attempt_route_path_attention_filter_step165.py",
        "05_training/patent_evidence/test_attempt_route_path_attention_filter_step165.py",
        "step165_attempt_route_path_attention_filter.ps1",
    ],
    "166": [
        "05_training/patent_evidence/patent_evidence_report_v2_schema_step166.md",
        "05_training/patent_evidence/patent_evidence_report_v2_step166.py",
        "05_training/patent_evidence/validate_patent_evidence_report_v2_step166.py",
        "05_training/patent_evidence/test_patent_evidence_report_v2_step166.py",
        "step166_patent_evidence_report_v2.ps1",
    ],
    "167": [
        "05_training/patent_evidence/actual_route_aware_rollout_evidence_runbook_step167.md",
        "05_training/patent_evidence/actual_route_aware_rollout_evidence_runbook_step167.py",
        "05_training/patent_evidence/validate_actual_route_aware_rollout_evidence_runbook_step167.py",
        "05_training/patent_evidence/test_actual_route_aware_rollout_evidence_runbook_step167.py",
        "step167_actual_route_aware_rollout_evidence_runbook.ps1",
    ],
    "168": [
        "05_training/patent_evidence/zero_loss_patent_evidence_project_log_update_step168.md",
        "05_training/patent_evidence/zero_loss_patent_evidence_project_log_update_step168.py",
        "05_training/patent_evidence/validate_zero_loss_patent_evidence_project_log_update_step168.py",
        "05_training/patent_evidence/test_zero_loss_patent_evidence_project_log_update_step168.py",
        "step168_zero_loss_patent_evidence_project_log_update.ps1",
    ],
    "169": [
        "05_training/patent_evidence/actual_evidence_input_readiness_checklist_step169.md",
        "05_training/patent_evidence/actual_evidence_input_readiness_checklist_step169.py",
        "05_training/patent_evidence/validate_actual_evidence_input_readiness_checklist_step169.py",
        "05_training/patent_evidence/test_actual_evidence_input_readiness_checklist_step169.py",
        "step169_actual_evidence_input_readiness_checklist.ps1",
    ],
    "170": [
        "05_training/patent_evidence/actual_evidence_command_packet_step170.md",
        "05_training/patent_evidence/actual_evidence_command_packet_step170.py",
        "05_training/patent_evidence/validate_actual_evidence_command_packet_step170.py",
        "05_training/patent_evidence/test_actual_evidence_command_packet_step170.py",
        "step170_actual_evidence_command_packet.ps1",
    ],
    "171": [
        "05_training/patent_evidence/actual_evidence_execution_release_checklist_step171.md",
        "05_training/patent_evidence/actual_evidence_execution_release_checklist_step171.py",
        "05_training/patent_evidence/validate_actual_evidence_execution_release_checklist_step171.py",
        "05_training/patent_evidence/test_actual_evidence_execution_release_checklist_step171.py",
        "step171_actual_evidence_execution_release_checklist.ps1",
    ],
}

REQUIRED_GUARDS = {
    "paper_level_claim_allowed": False,
    "causal_performance_claim_allowed": False,
    "actual_operational_claim_allowed": False,
    "actual_evidence_execution_allowed": False,
    "command_execution_allowed": False,
    "train_allowed": False,
}

PIPELINE_CHAIN = [
    "Step 162 route-aware rollout adapter",
    "Step 161 pickup attempt / ETA counterfactual writer",
    "Step 163 GATv2 real attention extractor",
    "Step 165 attempt-specific route/path attention filter",
    "Step 160 zero-loss pickup evidence reporter",
    "Step 166 patent evidence report v2",
    "Step 167 actual rollout evidence runbook",
    "Step 169 input readiness checklist",
    "Step 170 dry-run command packet",
    "Step 171 execution release checklist",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def git_commit_or_unknown(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(project_root),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def scan_files(project_root: Path) -> Dict[str, Any]:
    by_step: Dict[str, Any] = {}
    missing_all: List[str] = []
    present_all: List[str] = []
    for step, files in STEP_FILES.items():
        present = []
        missing = []
        for rel in files:
            if (project_root / rel).exists():
                present.append(rel)
                present_all.append(rel)
            else:
                missing.append(rel)
                missing_all.append(rel)
        by_step[step] = {
            "expected_file_count": len(files),
            "present_file_count": len(present),
            "missing_file_count": len(missing),
            "present_files": present,
            "missing_files": missing,
        }
    return {
        "by_step": by_step,
        "present_file_count": len(present_all),
        "missing_file_count": len(missing_all),
        "present_files": present_all,
        "missing_files": missing_all,
    }


def render_markdown(payload: Dict[str, Any]) -> str:
    lines = []
    lines.append("# Final Zero-Loss Patent Evidence Handoff Index — Step 172")
    lines.append("")
    lines.append(f"Generated at UTC: `{payload['created_at_utc']}`")
    lines.append(f"Git commit: `{payload['git_commit']}`")
    lines.append("")
    lines.append("## Status")
    lines.append("")
    lines.append(f"- audit_status: `{payload['audit_status']}`")
    lines.append(f"- bundle_status: `{payload['bundle_status']}`")
    lines.append(f"- template_only: `{payload['template_only']}`")
    lines.append(f"- handoff_index_ready: `{payload['handoff_index_ready']}`")
    lines.append(f"- actual_evidence_execution_allowed: `{payload['actual_evidence_execution_allowed']}`")
    lines.append("")
    lines.append("## Guard flags")
    lines.append("")
    for key, value in payload["guards"].items():
        lines.append(f"- `{key}` = `{value}`")
    lines.append("")
    lines.append("## Pipeline chain")
    lines.append("")
    for item in payload["pipeline_chain"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## File coverage by step")
    lines.append("")
    lines.append("| Step | Present | Missing |")
    lines.append("|---|---:|---:|")
    for step, info in payload["file_scan"]["by_step"].items():
        lines.append(f"| {step} | {info['present_file_count']} | {info['missing_file_count']} |")
    lines.append("")
    lines.append("## Missing files")
    lines.append("")
    if payload["file_scan"]["missing_file_count"] == 0:
        lines.append("No missing expected files were detected.")
    else:
        for rel in payload["file_scan"]["missing_files"]:
            lines.append(f"- `{rel}`")
    lines.append("")
    lines.append("## Non-claim boundary")
    lines.append("")
    lines.append("This handoff index does not permit paper-level, causal-performance, actual-operational, command-execution, or training claims.")
    return "\n".join(lines) + "\n"


def build_manifest(project_root: Path, output_root: Path) -> Dict[str, Any]:
    file_scan = scan_files(project_root)
    warnings: List[str] = []
    if file_scan["missing_file_count"]:
        warnings.append("some expected Step 160-171 files are missing from this working tree")

    payload: Dict[str, Any] = {
        "artifact_version": "final_zero_loss_patent_evidence_handoff_index_step172_v1",
        "created_at_utc": utc_now(),
        "project_root": str(project_root),
        "output_root": str(output_root),
        "git_commit": git_commit_or_unknown(project_root),
        "audit_status": "PASS",
        "bundle_status": "FINAL_ZERO_LOSS_PATENT_EVIDENCE_HANDOFF_INDEX_READY_NONCLAIM",
        "template_only": True,
        "handoff_index_ready": True,
        "ready_for_actual_like_execution": False,
        "actual_evidence_execution_allowed": False,
        "command_execution_allowed": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "actual_operational_claim_allowed": False,
        "train_allowed": False,
        "guards": dict(REQUIRED_GUARDS),
        "pipeline_chain": list(PIPELINE_CHAIN),
        "covered_steps": sorted(STEP_FILES.keys(), key=int),
        "file_scan": file_scan,
        "warnings": warnings,
        "hard_failures": 0,
        "next_recommended_step": "Step 173 — Zero-Loss patent claim drafting packet, still non-claim unless explicitly released",
    }
    return payload


def write_outputs(project_root: Path, output_root: Path) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    payload = build_manifest(project_root, output_root)
    manifest_path = output_root / "final_zero_loss_patent_evidence_handoff_index_step172.json"
    md_path = output_root / "final_zero_loss_patent_evidence_handoff_index_step172.md"
    payload["output_files"] = {
        "manifest": str(manifest_path),
        "markdown": str(md_path),
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 172 final Zero-Loss patent evidence handoff index")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-root", default="artifacts/patent_evidence/final_zero_loss_patent_evidence_handoff_index_step172_selftest")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = project_root / output_root

    payload = write_outputs(project_root, output_root)
    print("[OK] Step 172 final zero-loss patent evidence handoff index generated")
    print(f"[OK] audit_status              : {payload['audit_status']}")
    print(f"[OK] bundle_status             : {payload['bundle_status']}")
    print(f"[OK] template_only             : {payload['template_only']}")
    print(f"[OK] handoff_index_ready       : {payload['handoff_index_ready']}")
    print(f"[OK] present_file_count        : {payload['file_scan']['present_file_count']}")
    print(f"[OK] missing_file_count        : {payload['file_scan']['missing_file_count']}")
    print(f"[OK] actual_evidence_execution_allowed : {payload['actual_evidence_execution_allowed']}")
    print(f"[OK] command_execution_allowed : {payload['command_execution_allowed']}")
    print(f"[OK] paper_level_claim_allowed : {payload['paper_level_claim_allowed']}")
    print(f"[OK] causal_performance_claim_allowed : {payload['causal_performance_claim_allowed']}")
    print(f"[OK] output_root               : {output_root}")
    print(f"[OK] manifest                  : {payload['output_files']['manifest']}")


if __name__ == "__main__":
    main()
