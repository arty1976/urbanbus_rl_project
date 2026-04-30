from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PACKET_STATUS = "H200_OBSERVABILITY_HANDOFF_PACKET_INDEX_READY_MONITORING_ONLY_STILL_LOCKED"
DECISION = "OBSERVABILITY_PACKET_ONLY_NOT_RELEASED"
DASHBOARD_MODE = "MONITORING_ONLY"
CONTROL_POLICY = "NO_LIVE_MUTATION_CONFIG_BASED_NEXT_RUN_ONLY"
NEXT_GATE = "WAITING_FOR_REAL_H200_STEP149_EXPECT_H200_RESULT_AND_OPTIONAL_H200_OBSERVABILITY_PROBE"

GUARDS = {
    "actual_execution_allowed": False,
    "actual_execution_released": False,
    "train_allowed": False,
    "live_mutation_allowed": False,
    "paper_level_claim_allowed": False,
    "causal_performance_claim_allowed": False,
    "operator_approval_recorded": False,
    "operator_approval_granted": False,
}

COMPONENTS = [
    {
        "step": "152",
        "component": "monitoring_dashboard_contract",
        "manifest_relpath": "artifacts/observability/h200_monitoring_dashboard_contract_step152/h200_monitoring_dashboard_contract_step152_manifest.json",
        "primary_status_keys": ["contract_status", "status"],
        "purpose": "Defines monitoring-only observability contract and no-live-mutation policy.",
    },
    {
        "step": "153",
        "component": "tensorboard_jsonl_logger",
        "manifest_relpath": "artifacts/observability/h200_tensorboard_jsonl_logger_step153/h200_tensorboard_jsonl_logger_step153_manifest.json",
        "primary_status_keys": ["integration_status", "status"],
        "purpose": "Defines TensorBoard-compatible scalar contract plus JSONL/CSV fallback logging.",
    },
    {
        "step": "154",
        "component": "nvidia_smi_gpu_monitor",
        "manifest_relpath": "artifacts/observability/h200_nvidia_smi_gpu_monitor_step154/h200_nvidia_smi_gpu_monitor_step154_manifest.json",
        "primary_status_keys": ["monitor_status", "status"],
        "purpose": "Defines nvidia-smi GPU telemetry CSV/JSONL scaffold.",
    },
    {
        "step": "155",
        "component": "observability_dashboard_index_status_page",
        "manifest_relpath": "artifacts/observability/h200_observability_dashboard_index_step155/h200_observability_dashboard_index_step155_manifest.json",
        "primary_status_keys": ["index_status", "status"],
        "purpose": "Indexes Step 152-154 observability components into a status page.",
    },
    {
        "step": "156",
        "component": "observability_operator_runbook",
        "manifest_relpath": "artifacts/observability/h200_observability_operator_runbook_step156/h200_observability_operator_runbook_step156_manifest.json",
        "primary_status_keys": ["runbook_status", "status"],
        "purpose": "Provides H200 operator commands for monitoring-only observability checks.",
    },
]


def load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except UnicodeDecodeError:
            continue
        except json.JSONDecodeError as exc:
            return {"_json_error": str(exc)}
    return {"_json_error": "failed_to_decode_json"}


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def bool_from_payload(payload: Optional[Dict[str, Any]], key: str) -> Optional[bool]:
    if not isinstance(payload, dict):
        return None
    if key in payload:
        val = payload.get(key)
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            if val.lower() == "true":
                return True
            if val.lower() == "false":
                return False
    guards = payload.get("guards")
    if isinstance(guards, dict) and key in guards:
        val = guards.get(key)
        if isinstance(val, bool):
            return val
    return None


def get_first(payload: Optional[Dict[str, Any]], keys: List[str]) -> str:
    if not isinstance(payload, dict):
        return ""
    for key in keys:
        if key in payload:
            return str(payload[key])
    return ""


def build_component_rows(project_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in COMPONENTS:
        manifest_path = project_root / item["manifest_relpath"]
        payload = load_json_if_exists(manifest_path)
        present = payload is not None and "_json_error" not in (payload or {})
        status_value = get_first(payload, item["primary_status_keys"])
        status_key = ""
        if isinstance(payload, dict):
            for key in item["primary_status_keys"]:
                if key in payload:
                    status_key = key
                    break
        row = {
            "step": item["step"],
            "component": item["component"],
            "manifest_relpath": item["manifest_relpath"],
            "manifest_path": str(manifest_path),
            "present": bool(present),
            "json_error": (payload or {}).get("_json_error", "") if isinstance(payload, dict) else "",
            "status_key": status_key,
            "status_value": status_value,
            "dashboard_mode": get_first(payload, ["dashboard_mode"]),
            "control_policy": get_first(payload, ["control_policy"]),
            "train_allowed": bool_from_payload(payload, "train_allowed"),
            "actual_execution_allowed": bool_from_payload(payload, "actual_execution_allowed"),
            "live_mutation_allowed": bool_from_payload(payload, "live_mutation_allowed"),
            "purpose": item["purpose"],
        }
        rows.append(row)
    return rows


def write_component_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "step", "component", "manifest_relpath", "manifest_path", "present",
        "json_error", "status_key", "status_value", "dashboard_mode",
        "control_policy", "train_allowed", "actual_execution_allowed",
        "live_mutation_allowed", "purpose",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def write_command_index(path: Path) -> None:
    content = """#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/urbanbus_rl_project}"
cd "$PROJECT_ROOT"

echo "[INFO] H200 observability handoff packet index - monitoring only"
echo "[INFO] This script does not start training or reward ablation."

echo "[CHECK] Step 152 contract manifest"
test -f artifacts/observability/h200_monitoring_dashboard_contract_step152/h200_monitoring_dashboard_contract_step152_manifest.json

echo "[CHECK] Step 153 logger manifest"
test -f artifacts/observability/h200_tensorboard_jsonl_logger_step153/h200_tensorboard_jsonl_logger_step153_manifest.json

echo "[CHECK] Step 154 GPU monitor manifest"
test -f artifacts/observability/h200_nvidia_smi_gpu_monitor_step154/h200_nvidia_smi_gpu_monitor_step154_manifest.json

echo "[CHECK] Step 155 dashboard index manifest"
test -f artifacts/observability/h200_observability_dashboard_index_step155/h200_observability_dashboard_index_step155_manifest.json

echo "[CHECK] Step 156 operator runbook manifest"
test -f artifacts/observability/h200_observability_operator_runbook_step156/h200_observability_operator_runbook_step156_manifest.json

echo "[OPTIONAL] Run one H200 nvidia-smi probe"
python 05_training/observability/h200_nvidia_smi_gpu_monitor_step154.py \
  --project-root "$PROJECT_ROOT" \
  --output-root artifacts/observability/h200_nvidia_smi_gpu_monitor_step154_h200_probe \
  --probe-once || true

echo "[OPTIONAL] Start TensorBoard if installed"
echo "tensorboard --logdir artifacts/observability --host 0.0.0.0 --port 6006"

echo "[OK] observability handoff command index completed - monitoring only"
"""
    path.write_text(content, encoding="utf-8")


def write_notes(path: Path, component_rows: List[Dict[str, Any]]) -> None:
    present_count = sum(1 for r in component_rows if r.get("present"))
    lines = [
        "# H200 observability handoff local notes - Step 157",
        "",
        "## Status",
        "",
        f"- packet_status: `{PACKET_STATUS}`",
        f"- decision: `{DECISION}`",
        f"- dashboard_mode: `{DASHBOARD_MODE}`",
        f"- control_policy: `{CONTROL_POLICY}`",
        "- actual_execution_allowed: `false`",
        "- actual_execution_released: `false`",
        "- train_allowed: `false`",
        "- live_mutation_allowed: `false`",
        "",
        "## Component manifest presence",
        "",
        f"Found {present_count} of {len(component_rows)} expected observability component manifests.",
        "",
        "| Step | Component | Present | Status |",
        "|---|---|---:|---|",
    ]
    for row in component_rows:
        lines.append(f"| {row['step']} | {row['component']} | {row['present']} | {row.get('status_value','')} |")
    lines.extend(["", "## Next gate", "", f"`{NEXT_GATE}`", "", "The observability packet does not approve training, reward ablation, or paper-level claims."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-root", default="artifacts/observability/h200_observability_handoff_packet_index_step157")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    component_rows = build_component_rows(project_root)
    component_csv = output_root / "observability_handoff_component_index_step157.csv"
    write_component_csv(component_csv, component_rows)

    command_index = output_root / "h200_observability_handoff_command_index_step157.sh"
    write_command_index(command_index)

    notes_md = output_root / "h200_observability_handoff_local_notes_step157.md"
    write_notes(notes_md, component_rows)

    present_count = sum(1 for row in component_rows if row.get("present"))
    warnings = []
    if present_count < len(component_rows):
        missing = [f"Step {r['step']} {r['component']}" for r in component_rows if not r.get("present")]
        warnings.append({"code": "observability_component_manifest_missing", "items": missing})

    packet = {
        "artifact_version": "h200_observability_handoff_packet_index_step157_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "packet_status": PACKET_STATUS,
        "decision": DECISION,
        "dashboard_mode": DASHBOARD_MODE,
        "control_policy": CONTROL_POLICY,
        "next_gate": NEXT_GATE,
        **GUARDS,
        "component_count": len(component_rows),
        "component_manifest_present_count": present_count,
        "component_manifest_missing_count": len(component_rows) - present_count,
        "component_index_csv": str(component_csv),
        "command_index_sh": str(command_index),
        "local_notes_md": str(notes_md),
        "warnings": warnings,
    }

    packet_json = output_root / "observability_handoff_operator_packet_step157.json"
    dump_json(packet_json, packet)

    summary_md = output_root / "h200_observability_handoff_packet_index_step157.md"
    summary_md.write_text(
        "\n".join([
            "# Step 157 H200 observability handoff packet index",
            "",
            f"- packet_status: `{PACKET_STATUS}`",
            f"- decision: `{DECISION}`",
            f"- dashboard_mode: `{DASHBOARD_MODE}`",
            f"- control_policy: `{CONTROL_POLICY}`",
            "- actual_execution_allowed: `false`",
            "- actual_execution_released: `false`",
            "- train_allowed: `false`",
            "- live_mutation_allowed: `false`",
            f"- component_manifest_present_count: `{present_count}/{len(component_rows)}`",
            f"- next_gate: `{NEXT_GATE}`",
            "",
            "This packet is monitoring-only and does not release actual reward ablation.",
        ]) + "\n",
        encoding="utf-8",
    )

    manifest_path = output_root / "h200_observability_handoff_packet_index_step157_manifest.json"
    manifest = {
        "artifact_version": "h200_observability_handoff_packet_index_step157_v1",
        "created_at_utc": packet["created_at_utc"],
        "project_root": str(project_root),
        "output_root": str(output_root),
        "packet_status": PACKET_STATUS,
        "decision": DECISION,
        "dashboard_mode": DASHBOARD_MODE,
        "control_policy": CONTROL_POLICY,
        "next_gate": NEXT_GATE,
        **GUARDS,
        "component_count": len(component_rows),
        "component_manifest_present_count": present_count,
        "component_manifest_missing_count": len(component_rows) - present_count,
        "output_files": {
            "manifest": str(manifest_path),
            "summary_md": str(summary_md),
            "operator_packet_json": str(packet_json),
            "component_index_csv": str(component_csv),
            "command_index_sh": str(command_index),
            "local_notes_md": str(notes_md),
        },
        "warnings": warnings,
    }
    dump_json(manifest_path, manifest)

    latest = project_root / "05_training" / "observability" / "h200_observability_handoff_packet_index_step157.latest.json"
    dump_json(latest, {"latest_manifest": str(manifest_path), "packet_status": PACKET_STATUS, "decision": DECISION})

    print("[OK] Step 157 H200 observability handoff packet index generated")
    print(f"[OK] packet_status: {PACKET_STATUS}")
    print(f"[OK] decision     : {DECISION}")
    print(f"[OK] present      : {present_count}/{len(component_rows)} component manifests")
    print(f"[OK] train_allowed: {GUARDS['train_allowed']}")
    print(f"[OK] manifest     : {manifest_path}")


if __name__ == "__main__":
    main()
