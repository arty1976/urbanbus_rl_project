from __future__ import annotations

import argparse
import csv
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

RUNBOOK_STATUS = "H200_OBSERVABILITY_OPERATOR_RUNBOOK_READY_MONITORING_ONLY_STILL_LOCKED"
DASHBOARD_MODE = "MONITORING_ONLY"
CONTROL_POLICY = "NO_LIVE_MUTATION_CONFIG_BASED_NEXT_RUN_ONLY"
ARTIFACT_VERSION = "h200_observability_operator_runbook_step156_v1"

LOCKED_FLAGS = {
    "actual_execution_allowed": False,
    "actual_execution_released": False,
    "train_allowed": False,
    "live_mutation_allowed": False,
    "paper_level_claim_allowed": False,
    "causal_performance_claim_allowed": False,
    "operator_approval_recorded": False,
    "operator_approval_granted": False,
}

EXPECTED_COMPONENTS = [
    {
        "step": "152",
        "name": "monitoring_dashboard_contract",
        "manifest": "artifacts/observability/h200_monitoring_dashboard_contract_step152/h200_monitoring_dashboard_contract_step152_manifest.json",
        "required_status_key": "contract_status",
    },
    {
        "step": "153",
        "name": "tensorboard_jsonl_logger",
        "manifest": "artifacts/observability/h200_tensorboard_jsonl_logger_step153/h200_tensorboard_jsonl_logger_step153_manifest.json",
        "required_status_key": "integration_status",
    },
    {
        "step": "154",
        "name": "nvidia_smi_gpu_monitor",
        "manifest": "artifacts/observability/h200_nvidia_smi_gpu_monitor_step154/h200_nvidia_smi_gpu_monitor_step154_manifest.json",
        "required_status_key": "monitor_status",
    },
    {
        "step": "155",
        "name": "observability_dashboard_index",
        "manifest": "artifacts/observability/h200_observability_dashboard_index_step155/h200_observability_dashboard_index_step155_manifest.json",
        "required_status_key": "index_status",
    },
]

H200_OBSERVATION_COMMANDS = [
    {
        "name": "view_step155_status_page",
        "command": "cat artifacts/observability/h200_observability_dashboard_index_step155/h200_observability_status_page_step155.md",
        "purpose": "Read the file-based observability status page.",
    },
    {
        "name": "tail_jsonl_metrics",
        "command": "tail -f artifacts/observability/h200_tensorboard_jsonl_logger_step153/metrics_events_step153_sample.jsonl",
        "purpose": "Inspect JSONL scalar events. Replace sample path with actual run path when training is later released.",
    },
    {
        "name": "run_gpu_probe_once",
        "command": "python 05_training/observability/h200_nvidia_smi_gpu_monitor_step154.py --project-root /workspace/urbanbus_rl_project --output-root artifacts/observability/h200_nvidia_smi_gpu_monitor_step154_h200_probe --probe-once",
        "purpose": "Run one monitoring-only nvidia-smi probe on H200.",
    },
    {
        "name": "launch_tensorboard_read_only",
        "command": "tensorboard --logdir artifacts/observability --host 0.0.0.0 --port 6006",
        "purpose": "Launch TensorBoard for observation only if TensorBoard is installed.",
    },
]

LOCAL_OBSERVATION_COMMANDS = [
    {
        "name": "open_ssh_tunnel_for_tensorboard",
        "command": "ssh -L 6006:localhost:6006 <user>@<h200-host>",
        "purpose": "Open a local tunnel to the H200 TensorBoard port.",
    },
    {
        "name": "open_browser",
        "command": "http://localhost:6006",
        "purpose": "Open the TensorBoard UI from the local browser.",
    },
]

FORBIDDEN_COMMAND_SUBSTRINGS = [
    "train_mappo_actual.py",
    "--actual-execution-allowed true",
    "actual_execution_allowed=true",
    "train_allowed=true",
    "reward_weight_live_update",
    "live_mutation_allowed=true",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def collect_component_status(project_root: Path) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for comp in EXPECTED_COMPONENTS:
        manifest_path = project_root / comp["manifest"]
        payload = read_json(manifest_path)
        if payload is None:
            rows.append({
                "step": comp["step"],
                "name": comp["name"],
                "present": False,
                "manifest": str(manifest_path),
                "status_key": comp["required_status_key"],
                "status_value": "missing",
                "dashboard_mode": "unknown",
                "train_allowed": "unknown",
                "live_mutation_allowed": "unknown",
            })
            warnings.append(f"component_manifest_missing_step_{comp['step']}:{manifest_path}")
            continue

        status_key = comp["required_status_key"]
        status_value = str(payload.get(status_key, payload.get("status", "unknown")))
        rows.append({
            "step": comp["step"],
            "name": comp["name"],
            "present": True,
            "manifest": str(manifest_path),
            "status_key": status_key,
            "status_value": status_value,
            "dashboard_mode": str(payload.get("dashboard_mode", "unknown")),
            "train_allowed": str(payload.get("train_allowed", "unknown")),
            "live_mutation_allowed": str(payload.get("live_mutation_allowed", "unknown")),
        })

        if payload.get("dashboard_mode") not in (None, DASHBOARD_MODE):
            warnings.append(f"component_dashboard_mode_unexpected_step_{comp['step']}:{payload.get('dashboard_mode')}")
        if payload.get("train_allowed") is True:
            warnings.append(f"component_train_allowed_true_step_{comp['step']}")
        if payload.get("live_mutation_allowed") is True:
            warnings.append(f"component_live_mutation_allowed_true_step_{comp['step']}")

    present_count = sum(1 for row in rows if row["present"] is True)
    return {
        "rows": rows,
        "present_count": present_count,
        "expected_count": len(EXPECTED_COMPONENTS),
        "all_present": present_count == len(EXPECTED_COMPONENTS),
        "warnings": warnings,
    }


def validate_command_policy(commands: List[Dict[str, str]]) -> None:
    for item in commands:
        cmd = str(item.get("command", ""))
        lowered = cmd.lower()
        for forbidden in FORBIDDEN_COMMAND_SUBSTRINGS:
            if forbidden.lower() in lowered:
                raise RuntimeError(f"forbidden command substring found in {item.get('name')}: {forbidden}")


def render_status_page(manifest: Dict[str, Any]) -> str:
    components = manifest["component_status"]["rows"]
    lines = [
        "# H200 Observability Operator Runbook - Step 156",
        "",
        "## Status",
        "",
        f"- runbook_status: `{manifest['runbook_status']}`",
        f"- dashboard_mode: `{manifest['dashboard_mode']}`",
        f"- control_policy: `{manifest['control_policy']}`",
        f"- train_allowed: `{manifest['train_allowed']}`",
        f"- live_mutation_allowed: `{manifest['live_mutation_allowed']}`",
        f"- actual_execution_allowed: `{manifest['actual_execution_allowed']}`",
        "",
        "## Component status",
        "",
        "| Step | Component | Present | Status | Train Allowed | Live Mutation |",
        "|---|---|---:|---|---:|---:|",
    ]
    for row in components:
        lines.append(
            f"| {row['step']} | {row['name']} | {row['present']} | {row['status_value']} | {row['train_allowed']} | {row['live_mutation_allowed']} |"
        )

    lines.extend([
        "",
        "## H200 observation commands",
        "",
    ])
    for item in manifest["h200_observation_commands"]:
        lines.extend([
            f"### {item['name']}",
            "",
            f"Purpose: {item['purpose']}",
            "",
            "```bash",
            item["command"],
            "```",
            "",
        ])

    lines.extend([
        "## Local TensorBoard tunnel",
        "",
    ])
    for item in manifest["local_observation_commands"]:
        lines.extend([
            f"### {item['name']}",
            "",
            f"Purpose: {item['purpose']}",
            "",
            "```text",
            item["command"],
            "```",
            "",
        ])

    lines.extend([
        "## Prohibited actions",
        "",
        "- Do not change reward weights during a live run.",
        "- Do not change learning rate, batch size, seed allocation, or condition assignment during a live run.",
        "- Do not treat this runbook as actual reward ablation release approval.",
        "- Do not set train_allowed=true from this observability step.",
        "",
        "## Next allowed action",
        "",
        "Use this runbook only after H200-side observability files have been copied or generated. Actual training still requires a separate release gate.",
    ])
    return "\n".join(lines)


def render_h200_cheatsheet(manifest: Dict[str, Any]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Step 156 H200 observation-only command cheatsheet.",
        "# This file does not launch training and does not unlock any release flag.",
        "",
        "PROJECT_ROOT=${PROJECT_ROOT:-/workspace/urbanbus_rl_project}",
        "cd \"${PROJECT_ROOT}\"",
        "",
        "echo '[INFO] Step 156 observation-only commands'",
        "echo '[INFO] train_allowed=false'",
        "echo '[INFO] live_mutation_allowed=false'",
        "",
    ]
    for item in manifest["h200_observation_commands"]:
        lines.extend([
            f"echo ''",
            f"echo '[COMMAND] {item['name']}'",
            f"cat <<'CMD'",
            item["command"],
            "CMD",
            "",
        ])
    return "\n".join(lines)


def render_local_tunnel_cheatsheet(manifest: Dict[str, Any]) -> str:
    lines = [
        "$ErrorActionPreference = \"Stop\"",
        "",
        "Write-Host \"[INFO] Step 156 local TensorBoard tunnel cheatsheet\"",
        "Write-Host \"[INFO] Replace <user> and <h200-host> before running.\"",
        "",
    ]
    for item in manifest["local_observation_commands"]:
        lines.extend([
            f"Write-Host \"\"",
            f"Write-Host \"[COMMAND] {item['name']}\"",
            f"Write-Host @'",
            item["command"],
            "'@",
        ])
    return "\n".join(lines)


def render_checklist(manifest: Dict[str, Any]) -> str:
    lines = [
        "# Step 156 Operator Quick Checklist",
        "",
        "Before using the H200 observability runbook:",
        "",
        "- [ ] Confirm Step 152-155 source files are committed and pushed.",
        "- [ ] Confirm this runbook status is monitoring-only and still locked.",
        "- [ ] Confirm train_allowed is false.",
        "- [ ] Confirm live_mutation_allowed is false.",
        "- [ ] Confirm actual_execution_allowed is false.",
        "- [ ] Run the Step 154 nvidia-smi probe on H200 if GPU telemetry is needed.",
        "- [ ] Launch TensorBoard only as read-only observation if TensorBoard is installed.",
        "- [ ] Record any desired config change for the next run only, not the live run.",
        "",
        "Component presence summary:",
        "",
    ]
    for row in manifest["component_status"]["rows"]:
        mark = "x" if row["present"] else " "
        lines.append(f"- [{mark}] Step {row['step']} {row['name']}: {row['status_value']}")
    return "\n".join(lines)


def write_component_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step", "name", "present", "manifest", "status_key", "status_value",
        "dashboard_mode", "train_allowed", "live_mutation_allowed",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_manifest(project_root: Path, output_root: Path) -> Dict[str, Any]:
    component_status = collect_component_status(project_root)
    commands = H200_OBSERVATION_COMMANDS
    local_commands = LOCAL_OBSERVATION_COMMANDS
    validate_command_policy(commands)
    validate_command_policy(local_commands)

    output_files = {
        "manifest": str(output_root / "h200_observability_operator_runbook_step156_manifest.json"),
        "runbook_md": str(output_root / "h200_observability_operator_runbook_step156.md"),
        "h200_command_cheatsheet_sh": str(output_root / "h200_observability_command_cheatsheet_step156.sh"),
        "local_tensorboard_tunnel_cheatsheet_ps1": str(output_root / "local_tensorboard_tunnel_cheatsheet_step156.ps1"),
        "operator_quick_checklist_md": str(output_root / "operator_quick_checklist_step156.md"),
        "component_status_csv": str(output_root / "observability_operator_component_status_step156.csv"),
    }

    manifest: Dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "step": "156",
        "created_at_utc": utc_now(),
        "project_root": str(project_root),
        "output_root": str(output_root),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "runbook_status": RUNBOOK_STATUS,
        "dashboard_mode": DASHBOARD_MODE,
        "control_policy": CONTROL_POLICY,
        **LOCKED_FLAGS,
        "component_status": component_status,
        "h200_observation_commands": commands,
        "local_observation_commands": local_commands,
        "forbidden_command_substrings": FORBIDDEN_COMMAND_SUBSTRINGS,
        "warnings": component_status["warnings"],
        "hard_failures": 0,
        "output_files": output_files,
        "next_gate": "WAITING_FOR_REAL_H200_STEP149_EXPECT_H200_RESULT_OR_OBSERVABILITY_H200_PROBE",
    }
    return manifest


def write_outputs(manifest: Dict[str, Any]) -> None:
    output_root = Path(manifest["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    write_json(Path(manifest["output_files"]["manifest"]), manifest)
    Path(manifest["output_files"]["runbook_md"]).write_text(render_status_page(manifest), encoding="utf-8")
    Path(manifest["output_files"]["h200_command_cheatsheet_sh"]).write_text(render_h200_cheatsheet(manifest), encoding="utf-8")
    Path(manifest["output_files"]["local_tensorboard_tunnel_cheatsheet_ps1"]).write_text(render_local_tunnel_cheatsheet(manifest), encoding="utf-8")
    Path(manifest["output_files"]["operator_quick_checklist_md"]).write_text(render_checklist(manifest), encoding="utf-8")
    write_component_csv(Path(manifest["output_files"]["component_status_csv"]), manifest["component_status"]["rows"])

    latest_path = Path(manifest["project_root"]) / "05_training" / "observability" / "h200_observability_operator_runbook_step156.latest.json"
    write_json(latest_path, {
        "artifact_version": ARTIFACT_VERSION,
        "step": "156",
        "created_at_utc": manifest["created_at_utc"],
        "manifest": manifest["output_files"]["manifest"],
        "runbook_status": manifest["runbook_status"],
        "dashboard_mode": manifest["dashboard_mode"],
        "control_policy": manifest["control_policy"],
        "train_allowed": manifest["train_allowed"],
        "live_mutation_allowed": manifest["live_mutation_allowed"],
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Step 156 H200 observability operator runbook.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-root", default="artifacts/observability/h200_observability_operator_runbook_step156")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_root = (project_root / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root).resolve()

    manifest = build_manifest(project_root, output_root)
    write_outputs(manifest)

    print("[OK] Step 156 H200 observability operator runbook generated")
    print(f"[OK] runbook_status : {manifest['runbook_status']}")
    print(f"[OK] dashboard_mode  : {manifest['dashboard_mode']}")
    print(f"[OK] control_policy  : {manifest['control_policy']}")
    print(f"[OK] train_allowed   : {manifest['train_allowed']}")
    print(f"[OK] components      : {manifest['component_status']['present_count']}/{manifest['component_status']['expected_count']}")
    print(f"[OK] warnings        : {len(manifest['warnings'])}")
    print(f"[OK] manifest        : {manifest['output_files']['manifest']}")


if __name__ == "__main__":
    main()
