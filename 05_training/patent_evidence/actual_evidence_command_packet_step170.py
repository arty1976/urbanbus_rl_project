from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

ARTIFACT_VERSION = "actual_evidence_command_packet_step170_v1"
BUNDLE_STATUS = "ACTUAL_EVIDENCE_COMMAND_PACKET_READY_DRY_RUN_NONCLAIM"
BUNDLE_BLOCKED = "ACTUAL_EVIDENCE_COMMAND_PACKET_BLOCKED_NONCLAIM"
LOCKED_FALSE_FLAGS = [
    "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
    "actual_operational_claim_allowed",
    "train_allowed",
    "winner_selected",
    "actual_evidence_execution_allowed",
    "command_execution_allowed",
]

PIPELINE_SCRIPT_REL = {
    "step162_adapter": "05_training/patent_evidence/route_aware_rollout_adapter_step162.py",
    "step161_writer": "05_training/patent_evidence/route_aware_pickup_attempt_event_writer_step161.py",
    "step163_attention_extractor": "05_training/patent_evidence/gatv2_real_attention_extractor_step163.py",
    "step165_path_filter": "05_training/patent_evidence/attempt_route_path_attention_filter_step165.py",
    "step160_reporter": "05_training/patent_evidence/zero_loss_pickup_evidence_reporter_step160.py",
    "step166_report_v2": "05_training/patent_evidence/patent_evidence_report_v2_step166.py",
}


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


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def q(s: str) -> str:
    # PowerShell-safe double quote wrapper for paths in generated scripts.
    escaped = str(s).replace('"', '`"')
    return f'"{escaped}"'


def normalize_path(value: Optional[str]) -> str:
    return str(value or "").replace("/", "\\")


def make_sample_readiness_manifest(output_root: Path, project_root: Path) -> Path:
    sample_root = output_root / "sample_step169_like_inputs"
    sample_root.mkdir(parents=True, exist_ok=True)

    raw = sample_root / "raw_events.csv"
    window = sample_root / "window_rollup.csv"
    route = sample_root / "route_stop_sequence.csv"
    run_manifest = sample_root / "run_manifest.json"
    checkpoint = sample_root / "gatv2_checkpoint_stub.pt"
    pyg_pt = sample_root / "gatv2_snapshot_stub.pt"

    raw.write_text(
        "attempt_id,state_ts,condition_id,seed,route_id,direction_id,vehicle_id,current_stop_id,pickup_stop_id,dropoff_stop_id,existing_passenger_id,new_passenger_id,eta_without_new_pickup_sec,eta_with_new_pickup_sec\n"
        "a0,2026-01-02T08:00:00+09:00,A,1,R1,0,BUS1,S001,S002,S003,e0,n0,600,600\n",
        encoding="utf-8",
    )
    window.write_text(
        "condition_id,seed,window_id,state_ts,time_band\n"
        "A,1,w0,2026-01-02T08:00:00+09:00,peak\n",
        encoding="utf-8",
    )
    route.write_text(
        "route_id,direction_id,stop_id,stop_order,node_id,node_index\n"
        "R1,0,S001,1,S001,0\nR1,0,S002,2,S002,1\nR1,0,S003,3,S003,2\n",
        encoding="utf-8",
    )
    run_manifest.write_text(
        json.dumps({
            "run_id": "step170_sample_input",
            "condition_id": "A",
            "seed": 1,
            "paper_level_claim_allowed": False,
            "causal_performance_claim_allowed": False,
            "actual_operational_claim_allowed": False,
            "train_allowed": False,
            "winner_selected": False,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    checkpoint.write_bytes(b"step170 checkpoint placeholder for command packet only\n")
    pyg_pt.write_bytes(b"step170 pyg pt placeholder for dry-run command packet only\n")

    scripts = {}
    for key, rel in PIPELINE_SCRIPT_REL.items():
        p = project_root / rel
        scripts[key] = {"path": str(p), "exists": p.exists(), "sha256": sha256_file(p)}

    manifest = {
        "artifact_version": "actual_evidence_input_readiness_checklist_step169_v1_sample_for_step170",
        "audit_status": "PASS",
        "bundle_status": "ACTUAL_EVIDENCE_INPUTS_READY_FOR_ZERO_LOSS_PIPELINE_NONCLAIM",
        "mode": "sample_for_step170",
        "template_only": False,
        "ready_for_actual_like_execution": True,
        "ready_for_actual_evidence_pipeline": True,
        "actual_evidence_execution_allowed": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "actual_operational_claim_allowed": False,
        "train_allowed": False,
        "winner_selected": False,
        "project_root": str(project_root),
        "output_root": str(output_root),
        "input_checks": {
            "raw_events": {"path": str(raw), "exists": True, "status": "PASS"},
            "window_rollup": {"path": str(window), "exists": True, "status": "PASS"},
            "route_stop_sequence": {"path": str(route), "exists": True, "status": "PASS"},
            "run_manifest": {"path": str(run_manifest), "exists": True, "status": "PASS"},
        },
        "checkpoint_check": {
            "label": "gatv2_checkpoint",
            "path": str(checkpoint),
            "exists": True,
            "status": "PASS",
            "trained_model_claim_allowed": False,
        },
        "gatv2_snapshot_pt_check": {
            "label": "gatv2_snapshot_pt",
            "path": str(pyg_pt),
            "exists": True,
            "status": "PASS",
            "trained_model_claim_allowed": False,
        },
        "pipeline_scripts": scripts,
        "hard_failures": [],
        "warnings": ["sample_readiness_manifest_generated_by_step170"],
        "sample_inputs": {
            "raw_events": str(raw),
            "window_rollup": str(window),
            "route_stop_sequence": str(route),
            "run_manifest": str(run_manifest),
            "gatv2_checkpoint": str(checkpoint),
            "gatv2_snapshot_pt": str(pyg_pt),
        },
    }
    path = output_root / "sample_readiness_manifest_for_step170.json"
    dump_json(path, manifest)
    return path


def get_input_path(readiness: Dict[str, Any], key: str) -> str:
    # Supports both sample_inputs and input_checks layouts from Step 169.
    sample = readiness.get("sample_inputs", {}) or {}
    if sample.get(key):
        return str(sample[key])
    checks = readiness.get("input_checks", {}) or {}
    if key in checks and checks[key].get("path"):
        return str(checks[key]["path"])
    if key == "gatv2_checkpoint":
        return str((readiness.get("checkpoint_check", {}) or {}).get("path", ""))
    if key == "gatv2_snapshot_pt":
        return str((readiness.get("gatv2_snapshot_pt_check", {}) or {}).get("path", ""))
    return ""


def script_path(project_root: Path, key: str) -> str:
    return str(project_root / PIPELINE_SCRIPT_REL[key])


def build_commands(
    project_root: Path,
    readiness: Dict[str, Any],
    run_root: Path,
    python_expr: str,
    zero_loss_epsilon_sec: float,
    top_k_per_attempt: int,
    top_k_attention: int,
    file_format: str,
    attention_mode: str,
) -> List[Dict[str, Any]]:
    raw_events = get_input_path(readiness, "raw_events")
    window_rollup = get_input_path(readiness, "window_rollup")
    route_stop_sequence = get_input_path(readiness, "route_stop_sequence")
    run_manifest = get_input_path(readiness, "run_manifest")
    checkpoint = get_input_path(readiness, "gatv2_checkpoint")
    pyg_pt = get_input_path(readiness, "gatv2_snapshot_pt")

    s162 = run_root / "step162_route_aware_adapter"
    s161 = run_root / "step161_pickup_attempts"
    s163 = run_root / "step163_real_attention"
    s165 = run_root / "step165_path_filtered_attention"
    s160 = run_root / "step160_zero_loss_report"
    s166 = run_root / "step166_patent_report_v2"

    normalized = s162 / f"normalized_route_aware_rollout_events.{file_format}"
    s162_manifest = s162 / "route_aware_rollout_adapter_manifest.json"
    attempt_events = s161 / f"pickup_attempt_events.{file_format}"
    eta_counter = s161 / f"eta_counterfactual.{file_format}"
    s161_manifest = s161 / "route_aware_pickup_attempt_event_writer_manifest.json"
    real_attention = s163 / f"gatv2_attention.{file_format}"
    filtered_attention = s165 / "gatv2_attention.csv"
    s165_manifest = s165 / "attempt_route_path_attention_filter_manifest.json"
    s160_manifest = s160 / "zero_loss_evidence_manifest.json"

    commands: List[Dict[str, Any]] = []

    commands.append({
        "step": "Step 162",
        "name": "Normalize route-aware rollout events",
        "required_input_paths": [raw_events, window_rollup, run_manifest],
        "output_root": str(s162),
        "command": (
            f"& $Py {q(script_path(project_root, 'step162_adapter'))} `\n"
            f"  --mode from-route-aware-rollout `\n"
            f"  --raw-events {q(raw_events)} `\n"
            f"  --window-rollup {q(window_rollup)} `\n"
            f"  --source-manifest {q(run_manifest)} `\n"
            f"  --output-root {q(str(s162))} `\n"
            f"  --file-format {file_format}"
        ),
    })
    commands.append({
        "step": "Step 161",
        "name": "Write pickup attempts and ETA counterfactuals",
        "required_input_paths": [str(normalized), str(s162_manifest)],
        "output_root": str(s161),
        "command": (
            f"& $Py {q(script_path(project_root, 'step161_writer'))} `\n"
            f"  --mode from-rollout `\n"
            f"  --rollout-events {q(str(normalized))} `\n"
            f"  --source-manifest {q(str(s162_manifest))} `\n"
            f"  --output-root {q(str(s161))} `\n"
            f"  --zero-loss-epsilon-sec {zero_loss_epsilon_sec} `\n"
            f"  --file-format {file_format}"
        ),
    })

    if attention_mode == "real_from_pyg_pt":
        s163_cmd = (
            f"& $Py {q(script_path(project_root, 'step163_attention_extractor'))} `\n"
            f"  --mode from-pyg-pt `\n"
            f"  --pyg-pt {q(pyg_pt)} `\n"
            f"  --checkpoint {q(checkpoint)} `\n"
            f"  --attempt-events {q(str(attempt_events))} `\n"
            f"  --output-root {q(str(s163))} `\n"
            f"  --top-k-per-attempt {top_k_per_attempt} `\n"
            f"  --file-format {file_format}"
        )
        required = [pyg_pt, checkpoint, str(attempt_events)]
    else:
        s163_cmd = (
            f"& $Py {q(script_path(project_root, 'step163_attention_extractor'))} `\n"
            f"  --mode sample `\n"
            f"  --attempt-events {q(str(attempt_events))} `\n"
            f"  --output-root {q(str(s163))} `\n"
            f"  --top-k-per-attempt {top_k_per_attempt} `\n"
            f"  --file-format {file_format}"
        )
        required = [str(attempt_events)]
    commands.append({
        "step": "Step 163",
        "name": "Extract real GATv2 attention",
        "required_input_paths": required,
        "output_root": str(s163),
        "command": s163_cmd,
    })
    commands.append({
        "step": "Step 165",
        "name": "Filter attention by attempt-specific route/path",
        "required_input_paths": [str(attempt_events), str(eta_counter), str(real_attention), route_stop_sequence, run_manifest],
        "output_root": str(s165),
        "command": (
            f"& $Py {q(script_path(project_root, 'step165_path_filter'))} `\n"
            f"  --mode from-files `\n"
            f"  --pickup-attempt-events {q(str(attempt_events))} `\n"
            f"  --eta-counterfactual {q(str(eta_counter))} `\n"
            f"  --real-attention {q(str(real_attention))} `\n"
            f"  --route-stop-sequence {q(route_stop_sequence)} `\n"
            f"  --run-manifest {q(run_manifest)} `\n"
            f"  --output-root {q(str(s165))} `\n"
            f"  --top-k-per-attempt {top_k_per_attempt}"
        ),
    })
    commands.append({
        "step": "Step 160",
        "name": "Generate zero-loss pickup evidence bundle",
        "required_input_paths": [str(attempt_events), str(eta_counter), str(filtered_attention), run_manifest],
        "output_root": str(s160),
        "command": (
            f"& $Py {q(script_path(project_root, 'step160_reporter'))} `\n"
            f"  --pickup-attempt-events {q(str(attempt_events))} `\n"
            f"  --eta-counterfactual {q(str(eta_counter))} `\n"
            f"  --attention-weights {q(str(filtered_attention))} `\n"
            f"  --run-manifest {q(run_manifest)} `\n"
            f"  --output-root {q(str(s160))} `\n"
            f"  --zero-loss-epsilon-sec {zero_loss_epsilon_sec} `\n"
            f"  --top-k-attention {top_k_attention}"
        ),
    })
    commands.append({
        "step": "Step 166",
        "name": "Generate patent evidence report v2",
        "required_input_paths": [str(s160_manifest), str(s165_manifest)],
        "output_root": str(s166),
        "command": (
            f"& $Py {q(script_path(project_root, 'step166_report_v2'))} `\n"
            f"  --mode from-manifests `\n"
            f"  --step160-manifest {q(str(s160_manifest))} `\n"
            f"  --step165-manifest {q(str(s165_manifest))} `\n"
            f"  --output-root {q(str(s166))}"
        ),
    })
    return commands


def build_operator_script(commands: List[Dict[str, Any]], packet_manifest_path: Path) -> str:
    lines = [
        'param([switch]$Execute)',
        '$ErrorActionPreference = "Stop"',
        '$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path',
        'Set-Location $ProjectRoot',
        '$Py = "python"',
        'if (Test-Path ".\\05_training\\.venv\\Scripts\\python.exe") {',
        '    $Py = (Resolve-Path ".\\05_training\\.venv\\Scripts\\python.exe").Path',
        '}',
        'Write-Host "[INFO] Step 170 actual evidence command packet"',
        'Write-Host "[INFO] Default mode is dry-run. Use -Execute only after operator approval."',
        f'Write-Host "[INFO] packet_manifest: {packet_manifest_path}"',
        '',
    ]
    for idx, item in enumerate(commands, start=1):
        lines.append(f'Write-Host "[DRY-RUN] {idx}. {item["step"]}: {item["name"]}"')
        command = item["command"]
        # Use here-string for readability.
        lines.append('$Cmd = @"')
        lines.append(command)
        lines.append('"@')
        lines.append('Write-Host $Cmd')
        lines.append('if ($Execute) {')
        lines.append('    Invoke-Expression $Cmd')
        lines.append('    if ($LASTEXITCODE -ne 0) { throw "[FAIL] command failed" }')
        lines.append('}')
        lines.append('')
    lines.append('Write-Host "[DONE] Step 170 command packet dry-run complete."')
    return "\n".join(lines) + "\n"


def build_markdown(manifest: Dict[str, Any], commands: List[Dict[str, Any]]) -> str:
    lines = [
        "# Step 170 Actual Evidence Command Packet",
        "",
        "This command packet fixes the dry-run execution order for the Zero-Loss Pickup evidence pipeline.",
        "It does not unlock actual operational claims or paper-level claims.",
        "",
        "## Status",
        "",
        f"- audit_status: `{manifest['audit_status']}`",
        f"- bundle_status: `{manifest['bundle_status']}`",
        f"- dry_run_only: `{manifest['dry_run_only']}`",
        f"- command_execution_allowed: `{manifest['command_execution_allowed']}`",
        f"- paper_level_claim_allowed: `{manifest['paper_level_claim_allowed']}`",
        f"- causal_performance_claim_allowed: `{manifest['causal_performance_claim_allowed']}`",
        "",
        "## Command sequence",
        "",
    ]
    for idx, item in enumerate(commands, start=1):
        lines += [
            f"### {idx}. {item['step']} — {item['name']}",
            "",
            "```powershell",
            item["command"],
            "```",
            "",
        ]
    lines += [
        "## Claim guard",
        "",
        "The generated command packet is a dry-run command plan. Actual evidence execution requires an explicit operator approval/release step in a later gate.",
        "",
    ]
    return "\n".join(lines)


def generate_packet(args: argparse.Namespace) -> Dict[str, Any]:
    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.mode == "sample":
        readiness_path = make_sample_readiness_manifest(output_root, project_root)
    else:
        if not args.readiness_manifest:
            raise RuntimeError("--readiness-manifest is required for mode=from-readiness")
        readiness_path = Path(args.readiness_manifest).resolve()
        if not readiness_path.exists():
            raise RuntimeError(f"readiness manifest not found: {readiness_path}")

    readiness = load_json(readiness_path)
    run_id = args.run_id or str(readiness.get("run_id") or readiness.get("mode") or "step170_evidence_run")
    run_id = run_id.replace(" ", "_")
    run_root = output_root / "planned_run" / run_id

    hard_failures: List[str] = []
    warnings: List[str] = []

    if readiness.get("audit_status") != "PASS":
        hard_failures.append(f"readiness_audit_status_not_PASS:{readiness.get('audit_status')}")
    if bool(readiness.get("actual_evidence_execution_allowed", False)):
        hard_failures.append("readiness_manifest_unexpected_actual_evidence_execution_allowed_true")
    for flag in ["paper_level_claim_allowed", "causal_performance_claim_allowed", "actual_operational_claim_allowed", "train_allowed"]:
        if bool(readiness.get(flag, False)):
            hard_failures.append(f"readiness_guard_flag_true:{flag}")

    if args.attention_mode == "real_from_pyg_pt":
        if not get_input_path(readiness, "gatv2_snapshot_pt"):
            warnings.append("gatv2_snapshot_pt_not_declared; generated commands may require manual --pyg-pt substitution")

    commands = build_commands(
        project_root=project_root,
        readiness=readiness,
        run_root=run_root,
        python_expr="$Py",
        zero_loss_epsilon_sec=float(args.zero_loss_epsilon_sec),
        top_k_per_attempt=int(args.top_k_per_attempt),
        top_k_attention=int(args.top_k_attention),
        file_format=str(args.file_format),
        attention_mode=str(args.attention_mode),
    )

    for item in commands:
        for p in item.get("required_input_paths", []):
            if not p:
                warnings.append(f"empty_required_input_path:{item['step']}")

    command_sequence_path = output_root / "actual_evidence_command_sequence_step170.json"
    operator_script_path = output_root / "actual_evidence_command_packet_step170.ps1"
    report_path = output_root / "actual_evidence_command_packet_step170.md"
    manifest_path = output_root / "actual_evidence_command_packet_manifest_step170.json"

    audit_status = "PASS" if not hard_failures else "BLOCKED"
    bundle_status = BUNDLE_STATUS if audit_status == "PASS" else BUNDLE_BLOCKED

    manifest: Dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "audit_status": audit_status,
        "bundle_status": bundle_status,
        "mode": args.mode,
        "readiness_manifest": str(readiness_path),
        "readiness_manifest_sha256": sha256_file(readiness_path),
        "project_root": str(project_root),
        "output_root": str(output_root),
        "planned_run_root": str(run_root),
        "command_count": len(commands),
        "attention_mode": str(args.attention_mode),
        "zero_loss_epsilon_sec": float(args.zero_loss_epsilon_sec),
        "top_k_per_attempt": int(args.top_k_per_attempt),
        "top_k_attention": int(args.top_k_attention),
        "dry_run_only": True,
        "operator_approval_required": True,
        "actual_evidence_execution_allowed": False,
        "command_execution_allowed": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "actual_operational_claim_allowed": False,
        "train_allowed": False,
        "winner_selected": False,
        "hard_failures": hard_failures,
        "warnings": warnings,
        "output_files": {
            "command_sequence_json": str(command_sequence_path),
            "operator_script_ps1": str(operator_script_path),
            "report_md": str(report_path),
            "manifest": str(manifest_path),
        },
        "command_steps": [
            {"step": c["step"], "name": c["name"], "output_root": c["output_root"]}
            for c in commands
        ],
    }

    dump_json(command_sequence_path, {"artifact_version": ARTIFACT_VERSION, "commands": commands})
    operator_script_path.write_text(build_operator_script(commands, manifest_path), encoding="utf-8")
    report_path.write_text(build_markdown(manifest, commands), encoding="utf-8")
    dump_json(manifest_path, manifest)

    print("[OK] Step 170 actual evidence command packet generated")
    print(f"[OK] audit_status              : {manifest['audit_status']}")
    print(f"[OK] bundle_status             : {manifest['bundle_status']}")
    print(f"[OK] dry_run_only              : {manifest['dry_run_only']}")
    print(f"[OK] command_count             : {manifest['command_count']}")
    print(f"[OK] actual_evidence_execution_allowed : {manifest['actual_evidence_execution_allowed']}")
    print(f"[OK] command_execution_allowed : {manifest['command_execution_allowed']}")
    print(f"[OK] output_root               : {output_root}")
    print(f"[OK] manifest                  : {manifest_path}")
    print(f"[OK] paper_level_claim_allowed : {manifest['paper_level_claim_allowed']}")
    print(f"[OK] causal_performance_claim_allowed : {manifest['causal_performance_claim_allowed']}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 170 Actual Evidence Command Packet")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--mode", choices=["sample", "from-readiness"], default="sample")
    parser.add_argument("--readiness-manifest", default="")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", default="step170_dry_run")
    parser.add_argument("--zero-loss-epsilon-sec", type=float, default=0.0)
    parser.add_argument("--top-k-per-attempt", type=int, default=8)
    parser.add_argument("--top-k-attention", type=int, default=20)
    parser.add_argument("--file-format", choices=["csv", "parquet"], default="csv")
    parser.add_argument("--attention-mode", choices=["real_from_pyg_pt", "sample_real_attention_smoke"], default="sample_real_attention_smoke")
    args = parser.parse_args()
    generate_packet(args)


if __name__ == "__main__":
    main()
