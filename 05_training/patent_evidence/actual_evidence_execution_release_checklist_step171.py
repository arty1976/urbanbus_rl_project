from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


ARTIFACT_VERSION = "actual_evidence_execution_release_checklist_step171_v1"
BUNDLE_STATUS_LOCKED = "ACTUAL_EVIDENCE_EXECUTION_RELEASE_CHECKLIST_READY_STILL_LOCKED"
BUNDLE_STATUS_RELEASED = "ACTUAL_EVIDENCE_EXECUTION_RELEASE_CHECKLIST_RELEASED_FOR_EXECUTION_NONCLAIM"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with path.open("r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "approved", "grant", "granted"}:
        return True
    if s in {"0", "false", "no", "n", "deny", "denied", ""}:
        return False
    raise argparse.ArgumentTypeError(f"cannot parse bool: {value}")


def sample_readiness_manifest(path: Path) -> Dict[str, Any]:
    payload = {
        "artifact_version": "actual_evidence_input_readiness_checklist_step169_v1",
        "step": 169,
        "audit_status": "PASS",
        "bundle_status": "ACTUAL_EVIDENCE_INPUTS_READY_FOR_ZERO_LOSS_PIPELINE_NONCLAIM",
        "mode": "sample",
        "ready_for_actual_like_execution": True,
        "actual_evidence_execution_allowed": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "actual_operational_claim_allowed": False,
        "hard_failures": 0,
        "warnings": [],
        "input_files": {
            "raw_events": "sample/raw_events.csv",
            "window_rollup": "sample/window_rollup.csv",
            "route_stop_sequence": "sample/route_stop_sequence.csv",
            "run_manifest": "sample/run_manifest.json",
        },
    }
    dump_json(path, payload)
    return payload


def sample_command_packet_manifest(path: Path) -> Dict[str, Any]:
    commands = [
        {"step": 162, "name": "route_aware_rollout_adapter", "command": "python ...step162.py"},
        {"step": 161, "name": "pickup_attempt_writer", "command": "python ...step161.py"},
        {"step": 163, "name": "gatv2_real_attention_extractor", "command": "python ...step163.py"},
        {"step": 165, "name": "attempt_path_attention_filter", "command": "python ...step165.py"},
        {"step": 160, "name": "zero_loss_reporter", "command": "python ...step160.py"},
        {"step": 166, "name": "patent_evidence_report_v2", "command": "python ...step166.py"},
    ]
    payload = {
        "artifact_version": "actual_evidence_command_packet_step170_v1",
        "step": 170,
        "audit_status": "PASS",
        "bundle_status": "ACTUAL_EVIDENCE_COMMAND_PACKET_READY_DRY_RUN_NONCLAIM",
        "dry_run_only": True,
        "command_count": len(commands),
        "commands": commands,
        "actual_evidence_execution_allowed": False,
        "command_execution_allowed": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "actual_operational_claim_allowed": False,
        "hard_failures": 0,
        "warnings": [],
    }
    dump_json(path, payload)
    return payload


def evaluate_release(
    readiness: Dict[str, Any],
    command_packet: Dict[str, Any],
    operator_approval_recorded: bool,
    operator_approval_granted: bool,
    release_manifest_committed: bool,
) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    def add_check(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    add_check(
        "step169_ready_for_actual_like_execution",
        bool(readiness.get("ready_for_actual_like_execution", False)),
        f"value={readiness.get('ready_for_actual_like_execution')}",
    )
    add_check(
        "step169_claim_guards_locked",
        not bool(readiness.get("paper_level_claim_allowed", False))
        and not bool(readiness.get("causal_performance_claim_allowed", False))
        and not bool(readiness.get("actual_operational_claim_allowed", False)),
        "paper/causal/operational claim guards must remain false",
    )
    add_check(
        "step170_audit_status_pass",
        command_packet.get("audit_status") == "PASS",
        f"value={command_packet.get('audit_status')}",
    )
    add_check(
        "step170_dry_run_only",
        bool(command_packet.get("dry_run_only", False)),
        f"value={command_packet.get('dry_run_only')}",
    )
    add_check(
        "step170_command_count_ge_6",
        int(command_packet.get("command_count", 0)) >= 6,
        f"value={command_packet.get('command_count')}",
    )
    add_check(
        "step170_command_execution_still_locked_before_release",
        not bool(command_packet.get("command_execution_allowed", False)),
        f"value={command_packet.get('command_execution_allowed')}",
    )
    add_check(
        "step170_claim_guards_locked",
        not bool(command_packet.get("paper_level_claim_allowed", False))
        and not bool(command_packet.get("causal_performance_claim_allowed", False))
        and not bool(command_packet.get("actual_operational_claim_allowed", False)),
        "paper/causal/operational claim guards must remain false",
    )
    add_check(
        "operator_approval_recorded",
        bool(operator_approval_recorded),
        f"value={operator_approval_recorded}",
    )
    add_check(
        "operator_approval_granted",
        bool(operator_approval_granted),
        f"value={operator_approval_granted}",
    )
    add_check(
        "release_manifest_committed",
        bool(release_manifest_committed),
        f"value={release_manifest_committed}",
    )

    passed_count = sum(1 for c in checks if c["passed"])
    failed = [c for c in checks if not c["passed"]]
    hard_failures = len(failed)

    actual_evidence_execution_allowed = hard_failures == 0
    command_execution_allowed = actual_evidence_execution_allowed

    return {
        "checks": checks,
        "passed_check_count": int(passed_count),
        "failed_check_count": int(hard_failures),
        "hard_failures": int(hard_failures),
        "failed_checks": failed,
        "actual_evidence_execution_allowed": bool(actual_evidence_execution_allowed),
        "command_execution_allowed": bool(command_execution_allowed),
    }


def write_markdown(path: Path, manifest: Dict[str, Any]) -> None:
    lines = [
        "# Step 171 Actual Evidence Execution Release Checklist",
        "",
        f"- audit_status: `{manifest['audit_status']}`",
        f"- bundle_status: `{manifest['bundle_status']}`",
        f"- mode: `{manifest['mode']}`",
        f"- operator_approval_recorded: `{manifest['operator_approval_recorded']}`",
        f"- operator_approval_granted: `{manifest['operator_approval_granted']}`",
        f"- release_manifest_committed: `{manifest['release_manifest_committed']}`",
        f"- actual_evidence_execution_allowed: `{manifest['actual_evidence_execution_allowed']}`",
        f"- command_execution_allowed: `{manifest['command_execution_allowed']}`",
        "",
        "## Claim guards",
        "",
        f"- paper_level_claim_allowed: `{manifest['paper_level_claim_allowed']}`",
        f"- causal_performance_claim_allowed: `{manifest['causal_performance_claim_allowed']}`",
        f"- actual_operational_claim_allowed: `{manifest['actual_operational_claim_allowed']}`",
        "",
        "## Checks",
        "",
        "| check | passed | detail |",
        "|---|---:|---|",
    ]
    for c in manifest["checks"]:
        lines.append(f"| {c['name']} | {c['passed']} | {c.get('detail','')} |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "This manifest releases only evidence-generation command execution when all checks pass. It does not release paper-level, causal-performance, or operational claims.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_release_checklist(
    output_root: Path,
    mode: str,
    readiness_manifest_path: Optional[Path],
    command_packet_manifest_path: Optional[Path],
    operator_approval_granted: bool,
    release_manifest_committed: bool,
    operator_name: str,
    decision_note: str,
) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)

    prereq_root = output_root / "sample_prereqs"
    if mode == "sample":
        readiness_manifest_path = prereq_root / "step169_readiness_manifest_sample.json"
        command_packet_manifest_path = prereq_root / "step170_command_packet_manifest_sample.json"
        readiness = sample_readiness_manifest(readiness_manifest_path)
        command_packet = sample_command_packet_manifest(command_packet_manifest_path)
    else:
        if readiness_manifest_path is None or not readiness_manifest_path.exists():
            raise RuntimeError("--readiness-manifest is required and must exist outside sample mode")
        if command_packet_manifest_path is None or not command_packet_manifest_path.exists():
            raise RuntimeError("--command-packet-manifest is required and must exist outside sample mode")
        readiness = load_json(readiness_manifest_path)
        command_packet = load_json(command_packet_manifest_path)

    operator_approval_recorded = bool(operator_name.strip()) or bool(decision_note.strip()) or bool(operator_approval_granted)

    evaluation = evaluate_release(
        readiness=readiness,
        command_packet=command_packet,
        operator_approval_recorded=operator_approval_recorded,
        operator_approval_granted=operator_approval_granted,
        release_manifest_committed=release_manifest_committed,
    )

    audit_status = "PASS" if evaluation["hard_failures"] == 0 else "BLOCKED"
    bundle_status = BUNDLE_STATUS_RELEASED if evaluation["actual_evidence_execution_allowed"] else BUNDLE_STATUS_LOCKED

    manifest_path = output_root / "actual_evidence_execution_release_checklist_manifest.json"
    md_path = output_root / "actual_evidence_execution_release_checklist.md"

    manifest: Dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "step": 171,
        "generated_at_utc": utc_now(),
        "mode": mode,
        "audit_status": audit_status,
        "bundle_status": bundle_status,
        "readiness_manifest": str(readiness_manifest_path),
        "command_packet_manifest": str(command_packet_manifest_path),
        "operator_approval_required": True,
        "operator_approval_recorded": bool(operator_approval_recorded),
        "operator_approval_granted": bool(operator_approval_granted),
        "operator_name": operator_name,
        "decision_note": decision_note,
        "release_manifest_committed": bool(release_manifest_committed),
        "ready_for_actual_like_execution": bool(readiness.get("ready_for_actual_like_execution", False)),
        "dry_run_packet_validated": command_packet.get("audit_status") == "PASS",
        "command_count": int(command_packet.get("command_count", 0)),
        "dry_run_only": bool(command_packet.get("dry_run_only", False)),
        **evaluation,
        # Non-claim guards remain locked even if evidence execution is released.
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "actual_operational_claim_allowed": False,
        "train_allowed": False,
        "winner_selected": False,
        "output_files": {
            "manifest": str(manifest_path),
            "markdown": str(md_path),
        },
    }

    dump_json(manifest_path, manifest)
    write_markdown(md_path, manifest)

    print("[OK] Step 171 actual evidence execution release checklist generated")
    print(f"[OK] audit_status              : {manifest['audit_status']}")
    print(f"[OK] bundle_status             : {manifest['bundle_status']}")
    print(f"[OK] mode                      : {manifest['mode']}")
    print(f"[OK] operator_approval_granted : {manifest['operator_approval_granted']}")
    print(f"[OK] release_manifest_committed: {manifest['release_manifest_committed']}")
    print(f"[OK] actual_evidence_execution_allowed : {manifest['actual_evidence_execution_allowed']}")
    print(f"[OK] command_execution_allowed : {manifest['command_execution_allowed']}")
    print(f"[OK] hard_failures             : {manifest['hard_failures']}")
    print(f"[OK] manifest                  : {manifest_path}")
    print(f"[OK] paper_level_claim_allowed : {manifest['paper_level_claim_allowed']}")
    print(f"[OK] causal_performance_claim_allowed : {manifest['causal_performance_claim_allowed']}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 171 actual evidence execution release checklist")
    parser.add_argument("--mode", choices=["sample", "from-manifests"], default="sample")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--readiness-manifest", default="")
    parser.add_argument("--command-packet-manifest", default="")
    parser.add_argument("--operator-approval-granted", type=parse_bool, default=False)
    parser.add_argument("--release-manifest-committed", type=parse_bool, default=False)
    parser.add_argument("--operator-name", default="")
    parser.add_argument("--decision-note", default="")
    args = parser.parse_args()

    generate_release_checklist(
        output_root=Path(args.output_root),
        mode=args.mode,
        readiness_manifest_path=Path(args.readiness_manifest) if args.readiness_manifest else None,
        command_packet_manifest_path=Path(args.command_packet_manifest) if args.command_packet_manifest else None,
        operator_approval_granted=bool(args.operator_approval_granted),
        release_manifest_committed=bool(args.release_manifest_committed),
        operator_name=str(args.operator_name),
        decision_note=str(args.decision_note),
    )


if __name__ == "__main__":
    main()
