from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class ArrivalWorkflowError(RuntimeError):
    pass


PASS_STATUSES = {
    "PASS",
    "PASSED",
    "READY",
    "READY_TO_EXECUTE",
    "READY_FOR_PLAN",
    "ACTUAL_CHECKPOINT_READY",
}

BLOCKED_STATUSES = {
    "BLOCKED",
    "BLOCKED_PREFLIGHT",
    "FAIL",
    "FAILED",
    "ERROR",
    "STOP",
}

READY_PLAN_STATUS = "READY_TO_EXECUTE"
DRY_RUN_STATUS = "DRY_RUN_VALIDATED"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise ArrivalWorkflowError(f"failed to read JSON: {path}")


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def norm(value: Any) -> str:
    return str(value or "").strip().upper()


def collect_values_by_key(obj: Any, target_key: str) -> List[Any]:
    out: List[Any] = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == target_key:
                out.append(value)
            out.extend(collect_values_by_key(value, target_key))
    elif isinstance(obj, list):
        for item in obj:
            out.extend(collect_values_by_key(item, target_key))

    return out


def first_status(payload: Dict[str, Any], keys: List[str]) -> str:
    for key in keys:
        if key in payload:
            return norm(payload.get(key))

    for key in keys:
        nested = collect_values_by_key(payload, key)
        if nested:
            return norm(nested[0])

    return ""


def payload_contains_blocked_status(payload: Dict[str, Any]) -> bool:
    candidates: List[str] = []
    for key in ("status", "preflight_status", "plan_status", "runner_status", "arrival_workflow_status"):
        candidates.extend([norm(x) for x in collect_values_by_key(payload, key)])

    return any(x in BLOCKED_STATUSES or x.startswith("BLOCKED") for x in candidates)


def preflight_passed(payload: Dict[str, Any]) -> bool:
    if payload_contains_blocked_status(payload):
        return False

    status = first_status(payload, ["preflight_status", "status", "checkpoint_status", "runner_status"])
    return status in PASS_STATUSES


def plan_ready(payload: Dict[str, Any]) -> bool:
    if payload_contains_blocked_status(payload):
        return False

    status = first_status(payload, ["status", "plan_status", "execution_status", "runner_status"])
    return status == READY_PLAN_STATUS


def dry_run_validated(payload: Dict[str, Any]) -> bool:
    if payload_contains_blocked_status(payload):
        return False

    status = first_status(payload, ["runner_status", "status"])
    dry_run = bool(payload.get("dry_run", False))
    executed = bool(payload.get("executed", False))

    return status == DRY_RUN_STATUS and dry_run is True and executed is False


def same_checkpoint_path(input_path: Path, preflight_payload: Dict[str, Any]) -> bool:
    observed_values = collect_values_by_key(preflight_payload, "checkpoint_path")

    if not observed_values:
        # Some older preflight reports may omit checkpoint_path.
        # In that case, Step 59 will still perform its own consistency guard.
        return True

    expected = str(input_path.resolve())

    for value in observed_values:
        try:
            observed = str(Path(str(value)).resolve())
        except Exception:
            observed = str(value)

        if observed == expected:
            return True

    return False


def run_command(command: List[str], label: str) -> Dict[str, Any]:
    started_at = utc_now()
    completed = subprocess.run(command, text=True, capture_output=True)
    ended_at = utc_now()

    return {
        "label": label,
        "command": command,
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "returncode": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "status": "PASS" if completed.returncode == 0 else "FAIL",
    }


def require_command_passed(log: Dict[str, Any]) -> None:
    if int(log["returncode"]) != 0:
        raise ArrivalWorkflowError(
            f"{log['label']} command failed with returncode={log['returncode']}: {log.get('stderr', '').strip()}"
        )


def build_preflight_command(args: argparse.Namespace, checkpoint_path: Path, preflight_report: Path) -> List[str]:
    return [
        args.python_exe,
        str(Path(args.preflight_script)),
        "--checkpoint-path",
        str(checkpoint_path),
        "--mode",
        "actual",
        "--report-path",
        str(preflight_report),
    ]


def build_plan_command(
    args: argparse.Namespace,
    checkpoint_path: Path,
    preflight_report: Path,
    plan_json: Path,
) -> List[str]:
    return [
        args.python_exe,
        str(Path(args.plan_script)),
        "--preflight-report",
        str(preflight_report),
        "--checkpoint-path",
        str(checkpoint_path),
        "--output-plan",
        str(plan_json),
    ]


def build_dry_run_command(args: argparse.Namespace, plan_json: Path, dry_run_report: Path) -> List[str]:
    return [
        args.python_exe,
        str(Path(args.runner_script)),
        "--plan-json",
        str(plan_json),
        "--report-path",
        str(dry_run_report),
        "--dry-run",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 62 H200 actual checkpoint arrival workflow."
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Python executable used to invoke Step 58/59/60 scripts.",
    )
    parser.add_argument(
        "--preflight-script",
        default="05_training/policies/h200_actual_checkpoint_preflight_checklist.py",
    )
    parser.add_argument(
        "--plan-script",
        default="05_training/policies/a_family_mappo_actual_execution_plan.py",
    )
    parser.add_argument(
        "--runner-script",
        default="05_training/policies/run_a_family_mappo_actual_matrix_from_plan.py",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint_path)
    output_root = Path(args.output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    preflight_report = output_root / "h200_actual_checkpoint_preflight_report.json"
    plan_json = output_root / "a_family_mappo_actual_execution_plan.json"
    dry_run_report = output_root / "a_family_mappo_actual_matrix_runner_dry_run_report.json"
    arrival_report = output_root / "h200_actual_checkpoint_arrival_workflow_report.json"

    command_logs: List[Dict[str, Any]] = []

    def write_blocked(error: str) -> None:
        dump_json(
            arrival_report,
            {
                "artifact_version": "h200_actual_checkpoint_arrival_workflow_v1_step62",
                "created_at_utc": utc_now(),
                "arrival_workflow_status": "BLOCKED",
                "checkpoint_path": str(checkpoint_path),
                "output_root": str(output_root),
                "error": error,
                "reports": {
                    "preflight_report": str(preflight_report),
                    "plan_json": str(plan_json),
                    "dry_run_report": str(dry_run_report),
                    "arrival_report": str(arrival_report),
                },
                "command_logs": command_logs,
            },
        )

    try:
        if not checkpoint_path.exists():
            raise ArrivalWorkflowError(f"checkpoint file does not exist: {checkpoint_path}")

        for script_label, script_path in (
            ("preflight_script", Path(args.preflight_script)),
            ("plan_script", Path(args.plan_script)),
            ("runner_script", Path(args.runner_script)),
        ):
            if not script_path.exists():
                raise ArrivalWorkflowError(f"{script_label} not found: {script_path}")

        preflight_cmd = build_preflight_command(args, checkpoint_path, preflight_report)
        preflight_log = run_command(preflight_cmd, label="Step 58 actual preflight")
        command_logs.append(preflight_log)
        require_command_passed(preflight_log)

        if not preflight_report.exists():
            raise ArrivalWorkflowError(f"preflight report was not created: {preflight_report}")

        preflight_payload = load_json(preflight_report)

        if not preflight_passed(preflight_payload):
            raise ArrivalWorkflowError("preflight report is not PASS")

        if not same_checkpoint_path(checkpoint_path, preflight_payload):
            raise ArrivalWorkflowError("checkpoint_path mismatch between input checkpoint and preflight report")

        plan_cmd = build_plan_command(args, checkpoint_path, preflight_report, plan_json)
        plan_log = run_command(plan_cmd, label="Step 59 actual execution plan")
        command_logs.append(plan_log)
        require_command_passed(plan_log)

        if not plan_json.exists():
            raise ArrivalWorkflowError(f"execution plan was not created: {plan_json}")

        plan_payload = load_json(plan_json)

        if not plan_ready(plan_payload):
            raise ArrivalWorkflowError("execution plan is not READY_TO_EXECUTE")

        dry_run_cmd = build_dry_run_command(args, plan_json, dry_run_report)
        dry_run_log = run_command(dry_run_cmd, label="Step 60 matrix runner dry-run")
        command_logs.append(dry_run_log)
        require_command_passed(dry_run_log)

        if not dry_run_report.exists():
            raise ArrivalWorkflowError(f"dry-run report was not created: {dry_run_report}")

        dry_run_payload = load_json(dry_run_report)

        if not dry_run_validated(dry_run_payload):
            raise ArrivalWorkflowError("dry-run report is not DRY_RUN_VALIDATED")

        final_report = {
            "artifact_version": "h200_actual_checkpoint_arrival_workflow_v1_step62",
            "created_at_utc": utc_now(),
            "arrival_workflow_status": "READY_FOR_MANUAL_EXECUTE",
            "checkpoint_path": str(checkpoint_path),
            "output_root": str(output_root),
            "preflight_status": first_status(preflight_payload, ["preflight_status", "status", "checkpoint_status", "runner_status"]),
            "plan_status": first_status(plan_payload, ["status", "plan_status", "execution_status", "runner_status"]),
            "dry_run_status": first_status(dry_run_payload, ["runner_status", "status"]),
            "dry_run": bool(dry_run_payload.get("dry_run", False)),
            "executed": bool(dry_run_payload.get("executed", False)),
            "run_count": dry_run_payload.get("run_count"),
            "command_count": dry_run_payload.get("command_count"),
            "reports": {
                "preflight_report": str(preflight_report),
                "plan_json": str(plan_json),
                "dry_run_report": str(dry_run_report),
                "arrival_report": str(arrival_report),
            },
            "claim_boundary": {
                "actual_policy_path_validated_to_dry_run": True,
                "causal_performance_claim_allowed": False,
                "note": "Passing Step 62 does not prove causal performance superiority.",
            },
            "command_logs": command_logs,
        }

        dump_json(arrival_report, final_report)

        print("[OK] Step 62 H200 actual checkpoint arrival workflow PASS")
        print(f"[OK] checkpoint     : {checkpoint_path}")
        print(f"[OK] output_root    : {output_root}")
        print(f"[OK] preflight      : {preflight_report}")
        print(f"[OK] plan           : {plan_json}")
        print(f"[OK] dry_run_report : {dry_run_report}")
        print(f"[OK] arrival_report : {arrival_report}")
        print("[OK] status         : READY_FOR_MANUAL_EXECUTE")
        return 0

    except ArrivalWorkflowError as exc:
        write_blocked(str(exc))
        print(f"[BLOCKED] {exc}", file=sys.stderr)
        print(f"[BLOCKED] arrival_report: {arrival_report}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
