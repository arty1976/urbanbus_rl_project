from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


READY_STATUS = "READY_TO_EXECUTE"
A_FAMILY_CONDITIONS = {"A", "A90", "A80", "A70"}

ORDERED_COMMAND_KEYS = [
    "rollout_command",
    "metadata_validation_command",
    "canonical_command",
    "post_canonical_validation_command",
]

EXPECTED_REWARD_VERSION = "mappo_reward_v1"
EXPECTED_ENERGY_PROXY_MODEL_VERSION = "daegu_energy_proxy_v1"
EXPECTED_K_DIST = 0.0012
EXPECTED_K_ACC = 0.1800
EXPECTED_K_IDLE = 0.0080


class MatrixRunnerGuardError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise MatrixRunnerGuardError(f"failed to read JSON: {path}")


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def normalize_status(value: Any) -> str:
    return str(value or "").strip().upper()


def get_plan_status(plan: Dict[str, Any]) -> str:
    for key in ("status", "plan_status", "execution_status", "runner_status"):
        if key in plan:
            return normalize_status(plan.get(key))
    return ""


def get_runs(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("runs", "planned_runs", "run_plan", "matrix_runs"):
        value = plan.get(key)
        if isinstance(value, list):
            return value
    raise MatrixRunnerGuardError(
        "plan does not contain a run list. Expected one of: runs, planned_runs, run_plan, matrix_runs"
    )


def merged_guard(plan: Dict[str, Any], run: Dict[str, Any], key: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    top = plan.get(key)
    if isinstance(top, dict):
        out.update(top)

    nested = run.get(key)
    if isinstance(nested, dict):
        out.update(nested)

    return out


def lookup_guard_value(
    plan: Dict[str, Any],
    run: Dict[str, Any],
    guard_key: str,
    value_key: str,
    default: Any = None,
) -> Any:
    guard = merged_guard(plan, run, guard_key)
    if value_key in guard:
        return guard.get(value_key)
    if value_key in run:
        return run.get(value_key)
    if value_key in plan:
        return plan.get(value_key)
    return default


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return default


def as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def condition_seed_label(run: Dict[str, Any]) -> str:
    condition = str(run.get("condition_id", "UNKNOWN")).upper()
    seed = run.get("seed", "UNKNOWN")
    return f"{condition}/seed={seed}"


def is_noncausal_adapter_or_source(plan: Dict[str, Any], run: Dict[str, Any]) -> bool:
    text_parts = []

    for obj in (plan, run):
        for key in (
            "simulator_adapter",
            "simulator_adapter_path",
            "adapter",
            "adapter_path",
            "adapter_class",
            "source_mode",
            "policy_action_source_mode",
        ):
            if key in obj and obj.get(key) is not None:
                text_parts.append(str(obj.get(key)))

    for guard_key in ("causal_claim_guard", "actual_claim_guard"):
        for obj in (plan.get(guard_key), run.get(guard_key)):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if value is not None:
                        text_parts.append(f"{key}={value}")

    joined = " ".join(text_parts).lower()
    markers = [
        "historical",
        "replay",
        "noncausal",
        "non_causal",
        "non-causal",
        "smoke",
        "stub",
    ]
    return any(marker in joined for marker in markers)


def validate_plan_status(plan: Dict[str, Any]) -> None:
    status = get_plan_status(plan)
    if status != READY_STATUS:
        raise MatrixRunnerGuardError(
            f"plan is not READY_TO_EXECUTE. observed_status={status or '<missing>'}"
        )


def validate_condition(run: Dict[str, Any]) -> None:
    condition = str(run.get("condition_id", "")).strip().upper()
    if condition not in A_FAMILY_CONDITIONS:
        raise MatrixRunnerGuardError(
            f"non A-family condition is not allowed: condition_id={condition or '<missing>'}"
        )


def validate_command_set(run: Dict[str, Any]) -> List[Tuple[str, Any]]:
    commands: List[Tuple[str, Any]] = []

    for key in ORDERED_COMMAND_KEYS:
        if key not in run:
            raise MatrixRunnerGuardError(
                f"{condition_seed_label(run)} missing required command: {key}"
            )

        cmd = run.get(key)
        if isinstance(cmd, str):
            if not cmd.strip():
                raise MatrixRunnerGuardError(
                    f"{condition_seed_label(run)} has empty command: {key}"
                )
        elif isinstance(cmd, list):
            if not cmd or not all(isinstance(x, str) and x.strip() for x in cmd):
                raise MatrixRunnerGuardError(
                    f"{condition_seed_label(run)} has invalid command list: {key}"
                )
        else:
            raise MatrixRunnerGuardError(
                f"{condition_seed_label(run)} command must be string or list[str]: {key}"
            )

        commands.append((key, cmd))

    return commands


def validate_actual_claim_guard(plan: Dict[str, Any], run: Dict[str, Any]) -> Dict[str, Any]:
    label = condition_seed_label(run)

    policy_source_mode = str(
        lookup_guard_value(plan, run, "actual_claim_guard", "policy_source_mode", "")
    ).strip()

    checkpoint_validation_mode = str(
        lookup_guard_value(plan, run, "actual_claim_guard", "checkpoint_validation_mode", "")
    ).strip().lower()

    trained_model = as_bool(
        lookup_guard_value(plan, run, "actual_claim_guard", "trained_model", None),
        default=False,
    )

    performance_claim_allowed = as_bool(
        lookup_guard_value(plan, run, "actual_claim_guard", "performance_claim_allowed", None),
        default=False,
    )

    qwen_train = as_bool(
        lookup_guard_value(plan, run, "actual_claim_guard", "qwen_train", None),
        default=False,
    )

    qwen_inference = as_bool(
        lookup_guard_value(plan, run, "actual_claim_guard", "qwen_inference", None),
        default=False,
    )

    qwen_trigger_rate = as_float(
        lookup_guard_value(plan, run, "actual_claim_guard", "qwen_trigger_rate", 0.0),
        default=0.0,
    )

    reward_version = str(
        lookup_guard_value(plan, run, "actual_claim_guard", "reward_version", "")
    ).strip()

    energy_proxy_model_version = str(
        lookup_guard_value(plan, run, "actual_claim_guard", "energy_proxy_model_version", "")
    ).strip()

    k_dist = as_float(
        lookup_guard_value(plan, run, "actual_claim_guard", "k_dist_kwh_per_m", None),
        default=-1.0,
    )
    k_acc = as_float(
        lookup_guard_value(plan, run, "actual_claim_guard", "k_acc_kwh_per_event", None),
        default=-1.0,
    )
    k_idle = as_float(
        lookup_guard_value(plan, run, "actual_claim_guard", "k_idle_kwh_per_sec", None),
        default=-1.0,
    )

    if policy_source_mode != "mappo_actual":
        raise MatrixRunnerGuardError(
            f"{label} actual guard failed: policy_source_mode must be mappo_actual, got={policy_source_mode or '<missing>'}"
        )

    if checkpoint_validation_mode != "actual":
        raise MatrixRunnerGuardError(
            f"{label} actual guard failed: checkpoint_validation_mode must be actual, got={checkpoint_validation_mode or '<missing>'}"
        )

    if not trained_model:
        raise MatrixRunnerGuardError(f"{label} actual guard failed: trained_model must be true")

    if not performance_claim_allowed:
        raise MatrixRunnerGuardError(f"{label} actual guard failed: performance_claim_allowed must be true")

    if qwen_train:
        raise MatrixRunnerGuardError(f"{label} actual guard failed: qwen_train must be false")

    if qwen_inference:
        raise MatrixRunnerGuardError(f"{label} actual guard failed: qwen_inference must be false")

    if qwen_trigger_rate != 0.0:
        raise MatrixRunnerGuardError(
            f"{label} actual guard failed: qwen_trigger_rate must be 0.0, got={qwen_trigger_rate}"
        )

    if reward_version != EXPECTED_REWARD_VERSION:
        raise MatrixRunnerGuardError(
            f"{label} actual guard failed: reward_version must be {EXPECTED_REWARD_VERSION}, got={reward_version or '<missing>'}"
        )

    if energy_proxy_model_version != EXPECTED_ENERGY_PROXY_MODEL_VERSION:
        raise MatrixRunnerGuardError(
            f"{label} actual guard failed: energy_proxy_model_version must be {EXPECTED_ENERGY_PROXY_MODEL_VERSION}, got={energy_proxy_model_version or '<missing>'}"
        )

    if abs(k_dist - EXPECTED_K_DIST) > 1e-12:
        raise MatrixRunnerGuardError(
            f"{label} actual guard failed: k_dist_kwh_per_m mismatch, got={k_dist}"
        )

    if abs(k_acc - EXPECTED_K_ACC) > 1e-12:
        raise MatrixRunnerGuardError(
            f"{label} actual guard failed: k_acc_kwh_per_event mismatch, got={k_acc}"
        )

    if abs(k_idle - EXPECTED_K_IDLE) > 1e-12:
        raise MatrixRunnerGuardError(
            f"{label} actual guard failed: k_idle_kwh_per_sec mismatch, got={k_idle}"
        )

    return {
        "policy_source_mode": policy_source_mode,
        "checkpoint_validation_mode": checkpoint_validation_mode,
        "trained_model": trained_model,
        "performance_claim_allowed": performance_claim_allowed,
        "qwen_train": qwen_train,
        "qwen_inference": qwen_inference,
        "qwen_trigger_rate": qwen_trigger_rate,
        "reward_version": reward_version,
        "energy_proxy_model_version": energy_proxy_model_version,
        "k_dist_kwh_per_m": k_dist,
        "k_acc_kwh_per_event": k_acc,
        "k_idle_kwh_per_sec": k_idle,
        "passed": True,
    }


def validate_causal_claim_guard(plan: Dict[str, Any], run: Dict[str, Any]) -> Dict[str, Any]:
    label = condition_seed_label(run)

    # Step 60 fix:
    # Causal claim guards must be monotonic/strict.
    # A top-level plan guard must not be weakened by a per-run false value.
    top_guard = plan.get("causal_claim_guard")
    if not isinstance(top_guard, dict):
        top_guard = {}

    run_guard = run.get("causal_claim_guard")
    if not isinstance(run_guard, dict):
        run_guard = {}

    require_causal_claim = bool(
        as_bool(plan.get("require_causal_claim"), default=False)
        or as_bool(run.get("require_causal_claim"), default=False)
        or as_bool(top_guard.get("require_causal_claim"), default=False)
        or as_bool(run_guard.get("require_causal_claim"), default=False)
    )

    causal_performance_claim_allowed = bool(
        as_bool(plan.get("causal_performance_claim_allowed"), default=False)
        or as_bool(run.get("causal_performance_claim_allowed"), default=False)
        or as_bool(top_guard.get("causal_performance_claim_allowed"), default=False)
        or as_bool(run_guard.get("causal_performance_claim_allowed"), default=False)
    )

    noncausal_adapter_or_source = is_noncausal_adapter_or_source(plan, run)

    if noncausal_adapter_or_source and require_causal_claim:
        raise MatrixRunnerGuardError(
            f"{label} causal guard failed: historical/replay/noncausal adapter cannot require causal claim"
        )

    if noncausal_adapter_or_source and causal_performance_claim_allowed:
        raise MatrixRunnerGuardError(
            f"{label} causal guard failed: historical/replay/noncausal adapter cannot allow causal performance claim"
        )

    return {
        "require_causal_claim": require_causal_claim,
        "causal_performance_claim_allowed": causal_performance_claim_allowed,
        "noncausal_adapter_or_source": noncausal_adapter_or_source,
        "passed": True,
    }


def extract_expected_outputs(run: Dict[str, Any]) -> List[str]:
    value = run.get("expected_outputs", [])
    outputs: List[str] = []

    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                outputs.append(item)
            elif isinstance(item, dict):
                for v in item.values():
                    if isinstance(v, str):
                        outputs.append(v)
    elif isinstance(value, dict):
        for v in value.values():
            if isinstance(v, str):
                outputs.append(v)
            elif isinstance(v, list):
                outputs.extend([str(x) for x in v])

    for key in (
        "rollout_output",
        "metadata_validation_output",
        "canonical_output",
        "post_canonical_validation_output",
        "output_root",
        "rollout_root",
        "canonical_eval_root",
    ):
        if key in run and run.get(key) is not None:
            outputs.append(str(run.get(key)))

    return sorted(set(outputs))


def validate_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    validate_plan_status(plan)
    runs = get_runs(plan)

    if not runs:
        raise MatrixRunnerGuardError("plan has zero runs")

    command_records: List[Dict[str, Any]] = []
    expected_outputs: List[str] = []
    actual_guards: List[Dict[str, Any]] = []
    causal_guards: List[Dict[str, Any]] = []

    for run_index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise MatrixRunnerGuardError(f"run entry must be a dict: index={run_index}")

        validate_condition(run)
        commands = validate_command_set(run)
        actual_guard = validate_actual_claim_guard(plan, run)
        causal_guard = validate_causal_claim_guard(plan, run)

        actual_guards.append({
            "run_index": run_index,
            "condition_id": str(run.get("condition_id")).upper(),
            "seed": run.get("seed"),
            **actual_guard,
        })
        causal_guards.append({
            "run_index": run_index,
            "condition_id": str(run.get("condition_id")).upper(),
            "seed": run.get("seed"),
            **causal_guard,
        })

        for command_order, (command_key, command) in enumerate(commands, start=1):
            command_records.append({
                "run_index": run_index,
                "condition_id": str(run.get("condition_id")).upper(),
                "seed": run.get("seed"),
                "command_order": command_order,
                "command_key": command_key,
                "command": command,
            })

        expected_outputs.extend(extract_expected_outputs(run))

    return {
        "run_count": len(runs),
        "command_count": len(command_records),
        "commands": command_records,
        "expected_outputs": sorted(set(expected_outputs)),
        "actual_claim_guard": {
            "all_passed": True,
            "per_run": actual_guards,
        },
        "causal_claim_guard": {
            "all_passed": True,
            "per_run": causal_guards,
        },
    }


def run_command(command: Any, cwd: Path | None = None) -> int:
    if isinstance(command, list):
        completed = subprocess.run(command, cwd=str(cwd) if cwd else None)
    else:
        completed = subprocess.run(command, shell=True, cwd=str(cwd) if cwd else None)
    return int(completed.returncode)


def execute_plan(
    validation: Dict[str, Any],
    cwd: Path | None,
) -> List[Dict[str, Any]]:
    execution_log: List[Dict[str, Any]] = []

    for record in validation["commands"]:
        started_at = utc_now()
        rc = run_command(record["command"], cwd=cwd)
        ended_at = utc_now()

        log_row = {
            **record,
            "started_at_utc": started_at,
            "ended_at_utc": ended_at,
            "returncode": rc,
            "status": "PASS" if rc == 0 else "FAIL",
        }
        execution_log.append(log_row)

        if rc != 0:
            raise MatrixRunnerGuardError(
                f"command failed: {record['condition_id']}/seed={record['seed']} "
                f"{record['command_key']} returncode={rc}"
            )

    return execution_log


def build_report(
    plan_path: Path,
    plan: Dict[str, Any],
    validation: Dict[str, Any],
    dry_run: bool,
    executed: bool,
    execution_log: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "artifact_version": "a_family_mappo_actual_matrix_runner_v1_step60",
        "created_at_utc": utc_now(),
        "plan_path": str(plan_path),
        "plan_status": get_plan_status(plan),
        "dry_run": bool(dry_run),
        "executed": bool(executed),
        "runner_status": "DRY_RUN_VALIDATED" if dry_run else "EXECUTION_COMPLETED",
        "run_count": validation["run_count"],
        "command_count": validation["command_count"],
        "command_order_contract": list(ORDERED_COMMAND_KEYS),
        "commands": validation["commands"],
        "expected_outputs": validation["expected_outputs"],
        "causal_claim_guard": validation["causal_claim_guard"],
        "actual_claim_guard": validation["actual_claim_guard"],
        "execution_log": execution_log,
        "notes": [
            "Step 60 validates the mappo_actual plan guard and command order.",
            "Dry-run mode does not execute the real H200 MAPPO matrix.",
            "Historical/replay/noncausal adapters may run actual policy path but cannot support causal performance claims.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or dry-run A-family MAPPO actual matrix from Step 59 execution plan."
    )
    parser.add_argument("--plan-json", required=True, help="Path to Step 59 execution plan JSON")
    parser.add_argument("--report-path", default="", help="Output report JSON path")
    parser.add_argument("--dry-run", action="store_true", help="Validate commands without executing")
    parser.add_argument("--execute", action="store_true", help="Execute commands in guaranteed order")
    parser.add_argument("--cwd", default="", help="Optional working directory for command execution")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.dry_run == args.execute:
        print("[FAIL] Choose exactly one mode: --dry-run or --execute", file=sys.stderr)
        return 2

    plan_path = Path(args.plan_json)
    if not plan_path.exists():
        print(f"[FAIL] plan JSON not found: {plan_path}", file=sys.stderr)
        return 2

    report_path = Path(args.report_path) if args.report_path else (
        plan_path.parent / "a_family_mappo_actual_matrix_runner_report.json"
    )

    cwd = Path(args.cwd) if args.cwd else None
    if cwd is not None and not cwd.exists():
        print(f"[FAIL] cwd not found: {cwd}", file=sys.stderr)
        return 2

    try:
        plan = load_json(plan_path)
        validation = validate_plan(plan)

        execution_log: List[Dict[str, Any]] = []
        if args.execute:
            execution_log = execute_plan(validation, cwd=cwd)

        report = build_report(
            plan_path=plan_path,
            plan=plan,
            validation=validation,
            dry_run=args.dry_run,
            executed=args.execute,
            execution_log=execution_log,
        )
        dump_json(report_path, report)

        print("[OK] Step 60 A-family mappo_actual matrix runner validation PASS")
        print(f"[OK] mode          : {'dry-run' if args.dry_run else 'execute'}")
        print(f"[OK] plan          : {plan_path}")
        print(f"[OK] report        : {report_path}")
        print(f"[OK] run_count     : {validation['run_count']}")
        print(f"[OK] command_count : {validation['command_count']}")
        print(f"[OK] expected_outputs: {len(validation['expected_outputs'])}")
        print("[OK] command order : rollout -> metadata validation -> canonical -> post-canonical validation")
        print("[OK] actual claim guard PASS")
        print("[OK] causal claim guard PASS")
        return 0

    except MatrixRunnerGuardError as exc:
        failure_report = {
            "artifact_version": "a_family_mappo_actual_matrix_runner_v1_step60",
            "created_at_utc": utc_now(),
            "plan_path": str(plan_path),
            "runner_status": "BLOCKED",
            "error": str(exc),
        }
        try:
            dump_json(report_path, failure_report)
        except Exception:
            pass

        print(f"[BLOCKED] {exc}", file=sys.stderr)
        print(f"[BLOCKED] report: {report_path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
