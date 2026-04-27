from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


PLAN_VERSION = "a_family_mappo_actual_execution_plan_v1"
ALLOWED_CONDITIONS = ["A", "A90", "A80", "A70"]


class MAPPOActualExecutionPlanError(RuntimeError):
    pass


@dataclass(frozen=True)
class MAPPOActualExecutionPlanConfig:
    checkpoint_path: str
    preflight_report_path: str
    output_root: str
    conditions: List[str]
    seeds: List[int]
    device: str = "cpu"
    simulator_adapter: str = "adapters.historical_replay_adapter.HistoricalReplayAdapter"
    scenario_index: str = ""
    scenario_row_index: int = 0
    require_causal_claim: bool = False


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


def parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_seed_csv(text: str) -> List[int]:
    return [int(x) for x in parse_csv_list(text)]


def normalize_conditions(conditions: List[str]) -> List[str]:
    out: List[str] = []
    for condition in conditions:
        cid = str(condition).strip().upper()
        if cid not in ALLOWED_CONDITIONS:
            raise MAPPOActualExecutionPlanError(
                f"unsupported condition_id={condition}; allowed={ALLOWED_CONDITIONS}"
            )
        out.append(cid)

    if not out:
        raise MAPPOActualExecutionPlanError("at least one condition is required")

    return out


def normalize_seeds(seeds: List[int]) -> List[int]:
    out = [int(x) for x in seeds]
    if not out:
        raise MAPPOActualExecutionPlanError("at least one seed is required")
    if len(set(out)) != len(out):
        raise MAPPOActualExecutionPlanError(f"duplicate seeds are not allowed: {out}")
    return out


def is_historical_or_replay_adapter(adapter: str) -> bool:
    text = str(adapter).lower()
    return "historical" in text or "replay" in text


def compare_paths_loose(a: str, b: str) -> bool:
    if not a or not b:
        return False

    pa = str(Path(a))
    pb = str(Path(b))

    if pa == pb:
        return True

    return pa.replace("\\", "/").lower() == pb.replace("\\", "/").lower()


def validate_preflight_report(
    *,
    report_path: Path,
    checkpoint_path: Path,
    require_validator_ran: bool = True,
) -> Dict[str, Any]:
    if not report_path.exists():
        raise MAPPOActualExecutionPlanError(f"preflight report not found: {report_path}")

    report = read_json(report_path)

    blockers: List[str] = []

    if report.get("status") != "PASS":
        blockers.append(f"preflight status must be PASS, got {report.get('status')}")

    if bool(report.get("actual_checkpoint_ready", False)) is not True:
        blockers.append("actual_checkpoint_ready must be true")

    if bool(report.get("actual_policy_claim_ready_candidate", False)) is not True:
        blockers.append("actual_policy_claim_ready_candidate must be true")

    report_checkpoint = str(report.get("checkpoint_path", ""))
    if not compare_paths_loose(str(checkpoint_path), report_checkpoint):
        blockers.append(
            "checkpoint_path does not match preflight report: "
            f"requested={checkpoint_path}, report={report_checkpoint}"
        )

    actual_validator = report.get("actual_validator", None)
    if require_validator_ran:
        if not isinstance(actual_validator, dict):
            blockers.append("actual_validator report is required")
        elif bool(actual_validator.get("passed", False)) is not True:
            blockers.append("actual_validator.passed must be true")

    report_blockers = report.get("blockers", [])
    if report_blockers:
        blockers.append(f"preflight report still contains blockers: {report_blockers}")

    return {
        "valid": len(blockers) == 0,
        "blockers": blockers,
        "report": report,
    }


def build_rollout_command(
    *,
    training_dir: Path,
    output_root: Path,
    condition: str,
    seed: int,
    checkpoint_path: Path,
    device: str,
    simulator_adapter: str,
    scenario_index: str,
    scenario_row_index: int,
) -> List[str]:
    cmd = [
        sys.executable,
        str(training_dir / "run_a_family_policy_rollout_smoke.py"),
        "--condition-id",
        str(condition),
        "--policy-source-mode",
        "mappo_actual",
        "--checkpoint-path",
        str(checkpoint_path),
        "--checkpoint-validation-mode",
        "actual",
        "--seed",
        str(int(seed)),
        "--device",
        str(device),
        "--simulator-adapter",
        str(simulator_adapter),
        "--output-root",
        str(output_root / "rollouts"),
    ]

    if scenario_index:
        cmd.extend(["--scenario-index", str(scenario_index)])
        cmd.extend(["--scenario-row-index", str(int(scenario_row_index))])

    return cmd


def build_validation_command(
    *,
    training_dir: Path,
    rollup_path: Path,
    report_path: Path,
) -> List[str]:
    return [
        sys.executable,
        str(training_dir / "policies" / "validate_window_rollup_policy_metadata.py"),
        "--input",
        str(rollup_path),
        "--require-actual-ready",
        "--json-output",
        str(report_path),
    ]


def build_canonical_command(
    *,
    training_dir: Path,
    contract_path: Path,
    rollouts_root: Path,
    canonical_root: Path,
) -> List[str]:
    return [
        sys.executable,
        str(training_dir / "evaluation" / "canonical_kpi_aggregator.py"),
        "--mode",
        "official_rollup",
        "--contract",
        str(contract_path),
        "--input-root",
        str(rollouts_root),
        "--output-root",
        str(canonical_root),
        "--smoke",
    ]


def build_post_canonical_validation_command(
    *,
    training_dir: Path,
    canonical_window: Path,
    report_path: Path,
) -> List[str]:
    return [
        sys.executable,
        str(training_dir / "policies" / "validate_window_rollup_policy_metadata.py"),
        "--input",
        str(canonical_window),
        "--require-actual-ready",
        "--json-output",
        str(report_path),
    ]


def build_contract_payload(*, conditions: List[str], seeds: List[int]) -> Dict[str, Any]:
    return {
        "artifact_version": "experiment_A_family_mappo_actual_contract_v1",
        "phase": "Phase-1-actual-policy-noncausal-unless-causal-adapter",
        "condition_id": "A_FAMILY",
        "condition_ids": conditions,
        "qwen_train": False,
        "qwen_inference": False,
        "evaluation_horizon_minutes": 30,
        "seeds": seeds,
        "time_bands": ["peak", "offpeak", "night"],
        "shared_kpis": [
            "cv_headway",
            "avg_wait_seconds",
            "bunching_rate",
            "on_time_rate",
            "intervention_rate",
            "energy_proxy",
        ],
        "fairness_constraints": {
            "same_initial_state": True,
            "same_exogenous_events": True,
            "same_eval_window": True,
        },
        "note": (
            "Step 59 plan contract for A-family mappo_actual execution. "
            "Causal claim requires a causal simulator adapter."
        ),
    }


def build_a_family_mappo_actual_execution_plan(
    config: MAPPOActualExecutionPlanConfig,
) -> Dict[str, Any]:
    training_dir = Path(__file__).resolve().parents[1]

    conditions = normalize_conditions(config.conditions)
    seeds = normalize_seeds(config.seeds)

    checkpoint_path = Path(config.checkpoint_path)
    preflight_report_path = Path(config.preflight_report_path)
    output_root = Path(config.output_root)

    if not str(config.checkpoint_path).strip():
        raise MAPPOActualExecutionPlanError("checkpoint_path is required")

    if not str(config.preflight_report_path).strip():
        raise MAPPOActualExecutionPlanError("preflight_report_path is required")

    historical_or_replay = is_historical_or_replay_adapter(config.simulator_adapter)
    if bool(config.require_causal_claim) and historical_or_replay:
        raise MAPPOActualExecutionPlanError(
            "require_causal_claim=true but simulator_adapter is historical/replay"
        )

    preflight = validate_preflight_report(
        report_path=preflight_report_path,
        checkpoint_path=checkpoint_path,
        require_validator_ran=True,
    )

    if not preflight["valid"]:
        raise MAPPOActualExecutionPlanError(
            "preflight report is not acceptable for mappo_actual execution: "
            + "; ".join(preflight["blockers"])
        )

    contract_path = output_root / "experiment_A_family_mappo_actual_contract.json"
    rollouts_root = output_root / "rollouts"
    reports_root = output_root / "reports"
    canonical_root = output_root / "canonical_eval"

    run_matrix: List[Dict[str, Any]] = []

    for condition in conditions:
        for seed in seeds:
            run_dir = rollouts_root / f"{condition}_mappo_actual_seed_{int(seed):03d}"
            rollup_path = run_dir / "window_rollup.parquet"
            validation_report = reports_root / f"{condition}_seed_{int(seed):03d}_actual_window_rollup_policy_metadata_validation.json"

            run_matrix.append(
                {
                    "condition_id": condition,
                    "seed": int(seed),
                    "run_dir": str(run_dir),
                    "window_rollup": str(rollup_path),
                    "rollout_command": build_rollout_command(
                        training_dir=training_dir,
                        output_root=output_root,
                        condition=condition,
                        seed=int(seed),
                        checkpoint_path=checkpoint_path,
                        device=str(config.device),
                        simulator_adapter=str(config.simulator_adapter),
                        scenario_index=str(config.scenario_index or ""),
                        scenario_row_index=int(config.scenario_row_index),
                    ),
                    "metadata_validation_command": build_validation_command(
                        training_dir=training_dir,
                        rollup_path=rollup_path,
                        report_path=validation_report,
                    ),
                    "metadata_validation_report": str(validation_report),
                }
            )

    canonical_window = canonical_root / "kpi_by_window.parquet"
    post_canonical_report = reports_root / "actual_canonical_kpi_by_window_policy_metadata_validation.json"

    plan = {
        "artifact_version": PLAN_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "READY_TO_EXECUTE",
        "checkpoint_path": str(checkpoint_path),
        "preflight_report_path": str(preflight_report_path),
        "preflight_status": preflight["report"].get("status"),
        "conditions": conditions,
        "seeds": seeds,
        "run_count": int(len(run_matrix)),
        "policy_source_mode": "mappo_actual",
        "checkpoint_validation_mode": "actual",
        "device": str(config.device),
        "simulator_adapter": str(config.simulator_adapter),
        "historical_or_replay_adapter": bool(historical_or_replay),
        "actual_policy_claim_candidate": True,
        "causal_policy_claim_candidate": bool(not historical_or_replay and config.require_causal_claim),
        "causal_claim_allowed_by_adapter": bool(not historical_or_replay),
        "require_causal_claim": bool(config.require_causal_claim),
        "output_root": str(output_root),
        "contract_path": str(contract_path),
        "contract_payload": build_contract_payload(conditions=conditions, seeds=seeds),
        "rollouts_root": str(rollouts_root),
        "reports_root": str(reports_root),
        "canonical_root": str(canonical_root),
        "run_matrix": run_matrix,
        "canonical_command": build_canonical_command(
            training_dir=training_dir,
            contract_path=contract_path,
            rollouts_root=rollouts_root,
            canonical_root=canonical_root,
        ),
        "post_canonical_validation_command": build_post_canonical_validation_command(
            training_dir=training_dir,
            canonical_window=canonical_window,
            report_path=post_canonical_report,
        ),
        "post_canonical_validation_report": str(post_canonical_report),
        "limitations": {
            "does_not_train_mappo": True,
            "does_not_execute_matrix": True,
            "historical_replay_is_noncausal": bool(historical_or_replay),
            "causal_claim_requires_future_causal_adapter": bool(historical_or_replay),
        },
    }

    if historical_or_replay:
        plan["warnings"] = [
            "Historical/replay adapter can run the actual policy path but cannot support causal performance claims."
        ]
    else:
        plan["warnings"] = []

    return plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build guarded A-family mappo_actual execution plan"
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--preflight-report", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--conditions", default="A,A90,A80,A70")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--simulator-adapter", default="adapters.historical_replay_adapter.HistoricalReplayAdapter")
    parser.add_argument("--scenario-index", default="")
    parser.add_argument("--scenario-row-index", type=int, default=0)
    parser.add_argument("--require-causal-claim", action="store_true")
    parser.add_argument("--json-output", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        plan = build_a_family_mappo_actual_execution_plan(
            MAPPOActualExecutionPlanConfig(
                checkpoint_path=str(args.checkpoint_path),
                preflight_report_path=str(args.preflight_report),
                output_root=str(args.output_root),
                conditions=parse_csv_list(args.conditions),
                seeds=parse_seed_csv(args.seeds),
                device=str(args.device),
                simulator_adapter=str(args.simulator_adapter),
                scenario_index=str(args.scenario_index or ""),
                scenario_row_index=int(args.scenario_row_index),
                require_causal_claim=bool(args.require_causal_claim),
            )
        )

        output_root = Path(args.output_root)
        output_root.mkdir(parents=True, exist_ok=True)

        contract_path = Path(plan["contract_path"])
        dump_json(contract_path, plan["contract_payload"])

        json_output = Path(args.json_output) if args.json_output else output_root / "a_family_mappo_actual_execution_plan.json"
        dump_json(json_output, plan)

        print("[OK] A-family mappo_actual execution plan READY")
        print(f"[OK] checkpoint       : {plan['checkpoint_path']}")
        print(f"[OK] preflight_report : {plan['preflight_report_path']}")
        print(f"[OK] conditions       : {plan['conditions']}")
        print(f"[OK] seeds            : {plan['seeds']}")
        print(f"[OK] run_count        : {plan['run_count']}")
        print(f"[OK] plan             : {json_output}")
        print(f"[OK] contract         : {contract_path}")
        for warning in plan.get("warnings", []):
            print(f"[WARN] {warning}")
        return 0

    except Exception as exc:
        output_root = Path(args.output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        failure = {
            "artifact_version": PLAN_VERSION,
            "status": "BLOCKED",
            "error": str(exc),
            "checkpoint_path": str(args.checkpoint_path),
            "preflight_report_path": str(args.preflight_report),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        json_output = Path(args.json_output) if args.json_output else output_root / "a_family_mappo_actual_execution_plan_blocked.json"
        dump_json(json_output, failure)

        print("[BLOCKED] A-family mappo_actual execution plan was not created")
        print(f"[BLOCKED] error: {exc}")
        print(f"[BLOCKED] report: {json_output}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
