from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def append_training_dir_to_path() -> Path:
    training_dir = Path(__file__).resolve().parent
    if str(training_dir) not in sys.path:
        sys.path.insert(0, str(training_dir))
    return training_dir


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_cmd(cmd: List[str], *, expect_success: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    if expect_success and proc.returncode != 0:
        raise RuntimeError(
            "command failed\n"
            f"returncode={proc.returncode}\n"
            f"cmd={' '.join(str(x) for x in cmd)}\n"
            f"stdout={proc.stdout}"
        )

    if (not expect_success) and proc.returncode == 0:
        raise RuntimeError(
            "command was expected to fail but succeeded\n"
            f"cmd={' '.join(str(x) for x in cmd)}\n"
            f"stdout={proc.stdout}"
        )

    return proc


def parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_int_csv_list(text: str) -> List[int]:
    return [int(x) for x in parse_csv_list(text)]


def validate_conditions(conditions: List[str]) -> List[str]:
    allowed = {"A", "A90", "A80", "A70"}
    out = []
    for condition in conditions:
        cid = str(condition).strip().upper()
        if cid not in allowed:
            raise RuntimeError(f"unsupported condition_id={condition}; allowed={sorted(allowed)}")
        out.append(cid)
    return out


def make_contract(path: Path, *, conditions: List[str], seeds: List[int]) -> None:
    payload = {
        "artifact_version": "experiment_A_family_mappo_smoke_matrix_contract_v1",
        "phase": "Phase-1-smoke",
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
            "Step 56 smoke matrix contract. Uses mappo_smoke checkpoint validation mode. "
            "Not for actual performance claims."
        ),
    }
    dump_json(path, payload)


def create_smoke_checkpoint(path: Path, *, seed: int = 56) -> Path:
    import torch
    from policies.mappo_checkpoint_builder import save_mappo_checkpoint
    from policies.mappo_neural_inference_adapter import ActorCriticMLP

    path.parent.mkdir(parents=True, exist_ok=True)

    model = ActorCriticMLP(
        actor_obs_dim=16,
        critic_obs_dim=64,
        action_dim=2,
        hidden_dim=128,
    )

    save_mappo_checkpoint(
        path,
        model_state_dict=model.state_dict(),
        training_seed=int(seed),
        git_commit="STEP56_MATRIX_SMOKE_COMMIT",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        trained_model=False,
        performance_claim_allowed=False,
        extra_metadata={
            "selftest": True,
            "matrix_smoke": True,
            "not_for_performance_claims": True,
        },
    )

    return path


def read_parquet_summary(path: Path) -> Dict[str, Any]:
    import pandas as pd

    df = pd.read_parquet(path)
    return {
        "path": str(path),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "condition_ids": sorted(df["condition_id"].astype(str).str.upper().unique().tolist()) if "condition_id" in df.columns else [],
        "seeds": sorted([int(x) for x in df["seed"].unique().tolist()]) if "seed" in df.columns else [],
        "policy_source_counts": df["policy_source"].astype(str).value_counts().to_dict() if "policy_source" in df.columns else {},
        "source_mode_counts": df["source_mode"].astype(str).value_counts().to_dict() if "source_mode" in df.columns else {},
        "actual_policy_claim_ready_count": int(df["actual_policy_claim_ready"].astype(bool).sum()) if "actual_policy_claim_ready" in df.columns else 0,
        "causal_policy_claim_ready_count": int(df["causal_policy_claim_ready"].astype(bool).sum()) if "causal_policy_claim_ready" in df.columns else 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run A/A90/A80/A70 mappo_smoke rollout matrix and canonical KPI smoke"
    )
    parser.add_argument("--conditions", default="A,A90,A80,A70")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--scenario-index", default="")
    parser.add_argument("--scenario-row-index", type=int, default=0)
    parser.add_argument("--simulator-adapter", default="adapters.historical_replay_adapter.HistoricalReplayAdapter")
    parser.add_argument("--skip-canonical", action="store_true")
    return parser.parse_args()


def main() -> int:
    training_dir = append_training_dir_to_path()
    project_root = training_dir.parent

    args = parse_args()

    conditions = validate_conditions(parse_csv_list(args.conditions))
    seeds = parse_int_csv_list(args.seeds)

    output_root = (
        Path(args.output_root)
        if args.output_root
        else project_root / "artifacts" / "experiment_A_v1" / "a_family_mappo_smoke_matrix"
    )

    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else output_root / "checkpoints" / "step56_matrix_smoke_checkpoint.pt"
    rollouts_root = output_root / "rollouts"
    reports_root = output_root / "reports"
    canonical_root = output_root / "canonical_eval"
    contract_path = output_root / "experiment_A_family_mappo_smoke_matrix_contract.json"
    status_path = output_root / "status.json"
    manifest_path = output_root / "matrix_manifest.json"

    runner = training_dir / "run_a_family_policy_rollout_smoke.py"
    metadata_validator = training_dir / "policies" / "validate_window_rollup_policy_metadata.py"
    aggregator = training_dir / "evaluation" / "canonical_kpi_aggregator.py"

    status: Dict[str, Any] = {
        "status": "STARTING",
        "step": 56,
        "conditions": conditions,
        "seeds": seeds,
        "checkpoint_path": str(checkpoint_path),
        "output_root": str(output_root),
    }
    dump_json(status_path, status)

    try:
        if not checkpoint_path.exists():
            create_smoke_checkpoint(checkpoint_path, seed=56)

        make_contract(contract_path, conditions=conditions, seeds=seeds)

        run_records: List[Dict[str, Any]] = []
        validation_records: List[Dict[str, Any]] = []

        for condition in conditions:
            for seed in seeds:
                cmd = [
                    sys.executable,
                    str(runner),
                    "--condition-id",
                    condition,
                    "--policy-source-mode",
                    "mappo_smoke",
                    "--checkpoint-path",
                    str(checkpoint_path),
                    "--checkpoint-validation-mode",
                    "smoke",
                    "--seed",
                    str(int(seed)),
                    "--device",
                    str(args.device),
                    "--simulator-adapter",
                    str(args.simulator_adapter),
                    "--output-root",
                    str(rollouts_root),
                ]

                if args.scenario_index:
                    cmd.extend(["--scenario-index", str(args.scenario_index)])
                    cmd.extend(["--scenario-row-index", str(int(args.scenario_row_index))])

                proc = run_cmd(cmd, expect_success=True)

                run_dir = rollouts_root / f"{condition}_mappo_smoke_seed_{int(seed):03d}"
                window_rollup = run_dir / "window_rollup.parquet"
                run_status = run_dir / "status.json"

                if not window_rollup.exists():
                    raise RuntimeError(f"window_rollup not found after rollout: {window_rollup}")

                validation_report = reports_root / f"{condition}_seed_{int(seed):03d}_window_rollup_policy_metadata_validation.json"
                vproc = run_cmd(
                    [
                        sys.executable,
                        str(metadata_validator),
                        "--input",
                        str(window_rollup),
                        "--json-output",
                        str(validation_report),
                    ],
                    expect_success=True,
                )

                run_records.append(
                    {
                        "condition_id": condition,
                        "seed": int(seed),
                        "run_dir": str(run_dir),
                        "status": str(run_status),
                        "window_rollup": str(window_rollup),
                        "stdout_tail": proc.stdout[-2000:],
                    }
                )

                validation_records.append(
                    {
                        "condition_id": condition,
                        "seed": int(seed),
                        "validation_report": str(validation_report),
                        "stdout_tail": vproc.stdout[-2000:],
                    }
                )

        canonical_summary: Dict[str, Any] = {}

        if not args.skip_canonical:
            cproc = run_cmd(
                [
                    sys.executable,
                    str(aggregator),
                    "--mode",
                    "official_rollup",
                    "--contract",
                    str(contract_path),
                    "--input-root",
                    str(rollouts_root),
                    "--output-root",
                    str(canonical_root),
                    "--smoke",
                ],
                expect_success=True,
            )

            canonical_window = canonical_root / "kpi_by_window.parquet"
            if not canonical_window.exists():
                raise RuntimeError(f"canonical kpi_by_window not found: {canonical_window}")

            post_report = reports_root / "canonical_kpi_by_window_policy_metadata_validation.json"
            post_proc = run_cmd(
                [
                    sys.executable,
                    str(metadata_validator),
                    "--input",
                    str(canonical_window),
                    "--json-output",
                    str(post_report),
                ],
                expect_success=True,
            )

            canonical_summary = read_parquet_summary(canonical_window)

            expected_rows = int(len(conditions) * len(seeds))
            if canonical_summary["rows"] != expected_rows:
                raise RuntimeError(
                    f"canonical row count mismatch: expected={expected_rows}, got={canonical_summary['rows']}"
                )

            if canonical_summary["condition_ids"] != sorted(conditions):
                raise RuntimeError(
                    f"canonical condition_ids mismatch: expected={sorted(conditions)}, got={canonical_summary['condition_ids']}"
                )

            if canonical_summary["seeds"] != sorted(seeds):
                raise RuntimeError(
                    f"canonical seeds mismatch: expected={sorted(seeds)}, got={canonical_summary['seeds']}"
                )

            if canonical_summary["policy_source_counts"] != {"mappo_policy": expected_rows}:
                raise RuntimeError(
                    f"canonical policy_source_counts mismatch: {canonical_summary['policy_source_counts']}"
                )

            if canonical_summary["actual_policy_claim_ready_count"] != 0:
                raise RuntimeError("mappo_smoke matrix must not be actual claim-ready")

            if canonical_summary["causal_policy_claim_ready_count"] != 0:
                raise RuntimeError("mappo_smoke matrix must not be causal claim-ready")

            canonical_summary.update(
                {
                    "canonical_stdout_tail": cproc.stdout[-2000:],
                    "post_canonical_validation_report": str(post_report),
                    "post_canonical_validation_stdout_tail": post_proc.stdout[-2000:],
                }
            )

        manifest = {
            "artifact_version": "a_family_mappo_smoke_matrix_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "step": 56,
            "conditions": conditions,
            "seeds": seeds,
            "run_count": int(len(run_records)),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_validation_mode": "smoke",
            "policy_source_mode": "mappo_smoke",
            "contract": str(contract_path),
            "rollouts_root": str(rollouts_root),
            "reports_root": str(reports_root),
            "canonical_root": str(canonical_root),
            "run_records": run_records,
            "validation_records": validation_records,
            "canonical_summary": canonical_summary,
            "limitations": {
                "trained_model": False,
                "performance_claim_allowed": False,
                "actual_policy_claim_ready": False,
                "causal_policy_claim_ready": False,
                "not_for_performance_claims": True,
                "historical_replay_is_noncausal": True,
            },
        }

        dump_json(manifest_path, manifest)

        status = {
            "status": "PASS",
            "step": 56,
            "conditions": conditions,
            "seeds": seeds,
            "run_count": int(len(run_records)),
            "checkpoint_path": str(checkpoint_path),
            "policy_source_mode": "mappo_smoke",
            "checkpoint_validation_mode": "smoke",
            "canonical_rows": canonical_summary.get("rows"),
            "canonical_condition_ids": canonical_summary.get("condition_ids"),
            "canonical_policy_source_counts": canonical_summary.get("policy_source_counts"),
            "actual_policy_claim_ready_count": canonical_summary.get("actual_policy_claim_ready_count"),
            "causal_policy_claim_ready_count": canonical_summary.get("causal_policy_claim_ready_count"),
            "manifest": str(manifest_path),
        }
        dump_json(status_path, status)

        print("[OK] Step 56 A-family mappo_smoke rollout matrix completed")
        print(f"[OK] conditions       : {conditions}")
        print(f"[OK] seeds            : {seeds}")
        print(f"[OK] run_count        : {len(run_records)}")
        print(f"[OK] checkpoint       : {checkpoint_path}")
        print(f"[OK] rollouts_root    : {rollouts_root}")
        print(f"[OK] canonical_root   : {canonical_root}")
        print(f"[OK] manifest         : {manifest_path}")
        print(f"[OK] status           : {status_path}")
        print("SMOKE PASS")
        return 0

    except Exception as exc:
        status = {
            "status": "FAIL",
            "step": 56,
            "conditions": conditions,
            "seeds": seeds,
            "checkpoint_path": str(checkpoint_path),
            "output_root": str(output_root),
            "error": str(exc),
        }
        dump_json(status_path, status)

        print("[FAIL] Step 56 A-family mappo_smoke rollout matrix failed")
        print(f"[FAIL] status: {status_path}")
        print(f"[FAIL] error : {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
