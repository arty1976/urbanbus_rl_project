#!/usr/bin/env python3
"""H4M-AE post-promotion KPI evaluation scope and protocol freeze.

Read-only selection and freeze.  This program executes no KPI, runs no
inference, trains nothing, reopens no sealed data, mutates nothing and pushes
nothing.  It binds the promoted lineage, enumerates the comparison arms with
their evidenced status, freezes the candidate evaluation scope and the KPI,
aggregation and guardrail definitions from the canonical sources, and decides
whether a fair common four-arm comparison can be constructed today.
"""

from __future__ import annotations

import hashlib
import json
import py_compile
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo


STAGE = "PV8-R2A-R8E-R3-R-H4M-AE"
PASS_GATE = (
    "PASS_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_AE_"
    "POST_PROMOTION_KPI_EVALUATION_SCOPE_AND_PROTOCOL_FREEZE_COMPLETE"
)
BLOCK_GATE = "BLOCKED_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_AE_KPI_PROTOCOL_NOT_COMPARABLE"
NEXT_GATE_ON_PASS = "H4M-AF_POST_PROMOTION_FOUR_KPI_COMPARATIVE_EVALUATION_EXECUTION"
NEXT_GATE_ON_BLOCK = "H4M-AE-R1_KPI_MEASUREMENT_PATH_AVAILABILITY_AUDIT_AND_ARM_CONSTRUCTABILITY_REVIEW"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
SOURCE_REL = Path("05_training") / Path(__file__).name

H4I_R3_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4i_r3_fresh_training_contract_freeze_20260810_183250"
REPRESENTATIVE_REGISTRY = (
    ARTIFACTS_ROOT
    / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3r_representative_b1_regeneration_20260809_200442"
    / "r8er3r_representative_window_registry.parquet"
)
DL5_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl5_suseong_dl4_vs_baseline_kpi_evaluation_20260731_185643"
KPI_AGGREGATOR = "05_training/evaluation/canonical_kpi_aggregator.py"
ENERGY_MODEL = "05_training/rewards/energy_proxy_model_v1.py"
B0C_CONTRACT = "05_training/baselines/b0c_causal_shadow_baseline_contract.json"
CAUSAL_ROLLOUT = "05_training/run_causal_rollout.py"
POLICY_ROLLOUT_SMOKE = "05_training/run_a_family_policy_rollout_smoke.py"

EXPECTED = {
    "h4m_ad_source_commit": "3b9d6d8",
    "h4m_ad_gate": (
        "PASS_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_AD_"
        "SEALED_HOLDOUT_OUTCOME_REVIEW_AND_FINAL_PROMOTION_DECISION_COMPLETE"
    ),
    "promoted_status": "PROMOTED_REPAIRED_POLICY_LINEAGE",
    "final_promotion_contract_sha256": "6dfabf2452d83991cf58e5bf4e456144db744c5cbf313d4fd336aa31b00da48e",
    "r3_split_sha256": "cf7c21c1e85ae8717678fbce85cdbff27ef5e4ca133593de31ad4884aefd476c",
    "seeds": [1, 2, 3],
}

KPI_IDS = [
    "WAITING_TIME_REDUCTION",
    "IN_VEHICLE_TIME_REDUCTION",
    "FLEET_SIZE_REDUCTION",
    "ENERGY_CONSUMPTION_REDUCTION",
]

REQUIRED_ARTIFACTS = [
    "final_report.md",
    "manifest.json",
    "promoted_lineage_binding.json",
    "comparison_arms.json",
    "evaluation_scope.json",
    "evaluation_scope_hash.json",
    "kpi_definitions.json",
    "aggregation_protocol.json",
    "service_guardrails.json",
    "kpi_evaluation_contract.json",
    "gate_matrix.json",
]


def kst_now() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul")).replace(microsecond=0)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def canonical_sha(payload: Any) -> str:
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=jsonable)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=jsonable) + "\n", encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_run(args: Sequence[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, text=True, capture_output=True, check=check)


def latest_artifact(pattern: str, gate: str) -> Path:
    roots = sorted(ARTIFACTS_ROOT.glob(pattern))
    passing = [r for r in roots if (r / "gate_matrix.json").exists() and read_json(r / "gate_matrix.json").get("gate") == gate]
    if not passing:
        raise RuntimeError(f"no passing artifact for {pattern}")
    return passing[-1]


def source_contains(rel: str, token: str) -> bool:
    return token in (PROJECT_ROOT / rel).read_text(encoding="utf-8-sig", errors="replace")


# ---------------------------------------------------------------------------
def promoted_lineage_binding(ad_root: Path) -> Dict[str, Any]:
    payload = read_json(ad_root / "final_promotion_contract.json")
    contract = payload.get("contract", {})
    decision = read_json(ad_root / "final_promotion_decision.json")
    observed = {int(row["seed"]): sha256_file(Path(row["path"])) for row in contract.get("checkpoints", [])}
    checks = {
        "h4m_ad_gate_pass": read_json(ad_root / "gate_matrix.json").get("gate") == EXPECTED["h4m_ad_gate"],
        "status_promoted": decision.get("status") == EXPECTED["promoted_status"],
        "contract_sha_exact": payload.get("final_promotion_contract_sha256") == EXPECTED["final_promotion_contract_sha256"],
        "contract_recomputes_to_same_sha": canonical_sha(contract) == EXPECTED["final_promotion_contract_sha256"],
        "three_checkpoints_unchanged": len(contract.get("checkpoints", [])) == 3
        and all(observed[int(row["seed"])] == row["sha256"] for row in contract.get("checkpoints", [])),
        "sealed_holdout_marked_consumed": contract.get("post_promotion_constraints", {}).get("sealed_holdout_is_now_consumed") is True,
        "deployment_claims_not_authorised": contract.get("post_promotion_constraints", {}).get("deployment_or_kpi_claims_not_authorised") is True,
    }
    return {
        "stage": STAGE,
        "h4m_ad_artifact": str(ad_root),
        "final_promotion_contract_sha256": EXPECTED["final_promotion_contract_sha256"],
        "promoted_checkpoints": [
            {"seed": row["seed"], "path": row["path"], "sha256": row["sha256"], "current_sha256": observed[int(row["seed"])]}
            for row in contract.get("checkpoints", [])
        ],
        "implementation_lineage": {key: value for key, value in contract.items() if key.endswith("_sha256")},
        "test6_status": "EXHAUSTED_SEALED_HOLDOUT",
        "test6_reuse_permitted": False,
        "checks": checks,
        "passed": all(checks.values()),
    }


def comparison_arms() -> Dict[str, Any]:
    dl5_conditions = read_json(DL5_ROOT / "condition_registry.json")["conditions"]
    dl5_scope = read_json(DL5_ROOT / "evaluation_scope.json")
    b0c = read_json(PROJECT_ROOT / B0C_CONTRACT)
    causal_rollout_has_checkpoint = source_contains(CAUSAL_ROLLOUT, "checkpoint_path")
    smoke_uses_step_result = source_contains(POLICY_ROLLOUT_SMOKE, "step_result=") or source_contains(
        POLICY_ROLLOUT_SMOKE, "step_result["
    )
    arms = {
        "A_PROMOTED_REPAIRED_POLICY": {
            "role": "promoted repaired MAPPO policy lineage",
            "binding": "H4M-AD final promotion contract; three checkpoints with fixed SHA256",
            "kpi_execution_path_available": False,
            "evidence": {
                "canonical_kpi_rollout_runner": CAUSAL_ROLLOUT,
                "canonical_runner_loads_checkpoint": causal_rollout_has_checkpoint,
                "checkpoint_capable_runner": POLICY_ROLLOUT_SMOKE,
                "checkpoint_capable_runner_uses_simulator_step_result_for_kpis": smoke_uses_step_result,
                "note": "the checkpoint-capable runner synthesises headway, wait and bunching from the non-zero action count and discards the adapter step result, so its window_rollup is a pipeline smoke, not a measurement",
            },
            "status": "NO_VALID_KPI_MEASUREMENT_PATH_ON_THE_PROMOTED_SCOPE",
        },
        "B0_HISTORICAL_REFERENCE": {
            "role": "historical/current-operations reference",
            "binding": f"DL-5 condition registry entry {dl5_conditions[1]['condition_id']}",
            "executed_scope": dl5_scope.get("evaluation_timestamps_source"),
            "executed_snapshot_count": dl5_scope.get("test_snapshot_count"),
            "same_scope_as_promoted_lineage": False,
            "status": "EXECUTED_ON_A_DIFFERENT_SPLIT_LINEAGE",
        },
        "B0C_CAUSAL_SHADOW": {
            "role": "causal shadow historical baseline",
            "binding": B0C_CONTRACT,
            "contract_status": b0c.get("contract_status"),
            "status": "CONTRACT_ONLY_NEVER_EXECUTED",
        },
        "B1_NOOP": {
            "role": "no-op baseline",
            "binding": f"DL-5 condition registry entry {dl5_conditions[2]['condition_id']}",
            "executed_scope": dl5_scope.get("evaluation_timestamps_source"),
            "same_scope_as_promoted_lineage": False,
            "status": "EXECUTED_ON_A_DIFFERENT_SPLIT_LINEAGE",
        },
        "B2_RULE_BASED": {
            "role": "rule-based calibrated proxy",
            "binding": f"DL-5 condition registry entry {dl5_conditions[3]['condition_id']}",
            "executed_scope": dl5_scope.get("evaluation_timestamps_source"),
            "same_scope_as_promoted_lineage": False,
            "status": "EXECUTED_ON_A_DIFFERENT_SPLIT_LINEAGE",
        },
    }
    checks = {
        "all_four_required_arm_roles_identified": True,
        "promoted_arm_has_kpi_execution_path": arms["A_PROMOTED_REPAIRED_POLICY"]["kpi_execution_path_available"],
        "baseline_arms_share_promoted_scope": all(
            arms[name].get("same_scope_as_promoted_lineage") is True for name in ("B0_HISTORICAL_REFERENCE", "B1_NOOP", "B2_RULE_BASED")
        ),
        "b0c_executed": b0c.get("contract_status") not in {"DRAFT_LOCKED_NOT_EXECUTED", None},
        "no_cross_simulator_mixing_required": False,
    }
    return {
        "stage": STAGE,
        "arms": arms,
        "dl5_reference": {
            "artifact": DL5_ROOT.name,
            "conditions": dl5_conditions,
            "scope": dl5_scope,
            "causal_comparison_allowed": dl5_scope.get("causal_comparison_allowed"),
            "claim_guard": dl5_scope.get("claim_guard"),
        },
        "mixing_prohibition": "the DL-3/DL-4 holdout arms and the PV8 R3-R promoted lineage are different data ranges and must not be compared across scopes",
        "checks": checks,
        "passed": all(checks.values()),
    }


def evaluation_scope() -> Dict[str, Any]:
    import pandas as pd  # noqa: PLC0415 - registry metadata only

    split = read_json(H4I_R3_ROOT / "02_seed_split_frozen_contract.json")["ordered_window_ids"]
    registry = pd.read_parquet(REPRESENTATIVE_REGISTRY)
    registry_ids = sorted(str(value) for value in registry["window_id"].tolist())
    train = sorted(str(value) for value in split["train"])
    validation = sorted(str(value) for value in split["validation"])
    sealed = sorted(str(value) for value in split["test"])
    available = sorted(set(train) | set(validation))
    unused = sorted(set(registry_ids) - set(train) - set(validation) - set(sealed))
    scope = {
        "scope_id": "PV8_R3R_NON_TEST6_EVALUATION_SCOPE_V1",
        "window_ids_ordered": available,
        "window_count": len(available),
        "composition": {"train_windows": len(train), "validation_windows": len(validation)},
        "training_data_reused": True,
        "validation_data_reused": True,
        "exhausted_test6_excluded": True,
        "excluded_test6_window_ids": sealed,
        "unused_windows_available_in_registry": unused,
        "governing_split_sha256": EXPECTED["r3_split_sha256"],
        "agent_count": 8,
        "evaluation_horizon_minutes": 30,
        "common_conditions_required": "identical windows, demand and initial conditions, identical agent count and identical simulator causal semantics for every arm",
        "future_leakage": 0,
    }
    checks = {
        "test6_excluded": not (set(available) & set(sealed)),
        "scope_non_empty": len(available) > 0,
        "registry_fully_partitioned_by_split": not unused,
        "reuse_declared_explicitly": scope["training_data_reused"] is True and scope["validation_data_reused"] is True,
        "no_uncontaminated_windows_remain": not unused,
    }
    return {"stage": STAGE, "scope": scope, "checks": checks, "passed": checks["test6_excluded"] and checks["scope_non_empty"]}


def kpi_definitions() -> Dict[str, Any]:
    aggregator_text = (PROJECT_ROOT / KPI_AGGREGATOR).read_text(encoding="utf-8-sig", errors="replace")
    in_vehicle_tokens = ["in_vehicle", "ride_time", "alight_ts", "board_to_alight", "onboard_seconds"]
    in_vehicle_supported = any(token in aggregator_text for token in in_vehicle_tokens)
    definitions = {
        "WAITING_TIME_REDUCTION": {
            "measurable": True,
            "canonical_field": "avg_wait_seconds",
            "numerator": "wait_total_passenger_seconds",
            "denominator": "wait_passenger_count",
            "unit": "seconds per waiting passenger",
            "passenger_weighted": True,
            "window_aggregation": "passenger-second totals summed per window, then divided by the window passenger count",
            "seed_aggregation": "per seed first, then unweighted mean across the three seeds",
            "missing_or_zero_handling": "windows with zero waiting passengers contribute no passenger-seconds and are excluded from the ratio denominator",
            "direction_of_improvement": "LOWER_IS_BETTER",
            "tail_metric": {
                "field": "passenger_wait_p95_seconds",
                "reported_separately": True,
                "caveat": "the canonical aggregator falls back to avg_wait_seconds * 1.65 when no measured p95 column exists; the fallback must be reported as derived, never as a measured tail",
            },
            "source": KPI_AGGREGATOR,
        },
        "IN_VEHICLE_TIME_REDUCTION": {
            "measurable": False,
            "status": "NOT_YET_MEASURABLE",
            "reason": "no boarding-to-alighting elapsed time exists in the canonical KPI schema, the window rollup contract or the Reward V2 transition fields; alighting appears only as a Zero-Loss integrity counter, not as a timestamp",
            "searched_tokens": in_vehicle_tokens,
            "substitute_metric_used": False,
            "required_to_become_measurable": "an approved alighting timestamp or in-vehicle elapsed-time field emitted by the simulator and added to the canonical KPI contract",
        },
        "FLEET_SIZE_REDUCTION": {
            "measurable": True,
            "canonical_field": "fleet_reduction_ratio",
            "numerator": "baseline_bus_count - active_bus_count",
            "denominator": "baseline_bus_count",
            "unit": "ratio in [0, 1]",
            "passenger_weighted": False,
            "window_aggregation": "computed per window from the active and baseline bus counts, then averaged unweighted across windows",
            "seed_aggregation": "per seed first, then unweighted mean across the three seeds",
            "missing_or_zero_handling": "baseline_bus_count defaults to 8 and non-positive values are replaced by 8; the ratio is clipped to [0, 1]",
            "direction_of_improvement": "HIGHER_RATIO_IS_BETTER",
            "reduction_formula_note": "this KPI is already expressed as a reduction ratio, so the generic (baseline - A)/baseline percentage is not applied twice",
            "service_requirement_caveat": "a fleet reduction only counts if the frozen service guardrails hold on the same window",
            "source": KPI_AGGREGATOR,
        },
        "ENERGY_CONSUMPTION_REDUCTION": {
            "measurable": True,
            "canonical_field": "energy_proxy_total and energy_proxy_per_passenger",
            "numerator": "energy_kwh_equiv from distance, acceleration events and hold seconds",
            "denominator": "passenger_served_count for the per-passenger form",
            "unit": "kWh-equivalent, and kWh-equivalent per served passenger",
            "passenger_weighted": True,
            "window_aggregation": "energy totals summed per window; the per-passenger form divides by served passengers in the same window",
            "seed_aggregation": "per seed first, then unweighted mean across the three seeds",
            "missing_or_zero_handling": "windows with zero served passengers report the total only and are excluded from the per-passenger mean",
            "direction_of_improvement": "LOWER_IS_BETTER",
            "model": {
                "implementation": ENERGY_MODEL,
                "coefficients": ["k_dist_kwh_per_m", "k_acc_kwh_per_event", "k_idle_kwh_per_sec"],
                "status": "canonical approved energy proxy, not a physical measurement",
            },
            "source": KPI_AGGREGATOR,
        },
    }
    checks = {
        "all_four_kpis_resolved_or_marked": sorted(definitions) == sorted(KPI_IDS),
        "no_new_kpi_invented": all(
            row.get("source") == KPI_AGGREGATOR or row.get("measurable") is False for row in definitions.values()
        ),
        "in_vehicle_not_substituted": definitions["IN_VEHICLE_TIME_REDUCTION"]["substitute_metric_used"] is False,
        "in_vehicle_absence_verified_in_source": not in_vehicle_supported,
    }
    return {
        "stage": STAGE,
        "definitions": definitions,
        "measurable_count": sum(1 for row in definitions.values() if row.get("measurable")),
        "not_yet_measurable": [name for name, row in definitions.items() if not row.get("measurable")],
        "checks": checks,
        "passed": all(checks.values()),
    }


def aggregation_protocol() -> Dict[str, Any]:
    body = {
        "comparison_formula": {
            "absolute_delta": "A - baseline",
            "reduction_pct": "(baseline - A) / baseline * 100",
            "applies_to": ["WAITING_TIME_REDUCTION", "ENERGY_CONSUMPTION_REDUCTION"],
            "excluded": {
                "FLEET_SIZE_REDUCTION": "already a canonical reduction ratio; report the ratio difference, not a percentage of a percentage",
                "IN_VEHICLE_TIME_REDUCTION": "NOT_YET_MEASURABLE",
            },
            "zero_baseline_handling": "if a baseline value is zero the percentage is undefined and must be reported as null, never as zero or infinity",
        },
        "seed_protocol": {
            "checkpoints_used": 3,
            "seed_exclusion_permitted": False,
            "best_seed_selection_permitted": False,
            "report_order": "per seed first, then aggregate",
            "aggregate_rule": "unweighted mean across the three seeds, with the median reported alongside",
            "dispersion": "report the per-seed spread as min, max and standard deviation; no significance threshold is defined",
        },
        "statistical_rule": {
            "significance_threshold_defined_here": None,
            "threshold_creation_after_results_forbidden": True,
            "reason": "no pre-existing frozen significance criterion was found in the canonical evaluation lineage, so none is invented; results are reported descriptively",
        },
        "dataset_separation": {
            "test6_reuse": False,
            "cross_scope_pooling": False,
            "training_diagnostics_as_kpi_evidence": False,
        },
    }
    return {"stage": STAGE, "protocol": body, "protocol_sha256": canonical_sha(body), "passed": True}


def service_guardrails() -> Dict[str, Any]:
    body = {
        "source": "existing frozen integrity criteria carried from the H4M validation and sealed lineage; no new guardrail is invented",
        "hard_guardrails": {
            "illegal_action_count": 0,
            "k_mask_violation_count": 0,
            "zero_loss_violation_count": 0,
            "future_leakage_count": 0,
            "nan_inf_count": 0,
            "parameter_mutation_count": 0,
        },
        "behavioural_guardrails": {
            "service_collapse": "a KPI gain obtained by refusing service is not an improvement; passenger_service_rate must be reported next to every KPI",
            "unexplained_single_action_collapse": "the dominant action must not be identical across both target contexts without an evidenced explanation",
        },
        "interpretation_rule": "a KPI improvement purchased with any guardrail violation is not a promotion success and must be reported as a violation, not as a trade-off",
    }
    return {"stage": STAGE, "guardrails": body, "guardrails_sha256": canonical_sha(body), "passed": True}


def comparability_decision(arms: Mapping[str, Any], scope: Mapping[str, Any], kpis: Mapping[str, Any]) -> Dict[str, Any]:
    blockers = []
    if not arms["arms"]["A_PROMOTED_REPAIRED_POLICY"]["kpi_execution_path_available"]:
        blockers.append(
            {
                "id": "A_ARM_HAS_NO_VALID_KPI_MEASUREMENT_PATH",
                "detail": "the canonical KPI rollout runner drives placeholder A-family policies and never loads a checkpoint, while the checkpoint-capable runner synthesises headway, wait and bunching from the non-zero action count and discards the simulator step result",
                "consequence": "any A-arm KPI produced today would be a mechanical function of action count, not a measurement",
            }
        )
    if not arms["checks"]["baseline_arms_share_promoted_scope"]:
        blockers.append(
            {
                "id": "BASELINE_ARMS_EXECUTED_ON_A_DIFFERENT_SPLIT",
                "detail": "B0, B1 and B2 exist only as DL-3/DL-4 holdout executions with causal_comparison_allowed=false; none has been executed on the PV8 R3-R window set",
                "consequence": "comparing them against the promoted lineage would mix data ranges, which the protocol forbids",
            }
        )
    if not arms["checks"]["b0c_executed"]:
        blockers.append(
            {
                "id": "B0C_NEVER_EXECUTED",
                "detail": "the causal shadow baseline contract is DRAFT_LOCKED_NOT_EXECUTED",
                "consequence": "no causal-semantics reference exists for the promoted scope",
            }
        )
    if scope["scope"]["unused_windows_available_in_registry"] == []:
        blockers.append(
            {
                "id": "NO_UNCONTAMINATED_EVALUATION_WINDOWS_REMAIN",
                "detail": "the representative registry holds exactly 54 windows, partitioned into 44 train, 4 validation and 6 now-exhausted TEST6 windows",
                "consequence": "any KPI evaluation must reuse training or validation windows, which must be declared and cannot substitute for unseen data",
            }
        )
    comparable = not blockers
    return {
        "stage": STAGE,
        "fair_common_comparison_constructible_today": comparable,
        "blockers": blockers,
        "not_yet_measurable_kpis": kpis["not_yet_measurable"],
        "decision": "KPI_PROTOCOL_FROZEN_AND_COMPARABLE" if comparable else "KPI_PROTOCOL_FROZEN_BUT_NOT_YET_COMPARABLE",
        "explicitly_not_done": [
            "no proxy KPI definition was invented",
            "no TEST6 reuse was proposed",
            "no KPI was executed",
            "no arm was compared across different data ranges",
        ],
    }


def kpi_evaluation_contract(binding, arms, scope, kpis, protocol, guardrails, decision) -> Dict[str, Any]:
    body = {
        "promoted_lineage": {
            "status": EXPECTED["promoted_status"],
            "final_promotion_contract_sha256": EXPECTED["final_promotion_contract_sha256"],
            "checkpoints": binding["promoted_checkpoints"],
        },
        "test6": {"status": "EXHAUSTED_SEALED_HOLDOUT", "reuse_permitted": False},
        "comparison_arms": {name: {"role": row["role"], "binding": row["binding"], "status": row["status"]} for name, row in arms["arms"].items()},
        "evaluation_scope": scope["scope"],
        "evaluation_scope_sha256": canonical_sha(scope["scope"]),
        "kpi_definitions": kpis["definitions"],
        "kpi_definitions_sha256": canonical_sha(kpis["definitions"]),
        "aggregation_protocol": protocol["protocol"],
        "service_guardrails": guardrails["guardrails"],
        "comparability": {
            "constructible_today": decision["fair_common_comparison_constructible_today"],
            "blockers": decision["blockers"],
        },
        "execution_authorised": decision["fair_common_comparison_constructible_today"],
        "scientific_rule": "the sealed 144/144 result is evidence of policy action discrimination generalising to unseen sealed windows; it is not evidence of any operational KPI improvement",
    }
    return {"stage": STAGE, "contract": body, "kpi_evaluation_contract_sha256": canonical_sha(body)}


def source_provenance(created_at: str) -> Dict[str, Any]:
    head = git_run(["rev-parse", "HEAD"]).stdout.strip()
    head_files = [line for line in git_run(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]).stdout.splitlines() if line]
    status = git_run(["status", "--short"]).stdout.strip()
    with tempfile.TemporaryDirectory(prefix="h4mae_") as tmp:
        try:
            py_compile.compile(str(PROJECT_ROOT / SOURCE_REL), cfile=str(Path(tmp) / "ae.pyc"), doraise=True)
            error = None
        except Exception as exc:
            error = repr(exc)
    return {
        "stage": STAGE,
        "created_at": created_at,
        "source_commit": head,
        "head_commit_files": head_files,
        "source_only_local_commit": head_files == [SOURCE_REL.as_posix()] and status == "",
        "clean_worktree": status == "",
        "py_compile_passed": error is None,
        "github_push_performed": False,
    }


def gate_matrix(provenance, binding, arms, scope, kpis, protocol, guardrails, decision, contract) -> Dict[str, Any]:
    comparable = decision["fair_common_comparison_constructible_today"]
    criteria = {
        "source_only_commit": provenance.get("source_only_local_commit") is True,
        "promoted_lineage_exact_binding": binding.get("passed") is True,
        "test6_excluded": scope["checks"]["test6_excluded"] is True and binding["test6_reuse_permitted"] is False,
        "common_evaluation_scope_frozen": scope.get("passed") is True and bool(contract["contract"]["evaluation_scope_sha256"]),
        "all_comparison_arms_bound": arms.get("passed") is True,
        "four_kpi_definitions_resolved_or_marked": kpis.get("passed") is True,
        "aggregation_rules_frozen": protocol.get("passed") is True,
        "service_guardrails_frozen": guardrails.get("passed") is True,
        "no_kpi_executed": True,
        "no_proxy_invented": decision["not_yet_measurable_kpis"] == ["IN_VEHICLE_TIME_REDUCTION"]
        and kpis["checks"]["in_vehicle_not_substituted"] is True,
        "github_push_false": provenance.get("github_push_performed") is False,
    }
    passed = all(criteria.values()) and comparable
    return {
        "stage": STAGE,
        "gate": PASS_GATE if passed else BLOCK_GATE,
        "decision": decision["decision"],
        "fair_common_comparison_constructible_today": comparable,
        "kpi_evaluation_contract_sha256": contract.get("kpi_evaluation_contract_sha256"),
        "evaluation_scope_sha256": contract["contract"]["evaluation_scope_sha256"],
        "exact_next_gate": NEXT_GATE_ON_PASS if passed else NEXT_GATE_ON_BLOCK,
        "next_gate_auto_execution": False,
        "criteria": criteria,
        "failing_criteria": [key for key, value in criteria.items() if not value],
        "blockers": decision["blockers"],
        "final_flags": {
            "training_count": 0,
            "optimizer_step_count": 0,
            "kpi_executed": False,
            "TEST6_reopened": False,
            "proxy_definition_invented": False,
            "github_push_performed": False,
        },
    }


def make_manifest(root: Path, gate, provenance, contract, started) -> Dict[str, Any]:
    files = {p.relative_to(root).as_posix(): str(p) for p in root.rglob("*") if p.is_file() and p.name != "manifest.json"}
    return {
        "stage": STAGE,
        "artifact_root": str(root),
        "source_commit": provenance.get("source_commit"),
        "gate": gate.get("gate"),
        "decision": gate.get("decision"),
        "kpi_evaluation_contract_sha256": contract.get("kpi_evaluation_contract_sha256"),
        "evaluation_scope_sha256": gate.get("evaluation_scope_sha256"),
        "exact_next_gate": gate.get("exact_next_gate"),
        "required_artifacts_present": all((root / name).exists() for name in REQUIRED_ARTIFACTS if name != "manifest.json"),
        "immutable_evidence_sha256": {name: sha256_file(Path(path)) for name, path in files.items()},
        "mutable_lifecycle_state_files": [],
        "append_only_artifact": True,
        "read_only_freeze_stage": True,
        "elapsed_seconds": time.perf_counter() - started,
        "kpi_executed": False,
        "TEST6_reopened": False,
        "training_count": 0,
        "github_push_performed": False,
    }


def final_report(provenance, binding, arms, scope, kpis, protocol, guardrails, decision, contract, gate) -> str:
    return f"""# H4M-AE Post-Promotion KPI Evaluation Scope and Protocol Freeze

gate = {gate['gate']}
decision = {gate['decision']}
kpi_evaluation_contract_sha256 = {gate['kpi_evaluation_contract_sha256']}
evaluation_scope_sha256 = {gate['evaluation_scope_sha256']}
source_commit = {provenance['source_commit']}
kpi_executed = false
TEST6 = EXHAUSTED_SEALED_HOLDOUT, reuse forbidden
exact_next_gate = {gate['exact_next_gate']} (not executed automatically)

## Scientific rule

The sealed 144/144 result is evidence that the promoted policy's action discrimination generalises to
unseen sealed windows. It is not evidence of waiting-time, in-vehicle-time, fleet or energy improvement.
Those must be measured separately.

## Comparison arms

```json
{json.dumps({name: {'role': row['role'], 'status': row['status']} for name, row in arms['arms'].items()}, ensure_ascii=False, indent=2, default=jsonable)}
```

## Evaluation scope

```json
{json.dumps({k: v for k, v in scope['scope'].items() if k != 'window_ids_ordered'}, ensure_ascii=False, indent=2, default=jsonable)}
```

## KPI definitions

```json
{json.dumps({name: {'measurable': row.get('measurable'), 'canonical_field': row.get('canonical_field'), 'direction_of_improvement': row.get('direction_of_improvement'), 'status': row.get('status'), 'reason': row.get('reason')} for name, row in kpis['definitions'].items()}, ensure_ascii=False, indent=2, default=jsonable)}
```

## Aggregation and guardrails

```json
{json.dumps({'comparison_formula': protocol['protocol']['comparison_formula'], 'seed_protocol': protocol['protocol']['seed_protocol'], 'statistical_rule': protocol['protocol']['statistical_rule'], 'hard_guardrails': guardrails['guardrails']['hard_guardrails']}, ensure_ascii=False, indent=2, default=jsonable)}
```

## Comparability decision

```json
{json.dumps({'constructible_today': decision['fair_common_comparison_constructible_today'], 'blockers': decision['blockers'], 'not_yet_measurable': decision['not_yet_measurable_kpis'], 'explicitly_not_done': decision['explicitly_not_done']}, ensure_ascii=False, indent=2, default=jsonable)}
```

STOP: no KPI was executed, no proxy definition was invented, TEST6 was not reused, nothing was trained or
tuned, and nothing was pushed. The scope, KPI definitions, aggregation rules and guardrails above are frozen
and hashed so that a later execution cannot silently redefine them.
"""


def main() -> None:
    started = time.perf_counter()
    created_at = kst_now().isoformat()
    stamp = kst_now().strftime("%Y%m%d_%H%M%S%z")[:-2] + ":00"
    root = ARTIFACTS_ROOT / f"pv8_r2a_r8e_r3_r_h4m_ae_post_promotion_kpi_scope_protocol_freeze_{stamp}"
    root.mkdir(parents=True, exist_ok=True)

    ad_root = latest_artifact("pv8_r2a_r8e_r3_r_h4m_ad_*", EXPECTED["h4m_ad_gate"])
    provenance = source_provenance(created_at)
    binding = promoted_lineage_binding(ad_root)
    arms = comparison_arms()
    scope = evaluation_scope()
    kpis = kpi_definitions()
    protocol = aggregation_protocol()
    guardrails = service_guardrails()
    decision = comparability_decision(arms, scope, kpis)
    contract = kpi_evaluation_contract(binding, arms, scope, kpis, protocol, guardrails, decision)
    gate = gate_matrix(provenance, binding, arms, scope, kpis, protocol, guardrails, decision, contract)

    for name, payload in {
        "promoted_lineage_binding.json": binding,
        "comparison_arms.json": arms,
        "evaluation_scope.json": scope,
        "evaluation_scope_hash.json": {
            "scope_id": scope["scope"]["scope_id"],
            "evaluation_scope_sha256": contract["contract"]["evaluation_scope_sha256"],
            "window_count": scope["scope"]["window_count"],
            "window_ids_ordered": scope["scope"]["window_ids_ordered"],
            "excluded_test6_window_ids": scope["scope"]["excluded_test6_window_ids"],
        },
        "kpi_definitions.json": kpis,
        "aggregation_protocol.json": protocol,
        "service_guardrails.json": guardrails,
        "comparability_decision.json": decision,
        "kpi_evaluation_contract.json": contract,
        "source_provenance.json": provenance,
        "gate_matrix.json": gate,
    }.items():
        write_json(root / name, payload)
    (root / "final_report.md").write_text(
        final_report(provenance, binding, arms, scope, kpis, protocol, guardrails, decision, contract, gate), encoding="utf-8"
    )
    write_json(root / "manifest.json", make_manifest(root, gate, provenance, contract, started))

    print(f"[H4M-AE] artifact root: {root}")
    print(f"[H4M-AE] gate: {gate['gate']}")
    print(f"[H4M-AE] decision: {gate['decision']}")
    print(f"[H4M-AE] kpi_evaluation_contract_sha256: {gate['kpi_evaluation_contract_sha256']}")
    print(f"[H4M-AE] evaluation_scope_sha256: {gate['evaluation_scope_sha256']}")
    print(f"[H4M-AE] not yet measurable: {decision['not_yet_measurable_kpis']}")
    print(f"[H4M-AE] blockers: {[row['id'] for row in decision['blockers']]}")
    print(f"[H4M-AE] failing criteria: {gate['failing_criteria']}")
    print(f"[H4M-AE] exact next gate: {gate['exact_next_gate']} (not executed)")


if __name__ == "__main__":
    main()
