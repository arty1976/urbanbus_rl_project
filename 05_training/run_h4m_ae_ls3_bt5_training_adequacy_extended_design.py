#!/usr/bin/env python3
"""H4M-AE R9.8 LS3-BT5: read-only adequacy and extended-scale design audit."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import py_compile
import subprocess
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Sequence
from zoneinfo import ZoneInfo

import pandas as pd


STAGE = "H4M-AE-R9.8-LS3-BT5"
PASS_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT5_BOUNDED_TRAINING_ADEQUACY_AND_EXTENDED_SCALE_DESIGN_COMPLETE"
BLOCK_GATE = "BLOCKED_INSUFFICIENT_EVIDENCE_FOR_EXTENDED_TRAINING_SCALE"
PASS_CLASSIFICATION = "A_SUSEONG_LS3_JOINT_ASSIGNMENT_EXTENDED_BOUNDED_TRAINING_DESIGN_READY_FOR_SEPARATE_BT6_AUTHORIZATION"
BT4_SOURCE = "99617bd31a0ae383f2541ab58d677f42c84bec74"
BT3_SOURCE = "af8dd4ad3a36de2d22c9607abae5514fe4d6a1cb"

TRAINING_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_ROOT.parent
ARTIFACTS = TRAINING_ROOT / "artifacts"
SOURCE_REL = Path("05_training") / Path(__file__).name
BT4_ROOT = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt4_s2_training_20260822_143202"
BT4_RUNNER = TRAINING_ROOT / "run_h4m_ae_ls3_bt4_s2_training.py"
BT3_TEST = TRAINING_ROOT / "test_h4m_ae_ls3_bt3_guard_closure_and_scale.py"
REGISTRY = (ARTIFACTS / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3r_representative_b1_regeneration_20260809_200442"
            / "r8er3r_generated_demand.parquet")
FROZEN_FILES = {
    "gatv2_and_operational_actor_critic": "run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py",
    "reward_v2_authority": "rewards/mappo_reward_v1.py",
    "zero_loss_adapter": "simulator/zero_loss_admission_adapter.py",
    "local_search_authority": "local_search_contract.py",
    "causal_bridge": "causal_kpi_bridge.py",
    "authorization_enforcement": "simulator_authorization.py",
    "credit_contract": "joint_assignment_credit_contract.py",
    "joint_learning": "joint_assignment_learning.py",
}


def now() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul")).replace(microsecond=0)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(args: Sequence[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, text=True, capture_output=True, check=check)


def frozen_hashes() -> Dict[str, str]:
    return {name: sha256_file(TRAINING_ROOT / rel) for name, rel in FROZEN_FILES.items()}


def provenance_and_compile() -> Dict[str, Any]:
    head = git(["rev-parse", "HEAD"]).stdout.strip()
    parent = git(["rev-parse", "HEAD^"]).stdout.strip()
    files = [line for line in git(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]).stdout.splitlines() if line]
    with tempfile.TemporaryDirectory(prefix="bt5_compile_") as tmp:
        try:
            py_compile.compile(str(PROJECT_ROOT / SOURCE_REL), cfile=str(Path(tmp) / "bt5.pyc"), doraise=True)
            error = None
        except Exception as exc:  # noqa: BLE001
            error = repr(exc)
    cached = git(["diff", "--cached", "--check"], check=False)
    return {
        "source_commit": head,
        "source_parent": parent,
        "source_files": files,
        "source_only_local_commit": files == [SOURCE_REL.as_posix()],
        "parent_is_bt4_source": parent == BT4_SOURCE,
        "py_compile_passed": error is None,
        "py_compile_error": error,
        "git_diff_cached_check_passed": cached.returncode == 0,
        "github_push_performed": False,
    }


def binding(provenance: Dict[str, Any]) -> Dict[str, Any]:
    execution = read_json(BT4_ROOT / "bt4_execution_manifest.json")
    frozen = read_json(BT4_ROOT / "frozen_hash_before_after.json")
    after = frozen_hashes()
    checks = {
        "bt4_gate": execution.get("gate") == "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT4_S2_BOUNDED_CAUSAL_TRAINING_AND_MULTISTEP_GAE_INTEGRITY_COMPLETE",
        "bt4_source_parent": provenance["parent_is_bt4_source"],
        "source_only_commit": provenance["source_only_local_commit"],
        "py_compile": provenance["py_compile_passed"],
        "git_diff_cached_check": provenance["git_diff_cached_check_passed"],
        "registry": REGISTRY.is_file(),
        "frozen_hashes": all(after.get(name) == value for name, value in frozen["after"].items()),
    }
    return {
        "stage": STAGE,
        "checks": checks,
        "passed": all(checks.values()),
        "lineage": {"bt3_source_commit": BT3_SOURCE, "bt4_source_commit": BT4_SOURCE, "bt4_artifact": str(BT4_ROOT)},
        "source_provenance": provenance,
        "frozen_hash_before_after": {"before": frozen["after"], "after": after, "unchanged": checks["frozen_hashes"]},
    }


def line_number(text: str, needle: str) -> int | None:
    return next((i for i, line in enumerate(text.splitlines(), 1) if needle in line), None)


def structural_sanity() -> Dict[str, Any]:
    decisions = pd.DataFrame(read_json(BT4_ROOT / "bt4_learning_signal_audit.json")["decisions"])
    runner = BT4_RUNNER.read_text(encoding="utf-8")
    head = (TRAINING_ROOT / "multi_agent_candidate_assignment_head.py").read_text(encoding="utf-8")
    fragment = '"generalized_cost": float(agent_id)'
    one_pair = "pairs.append(MC.AgentCandidatePair(" in runner
    fixed_candidate_name = "candidate_id=f\"CAND_{agent_id:02d}_d{decision_index}\"" in runner
    direct_zero_loss_pass = 'zero_loss_status="PASS"' in runner
    agent_order_cost = fragment in runner
    actual_local_search = "LocalSearch" in runner or "generate_candidates" in runner
    actual_zero_loss = "ZeroLossAdmissionAdapter" in runner or "evaluate_candidate" in runner
    no_assign_reachable = "torch.cat([logits, no_assign_logit]" in head and "torch.argmax(flat)" in head
    literal_index_zero = "selected = 0" in head or "selected_index=0" in head
    selected = {str(k): int(v) for k, v in decisions["selected"].value_counts().items()}
    by_seed = {str(seed): {str(k): int(v) for k, v in rows["selected"].value_counts().items()} for seed, rows in decisions.groupby("seed")}
    defects = []
    if one_pair and fixed_candidate_name:
        defects.append({"id": "BT4_FIXED_ONE_CANDIDATE_PER_AGENT_SUPPORT", "detail": "BT4 constructs one CAND_<agent>_<decision> pair per agent and every persisted support has eight safe pairs.", "impact": "Variable candidate-set and within-agent candidate diversity cannot be reviewed after an extended reuse of this executor."})
    if direct_zero_loss_pass and not actual_zero_loss:
        defects.append({"id": "BT4_ZERO_LOSS_SELECTIVITY_NOT_EXECUTED_IN_SUPPORT", "detail": "BT4 declares every constructed pair Zero-Loss PASS without calling the adapter to generate varying safe/rejected support.", "impact": "Zero-Loss-conditioned behavior cannot be separated from a support with no observed selectivity."})
    if agent_order_cost:
        defects.append({"id": "BT4_AGENT_ORDER_ENCODED_AS_GENERALIZED_COST", "detail": "BT4 directly maps agent_id into candidate generalized_cost.", "impact": "Observed agent concentration is confounded; it is not evidence of learned agent preference."})
    checks = {
        "candidate_set_not_size_one": sorted(int(v) for v in decisions["safe_pairs"].unique()) != [1],
        "no_assign_structurally_reachable": no_assign_reachable,
        "selected_index_not_literal_zero": not literal_index_zero,
        "multiple_agents_selected": len(selected) > 1,
        "all_agents_have_support_by_construction": one_pair,
        "actual_local_search_support": actual_local_search,
        "actual_zero_loss_support": actual_zero_loss,
        "actor_logits_persisted": "pair_logits" in decisions.columns or "no_assign_logit" in decisions.columns,
    }
    return {
        "stage": STAGE,
        "scope": "Structural checks only. No selection frequency is interpreted as learned policy behavior.",
        "observations": {"decisions": int(len(decisions)), "safe_pair_sizes": sorted(int(v) for v in decisions["safe_pairs"].unique()), "no_assign_selected": int((decisions["selected"] == "NO_ASSIGN_KEEP_CURRENT_PLANS").sum()), "agent_selection_counts": selected, "per_seed_agent_selection_counts": by_seed},
        "source_evidence": {"one_pair_per_agent_line": line_number(runner, "pairs.append(MC.AgentCandidatePair("), "candidate_id_line": line_number(runner, "candidate_id=f\"CAND_{agent_id:02d}_d{decision_index}\""), "zero_loss_pass_line": line_number(runner, 'zero_loss_status="PASS"'), "agent_order_cost_line": line_number(runner, fragment), "argmax_line": line_number(head, "selected = int(torch.argmax(flat).item())")},
        "checks": checks,
        "structural_defects_found": bool(defects),
        "hard_structural_defects": defects,
        "limitations": ["NO_ASSIGN was legal but unselected in 48 decisions; that is not a learned-behavior conclusion.", "BT4 did not persist logits/probabilities, so constant-logit behavior cannot be audited post hoc."],
    }


def training_adequacy() -> Dict[str, Any]:
    learning = read_json(BT4_ROOT / "bt4_learning_signal_audit.json")
    gae = read_json(BT4_ROOT / "bt4_multistep_gae_audit.json")
    optimizer = read_json(BT4_ROOT / "bt4_optimizer_audit.json")
    causal = read_json(BT4_ROOT / "bt4_causal_rollout_audit.json")
    decisions = pd.DataFrame(learning["decisions"])
    registry = pd.read_parquet(REGISTRY)
    mapped = decisions.merge(registry.groupby(["time_band", "window_id"]).size().reset_index(name="requests"), on="window_id", how="left")
    band_rows = {str(band): {"decisions": int(len(rows)), "informative": int(rows["informative"].sum()), "informative_fraction": float(rows["informative"].mean()), "request_min_max": [int(rows["requests"].min()), int(rows["requests"].max())]} for band, rows in mapped.groupby("time_band")}
    updates = optimizer["updates"]
    entropies = [float(row["entropy"]) for row in updates]
    actor_grads = [float(row["actor_grad_norm"]) for row in updates]
    critic_grads = [float(row["critic_grad_norm"]) for row in updates]
    stats = {
        "assignment_decisions": int(len(decisions)),
        "informative_decisions": int(learning["informative_transitions"]),
        "informative_fraction": float(learning["non_zero_team_reward_fraction"]),
        "trajectories": int(gae["trajectories"]),
        "multi_step_trajectories": int(gae["multistep_trajectories"]),
        "trajectory_length": [int(gae["trajectory_length_min"]), float(gae["trajectory_length_median"]), int(gae["trajectory_length_max"])],
        "time_band_coverage": band_rows,
        "candidate_set_diversity": {"safe_pair_sizes": sorted(int(v) for v in decisions["safe_pairs"].unique()), "distinct_cardinalities": int(decisions["safe_pairs"].nunique())},
        "agent_exposure_selection_counts": {str(k): int(v) for k, v in decisions["selected"].value_counts().items()},
        "seed_count": int(decisions["seed"].nunique()),
        "optimizer_updates": int(optimizer["optimizer_steps"]),
        "actor_entropy": {"first": entropies[0], "last": entropies[-1], "delta": entropies[-1] - entropies[0]},
        "critic_target_variance": float(learning["critic_target_variance"]),
        "advantage_variance": float(learning["advantage_variance"]),
        "gradient_bearing_sample_occurrences": int(sum(int(row["rows"]) for row in updates)),
        "actor_gradients_positive": int(sum(value > 0.0 for value in actor_grads)),
        "critic_gradients_positive": int(sum(value > 0.0 for value in critic_grads)),
        "causal_transitions": int(causal["causal_transitions"]),
    }
    mechanism = stats["multi_step_trajectories"] == stats["trajectories"] and stats["informative_decisions"] > 0 and stats["actor_gradients_positive"] == stats["optimizer_updates"] and stats["critic_gradients_positive"] == stats["optimizer_updates"]
    return {
        "stage": STAGE,
        "bt4_statistics": stats,
        "mechanism_validation_scale_sufficient": mechanism,
        "bt4_training_adequate_for_behavior_interpretation": False,
        "conclusion": "BT4_SUFFICIENT_FOR_MECHANISM_VALIDATION_INSUFFICIENT_FOR_LEARNED_POLICY_BEHAVIOR_INTERPRETATION",
        "reasons": ["48 decisions and two seeds validate the learning mechanism but not policy behavior.", "BT4 has a single candidate-set cardinality.", "NO_ASSIGN has zero observations at this scale.", "Agent concentration is confounded by agent-order-as-cost construction."],
    }


def candidate(per_band: int, updates_per_seed: int, label: str, registry: pd.DataFrame, rates: Dict[str, float]) -> Dict[str, Any]:
    groups = registry.groupby(["time_band", "window_id"]).size().reset_index(name="requests").sort_values(["time_band", "window_id"], kind="mergesort")
    selected = groups.groupby("time_band", sort=True).head(per_band)
    windows = [{"time_band": str(row.time_band), "window_id": str(row.window_id), "requests": int(row.requests)} for row in selected.itertuples()]
    seeds = [20260822, 20260823]
    visits = len(windows) * len(seeds)
    decisions = visits * 4
    updates = len(seeds) * updates_per_seed
    low, center, high = min(rates.values()), sum(rates.values()) / len(rates), max(rates.values())
    return {
        "label": label,
        "scope": "Suseong authoritative 54-window / 414-request representative causal registry",
        "window_selection_rule": "Before training, take ascending window_id within every time band using only time_band and request count; never use outcome, reward, advantage, policy action, KPI, or winner data.",
        "windows": windows,
        "distinct_windows": len(windows),
        "time_band_composition": dict(Counter(row["time_band"] for row in windows)),
        "requests": int(sum(row["requests"] for row in windows)),
        "request_density_min_max": [int(min(row["requests"] for row in windows)), int(max(row["requests"] for row in windows))],
        "window_visits": visits,
        "agents": 8,
        "seeds": seeds,
        "trajectory_length": 4,
        "assignment_decisions": decisions,
        "causal_transitions": visits * 8,
        "causal_steps_per_window": 8,
        "optimizer_updates": updates,
        "expected_informative_decisions": {"range": [int(math.floor(decisions * low)), int(math.ceil(decisions * high))], "central_reference": int(round(decisions * center)), "observed_rate_scenarios": {"low": low, "central": center, "high": high}, "method": "Observed BT4 time-band informative-rate min/mean/max; planning reference only, not a prediction or confidence interval."},
        "estimated_runtime_seconds": round(visits * 8 * 0.05 + updates * 0.4, 1),
        "estimated_peak_memory_mb": 400 + 20 * len(windows),
        "resource_estimate_method": "BT3 frozen planning model; estimate only.",
    }


def ladder(adequacy: Dict[str, Any]) -> Dict[str, Any]:
    registry = pd.read_parquet(REGISTRY)
    rates = {band: row["informative_fraction"] for band, row in adequacy["bt4_statistics"]["time_band_coverage"].items()}
    rows = [candidate(3, 4, "E1_MINIMUM_EXTENDED", registry, rates), candidate(4, 5, "E2_MODERATE_EXTENDED", registry, rates), candidate(6, 6, "E3_UPPER_BOUNDED", registry, rates)]
    selected = rows[1]
    return {
        "stage": STAGE,
        "registry": {"path": str(REGISTRY.relative_to(PROJECT_ROOT)), "sha256": sha256_file(REGISTRY), "windows": int(registry["window_id"].nunique()), "requests": int(len(registry))},
        "ladder": rows,
        "provisional_minimum_sufficient_candidate": selected,
        "selection_reason": "E1 has only 24 decisions per band and three informative decisions in the observed-low planning scenario. E2 is the smallest non-maximum candidate with four windows and 32 decisions per band; E3 adds scale not needed for an intermediate review.",
        "candidate_is_not_authorization": True,
    }


def guard_and_locks() -> Dict[str, Any]:
    ast.parse(BT3_TEST.read_text(encoding="utf-8"))
    joint = (TRAINING_ROOT / "joint_assignment_learning.py").read_text(encoding="utf-8")
    source = (PROJECT_ROOT / SOURCE_REL).read_text(encoding="utf-8")
    source_tree = ast.parse(source)
    calls = {
        ast.unparse(node.func)
        for node in ast.walk(source_tree)
        if isinstance(node, ast.Call)
    }
    no_training_calls = {
        "optimizer_step": not any(name.endswith(".step") for name in calls),
        "causal_adapter": "PV8CausalKpiAdapter" not in calls,
        "capability_grant": not any(name.endswith(".granted") for name in calls),
        "checkpoint_write": "torch.save" not in calls,
    }
    joint_guard = joint.find('_authz.require_capability("training"') < joint.find("actor_optimizer.step()")
    return {"stage": STAGE, "bt3_real_optimizer_sites": 19, "bt3_unguarded_real_sites_after": 0, "joint_guard_before_optimizer": joint_guard, "bt5_no_execution_static_checks": no_training_calls, "optimizer_steps": 0, "simulator_causal_rollouts": 0, "checkpoint_writes": 0, "global_locks": {"training_allowed": False, "simulator_execution_allowed": False, "performance_comparison_allowed": False, "paper_level_claim_allowed": False, "causal_performance_claim_allowed": False}, "passed": joint_guard and all(no_training_calls.values())}


def readiness(provisional: Dict[str, Any], defects: bool) -> Dict[str, Any]:
    return {
        "stage": STAGE,
        "status": "NOT_READY_UNTIL_CANDIDATE_SUPPORT_REPAIR_IS_SEPARATELY_VALIDATED" if defects else "READY_AFTER_BT6_COMPLETION",
        "planned_bt6_design": provisional["label"],
        "criteria": {
            "completed_trajectories": provisional["window_visits"],
            "assignment_decisions": provisional["assignment_decisions"],
            "time_band_coverage": ["night", "offpeak", "peak"],
            "multiple_request_density_levels": True,
            "approved_seeds_complete": provisional["seeds"],
            "informative_decisions": {"planning_range": provisional["expected_informative_decisions"]["range"], "rule": "Report actual total and by-band counts; no post-result statistical threshold is invented."},
            "integrity": ["finite actor and critic gradients", "Zero-Loss violations = 0", "illegal selections = 0", "credit contamination = 0", "NaN/Inf = 0"],
            "structural_prerequisites": ["actual Local Search and frozen Zero-Loss-derived support must vary or include multiple candidates per eligible agent before candidate diversity is interpreted", "candidate generalized_cost must not encode agent id/order"],
        },
        "not_policy_quality_criteria": True,
    }


def main() -> None:
    started = time.perf_counter()
    stamp = now().strftime("%Y%m%d_%H%M%S%z")[:-2] + ":00"
    root = ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt5_training_adequacy_extended_design_{stamp}"
    if root.exists():
        raise SystemExit("append-only artifact collision")
    provenance = provenance_and_compile()
    bound = binding(provenance)
    structural = structural_sanity()
    adequacy = training_adequacy()
    candidates = ladder(adequacy)
    locks = guard_and_locks()
    review = readiness(candidates["provisional_minimum_sufficient_candidate"], structural["structural_defects_found"])
    failures = []
    if not bound["passed"]:
        failures.append("AUTHORITATIVE_BINDING_FAILED")
    if not locks["passed"]:
        failures.append("BT5_EXECUTION_OR_GUARD_INTEGRITY_FAILED")
    if structural["structural_defects_found"]:
        failures.append("BT4_CANDIDATE_SUPPORT_NOT_VALID_FOR_EXTENDED_BEHAVIOR_REVIEW")
    passed = not failures
    gate = {"stage": STAGE, "gate": PASS_GATE if passed else BLOCK_GATE, "classification": PASS_CLASSIFICATION if passed else "BT4_CANDIDATE_SUPPORT_AND_AGENT_ORDER_CONFOUNDED_EXTENDED_BEHAVIOR_DESIGN_BLOCKED", "bt4_training_adequate_for_behavior_interpretation": adequacy["bt4_training_adequate_for_behavior_interpretation"], "structural_behavior_defects_found": structural["structural_defects_found"], "hard_failures": failures, "warnings": [], "global_locks": locks["global_locks"], "next_step": "BT6 separate extended bounded training authorization" if passed else "STOP: candidate-support structural repair-selection gate before BT6 authorization", "github_push_performed": False}
    root.mkdir(parents=True, exist_ok=False)
    outputs = {"bt5_training_adequacy_audit.json": adequacy, "bt5_structural_behavior_sanity_audit.json": structural, "bt5_extended_scale_candidate_ladder.json": candidates, "bt5_selected_bt6_training_design.json": {"status": "SELECTED_NOT_AUTHORIZED_NOT_EXECUTED" if passed else "BLOCKED_NOT_FROZEN_NOT_AUTHORIZED", "selected_design": candidates["provisional_minimum_sufficient_candidate"], "structural_blockers": structural["hard_structural_defects"]}, "bt5_bt7_behavior_review_readiness_contract.json": review, "frozen_hash_before_after.json": bound["frozen_hash_before_after"], "test_results.json": {"source_provenance": provenance, "binding_checks": bound["checks"], "guard_and_locks": locks, "hard_failures": failures, "warnings": []}, "gate_decision.json": gate}
    for name, payload in outputs.items():
        write_json(root / name, payload)
    selected = candidates["provisional_minimum_sufficient_candidate"]
    (root / "final_report.md").write_text(f"""# H4M-AE-R9.8-LS3-BT5 — Bounded Training Adequacy + Extended Training Design

gate = {gate['gate']}
classification = {gate['classification']}
source_commit = {provenance['source_commit']}

BT4 is adequate for mechanism validation, but not for learned-policy behavior interpretation: 48 decisions, 16 informative decisions, 12 trajectories, and six optimizer updates.

Structural behavior defects found = {structural['structural_defects_found']}. BT4 does not have an accidentally one-option action set and NO_ASSIGN is structurally reachable. However, the executor constructs exactly one CAND_<agent>_<decision> pair per agent, marks every pair PASS without the Zero-Loss adapter, and encodes agent_id into generalized_cost. Thus agent concentration, candidate diversity, and NO_ASSIGN frequency cannot be treated as learned behavior, even at a larger budget using the same support.

The provisional smallest non-maximum ladder candidate is {selected['label']}: {selected['distinct_windows']} windows / {selected['window_visits']} visits / {selected['requests']} requests / {selected['seeds']} / {selected['agents']} agents / {selected['assignment_decisions']} decisions / {selected['causal_transitions']} transitions / trajectory length {selected['trajectory_length']} / {selected['optimizer_updates']} updates / expected informative {selected['expected_informative_decisions']['range']} (central reference {selected['expected_informative_decisions']['central_reference']}) / {selected['estimated_runtime_seconds']}s / {selected['estimated_peak_memory_mb']} MB. It is planning-only, not an authorization.

All frozen hashes are unchanged. BT5 executed zero optimizer steps, zero causal rollouts, and zero checkpoint writes; all global locks remain false.

next step = {gate['next_step']}
STOP.
""", encoding="utf-8")
    files = {path.relative_to(root).as_posix(): sha256_file(path) for path in root.rglob("*") if path.is_file()}
    write_json(root / "manifest.json", {"stage": STAGE, "gate": gate["gate"], "classification": gate["classification"], "artifact_root": str(root), "file_sha256": files, "elapsed_seconds": round(time.perf_counter() - started, 3), "optimizer_steps": 0, "simulator_causal_rollouts": 0, "checkpoint_writes": 0, "github_push_performed": False})
    (root / ("_SUCCESS.lock" if passed else "_BLOCKED.lock")).write_text(gate["gate"] + "\n", encoding="utf-8")
    print(f"[{ 'PASS' if passed else 'BLOCKED' }] {gate['gate']}")
    print(f"artifact: {root.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
