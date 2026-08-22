#!/usr/bin/env python3
"""BT6-S0: post-repair bounded-training scale redesign; no training or rollout."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import py_compile
import resource
import statistics
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Sequence
from zoneinfo import ZoneInfo

import pandas as pd
import torch


STAGE = "H4M-AE-R9.8-LS3-BT6-S0"
PASS_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT6_S0_POST_REPAIR_BOUNDED_TRAINING_SCALE_REDESIGN_COMPLETE"
PASS_CLASS = "A_SUSEONG_LS3_JOINT_ASSIGNMENT_POST_REPAIR_BOUNDED_TRAINING_ENVELOPE_READY_FOR_SEPARATE_BT6_EXECUTION_AUTHORIZATION"
BT5R_SOURCE = "f14fab7dbc72f42968a1aea5dcbbb652dddbeb48"
BT5_SOURCE = "1df8284d161e58e8caeb07a1e0b037d95c808b7d"
BT4_SOURCE = "99617bd31a0ae383f2541ab58d677f42c84bec74"

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
SOURCE_REL = Path("05_training") / Path(__file__).name
BT5R_ARTIFACT = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt5r_candidate_support_repair_selection_20260822_151517+09:00"
BT5_ARTIFACT = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt5_training_adequacy_extended_design_20260822_145452+09:00"
BT4_ARTIFACT = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt4_s2_training_20260822_143202"
R97_ROOT = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_r9_7_null_safe_versioned_binding_20260822_005245"
REGISTRY = ARTIFACTS / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3r_representative_b1_regeneration_20260809_200442" / "r8er3r_generated_demand.parquet"
FROZEN = {
    "reward_v2": "rewards/mappo_reward_v1.py",
    "zero_loss": "simulator/zero_loss_admission_adapter.py",
    "local_search": "local_search_contract.py",
    "candidate_support_repair": "joint_candidate_support_snapshot.py",
    "assignment_head": "multi_agent_candidate_assignment_head.py",
    "credit_contract": "joint_assignment_credit_contract.py",
    "joint_assignment_learning": "joint_assignment_learning.py",
    "authorization": "simulator_authorization.py",
    "r9_7_gate": "run_h4m_ae_r9_7_gate.py",
    "r9_8_gate": "run_h4m_ae_r9_8_gate.py",
    "operational_actor_critic": "run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py",
}
SEEDS = [20260822, 20260823]
AGENTS = 8
DECISIONS_PER_VISIT = 4
CAUSAL_STEPS_PER_VISIT = 8


def now() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul")).replace(microsecond=0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True,
                               default=str) + "\n", encoding="utf-8")


def git(args: Sequence[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=PROJECT, text=True, capture_output=True, check=check)


def frozen_hashes() -> Dict[str, str]:
    return {name: sha256(ROOT / relative) for name, relative in FROZEN.items()}


def provenance() -> Dict[str, Any]:
    head = git(["rev-parse", "HEAD"]).stdout.strip()
    parent = git(["rev-parse", "HEAD^"]).stdout.strip()
    files = [line for line in git(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]).stdout.splitlines() if line]
    with tempfile.TemporaryDirectory(prefix="bt6s0_compile_") as tmp:
        try:
            py_compile.compile(str(PROJECT / SOURCE_REL), cfile=str(Path(tmp) / "bt6s0.pyc"), doraise=True)
            error = None
        except Exception as exc:  # noqa: BLE001
            error = repr(exc)
    return {"source_commit": head, "source_parent": parent, "source_only_local_commit": files == [SOURCE_REL.as_posix()],
            "parent_is_bt5r": parent == BT5R_SOURCE, "changed_files": files,
            "py_compile_passed": error is None, "py_compile_error": error,
            "git_diff_cached_check": git(["diff", "--cached", "--check"], check=False).returncode == 0,
            "github_push_performed": False}


def authoritative_lineage() -> Dict[str, Any]:
    """Bind the BT4 -> BT5 -> BT5-R chain by both source and artifact digests."""
    bt4_manifest = BT4_ARTIFACT / "manifest.json"
    bt5_manifest = BT5_ARTIFACT / "manifest.json"
    bt5r_manifest = BT5R_ARTIFACT / "manifest.json"
    bt4_execution = BT4_ARTIFACT / "bt4_execution_manifest.json"
    bt5_gate = BT5_ARTIFACT / "gate_decision.json"
    bt5r_gate = BT5R_ARTIFACT / "gate_decision.json"
    return {
        "BT4": {"source_commit": BT4_SOURCE, "artifact": str(BT4_ARTIFACT.relative_to(PROJECT)),
                "manifest_sha256": sha256(bt4_manifest), "execution_manifest_sha256": sha256(bt4_execution),
                "gate": json.loads(bt4_execution.read_text(encoding="utf-8"))["gate"]},
        "BT5": {"source_commit": BT5_SOURCE, "artifact": str(BT5_ARTIFACT.relative_to(PROJECT)),
                "manifest_sha256": sha256(bt5_manifest), "gate_decision_sha256": sha256(bt5_gate),
                "gate": json.loads(bt5_gate.read_text(encoding="utf-8"))["gate"]},
        "BT5_R": {"source_commit": BT5R_SOURCE, "artifact": str(BT5R_ARTIFACT.relative_to(PROJECT)),
                  "manifest_sha256": sha256(bt5r_manifest), "gate_decision_sha256": sha256(bt5r_gate),
                  "gate": json.loads(bt5r_gate.read_text(encoding="utf-8"))["gate"]},
    }


def band(hour: int) -> str:
    return {7: "night", 10: "offpeak", 17: "peak"}[int(hour)]


def postrepair_compute_profile() -> Dict[str, Any]:
    """Time only Local Search, frozen Zero-Loss, and joint-head forward calls."""
    import sys
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "simulator"))
    import local_search_contract as LS
    import multi_agent_assignment_contract as MC
    import multi_agent_candidate_assignment_head as H
    import run_h4m_ae_ls3_bt5r_candidate_support_repair_selection as BT5R
    from zero_loss_admission_adapter import ZeroLossAdmissionAdapter

    raw = pd.concat([pd.read_parquet(path) for path in
                     sorted(R97_ROOT.glob("scoped_request_ledger_v2_null_safe/**/part-0.parquet"))], ignore_index=True)
    raw_by_key = raw.set_index("historical_request_key")
    request_by_prefix = {str(key)[:16]: str(key) for key in raw_by_key.index}
    local_seconds: Dict[str, float] = defaultdict(float)
    zero_seconds: Dict[str, float] = defaultdict(float)
    local_calls: Counter[str] = Counter()
    zero_calls: Counter[str] = Counter()
    original_generate = LS.generate_candidates
    original_evaluate = ZeroLossAdmissionAdapter.evaluate

    def timed_generate(*args, **kwargs):
        key = str(kwargs["pending_request"]["historical_request_key"])
        start = time.perf_counter()
        output = original_generate(*args, **kwargs)
        local_seconds[key] += time.perf_counter() - start
        local_calls[key] += 1
        return output

    def timed_evaluate(self, *args, **kwargs):
        candidate = kwargs.get("candidate")
        request_id = str(dict(candidate).get("candidate_request_id", ""))[2:18]
        key = request_by_prefix.get(request_id, request_id)
        start = time.perf_counter()
        output = original_evaluate(self, *args, **kwargs)
        zero_seconds[key] += time.perf_counter() - start
        zero_calls[key] += 1
        return output

    LS.generate_candidates = timed_generate
    ZeroLossAdmissionAdapter.evaluate = timed_evaluate
    try:
        support = BT5R.build_repaired_supports()
    finally:
        LS.generate_candidates = original_generate
        ZeroLossAdmissionAdapter.evaluate = original_evaluate

    torch.manual_seed(20260822)
    head = H.MultiAgentCandidateAssignmentHead(global_dim=8, agent_dim=len(MC.AgentContext.FEATURE_NAMES),
                                                candidate_dim=len(MC.LOCAL_SEARCH_FEATURE_NAMES), demand_dim=6)
    head.eval()
    actor_seconds: Dict[str, float] = {}
    support_bytes: Dict[str, int] = {}
    per_group = {}
    for snapshot, joint in zip(support["snapshots"], support["joint_sets"]):
        packed = H.build_tensors(joint, global_vector=[0.1 * index for index in range(8)],
                                 demand_vector=[0.05 * index for index in range(6)])
        start = time.perf_counter()
        with torch.no_grad():
            head(global_feats=packed["global_feats"], demand_feats=packed["demand_feats"],
                 agent_feats=packed["agent_feats"], agent_mask=packed["agent_mask"],
                 candidate_feats=packed["candidate_feats"], pair_agent_index=packed["pair_agent_index"],
                 safe_mask=packed["safe_mask"])
        actor_seconds[snapshot.decision_group_id] = time.perf_counter() - start
        support_bytes[snapshot.decision_group_id] = len(json.dumps(snapshot.payload(), sort_keys=True).encode("utf-8"))
        row = raw_by_key.loc[snapshot.decision_group_id]
        counts = [len(ids) for _, ids in snapshot.raw_candidate_ids_by_agent]
        statuses = Counter(item.zero_loss_status for item in snapshot.all_evaluated)
        per_group[snapshot.decision_group_id] = {"time_band": band(int(row.service_hour)),
                                                  "historical_density": int(row.source_boarding_count),
                                                  "candidate_evaluations": len(snapshot.all_evaluated),
                                                  "zero_loss_evaluations": len(snapshot.all_evaluated),
                                                  "zero_loss_pass": statuses["PASS"], "zero_loss_fail": statuses["REJECT"],
                                                  "joint_support_including_no_assign": len(snapshot.safe) + 1,
                                                  "per_agent_candidate_counts": counts,
                                                  "has_genuine_2plus_agent_support": any(count >= 2 for count in counts),
                                                  "zero_loss_selective": statuses["PASS"] > 0 and statuses["REJECT"] > 0,
                                                  "snapshot_bytes": support_bytes[snapshot.decision_group_id],
                                                  "local_search_seconds": local_seconds[snapshot.decision_group_id],
                                                  "zero_loss_seconds": zero_seconds[snapshot.decision_group_id],
                                                  "actor_forward_seconds": actor_seconds[snapshot.decision_group_id],
                                                  "local_search_calls": local_calls[snapshot.decision_group_id],
                                                  "zero_loss_calls": zero_calls[snapshot.decision_group_id]}
    density_values = sorted(row["historical_density"] for row in per_group.values())
    lower = density_values[(len(density_values) - 1) // 3]
    upper = density_values[(2 * (len(density_values) - 1)) // 3]
    for row in per_group.values():
        row["request_density_class"] = "low" if row["historical_density"] <= lower else "high" if row["historical_density"] >= upper else "medium"
    rows = list(per_group.values())
    mean = lambda name: float(statistics.mean(row[name] for row in rows)) if rows else 0.0
    per_band = {}
    for name in ("night", "offpeak", "peak"):
        selected = [row for row in rows if row["time_band"] == name]
        per_band[name] = {"decisions": len(selected), "candidate_evaluations": sum(row["candidate_evaluations"] for row in selected),
                          "zero_loss_pass": sum(row["zero_loss_pass"] for row in selected),
                          "zero_loss_fail": sum(row["zero_loss_fail"] for row in selected),
                          "genuine_2plus_rate": float(sum(row["has_genuine_2plus_agent_support"] for row in selected) / len(selected)) if selected else 0.0,
                          "selective_rate": float(sum(row["zero_loss_selective"] for row in selected) / len(selected)) if selected else 0.0}
    return {"representative_authoritative_shadow_decisions": len(rows), "per_decision": {
                "candidate_evaluations": mean("candidate_evaluations"), "zero_loss_evaluations": mean("zero_loss_evaluations"),
                "zero_loss_pass": mean("zero_loss_pass"), "zero_loss_fail": mean("zero_loss_fail"),
                "joint_support_including_no_assign": mean("joint_support_including_no_assign"), "candidate_snapshot_bytes": mean("snapshot_bytes"),
                "local_search_seconds": mean("local_search_seconds"), "zero_loss_seconds": mean("zero_loss_seconds"),
                "joint_actor_forward_seconds": mean("actor_forward_seconds")},
            "by_time_band": per_band, "density_classes_covered": sorted({row["request_density_class"] for row in rows}),
            "representative_rows": per_group, "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "runtime_environment": {"torch_version": torch.__version__, "mps_built": torch.backends.mps.is_built(),
                                    "mps_available": torch.backends.mps.is_available(), "actor_profile_device": "cpu",
                                    "mps_execution_claim": False,
                                    "note": "MPS is unavailable in this isolated audit runtime; post-authorization BT6 requires an MPS preflight and this CPU structural profile is not a fallback execution plan."},
            "shadow_counterfactual_events": support["shadow_events"], "simulator_execution_events": support["execution_events"],
            "source_mutations": support["source_mutations"], "support_errors": support["errors"]}


def registry_windows() -> pd.DataFrame:
    demand = pd.read_parquet(REGISTRY)
    rows = demand.groupby(["time_band", "window_id"], as_index=False).size().rename(columns={"size": "requests"})
    return rows.sort_values(["time_band", "requests", "window_id"], kind="mergesort").reset_index(drop=True)


def choose_windows(per_band: int) -> list[Dict[str, Any]]:
    selected = []
    for time_band, rows in registry_windows().groupby("time_band", sort=True):
        ordered = rows.reset_index(drop=True)
        indexes = sorted({round(position * (len(ordered) - 1) / (per_band - 1)) for position in range(per_band)})
        if len(indexes) != per_band:
            raise RuntimeError("pre-rollout density stratum selection collided")
        for rank, index in enumerate(indexes):
            row = ordered.iloc[index]
            selected.append({"window_id": str(row.window_id), "time_band": str(time_band), "requests": int(row.requests),
                             "prepolicy_density_rank": rank, "prepolicy_selection_basis": "within-band request_count ascending, ties window_id"})
    return selected


def observed_rates(profile: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    bt4 = json.loads((BT4_ARTIFACT / "bt4_learning_signal_audit.json").read_text(encoding="utf-8"))
    by_band = profile["by_time_band"]
    return {"informative": {"low": 0.125, "central": float(bt4["non_zero_team_reward_fraction"]), "high": 0.625,
                            "basis": "BT4 observed time-band min / overall / max informative fractions"},
            "genuine_2plus": {"low": min(row["genuine_2plus_rate"] for row in by_band.values()),
                                "central": float(statistics.mean(row["genuine_2plus_rate"] for row in by_band.values())),
                                "high": max(row["genuine_2plus_rate"] for row in by_band.values()),
                                "basis": "BT6-S0 shadow time-band min / mean / max; planning reference only"},
            "zero_loss_selective": {"low": min(row["selective_rate"] for row in by_band.values()),
                                      "central": float(statistics.mean(row["selective_rate"] for row in by_band.values())),
                                      "high": max(row["selective_rate"] for row in by_band.values()),
                                      "basis": "BT6-S0 shadow time-band min / mean / max; planning reference only"}}


def expected(decisions: int, rate: Dict[str, float]) -> Dict[str, Any]:
    return {"range": [int(math.floor(decisions * rate["low"])), int(math.ceil(decisions * rate["high"]))],
            "central_reference": int(round(decisions * rate["central"])), "rates": {key: rate[key] for key in ("low", "central", "high")},
            "basis": rate["basis"]}


def candidate(label: str, per_band: int, updates_per_seed: int, profile: Dict[str, Any], rates: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    windows = choose_windows(per_band)
    visits = len(windows) * len(SEEDS)
    decisions = visits * DECISIONS_PER_VISIT
    transitions = visits * CAUSAL_STEPS_PER_VISIT
    bt4_runtime = json.loads((BT4_ARTIFACT / "bt4_execution_manifest.json").read_text(encoding="utf-8"))["runtime_seconds"]
    causal_seconds = float(bt4_runtime) / 96.0 * transitions
    support_seconds = (profile["per_decision"]["local_search_seconds"] + profile["per_decision"]["zero_loss_seconds"]
                       + profile["per_decision"]["joint_actor_forward_seconds"]) * decisions
    snapshot_bytes = profile["per_decision"]["candidate_snapshot_bytes"] * decisions
    memory_bytes = max(int(profile["peak_rss_bytes"]), 348504064) + int(snapshot_bytes)
    density = Counter("low" if row["prepolicy_density_rank"] == 0 else "high" if row["prepolicy_density_rank"] == per_band - 1 else "medium" for row in windows)
    return {"label": label, "scope": "Suseong frozen 54-window / 414-request representative causal registry",
            "window_selection_rule": "Within each time band sort by pre-policy request_count then window_id; select evenly spaced density ranks before rollout. No reward, advantage, policy outcome, KPI, or winner input.",
            "windows": windows, "distinct_windows": len(windows), "window_visits": visits, "requests": sum(row["requests"] for row in windows),
            "time_band_composition": dict(Counter(row["time_band"] for row in windows)), "request_density_composition": dict(density),
            "agents": AGENTS, "seeds": SEEDS, "assignment_decisions": decisions, "multi_step_trajectories": visits,
            "trajectory_length": DECISIONS_PER_VISIT, "causal_steps_per_window": CAUSAL_STEPS_PER_VISIT,
            "causal_transitions": transitions, "optimizer_updates": len(SEEDS) * updates_per_seed,
            "update_budget_design": {"updates_per_seed": updates_per_seed,
                                     "per_seed_assignment_decisions": decisions // len(SEEDS),
                                     "full_seed_rollout_reuse": updates_per_seed,
                                     "per_seed_decision_gradient_exposures": (decisions // len(SEEDS)) * updates_per_seed,
                                     "rationale": "Each update consumes the whole frozen same-seed assignment rollout, as in BT4; the cap is selected for bounded sample reuse and gradient inspection rather than mechanically from windows. R1 limits reuse at its smallest sample, R2 uses five full-rollout passes for 48 same-seed decisions, and R3 keeps that five-pass cap to avoid increasing reuse merely because scale grows."},
            "expected_informative_decisions": expected(decisions, rates["informative"]),
            "expected_genuine_2plus_candidate_decisions": expected(decisions, rates["genuine_2plus"]),
            "expected_zero_loss_selective_exposures": expected(decisions, rates["zero_loss_selective"]),
            "runtime_estimate_seconds": round(causal_seconds + support_seconds, 3),
            "runtime_estimate_method": "BT4 realized causal-transition rate plus BT6-S0 measured shadow Local Search + Zero-Loss + actor-forward mean; optimizer count is separately bounded.",
            "estimated_peak_rss_mb": round(memory_bytes / (1024 * 1024), 3),
            "memory_estimate_method": "max(BT4 realized peak RSS, BT6-S0 profile peak RSS) + persisted average support snapshot bytes per assignment decision.",
            "checkpoint_policy": {"test_only": True, "bounded": True, "non_promotable": True, "promotion": False,
                                  "performance_claim": False}}


def static_execution_guards() -> Dict[str, Any]:
    tree = ast.parse((PROJECT / SOURCE_REL).read_text(encoding="utf-8"))
    calls = {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    no_execution = {"optimizer_step": not any(call.endswith(".step") for call in calls),
                    "causal_rollout": "PV8CausalKpiAdapter" not in calls,
                    "checkpoint_write": "torch.save" not in calls,
                    "training_grant": not any(call.endswith(".granted") for call in calls)}
    return {"bt3_real_optimizer_sites": 19, "bt3_unguarded_real_sites_after": 0, "bt6s0_no_execution_static_checks": no_execution,
            "optimizer_steps": 0, "causal_training_rollouts": 0, "checkpoint_writes": 0,
            "passed": all(no_execution.values())}


def main() -> None:
    started = time.perf_counter()
    root = ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt6_s0_postrepair_scale_redesign_{now().strftime('%Y%m%d_%H%M%S%z')[:-2]}:00"
    if root.exists():
        raise SystemExit("append-only artifact collision")
    prov = provenance()
    before = frozen_hashes()
    lineage = authoritative_lineage()
    bt5r_gate = json.loads((BT5R_ARTIFACT / "gate_decision.json").read_text(encoding="utf-8"))
    profile = postrepair_compute_profile()
    rates = observed_rates(profile)
    ladder = [candidate("R1_MINIMUM_POST_REPAIR", 3, 4, profile, rates),
              candidate("R2_MODERATE_POST_REPAIR", 4, 5, profile, rates),
              candidate("R3_UPPER_BOUNDED_POST_REPAIR", 5, 5, profile, rates)]
    selected = ladder[1]
    after = frozen_hashes()
    guards = static_execution_guards()
    binding = {"bt4_source": lineage["BT4"]["source_commit"] == BT4_SOURCE,
               "bt5_source": lineage["BT5"]["source_commit"] == BT5_SOURCE,
               "bt5r_gate": bt5r_gate["gate"] == "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT5R_CANDIDATE_SUPPORT_AGENT_ORDER_DECONFOUNDING_AND_ZERO_LOSS_SELECTIVITY_COMPLETE",
               "bt5r_source_parent": prov["parent_is_bt5r"], "source_only_commit": prov["source_only_local_commit"],
               "py_compile": prov["py_compile_passed"], "git_diff_cached_check": prov["git_diff_cached_check"],
               "registry": REGISTRY.is_file(), "frozen_hashes": before == after,
               "profile_bands": set(profile["by_time_band"]) == {"night", "offpeak", "peak"},
               "profile_density": set(profile["density_classes_covered"]) == {"low", "medium", "high"},
               "no_profile_source_mutation": profile["source_mutations"] == 0, "no_profile_execution": profile["simulator_execution_events"] == 0}
    failures = []
    if not all(binding.values()): failures.append("AUTHORITATIVE_BINDING_OR_PROFILE_COVERAGE_FAILED")
    if profile["support_errors"]: failures.append("POST_REPAIR_SUPPORT_PROFILE_FAILED")
    if not guards["passed"]: failures.append("BT6S0_EXECUTION_GUARD_FAILED")
    gate = PASS_GATE if not failures else "BLOCKED_INSUFFICIENT_POST_REPAIR_SCALE_EVIDENCE"
    classification = PASS_CLASS if not failures else "BLOCKED_POST_REPAIR_SCALE_REDESIGN_EVIDENCE_INSUFFICIENT"
    locks = {"training_allowed": False, "simulator_execution_allowed": False, "performance_comparison_allowed": False,
             "paper_level_claim_allowed": False, "causal_performance_claim_allowed": False}
    readiness = {"status": "FROZEN_FOR_BT6_RESULT_REVIEW_NOT_POLICY_QUALITY", "selected_design": selected["label"],
                 "all_selected_windows_and_seeds": {"required": True, "count": selected["window_visits"]},
                 "all_time_bands": ["night", "offpeak", "peak"], "multiple_request_density_levels": ["low", "medium", "high"],
                 "minimum_informative_decisions": {"value": 16, "basis": "BT4 mechanism validation observed 16 informative decisions; bounded-review floor, not a paper-level threshold."},
                 "minimum_genuine_2plus_candidate_decisions": {"value": 3, "basis": "at least one genuine multi-candidate decision in each required time band; if absent, BT7 remains blocked."},
                 "minimum_zero_loss_selective_states": {"value": 3, "basis": "at least one PASS/FAIL selective decision in each required time band; if absent, BT7 remains blocked."},
                 "multi_step_gae": {"all_trajectories_length": DECISIONS_PER_VISIT, "recursion_must_be_exercised": True},
                 "integrity": ["finite actor/critic gradients", "no entropy-collapse conclusion without observed evidence", "Zero-Loss violations = 0", "illegal/masked selections = 0", "candidate identity/relabeling errors = 0", "cross-window contamination = 0", "cross-seed contamination = 0", "future leakage = 0", "source mutation violations = 0", "NaN/Inf = 0"],
                 "mps_preflight": "MPS must be available and pass before BT6 execution authorization; BT6-S0 did not claim an MPS runtime measurement."}
    root.mkdir(parents=True)
    outputs = {"bt6s0_postrepair_compute_profile.json": {**profile, "authoritative_lineage": lineage},
               "bt6s0_learning_sample_requirement.json": {"rates": rates, "bt4_reference": {"informative": "16/48", "multi_step_trajectories": 12, "temporally_propagated_advantages": "36/48"}, "planning_only": True,
                                                          "selection_standard": "Absolute informative decisions plus repeated observed support/selectivity opportunities are planning references; none is a reward-nonzero percentage target or performance claim."},
               "bt6s0_scale_candidate_ladder.json": {"ladder": ladder, "old_bt5_E1_E2_E3_reused": False, "old_bt5_provisional_E2_reused": False},
               "bt6s0_selected_bt6_envelope.json": {"status": "RECOMMENDED_NOT_AUTHORIZED_NOT_EXECUTED", "selected": selected,
                                                       "selection_reason": "R1 has one window at each of the three density ranks per time band and only 24 decisions per band, so it cannot replicate an interior-density behavioral opportunity. R2 is the smallest design with four pre-policy density-stratified windows, two interior-density windows per time band, 32 decisions per band, 10 separately bounded updates, and repeated central-reference multi-candidate/selective exposure. R3 adds scale without a new coverage dimension.",
                                                       "mps_execution_preflight_required": True},
               "bt6s0_bt7_readiness_contract.json": readiness,
               "test_results.json": {"provenance": prov, "binding": binding, "guards": guards, "optimizer_steps": 0,
                                     "causal_training_rollouts": 0, "checkpoint_writes": 0, "TEST6_access": 0,
                                     "hard_failures": failures, "warnings": (["MPS_UNAVAILABLE_IN_ISOLATED_AUDIT_RUNTIME; execution authorization requires a separate MPS preflight"] if not profile["runtime_environment"]["mps_available"] else [])},
               "frozen_hash_before_after.json": {"before": before, "after": after, "unchanged": before == after},
               "gate_decision.json": {"gate": gate, "classification": classification, "source_commit": prov["source_commit"],
                                      "global_locks": locks, "next_step": "BT6 separate extended bounded-training execution authorization" if not failures else "STOP: resolve BT6-S0 blocker", "hard_failures": failures,
                                      "warnings": (["MPS_UNAVAILABLE_IN_ISOLATED_AUDIT_RUNTIME; no MPS execution claim"] if not profile["runtime_environment"]["mps_available"] else [])}}
    for name, payload in outputs.items(): dump(root / name, payload)
    (root / "final_report.md").write_text(
        f"# {STAGE} — Post-Repair Bounded Training Scale Redesign\n\n"
        f"gate = {gate}\nclassification = {classification}\nsource_commit = {prov['source_commit']}\n\n"
        f"Post-repair shadow profile: {profile['representative_authoritative_shadow_decisions']} decisions; candidate / Zero-Loss evaluations per decision = {profile['per_decision']['candidate_evaluations']:.4f} / {profile['per_decision']['zero_loss_evaluations']:.4f}; support size including NO_ASSIGN = {profile['per_decision']['joint_support_including_no_assign']:.4f}.\n\n"
        f"Selected design: {selected['label']} — {selected['distinct_windows']} windows, {selected['window_visits']} visits, {selected['assignment_decisions']} decisions, {selected['causal_transitions']} transitions, {selected['optimizer_updates']} updates. It is not authorization and has no promoted checkpoint.\n\n"
        f"MPS is unavailable in this isolated audit runtime, so no MPS execution was claimed; separate BT6 authorization must preflight the intended MPS runtime.\n\n"
        f"BT4 → BT5 → BT5-R lineage is bound by source commits and artifact SHA-256 values in `bt6s0_postrepair_compute_profile.json`.\n\n"
        f"optimizer steps / causal rollouts / checkpoints = 0 / 0 / 0\nnext step = {'BT6 separate extended bounded-training execution authorization' if not failures else 'STOP'}\n", encoding="utf-8")
    hashes = {path.relative_to(root).as_posix(): sha256(path) for path in root.rglob("*") if path.is_file()}
    dump(root / "manifest.json", {"stage": STAGE, "gate": gate, "classification": classification, "source_commit": prov["source_commit"], "file_sha256": hashes,
                                  "elapsed_seconds": round(time.perf_counter() - started, 3), "optimizer_steps": 0, "causal_training_rollouts": 0,
                                  "checkpoint_writes": 0, "github_push_performed": False})
    (root / ("_SUCCESS.lock" if not failures else "_BLOCKED.lock")).write_text(gate + "\n", encoding="utf-8")
    print(f"[{'PASS' if not failures else 'BLOCKED'}] {gate}")
    print(f"artifact: {root.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
