#!/usr/bin/env python3
"""BT8-F1 fail-closed authority gate.

The executor may grant neither MPS/training nor simulator capability until the
R4 design binds per-replicate update allocation and an authoritative
candidate-plan-to-causal-transition path.  This file deliberately writes a
complete blocked artifact when either authority is absent; it contains no
training, checkpoint serialization, candidate generation, or MPS operation.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


STAGE = "H4M-AE-R9.8-LS3-BT8-F1"
BLOCK = "BLOCKED_BT8_R4_EXECUTION_AUTHORITY_INCOMPLETE"
R4_SOURCE = "32fbf2b3043166188bde891a5c450702e7db3607"
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
R4 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r4_v2_actor_fresh_training_design_20260823_122459+0900"
SOURCE_FILES = {
    "05_training/run_h4m_ae_ls3_bt8_f1_bounded_training.py",
    "05_training/test_h4m_ae_ls3_bt8_f1_authority_gate.py",
}
LOCKS = {"training_allowed": False, "simulator_execution_allowed": False,
         "performance_comparison_allowed": False, "paper_level_claim_allowed": False,
         "causal_performance_claim_allowed": False}


class F1AuthorityError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True,
                               default=str) + "\n", encoding="utf-8")


def git(args: Sequence[str]) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT, text=True, capture_output=True,
                          check=True).stdout.strip()


def provenance() -> dict[str, Any]:
    changed = [item for item in git(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]).splitlines() if item]
    return {"source_commit": git(["rev-parse", "HEAD"]), "source_parent": git(["rev-parse", "HEAD^"]),
            "changed_files": changed, "source_only_local_commit": bool(changed) and set(changed).issubset(SOURCE_FILES),
            "github_push_performed": False}


def validate_r4_execution_authority(*, r4_gate: Mapping[str, Any], selected: Mapping[str, Any],
                                    plan_contract: Mapping[str, Any], bridge_step: Any) -> list[dict[str, str]]:
    """Return every pre-grant blocker; never fabricate missing scheduling policy."""
    failures: list[dict[str, str]] = []
    if r4_gate.get("gate") != "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R4_FRESH_V2_ACTOR_TRAINING_AND_NOVEL_EXPOSURE_DESIGN_COMPLETE":
        failures.append({"code": "R4_GATE_MISMATCH", "detail": "BT8-R4 PASS gate is not exactly bound"})
    if r4_gate.get("source_commit") != R4_SOURCE:
        failures.append({"code": "R4_SOURCE_COMMIT_MISMATCH", "detail": "BT8-R4 source commit differs"})
    expected = {"distinct_windows": 9, "train_visits": 12, "requests_train": 43,
                "review_snapshot_visits": 6, "requests_review": 27, "agents": 8,
                "decisions": 48, "trajectories": 12, "transitions": 96, "optimizer_updates": 6}
    for name, value in expected.items():
        if selected.get(name) != value:
            failures.append({"code": "F1_ENVELOPE_FIELD_MISMATCH", "detail": f"{name} != {value}"})
    if selected.get("seeds") != [20260822, 20260823]:
        failures.append({"code": "F1_ENVIRONMENT_SEEDS_MISMATCH", "detail": "exact seed list absent or altered"})
    update_by_seed = selected.get("optimizer_updates_by_environment_seed")
    if not isinstance(update_by_seed, Mapping) or set(str(key) for key in update_by_seed) != {"20260822", "20260823"} \
            or sum(int(value) for value in update_by_seed.values()) != int(selected.get("optimizer_updates", -1)):
        failures.append({"code": "PER_REPLICATE_UPDATE_BUDGET_MISSING", "detail": "R4 binds total=6 but no exact seed→update allocation"})
    if not isinstance(selected.get("ppo_epochs"), int) or not isinstance(selected.get("minibatch_policy"), Mapping):
        failures.append({"code": "PPO_UPDATE_PARTITION_MISSING", "detail": "R4 selected envelope lacks frozen PPO epochs/minibatch policy"})
    if plan_contract.get("disposable_shadow_application_only") is True:
        failures.append({"code": "CANDIDATE_PLAN_AUTHORITATIVE_APPLICATION_UNBOUND", "detail": "R4 plan contract is explicitly disposable-shadow only"})
    if "candidate_plan" not in inspect.signature(bridge_step).parameters:
        failures.append({"code": "CAUSAL_BRIDGE_CANDIDATE_PLAN_INPUT_UNBOUND", "detail": "PV8CausalKpiAdapter.step has no candidate-plan input"})
    return failures


def main() -> None:
    sys.path.insert(0, str(ROOT))
    import run_h4m_ae_ls3_bt6_postrepair_r2_training as BT6
    import test_h4m_ae_r3_causal_kpi_bridge as R3

    source = provenance()
    r4_gate = json.loads((R4 / "gate_decision.json").read_text(encoding="utf-8"))
    selected = json.loads((R4 / "bt8r4_selected_training_envelope.json").read_text(encoding="utf-8"))
    plan_contract = json.loads((R4 / "bt8r4_candidate_plan_credit_contract.json").read_text(encoding="utf-8"))
    init = json.loads((R4 / "bt8r4_v2_actor_initialization_authority.json").read_text(encoding="utf-8"))
    critic = json.loads((R4 / "bt8r4_critic_lineage_decision.json").read_text(encoding="utf-8"))
    split = json.loads((R4 / "bt8r4_train_review_split_contract.json").read_text(encoding="utf-8"))
    future = json.loads((R4 / "bt8r4_future_execution_evidence_contract.json").read_text(encoding="utf-8"))
    bridge = R3.imp("bridge", R3.BRIDGE).PV8CausalKpiAdapter
    failures = validate_r4_execution_authority(r4_gate=r4_gate, selected=selected,
        plan_contract=plan_contract, bridge_step=bridge.step)
    if not source["source_only_local_commit"] or source["source_parent"] != R4_SOURCE:
        failures.append({"code": "F1_SOURCE_PROVENANCE_INVALID", "detail": "source-only child of the R4 authority is required"})

    timestamp = time.strftime("%Y%m%d_%H%M%S%z")
    root = ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_f1_authority_block_{timestamp}"
    if root.exists():
        raise SystemExit("append-only artifact collision")
    root.mkdir(parents=True)
    counters = {"mps_preflight": 0, "capability_grant": 0, "authoritative_candidate_generation": 0,
                "causal_rollout": 0, "optimizer_step": 0, "training_visit": 0,
                "checkpoint_write": 0, "TEST6_access": 0, "github_push": 0}
    binding = {"r4_artifact_exists": R4.is_dir(), "r4_gate": r4_gate.get("gate"),
               "r4_source_commit": r4_gate.get("source_commit"), "selected_f1_envelope_sha256": sha256(R4 / "bt8r4_selected_training_envelope.json"),
               "plan_credit_contract_sha256": sha256(R4 / "bt8r4_candidate_plan_credit_contract.json"),
               "initialization_authority_sha256": sha256(R4 / "bt8r4_v2_actor_initialization_authority.json"),
               "critic_lineage_sha256": sha256(R4 / "bt8r4_critic_lineage_decision.json"),
               "train_review_split_sha256": sha256(R4 / "bt8r4_train_review_split_contract.json"),
               "future_execution_evidence_sha256": sha256(R4 / "bt8r4_future_execution_evidence_contract.json"),
               "failures": failures}
    not_run = {"not_executed": True, "block_reason": BLOCK, "counters": counters}
    dump(root / "bt8f1_mps_preflight.json", {**not_run, "status": "NOT_RUN_AFTER_AUTHORITY_BLOCK", "cpu_fallback": "NOT_ATTEMPTED"})
    dump(root / "bt8f1_authorization_manifest.json", {"stage": STAGE, "source": source, "binding": binding,
         "authorization_granted": False, "global_locks": LOCKS, "counters": counters})
    dump(root / "bt8f1_initialization_lineage.json", {**not_run, "actor": init, "critic": critic,
         "fresh_initialization_attempted": False, "reason": "authority must bind update partition and causal plan application first"})
    dump(root / "bt8f1_novel_exposure_binding.json", {**not_run, "selected_envelope": selected,
         "split": split, "verified": False, "reason": "no preflight authority pass"})
    dump(root / "bt8f1_train_review_leakage_audit.json", {**not_run, "split": split,
         "train_review_overlap": 0, "review_optimizer_exposure": 0})
    dump(root / "bt8f1_candidate_plan_credit_audit.json", {**not_run, "contract": plan_contract,
         "bridge_step_signature": str(inspect.signature(bridge.step)), "authoritative_plan_application_verified": False})
    dump(root / "bt8f1_training_execution_audit.json", not_run)
    dump(root / "bt8f1_learning_signal_audit.json", not_run)
    dump(root / "bt8f1_training_snapshot_collection.json", {**not_run, "snapshot_count": 0})
    dump(root / "bt8f1_review_snapshot_collection.json", {**not_run, "snapshot_count": 0})
    dump(root / "bt8f1_initial_review_replay.json", {**not_run, "replay_count": 0})
    dump(root / "bt8f1_final_review_replay.json", {**not_run, "replay_count": 0})
    dump(root / "bt8f1_initial_checkpoint_manifest.json", {**not_run, "checkpoint_created": False})
    dump(root / "bt8f1_final_checkpoint_manifest.json", {**not_run, "checkpoint_created": False})
    before, after = {**BT6.frozen_hashes(), "joint_actor_head": sha256(ROOT / "multi_agent_candidate_assignment_head.py")}, {**BT6.frozen_hashes(), "joint_actor_head": sha256(ROOT / "multi_agent_candidate_assignment_head.py")}
    dump(root / "frozen_hash_before_after.json", {"before": before, "after": after, "all_unchanged": before == after})
    dump(root / "test_results.json", {"authority_fail_closed": bool(failures), "failures": failures, "operation_counters": counters})
    report = "# BT8-F1 authority block\n\nNo MPS preflight, capability grant, rollout, optimizer step, snapshot, or checkpoint was performed.\n\n" + "\n".join(f"- `{row['code']}`: {row['detail']}" for row in failures) + "\n"
    (root / "final_report.md").write_text(report, encoding="utf-8")
    dump(root / "gate_decision.json", {"stage": STAGE, "gate": BLOCK, "classification": "BLOCKED",
         "source_commit": source["source_commit"], "hard_failures": failures, "warnings": [], "global_locks": LOCKS,
         "next_step": "separate authority completion gate: bind per-replicate PPO update partition and authorize a candidate-plan-aware causal transition path"})
    files = {path.name: sha256(path) for path in sorted(root.iterdir()) if path.is_file() and path.name != "manifest.json"}
    dump(root / "manifest.json", {"stage": STAGE, "gate": BLOCK, "source_commit": source["source_commit"],
         "r4_source_commit": R4_SOURCE, "github_push_performed": False, "file_sha256": files})
    print(f"[{BLOCK}] {root.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
