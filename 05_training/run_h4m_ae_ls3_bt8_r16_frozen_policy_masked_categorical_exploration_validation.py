#!/usr/bin/env python3
"""BT8-R16: implement and validate R15's frozen-policy E1 selector.

This runner is deliberately limited to immutable snapshot/checkpoint reads,
frozen MPS forwards, and pure action-selection probes.  It has no causal
rollout, reward settlement, optimizer, backward, parameter mutation, or
checkpoint-write path.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import torch


STAGE = "H4M-AE-R9.8-LS3-BT8-R16"
PASS_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R16_FROZEN_POLICY_MASKED_CATEGORICAL_EXPLORATION_IMPLEMENTATION_AND_EQUIVALENCE_VALIDATION_COMPLETE"
CLASSIFICATION = "A_E1_FROZEN_POLICY_MASKED_CATEGORICAL_EXPLORATION_IMPLEMENTED_AND_FROZEN_INFERENCE_EQUIVALENT"
BINDING_BLOCK = "BLOCKED_R16_UPSTREAM_EVIDENCE_BINDING_FAILURE"
INFERENCE_BLOCK = "BLOCKED_R16_INFERENCE_BEHAVIOR_CHANGED"
DISTRIBUTION_BLOCK = "BLOCKED_R16_POLICY_DISTRIBUTION_MUTATED"
NO_ASSIGN_BLOCK = "BLOCKED_R16_NO_ASSIGN_SEMANTICS_CHANGED"
ORDER_BLOCK = "BLOCKED_R16_ORDER_INVARIANCE_FAILURE"
RNG_BLOCK = "BLOCKED_R16_RNG_REPRODUCIBILITY_FAILURE"
LEGALITY_BLOCK = "BLOCKED_R16_ZERO_LOSS_OR_LEGALITY_REGRESSION"
EXPOSURE_BLOCK = "BLOCKED_R16_EXPLORATION_EXPOSURE_NOT_REPRODUCED"
SOURCE_BLOCK = "BLOCKED_R16_UNEXPECTED_SOURCE_OR_CHECKPOINT_MUTATION"
MPS_BLOCK = "BLOCKED_R16_MPS_FROZEN_VALIDATION_ENVIRONMENT_UNAVAILABLE"

R15_SOURCE = "8e24652fa0523e751077ca0c4500b20d626fc517"
R14_SOURCE = "36ccbaa84e06af3202560802fd3371066839a7c5"
R13_SOURCE = "a9818399c4a1a74734496a146b2b98fabab513b0"
R12_SOURCE = "6e22b95d80720255ca17b1ab0520417b89e5a5c9"
R11_SOURCE = "f467feef8246c6c77dc67a360aecc60e0102913e"
S3_CONTRACT_SHA256 = "66e2fb35de3aa767780d9f0f001d774919e059e580f4410fe1d01bd96b185393"
R15_CONTRACT_SHA256 = "cd864dcbba5ec42b26c6399aa92480596455aa104e752627676b3e6ac2de0c84"
R15_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R15_MINIMAL_EXPLORATION_DEADLOCK_REPAIR_SELECTION_AUDIT_COMPLETE"
R14_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R14_S3_SAME_INPUT_FROZEN_POLICY_FACTOR_REVIEW_COMPLETE"
R13_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R13_S3_FOUR_CELL_ON_POLICY_CAUSAL_EXECUTION_COMPLETE"
R12_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R12_S3_FACTORIAL_EXECUTION_AUTHORITY_SELECTION_COMPLETE"
R11_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R11_MINIMAL_SEED_DIVERGENCE_CAUSE_ISOLATION_AND_REPAIR_SELECTION_COMPLETE"
R15_CLASSIFICATION = "A_FROZEN_POLICY_MASKED_STOCHASTIC_SAMPLING_SUFFICIENT"
REVIEW_COLLECTION_DIGEST = "6c811022a5df4b3966ac14fce750f8bdd840a65a50e48285fe0c97ce157b889e"
TRAINING_COLLECTION_DIGEST = "954d50f384ebe4318dbe9565c4c30c5e6a0b6249dd06ca441ff5abc2af08fb63"
PROBE_SEEDS = tuple(range(32))
TIME_BANDS = ("night", "offpeak", "peak")
PROBABILITY_ABS_TOLERANCE = 0.0
LOG_PROBABILITY_ABS_TOLERANCE = 0.0

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
R15 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r15_minimal_exploration_deadlock_repair_selection_audit_20260827_194419+09:00"
R14 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r14_s3_frozen_factor_review_20260827_001145+09:00"
R13 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r13_s3_four_cell_execution_20260826_174729+09:00"
R12 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r12_s3_execution_authority_selection_20260826_145345+09:00"
R11 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r11_seed_divergence_repair_selection_20260826_123135+09:00"
F1 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_f1_fresh_v2_bounded_training_20260823_135127+09:00"

SELECTOR_REL = "05_training/joint_assignment_frozen_policy_selector.py"
RUNNER_REL = "05_training/run_h4m_ae_ls3_bt8_r16_frozen_policy_masked_categorical_exploration_validation.py"
TEST_REL = "05_training/test_joint_assignment_frozen_policy_selector.py"
SOURCE_FILES = {SELECTOR_REL, RUNNER_REL, TEST_REL}
CELL_ROLE = {"AC-R1": "AC", "AC-R2": "AC", "BD-R1": "BD", "BD-R2": "BD"}
EXPECTED_BD_EXPOSURE = {
    ("BD-R1", "night"): 123, ("BD-R1", "offpeak"): 137, ("BD-R1", "peak"): 201,
    ("BD-R2", "night"): 125, ("BD-R2", "offpeak"): 141, ("BD-R2", "peak"): 211,
}
EXPECTED_CHECKPOINTS = {
    "initial:AC-R1": "e13e3f0294234d3ee4b49878912a4b5ee18c5e53253e478201b161520dc6cba7",
    "initial:AC-R2": "ad618133374c1ccbe2c367f67117a3d6b205b9c48acd5ee7a2ba3b5549ed0df0",
    "initial:BD-R1": "565c07424f811170a8f3640cef622b3a87a235c763f7a790958c1c3aace9712d",
    "initial:BD-R2": "0fcd9c359bedc355718fabb9dbf87c3cbcf09d2efc07419bd848dcd4e12470dd",
    "final:AC-R1": "fc4a283d11bd436a4e42cdfebf2b07156dbfedbc703c715bce5618600bfd5968",
    "final:AC-R2": "f07240f2a11ea0707f5b2f775a56b723739742c51cfb0ff0b7383b995bc525b1",
    "final:BD-R1": "6a95135c2ff3e1637d23710783b8a7729c2a0fb0d025977c73452ed4045037da",
    "final:BD-R2": "4754a7d53a810e4562ee42d5e77dd55810ab88855d41d81e9eef02f9e2f5afae",
}
GLOBAL_LOCKS = {
    "training_allowed": False,
    "simulator_execution_allowed": False,
    "performance_comparison_allowed": False,
    "paper_level_claim_allowed": False,
    "causal_performance_claim_allowed": False,
}
EXECUTION_LOCKS = {
    "training_allowed": False,
    "optimizer_creation_allowed": False,
    "optimizer_step_allowed": False,
    "backward_allowed": False,
    "loss_update_allowed": False,
    "checkpoint_write_allowed": False,
    "checkpoint_mutation_allowed": False,
    "actor_parameter_mutation_allowed": False,
    "critic_parameter_mutation_allowed": False,
    "GATv2_parameter_mutation_allowed": False,
    "Reward_V2_mutation_allowed": False,
    "Zero_Loss_mutation_allowed": False,
    "candidate_generator_semantics_mutation_allowed": False,
    "E2_rescue_allowed": False,
    "E3_temperature_or_probability_floor_allowed": False,
    "TEST6_open_allowed": False,
    "GitHub_push_allowed": False,
    "automatic_training_authorization": False,
}
PROBE_COLUMNS = [
    "stage", "cell_id", "policy_family", "snapshot_digest", "snapshot_manifest_sha256", "decision_id", "window_id",
    "time_band", "environment_seed", "probe_seed", "selection_mode", "categorical_triggered", "trigger_reason",
    "deterministic_identity", "selected_identity", "selected_index", "source_probability", "returned_probability",
    "source_log_probability", "returned_log_probability", "log_probability_abs_delta", "support_size",
    "legal_candidate_count", "no_assign_in_support", "no_assign_probability", "no_assign_source_index",
    "no_assign_removed", "finite", "legal", "zero_loss_feasible", "r15_identity_match", "same_seed_identity_match",
    "canonical_rng_keyset_sha256", "actor_state_sha256",
]


class R16Error(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


def require(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise R16Error(code, detail)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def load_json(path: Path, code: str = BINDING_BLOCK) -> Any:
    require(path.is_file(), code, f"missing={path}")
    return json.loads(path.read_text(encoding="utf-8"))


def git(args: Sequence[str]) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT, text=True, capture_output=True, check=True).stdout.strip()


def git_blob_sha(revision: str, path: str) -> str:
    raw = subprocess.run(["git", "show", f"{revision}:{path}"], cwd=PROJECT, capture_output=True, check=True).stdout
    return hashlib.sha256(raw).hexdigest()


def manifest_audit(root: Path) -> dict[str, Any]:
    manifest = load_json(root / "manifest.json")
    hashes = manifest.get("file_sha256")
    require(isinstance(hashes, Mapping), BINDING_BLOCK, f"manifest_schema={root}")
    mismatches = [str(name) for name, digest in hashes.items()
                  if not (root / str(name)).is_file() or sha256(root / str(name)) != str(digest)]
    return {"manifest_sha256": sha256(root / "manifest.json"), "declared_file_count": len(hashes),
            "mismatches": mismatches, "all_match": not mismatches}


def module_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for _, tensor in sorted(module.state_dict().items()):
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def state_dicts_exact(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> bool:
    return set(left) == set(right) and all(torch.equal(left[name].detach().cpu(), right[name].detach().cpu()) for name in left)


def source_provenance() -> dict[str, Any]:
    changed = [row for row in git(["diff", "--name-only", f"{R15_SOURCE}..HEAD"]).splitlines() if row]
    return {
        "source_commit": git(["rev-parse", "HEAD"]),
        "source_parent": git(["rev-parse", "HEAD^"]),
        "source_lineage_descends_from_r15": git(["merge-base", R15_SOURCE, "HEAD"]) == R15_SOURCE,
        "changed_files_since_r15": changed,
        "source_only_local_commit": set(changed) == SOURCE_FILES,
        "git_status_porcelain": git(["status", "--porcelain=v1"]),
        "github_push_performed": False,
    }


def artifact_root() -> Path:
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M%S%z")[:-2] + ":00"
    return ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r16_frozen_policy_masked_categorical_exploration_validation_{stamp}"


def counters() -> dict[str, int]:
    return {
        "training": 0, "optimizer_creation": 0, "optimizer_step": 0, "backward": 0, "loss_update": 0,
        "causal_rollout": 0, "simulator_execution": 0, "candidate_generation": 0, "candidate_regeneration": 0,
        "local_search_rerun": 0, "zero_loss_reevaluation": 0, "reward_settlement": 0, "parameter_mutation": 0,
        "checkpoint_write": 0, "checkpoint_mutation": 0, "test6_access": 0, "github_push": 0,
        "inference_frozen_forwards": 0, "training_probe_frozen_forwards": 0, "order_probe_frozen_forwards": 0,
        "selector_calls": 0, "illegal_selection": 0, "zero_loss_violation": 0, "nan_or_inf": 0,
        "invalid_candidate_identity": 0, "empty_support_crash": 0,
    }


def identity_from_tie(selected: Any) -> str:
    return "NO_ASSIGN" if selected.is_no_assign else f"{selected.agent_id}::{selected.candidate_id}"


def pair_keys(payload: Mapping[str, Any]) -> list[tuple[str, str]]:
    return [(str(row["agent_id"]), str(row["candidate_id"])) for row in payload["metadata"]["candidate_ids"]]


def no_assign_source_index(selection: Any) -> int:
    return int(selection.probabilities.shape[0] - 1)


def canonical_actions_for(payload: Mapping[str, Any], result: Mapping[str, Any], TIE: Any) -> tuple[Any, ...]:
    metadata, tensors = payload["metadata"], payload["tensors"]
    support = int(metadata["selectable_pair_count"])
    return TIE.canonical_actions(
        candidate_ids=metadata["candidate_ids"], pair_scores=result["pair_logits"][0].detach().cpu().tolist(),
        safe_mask=tensors["safe_mask"][0, :support].detach().cpu().tolist(),
        no_assign_score=float(result["no_assign_logit"][0, 0].detach().cpu()),
    )


def normalize_r15_identity(value: str) -> str:
    return "NO_ASSIGN" if str(value) in {"NO_ASSIGN", "NO_ASSIGN_KEEP_CURRENT_PLANS"} else str(value)


def normalized_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        clean = {}
        for field in PROBE_COLUMNS:
            value = row.get(field)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                clean[field] = None
            elif hasattr(value, "item"):
                clean[field] = value.item()
            else:
                clean[field] = value
        result.append(clean)
    return result


def write_probe_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(normalized_rows(rows), key=lambda row: (str(row["cell_id"]), str(row["snapshot_digest"]), int(row["probe_seed"])))
    frame = pd.DataFrame(ordered, columns=PROBE_COLUMNS)
    frame.to_parquet(path, index=False, engine="pyarrow")
    raw = path.read_bytes()
    require(raw[:4] == b"PAR1" and raw[-4:] == b"PAR1", SOURCE_BLOCK, "parquet_magic")
    roundtrip = normalized_rows(pd.read_parquet(path, engine="pyarrow").to_dict("records"))
    require(roundtrip == ordered, SOURCE_BLOCK, "parquet_roundtrip")
    return {"path": path.name, "sha256": sha256(path), "row_count": len(ordered), "columns": PROBE_COLUMNS,
            "content_sha256": canonical_sha256(ordered), "engine": "pyarrow", "roundtrip_passed": True}


def run_focused_tests() -> dict[str, Any]:
    environment = dict(os.environ)
    environment["PYTEST_ADDOPTS"] = "-p no:cacheprovider"
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", str(ROOT / "test_joint_assignment_frozen_policy_selector.py")],
                            cwd=PROJECT, text=True, capture_output=True, env=environment)
    return {"command": f"{sys.executable} -m pytest -q 05_training/test_joint_assignment_frozen_policy_selector.py",
            "exit_code": int(result.returncode), "stdout": result.stdout.strip(), "stderr": result.stderr.strip(),
            "passed": result.returncode == 0}


def make_actor(*, H: Any, config: Mapping[str, Any], state: Mapping[str, torch.Tensor], device: torch.device) -> torch.nn.Module:
    actor = H.CandidateSensitiveMultiAgentCandidateAssignmentHead(
        global_dim=int(config["global_dim"]), demand_dim=int(config["demand_dim"]), agent_dim=int(config["agent_dim"]),
        candidate_dim=int(config["candidate_dim"]), hidden=int(config["hidden"]), heads=int(config["heads"])).to(device)
    actor.load_state_dict(state, strict=True)
    actor.eval()
    return actor


def max_delta(left: torch.Tensor, right: torch.Tensor) -> float:
    require(tuple(left.shape) == tuple(right.shape), INFERENCE_BLOCK, "tensor_shape")
    if torch.equal(left.detach().cpu(), right.detach().cpu()):
        return 0.0
    require(bool(torch.isfinite(left).all().item()) and bool(torch.isfinite(right).all().item()), INFERENCE_BLOCK,
            "nonfinite_tensor_delta")
    return float((left.detach().cpu() - right.detach().cpu()).abs().max().item()) if left.numel() else 0.0


def build_source_path_audit(*, source: Mapping[str, Any]) -> dict[str, Any]:
    existing = {
        "actor_head": "05_training/multi_agent_candidate_assignment_head.py",
        "t1_selector": "05_training/joint_assignment_frozen_tie_break.py",
        "frozen_review_forward": "05_training/run_h4m_ae_ls3_bt8_r1_frozen_policy_discrimination_review.py",
        "historical_s3_training": "05_training/run_h4m_ae_ls3_bt8_r13_s3_four_cell_execution.py",
    }
    source_hashes = {name: {"r15_sha256": git_blob_sha(R15_SOURCE, path), "current_sha256": sha256(PROJECT / path),
                            "unchanged": git_blob_sha(R15_SOURCE, path) == sha256(PROJECT / path)}
                     for name, path in existing.items()}
    callers = [row for row in git(["grep", "-n", "H.select", "--", "05_training"]).splitlines() if row]
    return {
        "source_commit": source["source_commit"], "current_inference_path": {
            "path": existing["frozen_review_forward"], "function": "frozen_forward", "steps": [
                "CandidateSensitiveMultiAgentCandidateAssignmentHead.forward", "H.masked_distribution",
                "TIE.select_exact_tie", "semantic candidate identity or NO_ASSIGN"],
            "semantics": "FROZEN_INFERENCE_T1 exact canonical tie selection"},
        "historical_training_path_preserved": {
            "path": existing["historical_s3_training"], "call_site": "H.select at preserved R13 line 596",
            "function": "multi_agent_candidate_assignment_head.select", "semantics": "historical positional masked argmax; untouched"},
        "new_r16_boundary": {
            "path": SELECTOR_REL, "function": "select_frozen_policy_action", "modes": [
                "FROZEN_INFERENCE_T1", "FROZEN_MASKED_CATEGORICAL_TRAINING"],
            "training_trigger": "generic R15 E1 predicate: deterministic T1 NO_ASSIGN and legal candidate support > 0",
            "unknown_mode": "fail closed", "r16_validation_explicit_call_site": RUNNER_REL,
            "historical_executor_retrofit": "forbidden: R13/F1 execution sources are frozen evidence",
            "future_call_site": "separately authorized R17 executor only"},
        "candidate_identity_representation": "(agent_id, candidate_id), mapped to original source_index only for source probability/log-prob lookup",
        "no_assign_representation": "TIE.NO_ASSIGN_IDENTITY = NO_ASSIGN_KEEP_CURRENT_PLANS; retained as terminal source index",
        "historical_h_select_callers": callers,
        "source_hashes": source_hashes,
    }


def summary_by_band(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["cell_id"]), str(row["time_band"]))].append(row)
    summary = []
    for (cell_id, band), values in sorted(grouped.items()):
        candidate = sum(str(row["selected_identity"]) != "NO_ASSIGN" for row in values)
        no_assign = len(values) - candidate
        summary.append({"cell_id": cell_id, "policy_family": CELL_ROLE[cell_id], "time_band": band,
                        "probe_rows": len(values), "candidate_exposure": candidate, "no_assign_samples": no_assign,
                        "positive_candidate_exposure": candidate > 0,
                        "categorical_triggered_rows": sum(bool(row["categorical_triggered"]) for row in values),
                        "min_non_no_assign_probability_mass": min(1.0 - float(row["no_assign_probability"]) for row in values),
                        "max_non_no_assign_probability_mass": max(1.0 - float(row["no_assign_probability"]) for row in values)})
    return summary


def write_block(*, root: Path, source: Mapping[str, Any], preflight: Mapping[str, Any], reason: str) -> None:
    parquet = write_probe_parquet(root / "training_selection_probe.parquet", [])
    outputs = {
        "evidence_binding_audit.json": preflight,
        "preimplementation_selection_path_audit.json": {"not_run": True},
        "implementation_source_hash_audit.json": {"not_run": True},
        "frozen_inference_equivalence.json": {"not_run": True},
        "categorical_distribution_equivalence.json": {"not_run": True},
        "training_selection_probe_summary.json": {"not_run": True, "parquet": parquet},
        "order_invariance_validation.json": {"not_run": True},
        "rng_reproducibility_validation.json": {"not_run": True},
        "no_assign_integrity_validation.json": {"not_run": True},
        "r16_implementation_contract.json": {"not_selected": True},
        "test_results.json": {"execution_counters": counters(), "hard_failures": [reason], "warnings": []},
        "frozen_hash_before_after.json": {"not_run": True},
        "gate_decision.json": {"stage": STAGE, "gate": reason, "classification": "BLOCKED", "source_commit": source["source_commit"],
                               "hard_failures": [reason], "warnings": [], "global_locks": GLOBAL_LOCKS, "execution_locks": EXECUTION_LOCKS,
                               "next_step": "STOP"},
    }
    for name, value in outputs.items():
        dump(root / name, value)
    (root / "final_report.md").write_text(f"# BT8-R16 blocked\n\n- gate: `{reason}`\n", encoding="utf-8")
    hashes = {item.relative_to(root).as_posix(): sha256(item) for item in root.rglob("*") if item.is_file() and item.name != "manifest.json"}
    dump(root / "manifest.json", {"stage": STAGE, "gate": reason, "source_commit": source["source_commit"], "file_sha256": hashes})
    (root / "_BLOCKED.lock").write_text(reason + "\n", encoding="utf-8")


def main() -> None:
    sys.path.insert(0, str(ROOT))
    import joint_assignment_frozen_policy_selector as S
    import joint_assignment_frozen_policy_snapshot as FPS
    import joint_assignment_frozen_tie_break as TIE
    import multi_agent_candidate_assignment_head as H
    import run_h4m_ae_ls3_bt8_r1_frozen_policy_discrimination_review as R1
    import run_h4m_ae_ls3_bt8_r12_s3_execution_authority_selection as R12MOD

    source = source_provenance()
    root = artifact_root()
    require(not root.exists(), BINDING_BLOCK, "append_only_artifact_collision")
    root.mkdir(parents=True)
    preflight: dict[str, Any] = {
        "stage": STAGE, "source": source, "required_sources": {"r15": R15_SOURCE, "r14": R14_SOURCE, "r13": R13_SOURCE,
        "r12": R12_SOURCE, "r11": R11_SOURCE}, "r15_contract_sha256": R15_CONTRACT_SHA256,
        "s3_contract_sha256": S3_CONTRACT_SHA256, "probe_seed_schedule": {"start": 0, "end": 31, "count": len(PROBE_SEEDS),
        "sha256": canonical_sha256(list(PROBE_SEEDS)), "fixed_before_probe": True}, "global_locks": GLOBAL_LOCKS,
        "execution_locks": EXECUTION_LOCKS,
    }
    try:
        gates = {"r15": load_json(R15 / "gate_decision.json"), "r14": load_json(R14 / "gate_decision.json"),
                 "r13": load_json(R13 / "gate_decision.json"), "r12": load_json(R12 / "gate_decision.json"),
                 "r11": load_json(R11 / "gate_decision.json")}
        manifests = {name: manifest_audit(path) for name, path in {"r15": R15, "r14": R14, "r13": R13, "r12": R12, "r11": R11}.items()}
        require(gates["r15"].get("gate") == R15_GATE and gates["r15"].get("classification") == R15_CLASSIFICATION
                and gates["r15"].get("source_commit") == R15_SOURCE, BINDING_BLOCK, "r15_gate")
        for name, gate, commit in (("r14", R14_GATE, R14_SOURCE), ("r13", R13_GATE, R13_SOURCE),
                                   ("r12", R12_GATE, R12_SOURCE), ("r11", R11_GATE, R11_SOURCE)):
            require(gates[name].get("gate") == gate and gates[name].get("source_commit") == commit, BINDING_BLOCK, f"{name}_gate")
        require(all(item["all_match"] for item in manifests.values()), BINDING_BLOCK, "upstream_manifest")
        r15_contract = load_json(R15 / "selected_minimal_exploration_repair_contract.json")
        r15_frozen = load_json(R15 / "frozen_hash_before_after.json")
        r14_frozen = load_json(R14 / "frozen_hash_before_after.json")
        r13_checkpoints = load_json(R13 / "bt8r13_checkpoint_manifest.json")
        r13_execution = load_json(R13 / "bt8r13_execution_manifest.json")
        r13_credit = load_json(R13 / "bt8r13_candidate_plan_credit.json")
        r12_contract = load_json(R12 / "bt8r12_s3_cell_contract.json")
        r11_contract = load_json(R11 / "bt8r11_selected_seed_repair_contract.json")
        require(r15_contract.get("contract_sha256") == R15_CONTRACT_SHA256 and r15_contract.get("repair_level") == "E1"
                and r15_contract.get("policy_probabilities_modified") is False, BINDING_BLOCK, "r15_contract")
        require(r13_execution.get("r11_s3_contract_sha256") == S3_CONTRACT_SHA256
                and r12_contract.get("s3_contract_sha256") == S3_CONTRACT_SHA256
                and r11_contract.get("sha256") == S3_CONTRACT_SHA256, BINDING_BLOCK, "s3_contract")
        require(r13_credit.get("identity_chain_all") is True and int(r13_credit.get("mismatches", -1)) == 0, BINDING_BLOCK, "r13_credit_identity")
        require(r14_frozen.get("all_unchanged") is True and r14_frozen.get("extra_all_unchanged") is True
                and r14_frozen.get("checkpoint_unchanged") is True and r15_frozen.get("all_unchanged") is True
                and r15_frozen.get("extra_all_unchanged") is True and r15_frozen.get("checkpoint_unchanged") is True,
                BINDING_BLOCK, "upstream_frozen_evidence")
        require(source["source_lineage_descends_from_r15"] and source["source_only_local_commit"]
                and not source["git_status_porcelain"], BINDING_BLOCK, "source_lineage_or_status")

        review_collection = load_json(F1 / "bt8f1_review_snapshots" / "collection_manifest.json")
        review_for_digest = dict(review_collection); collection_digest = review_for_digest.pop("collection_digest", None)
        require(collection_digest == FPS.canonical_sha256(review_for_digest) == REVIEW_COLLECTION_DIGEST, BINDING_BLOCK, "review_collection_digest")
        require(int(review_collection.get("snapshot_count", -1)) == len(review_collection.get("entries", [])) == 6
                and len({entry.get("snapshot_digest") for entry in review_collection["entries"]}) == 6, BINDING_BLOCK, "review_snapshots")
        review_records: list[dict[str, Any]] = []
        for entry in review_collection["entries"]:
            payload = FPS.load_snapshot(F1 / "bt8f1_review_snapshots" / str(entry["relative_path"]))
            require(payload["snapshot_digest"] == entry["snapshot_digest"], BINDING_BLOCK, "review_snapshot_digest")
            review_records.append({"payload": payload, "snapshot_digest": str(payload["snapshot_digest"]),
                                   "time_band": str(payload["metadata"]["time_band"]), "entry": entry})
        require({record["time_band"] for record in review_records} == set(TIME_BANDS), BINDING_BLOCK, "review_time_bands")

        raw_states: dict[str, Mapping[str, torch.Tensor]] = {}
        checkpoint_before: dict[str, str] = {}
        for kind in ("initial", "final"):
            for cell_id in CELL_ROLE:
                key, entry = f"{kind}:{cell_id}", r13_checkpoints[kind].get(cell_id)
                require(isinstance(entry, Mapping), BINDING_BLOCK, f"checkpoint_entry={key}")
                path = R13 / str(entry["path"])
                digest = sha256(path) if path.is_file() else ""
                require(digest == str(entry["sha256"]) == EXPECTED_CHECKPOINTS[key], BINDING_BLOCK, f"checkpoint_hash={key}")
                require(r15_frozen["checkpoint_before"].get(key) == digest, BINDING_BLOCK, f"r15_checkpoint_binding={key}")
                stored = torch.load(path, map_location="cpu", weights_only=False)
                require(isinstance(stored, Mapping) and isinstance(stored.get("actor"), Mapping), BINDING_BLOCK, f"checkpoint_schema={key}")
                raw_states[key] = stored["actor"]
                checkpoint_before[key] = digest
        require(state_dicts_exact(raw_states["initial:AC-R1"], raw_states["initial:AC-R2"]), BINDING_BLOCK, "ac_initial_pair")
        require(state_dicts_exact(raw_states["initial:BD-R1"], raw_states["initial:BD-R2"]), BINDING_BLOCK, "bd_initial_pair")
        require(state_dicts_exact(raw_states["initial:BD-R1"], raw_states["final:BD-R1"])
                and state_dicts_exact(raw_states["initial:BD-R1"], raw_states["final:BD-R2"]), BINDING_BLOCK, "bd_immutable_pair")

        training_collection = load_json(R13 / "bt8r13_training_snapshots" / "collection_manifest.json")
        training_for_digest = dict(training_collection); training_digest = training_for_digest.pop("collection_digest", None)
        entries = list(training_collection.get("entries", []))
        require(training_digest == FPS.canonical_sha256(training_for_digest) == TRAINING_COLLECTION_DIGEST, BINDING_BLOCK, "training_collection_digest")
        require(int(training_collection.get("snapshot_count", -1)) == len(entries) == 96
                and len({entry.get("snapshot_digest") for entry in entries}) == 96, BINDING_BLOCK, "training_snapshots")
        credit_by_snapshot = {str(row["snapshot_digest"]): row for row in r13_credit.get("rows", [])}
        require(len(credit_by_snapshot) == 96, BINDING_BLOCK, "training_credit_rows")
        config: Mapping[str, Any] | None = None
        training_records: list[dict[str, Any]] = []
        for entry in entries:
            payload = FPS.load_snapshot(R13 / "bt8r13_training_snapshots" / str(entry["relative_path"]))
            metadata, tensors = payload["metadata"], payload["tensors"]
            tokens = str(metadata["decision_id"]).split(":")
            require(len(tokens) >= 2 and tokens[1] in CELL_ROLE, BINDING_BLOCK, "training_cell_identity")
            cell_id, support = tokens[1], int(metadata["selectable_pair_count"])
            require(payload["snapshot_digest"] == entry["snapshot_digest"] and support > 0
                    and bool(tensors["safe_mask"][0, :support].all().item()), BINDING_BLOCK, "training_snapshot_support")
            require(metadata["candidate_ids"] == metadata["candidate_order"] and int(metadata["no_assign_index"]) == support,
                    BINDING_BLOCK, "training_candidate_identity")
            require(str(payload["snapshot_digest"]) in credit_by_snapshot
                    and str(credit_by_snapshot[str(payload["snapshot_digest"])]["cell_id"]) == cell_id, BINDING_BLOCK, "training_credit_binding")
            require(metadata["actor_config"] == (config if config is not None else metadata["actor_config"]), BINDING_BLOCK, "actor_config")
            config = metadata["actor_config"]
            training_records.append({"payload": payload, "entry": entry, "snapshot_digest": str(payload["snapshot_digest"]),
                                     "cell_id": cell_id, "policy_family": CELL_ROLE[cell_id], "decision_id": str(metadata["decision_id"]),
                                     "window_id": str(metadata["window_id"]), "time_band": str(metadata["time_band"]),
                                     "environment_seed": int(metadata["seed"]), "support_size": support})
        require(config is not None and config.get("actor_head_id") == H.CANDIDATE_SENSITIVE_HEAD_ID
                and config.get("actor_head_version") == H.CANDIDATE_SENSITIVE_HEAD_VERSION, BINDING_BLOCK, "v2_actor_config")
        train_counts = Counter((record["cell_id"], record["time_band"]) for record in training_records)
        require(all(train_counts[(cell, band)] == 8 for cell in CELL_ROLE for band in TIME_BANDS), BINDING_BLOCK, "training_time_bands")

        r15_probe = pd.read_parquet(R15 / "exploration_probe_results.parquet", engine="pyarrow")
        e1_frame = r15_probe.loc[r15_probe["repair_level"] == "E1"]
        require(len(e1_frame) == 96 * len(PROBE_SEEDS), BINDING_BLOCK, "r15_e1_probe_rows")
        r15_identity = {(str(row.cell_id), str(row.snapshot_digest), int(row.probe_seed)): normalize_r15_identity(str(row.sampled_identity))
                        for row in e1_frame.itertuples(index=False)}
        require(len(r15_identity) == 96 * len(PROBE_SEEDS), BINDING_BLOCK, "r15_e1_probe_identity")

        current_frozen = R12MOD.frozen_hashes()
        current_extra = {"candidate_plan_bridge": sha256(ROOT / "joint_candidate_plan_causal_bridge.py"),
                         "e1_eligibility": sha256(ROOT / "joint_assignment_e1_eligibility.py"),
                         "t1_selector": sha256(ROOT / "joint_assignment_frozen_tie_break.py")}
        require(current_frozen == r15_frozen.get("before") and current_extra == r15_frozen.get("extra_before"),
                BINDING_BLOCK, "current_frozen_hash_binding")
        source_path = build_source_path_audit(source=source)
        require(all(value["unchanged"] for value in source_path["source_hashes"].values()), BINDING_BLOCK, "frozen_source_hash_binding")
        preflight |= {"upstream_manifests": manifests, "authority_gates": {name: True for name in gates},
                      "review_collection_digest": collection_digest, "review_snapshot_count": len(review_records),
                      "training_collection_digest": training_digest, "training_snapshot_count": len(training_records),
                      "training_time_band_counts": {f"{cell}:{band}": train_counts[(cell, band)] for cell in CELL_ROLE for band in TIME_BANDS},
                      "checkpoint_before": checkpoint_before, "frozen_hash_binding": True,
                      "source_path_audit_ready": True, "r15_e1_probe_rows": len(r15_identity)}
    except R16Error as exc:
        write_block(root=root, source=source, preflight=preflight | {"hard_failures": [str(exc)]}, reason=exc.code)
        print(f"[BLOCKED] {exc.code}"); print(f"artifact: {root.relative_to(PROJECT)}"); return
    except Exception as exc:  # noqa: BLE001
        write_block(root=root, source=source, preflight=preflight | {"hard_failures": [repr(exc)]}, reason=BINDING_BLOCK)
        print(f"[BLOCKED] {BINDING_BLOCK}"); print(f"artifact: {root.relative_to(PROJECT)}"); return

    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        write_block(root=root, source=source, preflight=preflight | {"mps_built": torch.backends.mps.is_built(), "mps_available": torch.backends.mps.is_available()},
                    reason=MPS_BLOCK)
        print(f"[BLOCKED] {MPS_BLOCK}"); print(f"artifact: {root.relative_to(PROJECT)}"); return

    device = torch.device("mps:0")
    execution = counters()
    hard: list[str] = []
    try:
        tests = run_focused_tests()
        require(tests["passed"], SOURCE_BLOCK, "focused_selector_tests")

        actor_before: dict[str, str] = {}
        actor_after_review: dict[str, str] = {}
        inference_rows: list[dict[str, Any]] = []
        inference_order = Counter({"candidate": 0, "agent": 0, "combined": 0})
        for state_key, state in raw_states.items():
            actor = make_actor(H=H, config=config, state=state, device=device)
            actor_digest = module_digest(actor)
            actor_before[state_key] = actor_digest
            for record in review_records:
                payload = record["payload"]
                legacy = R1.frozen_forward(actor, payload, device=device, head=H, tie=TIE)
                execution["inference_frozen_forwards"] += 1
                support = int(legacy["support_size"])
                mask = payload["tensors"]["safe_mask"][:, :support].to(device)
                post = S.select_frozen_policy_action(pair_keys=pair_keys(payload), pair_logits=legacy["pair_logits"].to(device),
                                                      no_assign_logit=legacy["no_assign_logit"].to(device), safe_mask=mask,
                                                      mode=S.FROZEN_INFERENCE_T1,
                                                      masked_probabilities=legacy["probabilities"].to(device))
                pre_masked = torch.where(mask[0].detach().cpu(), legacy["pair_logits"][0], torch.full_like(legacy["pair_logits"][0], float("-inf")))
                post_masked = torch.where(post.safe_mask.detach().cpu(), post.pair_logits.detach().cpu(), torch.full_like(post.pair_logits.detach().cpu(), float("-inf")))
                exact = (torch.equal(post.pair_logits.detach().cpu(), legacy["pair_logits"][0])
                         and torch.equal(post.no_assign_logit.detach().cpu(), legacy["no_assign_logit"][0])
                         and torch.equal(post.probabilities.detach().cpu(), legacy["probabilities"][0])
                         and torch.equal(post.safe_mask.detach().cpu(), mask[0].detach().cpu())
                         and torch.equal(pre_masked, post_masked)
                         and identity_from_tie(post.selected) == identity_from_tie(legacy["selection"].selected)
                         and post.selected_index == int(legacy["selection"].selected.source_index)
                         and bool(post.deterministic_selection.exact_tie) == bool(legacy["selection"].exact_tie))
                require(exact, INFERENCE_BLOCK, f"review={record['snapshot_digest']}:{state_key}")
                inference_rows.append({"checkpoint": state_key, "snapshot_digest": record["snapshot_digest"], "time_band": record["time_band"],
                                       "tensor_delta": 0.0, "logit_delta": max_delta(post.pair_logits.detach().cpu(), legacy["pair_logits"][0]),
                                       "masked_logit_delta": max_delta(post_masked, pre_masked),
                                       "probability_delta": max_delta(post.probabilities.detach().cpu(), legacy["probabilities"][0]),
                                       "selection_equal": True, "t1_exact_tie_equal": True,
                                       "no_assign_equal": post.selected_is_no_assign == legacy["selection"].selected.is_no_assign,
                                       "legal_support_equal": True, "zero_loss_support_equal": True})
                base_identity = identity_from_tie(post.selected)
                for permutation_name, permuted in {
                    "candidate": R1.candidate_permutation(payload),
                    "agent": R1.agent_permutation(payload),
                    "combined": R1.candidate_permutation(R1.agent_permutation(payload)),
                }.items():
                    result = R1.frozen_forward(actor, permuted, device=device, head=H, tie=TIE)
                    execution["order_probe_frozen_forwards"] += 1
                    psupport = int(result["support_size"])
                    ppost = S.select_frozen_policy_action(pair_keys=pair_keys(permuted), pair_logits=result["pair_logits"].to(device),
                                                           no_assign_logit=result["no_assign_logit"].to(device),
                                                           safe_mask=permuted["tensors"]["safe_mask"][:, :psupport].to(device),
                                                           mode=S.FROZEN_INFERENCE_T1,
                                                           masked_probabilities=result["probabilities"].to(device))
                    inference_order[permutation_name] += int(identity_from_tie(ppost.selected) != base_identity)
            actor_after_review[state_key] = module_digest(actor)
            require(actor_after_review[state_key] == actor_before[state_key], SOURCE_BLOCK, f"review_actor_mutation={state_key}")
            del actor
        torch.mps.synchronize()
        require(not any(inference_order.values()), ORDER_BLOCK, "inference_permutation")

        training_actors = {
            "AC": make_actor(H=H, config=config, state=raw_states["initial:AC-R1"], device=device),
            "BD": make_actor(H=H, config=config, state=raw_states["initial:BD-R1"], device=device),
        }
        training_actor_digest = {family: module_digest(actor) for family, actor in training_actors.items()}
        selector_rng_before = torch.random.get_rng_state().clone()
        probe_rows: list[dict[str, Any]] = []
        base_selection: dict[tuple[str, int], str] = {}
        distribution_deltas: list[float] = []
        log_deltas: list[float] = []
        support_mismatches = 0
        no_assign_support_mismatches = 0
        no_assign_probability_overwrites = 0
        no_assign_logit_overwrites = 0
        same_seed_mismatches = 0
        ac_trigger_count = 0
        identities_by_snapshot: dict[str, set[str]] = defaultdict(set)
        for record in training_records:
            actor = training_actors[record["policy_family"]]
            legacy = R1.frozen_forward(actor, record["payload"], device=device, head=H, tie=TIE)
            execution["training_probe_frozen_forwards"] += 1
            support = int(legacy["support_size"])
            mask = record["payload"]["tensors"]["safe_mask"][:, :support].to(device)
            source_probability = legacy["probabilities"][0]
            source_probability_view = legacy["probabilities"].to(device)[0]
            source_no_assign_logit = legacy["no_assign_logit"][0, 0]
            candidates = canonical_actions_for(record["payload"], legacy, TIE)
            legal_candidate_count = sum(not action.is_no_assign for action in candidates)
            for seed in PROBE_SEEDS:
                selected = S.select_frozen_policy_action(pair_keys=pair_keys(record["payload"]), pair_logits=legacy["pair_logits"].to(device),
                                                          no_assign_logit=legacy["no_assign_logit"].to(device), safe_mask=mask,
                                                          mode=S.FROZEN_MASKED_CATEGORICAL_TRAINING,
                                                          snapshot_identity=record["snapshot_digest"], probe_seed=seed,
                                                          masked_probabilities=legacy["probabilities"].to(device))
                replay = S.select_frozen_policy_action(pair_keys=pair_keys(record["payload"]), pair_logits=legacy["pair_logits"].to(device),
                                                        no_assign_logit=legacy["no_assign_logit"].to(device), safe_mask=mask,
                                                        mode=S.FROZEN_MASKED_CATEGORICAL_TRAINING,
                                                        snapshot_identity=record["snapshot_digest"], probe_seed=seed,
                                                        masked_probabilities=legacy["probabilities"].to(device))
                execution["selector_calls"] += 2
                selected_identity = identity_from_tie(selected.selected)
                deterministic_identity = identity_from_tie(selected.deterministic_selection.selected)
                expected_identity = r15_identity[(record["cell_id"], record["snapshot_digest"], seed)]
                same_seed = selected_identity == identity_from_tie(replay.selected)
                same_seed_mismatches += int(not same_seed)
                base_selection[(record["snapshot_digest"], seed)] = selected_identity
                identities_by_snapshot[record["snapshot_digest"]].add(selected_identity)
                probability_equal = torch.equal(selected.probabilities.detach().cpu(), source_probability)
                delta = max_delta(selected.probabilities.detach().cpu(), source_probability)
                distribution_deltas.append(delta)
                support_mismatches += int(not probability_equal or int(selected.probabilities.shape[0]) != support + 1)
                no_assign_support_mismatches += int(not bool(selected.probabilities[-1].detach().cpu() > 0))
                no_assign_probability_overwrites += int(float(selected.probabilities[-1].detach().cpu()) != float(source_probability[-1]))
                no_assign_logit_overwrites += int(float(selected.no_assign_logit[0].detach().cpu()) != float(source_no_assign_logit.detach().cpu()))
                selected_probability = selected.probabilities[selected.selected_index]
                # The source probability view was produced by the frozen
                # actor-forward path and is passed unchanged to the selector.
                # Evaluate both log references on that same view/device rather
                # than confuse the equivalence gate with CPU-vs-MPS log kernels.
                source_log = torch.log(source_probability_view[selected.selected_index])
                log_delta = abs(float(selected.log_probability.detach().cpu()) - float(source_log.detach().cpu()))
                log_deltas.append(log_delta)
                legal = selected.selected_is_no_assign or bool(mask[0, selected.selected_index].item())
                finite = bool(torch.isfinite(selected.probabilities).all().item() and torch.isfinite(selected.log_probability).item())
                execution["illegal_selection"] += int(not legal)
                execution["nan_or_inf"] += int(not finite)
                execution["zero_loss_violation"] += int(not legal)
                execution["invalid_candidate_identity"] += int((not selected.selected_is_no_assign)
                                                                 and selected.selected_pair not in set(pair_keys(record["payload"])))
                if record["policy_family"] == "AC": ac_trigger_count += int(selected.categorical_triggered)
                probe_rows.append({
                    "stage": STAGE, "cell_id": record["cell_id"], "policy_family": record["policy_family"],
                    "snapshot_digest": record["snapshot_digest"], "snapshot_manifest_sha256": str(record["entry"]["snapshot_manifest_sha256"]),
                    "decision_id": record["decision_id"], "window_id": record["window_id"], "time_band": record["time_band"],
                    "environment_seed": record["environment_seed"], "probe_seed": seed, "selection_mode": selected.selection_mode,
                    "categorical_triggered": selected.categorical_triggered, "trigger_reason": selected.trigger_reason,
                    "deterministic_identity": deterministic_identity, "selected_identity": selected_identity,
                    "selected_index": selected.selected_index, "source_probability": float(source_probability[selected.selected_index]),
                    "returned_probability": float(selected_probability.detach().cpu()), "source_log_probability": float(source_log.detach().cpu()),
                    "returned_log_probability": float(selected.log_probability.detach().cpu()), "log_probability_abs_delta": log_delta,
                    "support_size": support, "legal_candidate_count": legal_candidate_count, "no_assign_in_support": True,
                    "no_assign_probability": float(selected.probabilities[-1].detach().cpu()), "no_assign_source_index": no_assign_source_index(selected),
                    "no_assign_removed": False, "finite": finite, "legal": legal, "zero_loss_feasible": legal,
                    "r15_identity_match": selected_identity == expected_identity, "same_seed_identity_match": same_seed,
                    "canonical_rng_keyset_sha256": selected.canonical_rng_keyset_sha256 or "", "actor_state_sha256": training_actor_digest[record["policy_family"]],
                })
        selector_rng_after = torch.random.get_rng_state().clone()
        torch.mps.synchronize()
        require(torch.equal(selector_rng_before, selector_rng_after) and same_seed_mismatches == 0, RNG_BLOCK, "selector_rng")
        require(max(distribution_deltas, default=0.0) <= PROBABILITY_ABS_TOLERANCE and support_mismatches == 0
                and max(log_deltas, default=0.0) <= LOG_PROBABILITY_ABS_TOLERANCE, DISTRIBUTION_BLOCK,
                f"source_probability_or_logprob:prob_max={max(distribution_deltas, default=0.0)}:support={support_mismatches}:log_max={max(log_deltas, default=0.0)}")
        require(no_assign_support_mismatches == 0 and no_assign_probability_overwrites == 0 and no_assign_logit_overwrites == 0,
                NO_ASSIGN_BLOCK, "no_assign")
        require(all(bool(row["r15_identity_match"]) for row in probe_rows), EXPOSURE_BLOCK, "r15_canonical_replay")
        require(ac_trigger_count == 0, DISTRIBUTION_BLOCK, "ac_control_trigger")

        training_order = Counter({"candidate": 0, "agent": 0, "combined": 0})
        for record in training_records:
            actor = training_actors[record["policy_family"]]
            payload = record["payload"]
            for permutation_name, permuted in {
                "candidate": R1.candidate_permutation(payload),
                "agent": R1.agent_permutation(payload),
                "combined": R1.candidate_permutation(R1.agent_permutation(payload)),
            }.items():
                legacy = R1.frozen_forward(actor, permuted, device=device, head=H, tie=TIE)
                execution["order_probe_frozen_forwards"] += 1
                psupport = int(legacy["support_size"])
                pmask = permuted["tensors"]["safe_mask"][:, :psupport].to(device)
                for seed in PROBE_SEEDS:
                    selected = S.select_frozen_policy_action(pair_keys=pair_keys(permuted), pair_logits=legacy["pair_logits"].to(device),
                                                              no_assign_logit=legacy["no_assign_logit"].to(device), safe_mask=pmask,
                                                              mode=S.FROZEN_MASKED_CATEGORICAL_TRAINING,
                                                              snapshot_identity=record["snapshot_digest"], probe_seed=seed,
                                                              masked_probabilities=legacy["probabilities"].to(device))
                    execution["selector_calls"] += 1
                    training_order[permutation_name] += int(identity_from_tie(selected.selected) != base_selection[(record["snapshot_digest"], seed)])
        torch.mps.synchronize()
        require(not any(training_order.values()), ORDER_BLOCK, "training_permutation")
        exposure = summary_by_band(probe_rows)
        exposure_by_key = {(row["cell_id"], row["time_band"]): int(row["candidate_exposure"]) for row in exposure}
        require(all(exposure_by_key.get(key) == expected for key, expected in EXPECTED_BD_EXPOSURE.items()), EXPOSURE_BLOCK, "bd_time_band_exposure")
        require(all(row["candidate_exposure"] > 0 for row in exposure if str(row["cell_id"]).startswith("BD-")), EXPOSURE_BLOCK, "bd_positive_exposure")
        require(execution["illegal_selection"] == execution["zero_loss_violation"] == execution["nan_or_inf"] == execution["invalid_candidate_identity"] == 0,
                LEGALITY_BLOCK, "probe_integrity")

        actor_after = {family: module_digest(actor) for family, actor in training_actors.items()}
        execution["parameter_mutation"] = sum(training_actor_digest[key] != actor_after[key] for key in training_actors)
        for actor in training_actors.values(): del actor
        checkpoint_after = {key: sha256(R13 / str(r13_checkpoints[kind][cell]["path"]))
                            for kind in ("initial", "final") for cell in CELL_ROLE for key in [f"{kind}:{cell}"]}
        execution["checkpoint_mutation"] = sum(checkpoint_before[key] != checkpoint_after[key] for key in checkpoint_before)
        frozen_after = R12MOD.frozen_hashes()
        extra_after = {"candidate_plan_bridge": sha256(ROOT / "joint_candidate_plan_causal_bridge.py"),
                       "e1_eligibility": sha256(ROOT / "joint_assignment_e1_eligibility.py"),
                       "t1_selector": sha256(ROOT / "joint_assignment_frozen_tie_break.py")}
        require(current_frozen == frozen_after and current_extra == extra_after and checkpoint_before == checkpoint_after
                and execution["parameter_mutation"] == execution["checkpoint_mutation"] == 0, SOURCE_BLOCK, "frozen_or_checkpoint_mutation")
        forbidden = ("training", "optimizer_creation", "optimizer_step", "backward", "loss_update", "causal_rollout", "simulator_execution",
                     "candidate_generation", "candidate_regeneration", "local_search_rerun", "zero_loss_reevaluation", "reward_settlement",
                     "parameter_mutation", "checkpoint_write", "checkpoint_mutation", "test6_access", "github_push")
        require(all(execution[key] == 0 for key in forbidden), SOURCE_BLOCK, "forbidden_execution_counter")

        parquet = write_probe_parquet(root / "training_selection_probe.parquet", probe_rows)
        implementation_hashes = {
            SELECTOR_REL: {"before_r15_exists": False, "before_r15_sha256": None, "after_sha256": sha256(ROOT / Path(SELECTOR_REL).name)},
            RUNNER_REL: {"before_r15_exists": False, "before_r15_sha256": None, "after_sha256": sha256(ROOT / Path(RUNNER_REL).name)},
            TEST_REL: {"before_r15_exists": False, "before_r15_sha256": None, "after_sha256": sha256(ROOT / Path(TEST_REL).name)},
        }
        contract = {
            "selected_repair": "E1_FROZEN_POLICY_MASKED_CATEGORICAL_SAMPLING", "source_contract_sha256": R15_CONTRACT_SHA256,
            "selector_contract_id": S.R15_E1_CONTRACT_ID, "E2_enabled": False, "E3_enabled": False,
            "inference_mode": S.FROZEN_INFERENCE_T1, "training_selection_mode": S.FROZEN_MASKED_CATEGORICAL_TRAINING,
            "training_trigger": "generic R15 E1 predicate only; no BD/cell label special case",
            "policy_probabilities_modified": False, "policy_parameters_modified": False,
            "training_authorized": False, "optimizer_authorized": False, "actual_execution_authorized": False,
            "separate_bounded_training_authorization_required": True, "automatic_training_authorization": False,
            "historical_training_executor_retrofitted": False,
            "integration_boundary": "implemented and exercised by this R16 validation; a new R17 executor must explicitly request training mode",
            "historical_executor_preservation_reason": "R13/F1 execution sources are bound frozen evidence and cannot be retrofitted",
            "contract_sha256": "PENDING",
        }
        contract["contract_sha256"] = canonical_sha256({key: value for key, value in contract.items() if key != "contract_sha256"})
        order = {"inference": {"candidate_order_failures": int(inference_order["candidate"]), "agent_order_failures": int(inference_order["agent"]),
                                 "combined_order_failures": int(inference_order["combined"])},
                 "training": {"candidate_order_failures": int(training_order["candidate"]), "agent_order_failures": int(training_order["agent"]),
                              "combined_order_failures": int(training_order["combined"])},
                 "total": {"candidate_order_failures": int(inference_order["candidate"] + training_order["candidate"]),
                           "agent_order_failures": int(inference_order["agent"] + training_order["agent"]),
                           "combined_order_failures": int(inference_order["combined"] + training_order["combined"])},
                 "passed": not any(inference_order.values()) and not any(training_order.values()),
                 "rule": "semantic selected identity, not list index, must match for every fixed seed; no score tolerance was introduced."}
        rng = {"same_seed_replay_mismatch_count": same_seed_mismatches, "global_torch_rng_unchanged": True,
               "canonical_rng": "SHA-256 semantic exponential race; no Python/NumPy/Torch global RNG consumption",
               "different_seed_diverse_snapshot_count": sum(len(values) > 1 for values in identities_by_snapshot.values()),
               "different_seed_diversity_may_occur": True}
        outputs = {
            "evidence_binding_audit.json": preflight | {"mps_built": True, "mps_available": True, "device": "mps:0"},
            "preimplementation_selection_path_audit.json": source_path,
            "implementation_source_hash_audit.json": {"modified_source_hashes": implementation_hashes,
                                                        "unchanged_bound_source_hashes": source_path["source_hashes"],
                                                        "minimum_source_surface": sorted(SOURCE_FILES)},
            "frozen_inference_equivalence.json": {"review_snapshot_count": len(review_records), "checkpoint_count": len(raw_states),
                                                    "comparison_rows": inference_rows, "tensor_delta_max": 0.0,
                                                    "actor_tensor_digest_before": actor_before, "actor_tensor_digest_after": actor_after_review,
                                                    "actor_tensor_mutation_count": sum(actor_before[key] != actor_after_review[key] for key in actor_before),
                                                    "logit_delta_max": max((float(row["logit_delta"]) for row in inference_rows), default=0.0),
                                                    "masked_logit_delta_max": max((float(row["masked_logit_delta"]) for row in inference_rows), default=0.0),
                                                    "probability_delta_max": max((float(row["probability_delta"]) for row in inference_rows), default=0.0),
                                                    "selection_mismatches": 0, "t1_tie_semantics_mismatches": 0, "passed": True},
            "categorical_distribution_equivalence.json": {"reviewed_training_snapshot_count": len(training_records), "probe_row_count": len(probe_rows),
                                                            "probability_abs_tolerance": PROBABILITY_ABS_TOLERANCE,
                                                            "max_abs_probability_delta": max(distribution_deltas, default=0.0), "support_mismatch": support_mismatches,
                                                            "no_assign_support_mismatch": no_assign_support_mismatches,
                                                            "log_probability_abs_tolerance": LOG_PROBABILITY_ABS_TOLERANCE,
                                                            "max_abs_log_probability_delta": max(log_deltas, default=0.0), "passed": True},
            "training_selection_probe_summary.json": {"parquet": parquet, "r15_exact_identity_replay": True, "time_band_exposure": exposure,
                                                        "bd_expected_exposure": {f"{cell}:{band}": count for (cell, band), count in EXPECTED_BD_EXPOSURE.items()},
                                                        "ac_categorical_trigger_count": ac_trigger_count, "no_reward_or_kpi_interpretation": True},
            "order_invariance_validation.json": order,
            "rng_reproducibility_validation.json": rng,
            "no_assign_integrity_validation.json": {"no_assign_removed": 0, "no_assign_probability_overwritten": no_assign_probability_overwrites,
                                                       "no_assign_logit_overwritten": no_assign_logit_overwrites, "no_assign_inference_semantics_changed": 0,
                                                       "no_assign_only_fixture_preserved": True, "passed": True},
            "r16_implementation_contract.json": contract,
            "test_results.json": {"focused_selector_tests": tests, "execution_counters": execution, "hard_failures": [], "warnings": [],
                                  "training_authorized": False, "github_push_performed": False, "performance_or_kpi_interpretation_performed": False},
            "frozen_hash_before_after.json": {"before": current_frozen, "after": frozen_after, "all_unchanged": current_frozen == frozen_after,
                                                "extra_before": current_extra, "extra_after": extra_after, "extra_all_unchanged": current_extra == extra_after,
                                                "checkpoint_before": checkpoint_before, "checkpoint_after": checkpoint_after,
                                                "checkpoint_unchanged": checkpoint_before == checkpoint_after},
            "gate_decision.json": {"stage": STAGE, "gate": PASS_GATE, "classification": CLASSIFICATION, "source_commit": source["source_commit"],
                                   "hard_failures": [], "warnings": [], "global_locks": GLOBAL_LOCKS, "execution_locks": EXECUTION_LOCKS,
                                   "next_step": "R17 = E1 bounded-training execution authorization / envelope freeze review"},
        }
        for name, value in outputs.items(): dump(root / name, value)
        (root / "final_report.md").write_text(
            f"# BT8-R16 final report\n\n- gate: `{PASS_GATE}`\n- classification: `{CLASSIFICATION}`\n- source commit: `{source['source_commit']}`\n\n"
            "R16 implemented only the generic R15 E1 selection boundary. No training, optimizer, causal rollout, reward settlement, checkpoint mutation, or authorization was performed.\n\n"
            "## Required answers\n\n"
            "1. **Q1 — E1 implementation:** Yes. It adds no Actor logit, weight, or probability modification.\n"
            "2. **Q2 — frozen inference/T1:** Yes. The pre/post tensor, logit, masked-logit, probability, support, selected identity, and T1 tie checks are exact.\n"
            "3. **Q3 — NO_ASSIGN:** Yes. It remains a legal terminal action with its original logit and probability.\n"
            "4. **Q4 — training distribution:** Yes. The generic R15 deadlock branch samples the original full masked categorical distribution, including NO_ASSIGN.\n"
            "5. **Q5 — log probability:** Yes. The returned value is `log(source masked probability[selected source index])`.\n"
            "6. **Q6 — same seed:** Yes. Canonical identity-keyed replay is exact and does not advance global Torch RNG.\n"
            "7. **Q7 — order invariance:** Yes. Candidate, agent, and combined semantic-identity failures are zero.\n"
            "8. **Q8 — BD exposure:** Yes. Every BD-R1/BD-R2 time band reproduces the positive R15 candidate exposure.\n"
            "9. **Q9 — immutable authorities:** Yes. Reward V2, Zero-Loss, Local Search, GATv2, bound head/T1 sources, and all eight checkpoints are unchanged.\n"
            "10. **Q10 — next step:** Yes. Only separately authorized `R17 = E1 bounded-training execution authorization / envelope freeze review` is permissible; no historical executor was retrofitted.\n",
            encoding="utf-8")
        manifest = {item.relative_to(root).as_posix(): sha256(item) for item in root.rglob("*") if item.is_file() and item.name != "manifest.json"}
        dump(root / "manifest.json", {"stage": STAGE, "gate": PASS_GATE, "classification": CLASSIFICATION,
                                       "source_commit": source["source_commit"], "file_sha256": manifest, "github_push_performed": False})
        (root / "_SUCCESS.lock").write_text(PASS_GATE + "\n", encoding="utf-8")
        print(f"[PASS] {PASS_GATE}")
        print(f"classification: {CLASSIFICATION}")
        print(f"artifact: {root.relative_to(PROJECT)}")
    except R16Error as exc:
        write_block(root=root, source=source, preflight=preflight | {"hard_failures": [str(exc)]}, reason=exc.code)
        print(f"[BLOCKED] {exc.code}"); print(f"artifact: {root.relative_to(PROJECT)}")
    except Exception as exc:  # noqa: BLE001
        write_block(root=root, source=source, preflight=preflight | {"hard_failures": [repr(exc)]}, reason=SOURCE_BLOCK)
        print(f"[BLOCKED] {SOURCE_BLOCK}"); print(f"artifact: {root.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
