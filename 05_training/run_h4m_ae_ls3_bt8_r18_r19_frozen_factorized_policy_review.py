#!/usr/bin/env python3
"""R18-R19 same-input frozen factorized-policy initial/final review.

This audit is inference only.  It rebuilds the deterministic factorized initial
Actor and loads the R18-R18B final Actor, then runs both on exactly the same
three SHA-bound frozen review snapshots per arm.  Gate movement and conditional
candidate-head discrimination are reported as distinct quantities.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
sys.path.insert(0, str(ROOT))

import joint_assignment_frozen_policy_snapshot as FPS  # noqa: E402
import joint_assignment_frozen_tie_break as TIE  # noqa: E402
import multi_agent_candidate_assignment_head as H  # noqa: E402
import run_h4m_ae_ls3_bt8_r18_e1_bounded_training as R18  # noqa: E402
import run_h4m_ae_ls3_bt8_r18_r18b_external_mps_one_shot_authorization as R18B  # noqa: E402


STAGE = "H4M-AE-R9.8-LS3-BT8-R18-R19"
PASS_GATE = (
    "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_R19_"
    "SAME_INPUT_FROZEN_FACTORIZED_POLICY_GATE_AND_CANDIDATE_HEAD_REVIEW_COMPLETE"
)
CLASSIFICATION = "A_GATE_SHIFT_AND_CONDITIONAL_CANDIDATE_DISCRIMINATION_SEPARATED_ON_SAME_FROZEN_INPUTS"
BLOCK_BINDING = "BLOCKED_R18R19_UPSTREAM_EVIDENCE_BINDING_FAILURE"
BLOCK_MPS = "BLOCKED_R18R19_EXTERNAL_MPS_PREFLIGHT_FAILURE"
BLOCK_REPLAY = "BLOCKED_R18R19_FROZEN_FACTORIZED_REPLAY_FAILURE"
BLOCK_DECOMPOSITION = "BLOCKED_R18R19_GATE_CANDIDATE_DECOMPOSITION_FAILURE"

R18R18B_AUTH_ROOT = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r18b_external_mps_one_shot_authorization_20260828_203601+09:00"
R18R18B_EXEC_ROOT = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r18b_factorized_actor_bounded_training_execution_20260828_203601+09:00"
AUTH_PATH = R18R18B_AUTH_ROOT / "r18r18b_factorized_one_shot_authorization_manifest.json"
SOURCE_AT_EXECUTION = "562ac7b0501a071cac9f60626eef79381b803387"
TOL = 2e-6


class R18R19Error(RuntimeError):
    def __init__(self, gate: str, detail: str) -> None:
        super().__init__(f"{gate}: {detail}")
        self.gate = gate
        self.detail = detail


def require(condition: bool, detail: str, gate: str = BLOCK_BINDING) -> None:
    if not condition:
        raise R18R19Error(gate, detail)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
                     default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def load(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing={path}")
    return json.loads(path.read_text(encoding="utf-8"))


def git(args: Sequence[str]) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT, text=True, capture_output=True, check=True).stdout.strip()


def artifact_root() -> Path:
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M%S%z")[:-2] + ":00"
    return ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r19_frozen_factorized_policy_review_{stamp}"


def identity(candidate: Mapping[str, Any]) -> str:
    return f"{candidate['agent_id']}::{candidate['candidate_id']}"


def load_authoritative_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    authorization = load(AUTH_PATH)
    supplied = authorization.get("authorization_sha256")
    expected = canonical_sha256({key: value for key, value in authorization.items() if key != "authorization_sha256"})
    require(supplied == expected, "authorization_sha256")
    require(authorization.get("authorization") == R18.R18R18B_FACTORIZED_AUTHORIZATION
            and authorization.get("source_commit") == SOURCE_AT_EXECUTION, "authorization_identity")
    gate = load(R18R18B_EXEC_ROOT / "gate_decision.json")
    require(gate.get("gate") == R18.R18R18B_FACTORIZED_PASS_GATE, f"r18r18b_gate={gate.get('gate')}")
    execution = load(R18R18B_EXEC_ROOT / "execution_manifest.json")
    counters = dict(execution.get("counters", {}))
    require(counters.get("training") == 1 and counters.get("actor_optimizer_step") == 6
            and counters.get("critic_optimizer_step") == 6 and counters.get("raw_optimizer_step") == 12,
            "r18r18b_execution_budget")
    checkpoint = load(R18R18B_EXEC_ROOT / "checkpoint_manifest.json")
    require(checkpoint.get("test_only") is True and checkpoint.get("bounded") is True
            and checkpoint.get("non_promotable") is True and checkpoint.get("promotion") is False,
            "checkpoint_policy")
    bound = dict(authorization["module_freeze_contract"]["frozen_source_hashes"])
    expected_hashes = {str(key): str(value) for key, value in dict(bound["expected"]).items()}
    actual_hashes = {rel: sha256(PROJECT / rel) for rel in expected_hashes}
    require(actual_hashes == expected_hashes, "r18r18b_frozen_source_hashes")
    require(git(["status", "--porcelain=v1"]) == "", "dirty_tree_before_review")
    changed = [item for item in git(["diff", "--name-only", f"{SOURCE_AT_EXECUTION}..HEAD"]).splitlines() if item]
    require(set(changed) == {"05_training/run_h4m_ae_ls3_bt8_r18_r19_frozen_factorized_policy_review.py"},
            f"review_source_scope={changed}")
    return authorization, checkpoint, {
        "passed": True,
        "r18r18b_authorization_sha256": expected,
        "r18r18b_execution_gate": gate.get("gate"),
        "r18r18b_execution_manifest_sha256": sha256(R18R18B_EXEC_ROOT / "execution_manifest.json"),
        "r18r18b_checkpoint_manifest_sha256": sha256(R18R18B_EXEC_ROOT / "checkpoint_manifest.json"),
        "frozen_source_hashes": {"expected": expected_hashes, "actual": actual_hashes, "all_unchanged": True},
        "r18r18b_execution_source_commit": SOURCE_AT_EXECUTION,
        "r18r19_review_source_commit": git(["rev-parse", "HEAD"]),
        "review_source_scope": changed,
    }


def build_initial_actor(*, arm: Mapping[str, Any], checkpoints: Mapping[str, Any], config: Mapping[str, Any],
                        device: torch.device) -> torch.nn.Module:
    key = str(arm["initial_checkpoint_key"])
    record = dict(checkpoints[key])
    path = Path(str(record["path"]))
    require(path.is_file() and sha256(path) == record["sha256"], f"initial_checkpoint={key}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    require(set(payload).issuperset({"actor", "critic", "meta"}) and not any("optimizer" in str(name).lower() for name in payload),
            f"initial_checkpoint_schema={key}")
    torch.manual_seed(int(arm["factorized_gate_initialization_seed"]))
    actor = H.FactorizedAssignThenCandidateAssignmentHead(
        global_dim=int(config["global_dim"]), demand_dim=int(config["demand_dim"]), agent_dim=int(config["agent_dim"]),
        candidate_dim=int(config["candidate_dim"]), hidden=int(config["hidden"]), heads=int(config["heads"]),
        detach_gate_context=True,
    ).to(device).eval()
    result = actor.load_state_dict(payload["actor"], strict=False)
    require(result.missing_keys and result.unexpected_keys
            and all(str(name).startswith("gate_scorer.") for name in result.missing_keys)
            and all(str(name).startswith("no_assign_scorer.") for name in result.unexpected_keys),
            f"initial_factorized_projection={arm['arm_id']}")
    return actor


def build_final_actor(*, record: Mapping[str, Any], config: Mapping[str, Any], device: torch.device) -> torch.nn.Module:
    path = Path(str(record["path"]))
    require(path.is_file() and sha256(path) == record["sha256"], f"final_checkpoint={path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    meta = dict(payload.get("meta", {}))
    require(meta.get("actor_architecture") == "FACTORIZED_ASSIGN_THEN_CANDIDATE", f"final_actor_architecture={path}")
    actor = H.FactorizedAssignThenCandidateAssignmentHead(
        global_dim=int(config["global_dim"]), demand_dim=int(config["demand_dim"]), agent_dim=int(config["agent_dim"]),
        candidate_dim=int(config["candidate_dim"]), hidden=int(config["hidden"]), heads=int(config["heads"]),
        detach_gate_context=True,
    ).to(device).eval()
    actor.load_state_dict(payload["actor"], strict=True)
    require(R18._module_digest(actor) == meta.get("final_actor_digest") == record.get("final_actor_digest"),
            f"final_actor_digest={record['arm_id']}")
    return actor


def selected_identity(*, candidate_ids: Sequence[Mapping[str, Any]], candidate_scores: Sequence[float],
                      safe_mask: Sequence[bool], no_assign_score: float) -> str:
    selection = TIE.select_exact_tie(candidate_ids=candidate_ids, pair_scores=list(candidate_scores),
                                     safe_mask=list(safe_mask), no_assign_score=float(no_assign_score))
    return "NO_ASSIGN" if selection.selected.is_no_assign else f"{selection.selected.agent_id}::{selection.selected.candidate_id}"


def factorized_view(*, actor: torch.nn.Module, payload: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    metadata = dict(payload["metadata"])
    selectable = int(metadata["selectable_pair_count"])
    candidate_ids = list(metadata["candidate_ids"])
    tensors = {name: value.to(device) for name, value in payload["tensors"].items()}
    with torch.no_grad():
        dist = actor.forward_factorized(**tensors)
    safe = tensors["safe_mask"][0, :selectable]
    require(len(candidate_ids) == selectable and bool(safe.all().item()), "review_support")
    gate = dist.gate_log_probs[0].exp().detach().cpu()
    conditional = dist.conditional_candidate_log_probs[0, :selectable].exp().detach().cpu()
    final = dist.action_probabilities[0, :selectable + 1].detach().cpu()
    candidate_logits = dist.candidate_logits[0, :selectable].detach().cpu()
    candidate_action_scores = dist.candidate_action_log_probs[0, :selectable].detach().cpu()
    no_assign_score = float(dist.no_assign_action_log_prob[0, 0].detach().cpu())
    assign_logit = float(dist.assign_logit[0, 0].detach().cpu())
    no_assign_logit = float(dist.no_assign_logit[0, 0].detach().cpu())
    require(math.isfinite(assign_logit) and math.isfinite(no_assign_logit)
            and bool(torch.isfinite(candidate_logits).all().item()) and bool(torch.isfinite(final).all().item()),
            "nonfinite_factorized_view", BLOCK_REPLAY)
    require(abs(float(gate.sum()) - 1.0) <= TOL and abs(float(conditional.sum()) - 1.0) <= TOL
            and abs(float(final.sum()) - 1.0) <= TOL, "probability_normalization", BLOCK_REPLAY)
    selected = selected_identity(candidate_ids=candidate_ids, candidate_scores=candidate_action_scores.tolist(),
                                 safe_mask=safe.detach().cpu().tolist(), no_assign_score=no_assign_score)
    ranked = sorted(range(selectable), key=lambda index: (-float(conditional[index]), identity(candidate_ids[index])))
    top = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    return {
        "snapshot_digest": str(payload["snapshot_digest"]),
        "candidate_support_digest": str(metadata["candidate_support_digest"]),
        "decision_id": str(metadata["decision_id"]),
        "window_id": str(metadata["window_id"]),
        "selectable_candidate_count": selectable,
        "candidate_semantic_identities": [identity(row) for row in candidate_ids],
        "assign_logit": assign_logit,
        "no_assign_logit": no_assign_logit,
        "assign_probability": float(gate[0]),
        "no_assign_probability": float(gate[1]),
        "gate_assign_minus_no_assign_logit_margin": assign_logit - no_assign_logit,
        "conditional_probabilities": [float(value) for value in conditional],
        "candidate_logits": [float(value) for value in candidate_logits],
        "reconstructed_action_probabilities": [float(value) for value in final],
        "selected_identity": selected,
        "conditional_top_identity": identity(candidate_ids[top]),
        "conditional_top_probability": float(conditional[top]),
        "conditional_top_minus_second_margin": None if second is None else float(conditional[top] - conditional[second]),
        "conditional_probability_range": float(conditional.max() - conditional.min()),
        "conditional_probability_std": float(conditional.std(unbiased=False)),
    }


def counterfactual_selected(*, candidate_ids: Sequence[str], probabilities: Sequence[float]) -> str:
    top = max(range(len(probabilities)), key=lambda index: (float(probabilities[index]), candidate_ids[index] if index < len(candidate_ids) - 1 else "ZZZ_NO_ASSIGN"))
    return candidate_ids[top]


def compare_view(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    require(before["snapshot_digest"] == after["snapshot_digest"]
            and before["candidate_support_digest"] == after["candidate_support_digest"]
            and before["candidate_semantic_identities"] == after["candidate_semantic_identities"],
            "same_input_support_identity", BLOCK_REPLAY)
    candidate_ids = list(before["candidate_semantic_identities"])
    q_before = list(before["conditional_probabilities"])
    q_after = list(after["conditional_probabilities"])
    assign_before, assign_after = float(before["assign_probability"]), float(after["assign_probability"])
    p_before = list(before["reconstructed_action_probabilities"])
    p_after = list(after["reconstructed_action_probabilities"])
    candidate_rows = []
    residuals = []
    for index, candidate_id in enumerate(candidate_ids):
        gate_component = (assign_after - assign_before) * q_before[index]
        candidate_head_component = assign_after * (q_after[index] - q_before[index])
        observed = p_after[index] - p_before[index]
        residual = observed - gate_component - candidate_head_component
        residuals.append(abs(residual))
        candidate_rows.append({
            "semantic_candidate_id": candidate_id,
            "conditional_probability_initial": q_before[index],
            "conditional_probability_final": q_after[index],
            "conditional_probability_delta": q_after[index] - q_before[index],
            "reconstructed_probability_initial": p_before[index],
            "reconstructed_probability_final": p_after[index],
            "reconstructed_probability_delta": observed,
            "gate_component": gate_component,
            "candidate_head_component": candidate_head_component,
            "additive_residual": residual,
        })
    gate_only = [assign_after * value for value in q_before] + [1.0 - assign_after]
    head_only = [assign_before * value for value in q_after] + [1.0 - assign_before]
    actions = [*candidate_ids, "NO_ASSIGN"]
    actual_before = str(before["selected_identity"])
    actual_after = str(after["selected_identity"])
    gate_counterfactual = counterfactual_selected(candidate_ids=actions, probabilities=gate_only)
    head_counterfactual = counterfactual_selected(candidate_ids=actions, probabilities=head_only)
    if actual_after == actual_before:
        selection_attribution = "T1_UNCHANGED"
    elif gate_counterfactual == actual_after and head_counterfactual == actual_before:
        selection_attribution = "GATE_SHIFT"
    elif head_counterfactual == actual_after and gate_counterfactual == actual_before:
        selection_attribution = "CANDIDATE_HEAD_DISCRIMINATION"
    else:
        selection_attribution = "MIXED_GATE_AND_CANDIDATE_HEAD"
    require(max(residuals, default=0.0) <= TOL, f"probability_decomposition_residual={max(residuals, default=0.0)}",
            BLOCK_DECOMPOSITION)
    return {
        "arm_id": None,
        "decision_id": before["decision_id"],
        "window_id": before["window_id"],
        "snapshot_digest": before["snapshot_digest"],
        "candidate_support_digest": before["candidate_support_digest"],
        "initial": before,
        "final": after,
        "gate_shift": {
            "assign_probability_delta": assign_after - assign_before,
            "no_assign_probability_delta": float(after["no_assign_probability"]) - float(before["no_assign_probability"]),
            "assign_minus_no_assign_logit_margin_delta": float(after["gate_assign_minus_no_assign_logit_margin"]) - float(before["gate_assign_minus_no_assign_logit_margin"]),
        },
        "candidate_head_discrimination": {
            "conditional_top_identity_changed": before["conditional_top_identity"] != after["conditional_top_identity"],
            "conditional_top_margin_delta": (
                None if before["conditional_top_minus_second_margin"] is None else
                float(after["conditional_top_minus_second_margin"] - before["conditional_top_minus_second_margin"])
            ),
            "conditional_probability_range_delta": float(after["conditional_probability_range"] - before["conditional_probability_range"]),
            "conditional_probability_std_delta": float(after["conditional_probability_std"] - before["conditional_probability_std"]),
        },
        "candidate_probability_decomposition": candidate_rows,
        "candidate_head_component_sum": sum(row["candidate_head_component"] for row in candidate_rows),
        "gate_component_sum": sum(row["gate_component"] for row in candidate_rows),
        "max_abs_additive_residual": max(residuals, default=0.0),
        "t1": {
            "initial_selected_identity": actual_before,
            "final_selected_identity": actual_after,
            "changed": actual_before != actual_after,
            "gate_only_counterfactual_selected_identity": gate_counterfactual,
            "candidate_head_only_counterfactual_selected_identity": head_counterfactual,
            "attribution": selection_attribution,
        },
    }


def aggregate(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    gate = [dict(row["gate_shift"]) for row in rows]
    discrimination = [dict(row["candidate_head_discrimination"]) for row in rows]
    selection = [dict(row["t1"]) for row in rows]
    decomposition = [{
        "arm_id": row["arm_id"], "decision_id": row["decision_id"], "snapshot_digest": row["snapshot_digest"],
        "candidate_head_component_sum": row["candidate_head_component_sum"], "gate_component_sum": row["gate_component_sum"],
        "max_abs_additive_residual": row["max_abs_additive_residual"], "rows": row["candidate_probability_decomposition"],
    } for row in rows]
    return (
        {"review_rows": len(rows), "assign_probability_delta_mean": sum(row["assign_probability_delta"] for row in gate) / len(gate),
         "no_assign_probability_delta_mean": sum(row["no_assign_probability_delta"] for row in gate) / len(gate),
         "gate_margin_delta_mean": sum(row["assign_minus_no_assign_logit_margin_delta"] for row in gate) / len(gate),
         "assign_probability_increased_rows": sum(row["assign_probability_delta"] > 0 for row in gate),
         "no_assign_probability_increased_rows": sum(row["no_assign_probability_delta"] > 0 for row in gate),
         "rows": gate},
        {"review_rows": len(rows), "conditional_top_identity_changed_rows": sum(row["conditional_top_identity_changed"] for row in discrimination),
         "multi_candidate_rows": sum(row["conditional_top_margin_delta"] is not None for row in discrimination),
         "conditional_top_margin_delta_mean_over_multi_candidate": (
             None if not [row for row in discrimination if row["conditional_top_margin_delta"] is not None] else
             sum(float(row["conditional_top_margin_delta"]) for row in discrimination if row["conditional_top_margin_delta"] is not None)
             / sum(row["conditional_top_margin_delta"] is not None for row in discrimination)
         ), "rows": discrimination},
        {"review_rows": len(rows), "t1_selection_changed_rows": sum(row["changed"] for row in selection),
         "attribution_counts": {label: sum(row["attribution"] == label for row in selection)
                                for label in ("T1_UNCHANGED", "GATE_SHIFT", "CANDIDATE_HEAD_DISCRIMINATION", "MIXED_GATE_AND_CANDIDATE_HEAD")},
         "rows": selection},
        {"review_rows": len(rows), "max_abs_additive_residual": max(row["max_abs_additive_residual"] for row in decomposition),
         "candidate_head_component_sum_max_abs": max(abs(float(row["candidate_head_component_sum"])) for row in decomposition),
         "rows": decomposition},
    )


def write_manifest(root: Path, gate: str, source_commit: str, classification: str) -> None:
    files = {item.relative_to(root).as_posix(): sha256(item)
             for item in root.rglob("*") if item.is_file() and item.name != "manifest.json"}
    dump(root / "manifest.json", {"stage": STAGE, "gate": gate, "classification": classification,
                                   "source_commit": source_commit, "github_push_performed": False, "file_sha256": files})


def main() -> None:
    root = artifact_root()
    root.mkdir(parents=True)
    counters = {"training": 0, "rollout": 0, "simulator": 0, "reward_recomputation": 0, "optimizer_creation": 0,
                "optimizer_step": 0, "backward": 0, "checkpoint_write": 0, "policy_mutation": 0,
                "candidate_generation": 0, "frozen_forward_calls": 0, "github_push": 0}
    source_commit = None
    try:
        authorization, checkpoint_manifest, binding = load_authoritative_inputs()
        source_commit = str(binding["r18r19_review_source_commit"])
        mps = R18B.external_mps_preflight()
        device = torch.device("mps:0")
        entries = [dict(row) for row in dict(authorization["upstream"])["review_snapshot_binding"]["entries"]
                   if int(row["seed"]) == 20260822]
        require(len(entries) == 3 and len({str(row["snapshot_digest"]) for row in entries}) == 3,
                "review_snapshot_binding")
        payloads = [FPS.load_snapshot(Path(str(entry["snapshot_root"]))) for entry in entries]
        require(all(payload["snapshot_digest"] == entry["snapshot_digest"] for payload, entry in zip(payloads, entries)),
                "snapshot_digest_binding")
        config = dict(payloads[0]["metadata"]["actor_config"])
        require(all(dict(payload["metadata"]["actor_config"]) == config for payload in payloads), "review_actor_config")
        arms = list(dict(authorization["envelope"])["selected_arms"])
        final_records = dict(checkpoint_manifest["final"])
        initial_inputs = dict(dict(authorization["checkpoint_contract"])["initial_inputs"])
        rows = []
        actor_digests = {}
        for arm in arms:
            arm_id = str(arm["arm_id"])
            initial = build_initial_actor(arm=arm, checkpoints=initial_inputs, config=config, device=device)
            final = build_final_actor(record=dict(final_records[arm_id]), config=config, device=device)
            initial_digest, final_digest = R18._module_digest(initial), R18._module_digest(final)
            require(initial_digest == final_records[arm_id]["initial_actor_digest"], f"initial_actor_digest={arm_id}", BLOCK_REPLAY)
            require(final_digest == final_records[arm_id]["final_actor_digest"] and initial_digest != final_digest,
                    f"final_actor_digest={arm_id}", BLOCK_REPLAY)
            actor_digests[arm_id] = {"initial_actor_digest": initial_digest, "final_actor_digest": final_digest}
            for payload in payloads:
                initial_before, final_before = R18._module_digest(initial), R18._module_digest(final)
                before = factorized_view(actor=initial, payload=payload, device=device)
                after = factorized_view(actor=final, payload=payload, device=device)
                require(R18._module_digest(initial) == initial_before and R18._module_digest(final) == final_before,
                        f"inference_mutated_actor={arm_id}", BLOCK_REPLAY)
                require(all(parameter.grad is None for parameter in initial.parameters())
                        and all(parameter.grad is None for parameter in final.parameters()),
                        f"inference_retained_gradient={arm_id}", BLOCK_REPLAY)
                row = compare_view(before, after)
                row["arm_id"] = arm_id
                rows.append(row)
                counters["frozen_forward_calls"] += 2
        require(len(rows) == 6 and all(row["max_abs_additive_residual"] <= TOL for row in rows),
                "review_row_count_or_decomposition", BLOCK_DECOMPOSITION)
        gate, candidate, selection, decomposition = aggregate(rows)
        require(float(decomposition["candidate_head_component_sum_max_abs"]) <= TOL,
                "candidate_head_components_must_redistribute_only", BLOCK_DECOMPOSITION)
        outputs = {
            "evidence_binding_audit.json": binding,
            "external_mps_preflight.json": mps,
            "same_input_factorized_review_rows.json": {"review_rows": rows, "actor_digests": actor_digests},
            "gate_shift_audit.json": gate,
            "candidate_head_discrimination_audit.json": candidate,
            "probability_decomposition_audit.json": decomposition,
            "selection_attribution_audit.json": selection,
            "test_results.json": {"passed": True, "execution_counters": counters, "hard_failures": [], "warnings": [],
                                  "training_semantics_changed": False, "policy_mutation": False},
            "gate_decision.json": {"stage": STAGE, "gate": PASS_GATE, "classification": CLASSIFICATION,
                                   "source_commit": source_commit, "next_step": "STOP; no training authorization"},
        }
        for name, value in outputs.items():
            dump(root / name, value)
        (root / "final_report.md").write_text(
            "# R18-R19 frozen factorized-policy review\n\n"
            f"- gate: `{PASS_GATE}`\n- source: `{source_commit}`\n"
            "- same frozen inputs: 3 snapshots × 2 arms\n"
            "- gate shift and conditional candidate-head discrimination were measured separately.\n"
            "- training / rollout / optimizer / backward / checkpoint write: `0`\n",
            encoding="utf-8",
        )
        write_manifest(root, PASS_GATE, source_commit, CLASSIFICATION)
        (root / "_SUCCESS.lock").write_text(PASS_GATE + "\n", encoding="utf-8")
        print(f"[PASS] {PASS_GATE}")
        print(root)
    except R18R19Error as exc:
        dump(root / "test_results.json", {"passed": False, "execution_counters": counters, "hard_failures": [exc.detail], "warnings": []})
        dump(root / "gate_decision.json", {"stage": STAGE, "gate": exc.gate, "classification": "BLOCKED",
                                             "source_commit": source_commit, "hard_failures": [exc.detail], "next_step": "STOP"})
        (root / "final_report.md").write_text(f"# R18-R19 blocked\n\n- gate: `{exc.gate}`\n- detail: `{exc.detail}`\n", encoding="utf-8")
        write_manifest(root, exc.gate, source_commit or "UNKNOWN", "BLOCKED")
        print(f"[BLOCKED] {exc.gate}")
        print(root)


if __name__ == "__main__":
    main()
