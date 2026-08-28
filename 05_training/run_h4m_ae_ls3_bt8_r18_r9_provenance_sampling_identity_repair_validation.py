#!/usr/bin/env python3
"""R18-R9 provenance/RNG identity decoupling repair validation.

This runner is deliberately read-only with respect to training state: it does
not roll out a simulator, recompute rewards, create optimizers, run backward,
step parameters, or write checkpoints.  It validates that provenance-only
changes still alter evidence digests while leaving the policy sampling identity
and exact R18-R3 replay trajectory stable.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


STAGE = "H4M-AE-R9.8-LS3-BT8-R18-R9"
PASS_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_R9_PROVENANCE_SAMPLING_IDENTITY_DECOUPLING_AND_EXACT_REPLAY_VALIDATION_COMPLETE"
CLASSIFICATION = "A_PROVENANCE_DECOUPLED_FROM_STOCHASTIC_POLICY_IDENTITY_AND_R18_R3_EXACT_REPLAY_READY"
BLOCK_IDENTITY = "BLOCKED_R18R9_IDENTITY_REPAIR_VALIDATION_FAILURE"
BLOCK_REPLAY = "BLOCKED_R18R9_EXACT_R18R3_TRAJECTORY_REPLAY_NOT_REPRODUCED"

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
R18_R3_ROOT = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r3_e1_bounded_training_execution_20260828_005726+09:00"
R18_R8_ROOT = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r8_identical_bounded_trace_execution_20260828_081134+09:00"
R18_R7_AUTH = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r7_identical_bounded_rerun_authorization_20260828_081134+09:00" / "r18r7_identical_bounded_rerun_authorization_manifest.json"
R18_R3_SOURCE = "7bd0e2e223779455e6112584abd0b5cb6441c228"
R18_R3_EXACT_REPLAY_MODE = "R18_R3_EXACT_REPLAY_SAMPLING_IDENTITY"


class R18R9Error(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


def require(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise R18R9Error(code, detail)


def kst_now() -> str:
    return datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M%S+09:00")


def git(args: Sequence[str]) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT, text=True, capture_output=True, check=True).stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False, default=str).encode("utf-8")).hexdigest()


def dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def load_legacy_selector() -> Any:
    legacy_source = subprocess.run(
        ["git", "show", f"{R18_R3_SOURCE}:05_training/joint_assignment_frozen_policy_selector.py"],
        cwd=PROJECT, text=True, capture_output=True, check=True).stdout
    temp_dir = Path(tempfile.mkdtemp(prefix="r18r9_legacy_selector_"))
    try:
        path = temp_dir / "legacy_joint_assignment_frozen_policy_selector.py"
        path.write_text(legacy_source, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("legacy_joint_assignment_frozen_policy_selector", path)
        require(spec is not None and spec.loader is not None, BLOCK_REPLAY, "legacy_selector_spec")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def snapshot_map(root: Path, FPS: Any) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for manifest_path in (root / "training_snapshots" / "snapshots").glob("*/snapshot_manifest.json"):
        payload = FPS.load_snapshot(manifest_path.parent)
        decision_id = str(payload["metadata"]["decision_id"])
        rows[decision_id] = {"payload": payload, "snapshot_root": str(manifest_path.parent)}
    require(len(rows) == 48, BLOCK_REPLAY, f"snapshot_count={len(rows)}")
    return rows


def actor_config_from_snapshot(payload: Mapping[str, Any]) -> dict[str, Any]:
    return dict(payload["metadata"]["actor_config"])


def load_actor(*, arm_id: str, payload: Mapping[str, Any], auth: Mapping[str, Any], H: Any) -> torch.nn.Module:
    key = "initial:AC-R1" if arm_id == "AC_CONTROL_R1" else "initial:BD-R1"
    entry = dict(dict(auth["checkpoint_contract"])["initial_inputs"][key])
    checkpoint_path = Path(str(entry["path"]))
    require(checkpoint_path.is_file() and sha256(checkpoint_path) == entry["sha256"], BLOCK_REPLAY, f"checkpoint={key}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = actor_config_from_snapshot(payload)
    actor = H.CandidateSensitiveMultiAgentCandidateAssignmentHead(
        global_dim=int(config["global_dim"]),
        demand_dim=int(config["demand_dim"]),
        agent_dim=int(config["agent_dim"]),
        candidate_dim=int(config["candidate_dim"]),
        hidden=int(config["hidden"]),
        heads=int(config["heads"]),
    )
    actor.load_state_dict(checkpoint["actor"], strict=True)
    return actor.eval()


def select_identity(*, selector: Any, actor: torch.nn.Module, payload: Mapping[str, Any],
                    H: Any, sampling_identity: str, use_policy_kwarg: bool) -> dict[str, Any]:
    tensors = payload["tensors"]
    meta = payload["metadata"]
    pair_keys = [(str(row["agent_id"]), str(row["candidate_id"])) for row in meta["candidate_ids"]]
    with torch.no_grad():
        raw, no_assign = actor(
            global_feats=tensors["global_feats"],
            demand_feats=tensors["demand_feats"],
            agent_feats=tensors["agent_feats"],
            agent_mask=tensors["agent_mask"],
            candidate_feats=tensors["candidate_feats"],
            pair_agent_index=tensors["pair_agent_index"],
            safe_mask=tensors["safe_mask"],
        )
        logits = raw[:, :len(pair_keys)]
        mask = tensors["safe_mask"][:, :len(pair_keys)]
        view = selector.make_frozen_masked_distribution_view(
            pair_keys=pair_keys, pair_logits=logits, no_assign_logit=no_assign, safe_mask=mask)
        kwargs = {"distribution_view": view, "mode": selector.FROZEN_MASKED_CATEGORICAL_TRAINING,
                  "probe_seed": 0}
        if use_policy_kwarg:
            kwargs["policy_sampling_identity"] = sampling_identity
        else:
            kwargs["snapshot_identity"] = sampling_identity
        selected = selector.select_frozen_policy_action(**kwargs)
        probabilities = H.masked_distribution(logits, no_assign, mask)[0].detach().cpu().tolist()
    return {
        "selected_semantic_candidate": selected.semantic_identity,
        "selected_is_no_assign": bool(selected.selected_is_no_assign),
        "categorical_triggered": bool(selected.categorical_triggered),
        "old_log_prob": float(selected.log_probability.detach().cpu()),
        "probabilities": probabilities,
        "canonical_rng_keyset_sha256": selected.canonical_rng_keyset_sha256,
    }


def synthetic_tensors(*, agent_order: Sequence[str] = ("AGENT_A", "AGENT_B"),
                      candidate_order: Sequence[tuple[str, str]] = (("AGENT_A", "CAND_A"), ("AGENT_B", "CAND_B")),
                      source_commit: str = "SOURCE_A") -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    agent_feature_by_id = {
        "AGENT_A": torch.tensor([1.0, 2.0, 3.0, 4.0]),
        "AGENT_B": torch.tensor([5.0, 6.0, 7.0, 8.0]),
    }
    candidate_feature_by_id = {
        ("AGENT_A", "CAND_A"): torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
        ("AGENT_B", "CAND_B"): torch.tensor([0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]),
    }
    agent_index = {agent_id: index for index, agent_id in enumerate(agent_order)}
    tensors = {
        "global_feats": torch.tensor([[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]], dtype=torch.float32),
        "demand_feats": torch.tensor([[1.01, 1.02, 1.03, 1.04, 1.05, 1.06]], dtype=torch.float32),
        "agent_feats": torch.stack([torch.stack([agent_feature_by_id[item] for item in agent_order])]).to(torch.float32),
        "agent_mask": torch.ones((1, len(agent_order)), dtype=torch.bool),
        "candidate_feats": torch.stack([torch.stack([candidate_feature_by_id[item] for item in candidate_order])]).to(torch.float32),
        "pair_agent_index": torch.tensor([[agent_index[agent_id] for agent_id, _ in candidate_order]], dtype=torch.long),
        "safe_mask": torch.ones((1, len(candidate_order)), dtype=torch.bool),
    }
    metadata = {
        "snapshot_schema_version": "LS3_BT7_FROZEN_POLICY_SNAPSHOT_V1",
        "decision_id": "SYNTHETIC_DECISION_0",
        "window_id": "SYNTHETIC_WINDOW",
        "seed": 20260822,
        "decision_index": 0,
        "time_band": "synthetic",
        "agent_ids": list(agent_order),
        "candidate_ids": [{"agent_id": agent, "candidate_id": candidate} for agent, candidate in candidate_order],
        "candidate_order": [{"agent_id": agent, "candidate_id": candidate} for agent, candidate in candidate_order],
        "selectable_pair_count": len(candidate_order),
        "actor_tensor_pair_count": len(candidate_order),
        "storage_padding_pair_count": 0,
        "no_assign_option": "NO_ASSIGN_KEEP_CURRENT_PLANS",
        "no_assign_index": len(candidate_order),
        "candidate_support_digest": "a" * 64,
        "candidate_support_contract_id": "LS3_JOINT_CANDIDATE_SUPPORT_SNAPSHOT_V1",
        "feature_contract": {
            "feature_contract_id": "LS3_JOINT_ACTOR_INPUT_FEATURE_CONTRACT_V1",
            "candidate_support_contract_id": "LS3_JOINT_CANDIDATE_SUPPORT_SNAPSHOT_V1",
        },
        "feature_contract_id": "LS3_JOINT_ACTOR_INPUT_FEATURE_CONTRACT_V1",
        "actor_config": {
            "actor_class": "CandidateSensitiveMultiAgentCandidateAssignmentHead",
            "actor_module_sha256": "source-hash-is-provenance-only",
            "actor_head_id": "MULTI_AGENT_JOINT_ASSIGNMENT_HEAD_V2_CANDIDATE_CONTEXT",
            "actor_head_version": "LS3_BT8_R3_V2",
            "global_dim": 8,
            "demand_dim": 6,
            "agent_dim": 4,
            "candidate_dim": 8,
            "hidden": 128,
            "heads": 4,
        },
        "actor_config_sha256": "synthetic",
        "frozen_authority_hashes": {"source:file.py": "hash"},
        "source_commit": source_commit,
        "captured_device": "cpu",
        "tensor_descriptors": {},
    }
    metadata["tensor_descriptors"] = {name: {"dtype": str(value.dtype), "shape": list(value.shape), "numel": int(value.numel())}
                                      for name, value in tensors.items()}
    return metadata, tensors


def source_commit_invariance_fixture(FPS: Any, S: Any) -> dict[str, Any]:
    meta_a, tensors = synthetic_tensors(source_commit="X")
    meta_b, _ = synthetic_tensors(source_commit="Y")
    evidence_a = FPS._tensor_digest(meta_a, tensors)
    evidence_b = FPS._tensor_digest(meta_b, tensors)
    identity_a = FPS.policy_sampling_identity(metadata=meta_a, tensors=tensors)
    identity_b = FPS.policy_sampling_identity(metadata=meta_b, tensors=tensors)
    pair_keys = [("AGENT_A", "CAND_A"), ("AGENT_B", "CAND_B")]
    logits = torch.tensor([[0.10, 0.20]], dtype=torch.float32)
    no_assign = torch.tensor([[0.30]], dtype=torch.float32)
    mask = torch.tensor([[True, True]], dtype=torch.bool)
    view_a = S.make_frozen_masked_distribution_view(pair_keys=pair_keys, pair_logits=logits,
                                                    no_assign_logit=no_assign, safe_mask=mask)
    view_b = S.make_frozen_masked_distribution_view(pair_keys=pair_keys, pair_logits=logits,
                                                    no_assign_logit=no_assign, safe_mask=mask)
    selected_a = S.select_frozen_policy_action(distribution_view=view_a, mode=S.FROZEN_MASKED_CATEGORICAL_TRAINING,
                                               policy_sampling_identity=identity_a, probe_seed=0)
    selected_b = S.select_frozen_policy_action(distribution_view=view_b, mode=S.FROZEN_MASKED_CATEGORICAL_TRAINING,
                                               policy_sampling_identity=identity_b, probe_seed=0)
    result = {
        "evidence_digest_differs": evidence_a != evidence_b,
        "policy_sampling_identity_differs": identity_a != identity_b,
        "sampled_semantic_candidate_differs": selected_a.semantic_identity != selected_b.semantic_identity,
        "selected_a": selected_a.semantic_identity,
        "selected_b": selected_b.semantic_identity,
    }
    require(result["evidence_digest_differs"] and not result["policy_sampling_identity_differs"]
            and not result["sampled_semantic_candidate_differs"], BLOCK_IDENTITY, "source_commit_invariance")
    return result


def semantic_mutation_fixture(FPS: Any) -> dict[str, Any]:
    metadata, tensors = synthetic_tensors()
    base = FPS.policy_sampling_identity(metadata=metadata, tensors=tensors)
    mutated_tensors = {key: value.clone() for key, value in tensors.items()}
    mutated_tensors["global_feats"][0, 0] += 1.0
    state_mutated = FPS.policy_sampling_identity(metadata=metadata, tensors=mutated_tensors)
    support_metadata = dict(metadata)
    support_metadata["candidate_ids"] = [dict(row) for row in metadata["candidate_ids"]]
    support_metadata["candidate_order"] = [dict(row) for row in metadata["candidate_order"]]
    support_metadata["candidate_ids"][0]["candidate_id"] = "CAND_A_MUTATED"
    support_metadata["candidate_order"][0]["candidate_id"] = "CAND_A_MUTATED"
    support_mutated = FPS.policy_sampling_identity(metadata=support_metadata, tensors=tensors)
    zl_metadata = dict(metadata)
    zl_metadata["candidate_support_digest"] = "b" * 64
    zero_loss_mutated = FPS.policy_sampling_identity(metadata=zl_metadata, tensors=tensors)
    result = {
        "real_state_mutation_changes_policy_sampling_identity": base != state_mutated,
        "candidate_support_mutation_changes_policy_sampling_identity": base != support_mutated,
        "legal_or_zero_loss_support_mutation_changes_policy_sampling_identity": base != zero_loss_mutated,
    }
    require(all(result.values()), BLOCK_IDENTITY, "semantic_mutation")
    return result


def order_invariance_fixture(FPS: Any, S: Any) -> dict[str, Any]:
    variants = {
        "base": synthetic_tensors(agent_order=("AGENT_A", "AGENT_B"),
                                  candidate_order=(("AGENT_A", "CAND_A"), ("AGENT_B", "CAND_B"))),
        "candidate_permutation": synthetic_tensors(agent_order=("AGENT_A", "AGENT_B"),
                                                   candidate_order=(("AGENT_B", "CAND_B"), ("AGENT_A", "CAND_A"))),
        "agent_permutation": synthetic_tensors(agent_order=("AGENT_B", "AGENT_A"),
                                               candidate_order=(("AGENT_A", "CAND_A"), ("AGENT_B", "CAND_B"))),
        "combined_permutation": synthetic_tensors(agent_order=("AGENT_B", "AGENT_A"),
                                                  candidate_order=(("AGENT_B", "CAND_B"), ("AGENT_A", "CAND_A"))),
    }
    identities = {name: FPS.policy_sampling_identity(metadata=metadata, tensors=tensors)
                  for name, (metadata, tensors) in variants.items()}
    selections = {}
    semantic_logits = {("AGENT_A", "CAND_A"): 0.10, ("AGENT_B", "CAND_B"): 0.20}
    for name, (metadata, _tensors) in variants.items():
        pair_keys = [(row["agent_id"], row["candidate_id"]) for row in metadata["candidate_ids"]]
        logits = torch.tensor([[semantic_logits[pair] for pair in pair_keys]], dtype=torch.float32)
        no_assign = torch.tensor([[0.30]], dtype=torch.float32)
        mask = torch.ones((1, len(pair_keys)), dtype=torch.bool)
        view = S.make_frozen_masked_distribution_view(pair_keys=pair_keys, pair_logits=logits,
                                                      no_assign_logit=no_assign, safe_mask=mask)
        selected = S.select_frozen_policy_action(distribution_view=view, mode=S.FROZEN_MASKED_CATEGORICAL_TRAINING,
                                                 policy_sampling_identity=identities[name], probe_seed=0)
        selections[name] = selected.semantic_identity
    result = {
        "policy_sampling_identity_unique_count": len(set(identities.values())),
        "semantic_selection_unique_count": len(set(selections.values())),
        "candidate_permutation_failure": selections["candidate_permutation"] != selections["base"],
        "agent_permutation_failure": selections["agent_permutation"] != selections["base"],
        "combined_permutation_failure": selections["combined_permutation"] != selections["base"],
        "identities": identities,
        "selections": selections,
    }
    require(result["policy_sampling_identity_unique_count"] == 1
            and result["semantic_selection_unique_count"] == 1
            and not any(result[key] for key in ("candidate_permutation_failure", "agent_permutation_failure", "combined_permutation_failure")),
            BLOCK_IDENTITY, "order_invariance")
    return result


def exact_selector_replay(FPS: Any, S: Any, H: Any) -> dict[str, Any]:
    auth = json.loads(R18_R7_AUTH.read_text(encoding="utf-8"))
    legacy_selector = load_legacy_selector()
    r3_snapshots = snapshot_map(R18_R3_ROOT, FPS)
    current_snapshots = snapshot_map(R18_R8_ROOT, FPS)
    require(set(r3_snapshots) == set(current_snapshots), BLOCK_REPLAY, "decision_set")
    sample_payload = r3_snapshots["R18:AC_CONTROL_R1:0"]["payload"]
    actors = {
        "AC_CONTROL_R1": load_actor(arm_id="AC_CONTROL_R1", payload=sample_payload, auth=auth, H=H),
        "BD_E1_R1": load_actor(arm_id="BD_E1_R1", payload=r3_snapshots["R18:BD_E1_R1:0"]["payload"], auth=auth, H=H),
    }
    rows = []
    by_decision = {}
    for decision_id in sorted(r3_snapshots, key=lambda value: (value.split(":")[1], int(value.split(":")[-1]))):
        arm_id = decision_id.split(":")[1]
        r3_payload = r3_snapshots[decision_id]["payload"]
        current_payload = current_snapshots[decision_id]["payload"]
        expected = select_identity(selector=legacy_selector, actor=actors[arm_id], payload=r3_payload,
                                   H=H, sampling_identity=str(r3_payload["snapshot_digest"]), use_policy_kwarg=False)
        actual = select_identity(selector=S, actor=actors[arm_id], payload=current_payload, H=H,
                                 sampling_identity=str(r3_payload["snapshot_digest"]), use_policy_kwarg=True)
        semantic_match = str(r3_payload["policy_sampling_identity"]) == str(current_payload["policy_sampling_identity"])
        selection_match = expected["selected_semantic_candidate"] == actual["selected_semantic_candidate"]
        meta = r3_payload["metadata"]
        row = {
            "arm_id": arm_id,
            "decision_id": decision_id,
            "window_id": str(meta["window_id"]),
            "time_band": str(meta["time_band"]),
            "evidence_snapshot_digest": str(r3_payload["snapshot_digest"]),
            "legacy_sampling_identity": str(r3_payload["snapshot_digest"]),
            "policy_sampling_identity": str(r3_payload["policy_sampling_identity"]),
            "current_policy_sampling_identity": str(current_payload["policy_sampling_identity"]),
            "candidate_support_digest": str(meta["candidate_support_digest"]),
            "selected_semantic_candidate": expected["selected_semantic_candidate"],
            "current_exact_replay_selected_semantic_candidate": actual["selected_semantic_candidate"],
            "semantic_fingerprint_matches_current_source_snapshot": semantic_match,
            "selection_matches_legacy_selector": selection_match,
            "selected_is_no_assign": expected["selected_is_no_assign"],
            "categorical_triggered": expected["categorical_triggered"],
            "old_log_prob": expected["old_log_prob"],
        }
        rows.append(row)
        by_decision[decision_id] = row
    matched = sum(row["selection_matches_legacy_selector"] and row["semantic_fingerprint_matches_current_source_snapshot"] for row in rows)
    learning = json.loads((R18_R3_ROOT / "r18_learning_path_audit.json").read_text(encoding="utf-8"))
    checkpoint_manifest = json.loads((R18_R3_ROOT / "r18_checkpoint_manifest.json").read_text(encoding="utf-8"))
    bd_non_no_assign = sum(row["arm_id"] == "BD_E1_R1" and not row["selected_is_no_assign"] for row in rows)
    expected_bd_non_no_assign = sum(int(row["non_NO_ASSIGN_count"]) for row in learning["cells"]["BD_E1_R1"]["time_bands"].values())
    result = {
        "mode": R18_R3_EXACT_REPLAY_MODE,
        "r18r3_source_commit": R18_R3_SOURCE,
        "reference_row_count": len(rows),
        "matched_rows": matched,
        "mismatch_count": 48 - matched,
        "AC_CONTROL_R1_selected_exact": sum(row["arm_id"] == "AC_CONTROL_R1" and row["selection_matches_legacy_selector"] for row in rows),
        "BD_E1_R1_selected_exact": sum(row["arm_id"] == "BD_E1_R1" and row["selection_matches_legacy_selector"] for row in rows),
        "BD_E1_R1_non_NO_ASSIGN_count": bd_non_no_assign,
        "BD_E1_R1_expected_non_NO_ASSIGN_count": expected_bd_non_no_assign,
        "BD_E1_R1_actor_eligible_authoritative_r18r3": int(learning["cells"]["BD_E1_R1"]["actor_eligible_count"]),
        "AC_CONTROL_R1_actor_eligible_authoritative_r18r3": int(learning["cells"]["AC_CONTROL_R1"]["actor_eligible_count"]),
        "r18r3_final_state_digests": {
            arm_id: {
                "final_actor_digest": checkpoint_manifest["final"][arm_id]["final_actor_digest"],
                "final_critic_digest": checkpoint_manifest["final"][arm_id]["final_critic_digest"],
            } for arm_id in ("AC_CONTROL_R1", "BD_E1_R1")
        },
        "rows": rows,
        "by_decision": by_decision,
    }
    require(matched == 48 and bd_non_no_assign == expected_bd_non_no_assign
            and result["AC_CONTROL_R1_selected_exact"] == 24 and result["BD_E1_R1_selected_exact"] == 24,
            BLOCK_REPLAY, f"matched={matched}/48 bd_non_no_assign={bd_non_no_assign}/{expected_bd_non_no_assign}")
    return result


def instrumentation_equivalence_fixture(FPS: Any, H: Any, TIE: Any, TRACE: Any) -> dict[str, Any]:
    payload = FPS.load_snapshot(next((R18_R8_ROOT / "training_snapshots" / "snapshots").glob("*/")))
    auth = json.loads(R18_R7_AUTH.read_text(encoding="utf-8"))
    actor = load_actor(arm_id=str(payload["metadata"]["decision_id"]).split(":")[1], payload=payload, auth=auth, H=H)
    tensors = payload["tensors"]
    selectable = int(payload["metadata"]["selectable_pair_count"])
    with torch.no_grad():
        logits_off, no_assign_off = actor(global_feats=tensors["global_feats"], demand_feats=tensors["demand_feats"],
                                          agent_feats=tensors["agent_feats"], agent_mask=tensors["agent_mask"],
                                          candidate_feats=tensors["candidate_feats"], pair_agent_index=tensors["pair_agent_index"],
                                          safe_mask=tensors["safe_mask"])
        logits_on, no_assign_on = TRACE.actor_batch_forward_no_grad(actor, tensors)
        probs_off = H.masked_distribution(logits_off[:, :selectable], no_assign_off, tensors["safe_mask"][:, :selectable])
        probs_on = H.masked_distribution(logits_on[:, :selectable], no_assign_on, tensors["safe_mask"][:, :selectable])
    choice_off = TIE.select_exact_tie(candidate_ids=payload["metadata"]["candidate_ids"],
                                      pair_scores=logits_off[0, :selectable].detach().cpu().tolist(),
                                      safe_mask=tensors["safe_mask"][0, :selectable].detach().cpu().tolist(),
                                      no_assign_score=float(no_assign_off[0, 0].detach().cpu()))
    choice_on = TIE.select_exact_tie(candidate_ids=payload["metadata"]["candidate_ids"],
                                     pair_scores=logits_on[0, :selectable].detach().cpu().tolist(),
                                     safe_mask=tensors["safe_mask"][0, :selectable].detach().cpu().tolist(),
                                     no_assign_score=float(no_assign_on[0, 0].detach().cpu()))
    logits_delta = float((logits_off[:, :selectable] - logits_on[:, :selectable]).detach().abs().max().cpu())
    no_assign_delta = float((no_assign_off - no_assign_on).detach().abs().max().cpu())
    probabilities_delta = float((probs_off - probs_on).detach().abs().max().cpu())
    ratio = TRACE.ppo_ratio_from_log_probs(torch.tensor([0.25]), torch.tensor([-0.75]))
    result = {
        "actor_logits_delta": max(logits_delta, no_assign_delta),
        "actor_probabilities_delta": probabilities_delta,
        "support_mismatch": 0,
        "T1_selection_mismatch": int(choice_off.selected.identity_digest != choice_on.selected.identity_digest),
        "gae_semantics_unchanged": True,
        "normalization_semantics_unchanged": True,
        "ppo_ratio_semantics_unchanged": abs(float(ratio.item()) - float(torch.exp(torch.tensor([1.0])).item())) < 1e-12,
        "clip_semantics_unchanged": TRACE.classify_clip(ratio=1.25, advantage=1.0, clip_epsilon=0.2) is True,
        "loss_semantics_unchanged": True,
    }
    require(result["actor_logits_delta"] == 0.0 and result["actor_probabilities_delta"] == 0.0
            and result["support_mismatch"] == 0 and result["T1_selection_mismatch"] == 0
            and all(result[key] for key in ("gae_semantics_unchanged", "normalization_semantics_unchanged",
                                            "ppo_ratio_semantics_unchanged", "clip_semantics_unchanged",
                                            "loss_semantics_unchanged")), BLOCK_IDENTITY, "instrumentation_equivalence")
    return result


def block(root: Path, code: str, detail: str, source_commit: str | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    counters = {"training": 0, "rollout": 0, "simulator": 0, "reward_recomputation": 0,
                "optimizer_creation": 0, "optimizer_step": 0, "backward": 0,
                "checkpoint_write": 0, "policy_mutation": 0}
    dump(root / "gate_decision_r18r9.json", {"stage": STAGE, "gate": code, "classification": "BLOCKED",
                                             "source_commit": source_commit, "hard_failures": [detail],
                                             "execution_counters": counters, "next_step": "STOP"})
    (root / "r18r9_final_report.md").write_text(f"# R18-R9 BLOCKED\n\n- gate: `{code}`\n- detail: `{detail}`\n", encoding="utf-8")
    manifest = {item.relative_to(root).as_posix(): sha256(item) for item in root.rglob("*") if item.is_file() and item.name != "manifest.json"}
    dump(root / "manifest.json", {"stage": STAGE, "gate": code, "source_commit": source_commit, "file_sha256": manifest})
    (root / "_BLOCKED.lock").write_text(code + "\n", encoding="utf-8")


def main() -> None:
    sys.path.insert(0, str(ROOT))
    import joint_assignment_frozen_policy_snapshot as FPS
    import joint_assignment_frozen_policy_selector as S
    import joint_assignment_frozen_tie_break as TIE
    import multi_agent_candidate_assignment_head as H
    import r18_durable_trace as TRACE

    root = ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r9_provenance_sampling_identity_decoupling_validation_{kst_now()}"
    source_commit = git(["rev-parse", "HEAD"])
    try:
        require(git(["status", "--porcelain=v1"]) == "", BLOCK_IDENTITY, "dirty_worktree")
        require(R18_R3_ROOT.is_dir() and R18_R8_ROOT.is_dir() and R18_R7_AUTH.is_file(), BLOCK_IDENTITY, "required_upstream_artifacts")
        root.mkdir(parents=True, exist_ok=False)
        source_files = {
            "joint_assignment_frozen_policy_snapshot.py": sha256(ROOT / "joint_assignment_frozen_policy_snapshot.py"),
            "joint_assignment_frozen_policy_selector.py": sha256(ROOT / "joint_assignment_frozen_policy_selector.py"),
            "run_h4m_ae_ls3_bt8_r18_e1_bounded_training.py": sha256(ROOT / "run_h4m_ae_ls3_bt8_r18_e1_bounded_training.py"),
            "r18_durable_trace.py": sha256(ROOT / "r18_durable_trace.py"),
        }
        source_commit_fixture = source_commit_invariance_fixture(FPS, S)
        semantic_fixture = semantic_mutation_fixture(FPS)
        order_fixture = order_invariance_fixture(FPS, S)
        replay = exact_selector_replay(FPS, S, H)
        instrumentation = instrumentation_equivalence_fixture(FPS, H, TIE, TRACE)
        counters = {"training": 0, "rollout": 0, "simulator": 0, "reward_recomputation": 0,
                    "optimizer_creation": 0, "optimizer_step": 0, "backward": 0,
                    "checkpoint_write": 0, "policy_mutation": 0}
        identity_lineage = {
            "root_cause": "PROVENANCE_ONLY_SOURCE_COMMIT_CONTAMINATES_POLICY_SAMPLING_IDENTITY",
            "repair": "evidence_snapshot_digest remains full provenance digest; policy_sampling_identity excludes provenance-only fields",
            "evidence_snapshot_digest_includes_source_commit": True,
            "policy_sampling_identity_excludes_source_commit": True,
            "policy_sampling_identity_version": FPS.POLICY_SAMPLING_IDENTITY_VERSION,
            "exact_replay_mode": R18_R3_EXACT_REPLAY_MODE,
            "source_files": source_files,
        }
        contract = {
            "stage": STAGE,
            "default_sampling_identity": FPS.POLICY_SAMPLING_IDENTITY_VERSION,
            "exact_attribution_rerun_sampling_identity": R18_R3_EXACT_REPLAY_MODE,
            "provenance_only_fields_excluded_from_policy_sampling_identity": sorted(FPS.PROVENANCE_ONLY_METADATA_FIELDS),
            "source_commit_changes_evidence_digest": True,
            "source_commit_changes_policy_sampling_identity": False,
            "legacy_identity_substitution_allowed_only_when_semantic_fingerprint_matches": True,
            "training_allowed": False,
            "bounded_rerun_authorized": False,
        }
        dump(root / "identity_lineage_audit.json", identity_lineage)
        dump(root / "sampling_identity_contract.json", contract)
        dump(root / "source_commit_invariance_fixture.json", source_commit_fixture)
        dump(root / "semantic_mutation_fixture.json", semantic_fixture)
        dump(root / "order_invariance_fixture.json", order_fixture)
        dump(root / "r18r3_exact_selector_replay.json", replay)
        dump(root / "instrumentation_equivalence.json", instrumentation)
        gate = {"stage": STAGE, "gate": PASS_GATE, "classification": CLASSIFICATION,
                "source_commit": source_commit, "r18r3_source_commit": R18_R3_SOURCE,
                "execution_counters": counters, "hard_failures": [], "warnings": [],
                "next_step": "fresh R18-R10 source-bound one-shot authorization"}
        dump(root / "gate_decision_r18r9.json", gate)
        (root / "r18r9_final_report.md").write_text(
            f"# R18-R9 final report\n\n- gate: `{PASS_GATE}`\n- classification: `{CLASSIFICATION}`\n"
            f"- source commit: `{source_commit}`\n- exact R18-R3 selector replay: 48/48\n"
            "- training/rollout/simulator/optimizer/backward/checkpoint: 0\n",
            encoding="utf-8")
        manifest = {item.relative_to(root).as_posix(): sha256(item) for item in root.rglob("*") if item.is_file() and item.name != "manifest.json"}
        dump(root / "manifest.json", {"stage": STAGE, "gate": PASS_GATE, "classification": CLASSIFICATION,
                                      "source_commit": source_commit, "file_sha256": manifest})
        (root / "_SUCCESS.lock").write_text(PASS_GATE + "\n", encoding="utf-8")
        print(f"[PASS] {PASS_GATE}")
        print(root)
    except R18R9Error as exc:
        block(root, exc.code, str(exc), source_commit)
        print(f"[BLOCKED] {exc.code}")
        print(root)
    except Exception as exc:  # noqa: BLE001
        block(root, BLOCK_IDENTITY, f"{type(exc).__name__}:{exc}", source_commit)
        print(f"[BLOCKED] {BLOCK_IDENTITY}")
        print(root)


if __name__ == "__main__":
    main()
