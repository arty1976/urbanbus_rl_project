#!/usr/bin/env python3
"""Focused non-causal fixtures for the BT7-R0 frozen-policy snapshot contract."""

from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict

import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import joint_assignment_frozen_policy_snapshot as FPS  # noqa: E402
import multi_agent_candidate_assignment_head as H  # noqa: E402


def _expect_error(expected: str, fn: Callable[[], Any]) -> bool:
    try:
        fn()
    except FPS.FrozenPolicySnapshotError as exc:
        return exc.code == expected
    return False


def _forward(actor: torch.nn.Module, packed: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    with torch.no_grad():
        logits, no_assign = actor(global_feats=packed["global_feats"], demand_feats=packed["demand_feats"],
                                  agent_feats=packed["agent_feats"], agent_mask=packed["agent_mask"],
                                  candidate_feats=packed["candidate_feats"],
                                  pair_agent_index=packed["pair_agent_index"], safe_mask=packed["safe_mask"])
        probabilities = H.masked_distribution(logits, no_assign, packed["safe_mask"])
    return logits.detach(), no_assign.detach(), probabilities.detach(), int(torch.argmax(probabilities[0]).item())


def run_fixture_validation() -> Dict[str, Any]:
    """Exercise capture, lossless serialization, pure replay, and all fail-closed paths."""
    torch.manual_seed(20260822)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    config = FPS.actor_config(global_dim=8, demand_dim=6, agent_dim=4, candidate_dim=8,
                              actor_module_sha256=FPS.sha256_file(ROOT / "multi_agent_candidate_assignment_head.py"),
                              actor_head_id=H.HEAD_ID, actor_head_version=H.HEAD_VERSION)
    feature_contract = {
        "feature_contract_id": FPS.FEATURE_CONTRACT_ID,
        "global_feature_dim": 8, "demand_feature_dim": 6, "agent_feature_dim": 4,
        "candidate_feature_dim": 8, "feature_normalization": "already materialized at pre-forward boundary; replay forbids recomputation",
        "candidate_support_contract_id": FPS.CANDIDATE_SUPPORT_CONTRACT_ID,
        "candidate_support_contract": "post-Zero-Loss finalized support only",
    }
    authorities = {"reward_v2": "a" * 64, "zero_loss": "b" * 64, "local_search": "c" * 64}
    binding = {"snapshot_schema_version": FPS.SNAPSHOT_SCHEMA_VERSION, "source_commit": "fixture-only",
               "actor_config": config, "actor_config_sha256": FPS.actor_config_sha256(config),
               "feature_contract": feature_contract, "frozen_authority_hashes": authorities,
               "captured_device": str(device)}
    actor = H.MultiAgentCandidateAssignmentHead(global_dim=8, demand_dim=6, agent_dim=4, candidate_dim=8).to(device).eval()
    packed = {
        "global_feats": torch.linspace(0.0, 0.7, 8, dtype=torch.float32).reshape(1, 8).to(device),
        "demand_feats": torch.linspace(0.0, 0.5, 6, dtype=torch.float32).reshape(1, 6).to(device),
        "agent_feats": torch.arange(12, dtype=torch.float32).reshape(1, 3, 4).to(device) / 10.0,
        "agent_mask": torch.tensor([[True, True, True]], dtype=torch.bool, device=device),
        "candidate_feats": torch.arange(24, dtype=torch.float32).reshape(1, 3, 8).to(device) / 10.0,
        "pair_agent_index": torch.tensor([[0, 1, 2]], dtype=torch.long, device=device),
        "safe_mask": torch.tensor([[True, True, True]], dtype=torch.bool, device=device),
    }
    agent_ids = ["AGENT_000", "AGENT_001", "AGENT_002"]
    candidate_ids = [("AGENT_000", "CAND_A"), ("AGENT_001", "CAND_B"), ("AGENT_002", "CAND_C")]
    before_tensors = {key: value.detach().clone() for key, value in packed.items()}
    rng_before = torch.get_rng_state().clone()
    before_logits, before_no_assign, before_probabilities, before_argmax = _forward(actor, packed)
    payload = FPS.capture_actor_input(
        decision_id="fixture-window:seed20260822:decision0", window_id="fixture-window", seed=20260822,
        decision_index=0, time_band="night", actor_inputs=packed, agent_ids=agent_ids,
        candidate_ids=candidate_ids, selectable_pair_count=3, candidate_support_digest="d" * 64,
        no_assign_option="NO_ASSIGN_KEEP_CURRENT_PLANS", actor_config_value=config,
        feature_contract=feature_contract, frozen_authority_hashes=authorities,
        source_commit="fixture-only", captured_device=str(device))
    after_logits, after_no_assign, after_probabilities, after_argmax = _forward(actor, packed)
    source_unchanged = all(torch.equal(before_tensors[key], packed[key]) for key in packed)
    rng_unchanged = bool(torch.equal(rng_before, torch.get_rng_state()))

    with tempfile.TemporaryDirectory(prefix="bt7r0_snapshot_") as raw:
        temporary = Path(raw)
        writer = FPS.SnapshotCollectionWriter(root=temporary / "frozen_policy_snapshots", collection_binding=binding)
        entry = writer.capture(
            decision_id="fixture-window:seed20260822:decision0", window_id="fixture-window", seed=20260822,
            decision_index=0, time_band="night", actor_inputs=packed, agent_ids=agent_ids,
            candidate_ids=candidate_ids, selectable_pair_count=3, candidate_support_digest="d" * 64,
            no_assign_option="NO_ASSIGN_KEEP_CURRENT_PLANS", actor_config_value=config,
            feature_contract=feature_contract, frozen_authority_hashes=authorities,
            source_commit="fixture-only", captured_device=str(device))
        checkpoint = temporary / "fixture_frozen_checkpoint.pt"
        torch.save({"actor": actor.state_dict(), "meta": {"fixture_only": True, "not_training": True}}, checkpoint)
        collection = writer.finalize(checkpoint_path=checkpoint)
        snapshot_root = (temporary / "frozen_policy_snapshots" / entry["relative_path"])
        replay_actor = H.MultiAgentCandidateAssignmentHead(global_dim=8, demand_dim=6, agent_dim=4, candidate_dim=8)
        replay = FPS.replay_snapshot(snapshot_root=snapshot_root,
                                     collection_manifest_path=temporary / "frozen_policy_snapshots" / "collection_manifest.json",
                                     checkpoint_path=checkpoint, actor=replay_actor, actor_config_value=config,
                                     feature_contract=feature_contract, frozen_authority_hashes=authorities,
                                     expected_device=str(device))
        restored = FPS.load_snapshot(snapshot_root)
        dtype_shape_preserved = all(str(restored["tensors"][name].dtype) == str(packed[name].dtype)
                                    and list(restored["tensors"][name].shape) == list(packed[name].shape)
                                    for name in FPS.TENSOR_FIELDS)

        # Isolated payload validation exercises malformed capture before any file or actor forward exists.
        missing = copy.deepcopy(payload); del missing["tensors"]["global_feats"]
        wrong_shape = copy.deepcopy(payload); wrong_shape["tensors"]["candidate_feats"] = torch.zeros((1, 3, 7))
        wrong_dtype = copy.deepcopy(payload); wrong_dtype["tensors"]["pair_agent_index"] = torch.zeros((1, 3), dtype=torch.float32)
        no_assign_removed = copy.deepcopy(payload); no_assign_removed["metadata"]["no_assign_option"] = ""
        digest_tamper = copy.deepcopy(payload); digest_tamper["snapshot_digest"] = "0" * 64

        # Archive tampering has to fail through its canonical digest, even if an attacker changes only identities/order.
        altered_candidate_order = temporary / "altered_candidate_order"
        altered_candidate_id = temporary / "altered_candidate_id"
        altered_pair_index = temporary / "altered_pair_index"
        altered_safe_mask = temporary / "altered_safe_mask"
        altered_agent_mask = temporary / "altered_agent_mask"
        for destination in (altered_candidate_order, altered_candidate_id, altered_pair_index, altered_safe_mask, altered_agent_mask):
            shutil.copytree(snapshot_root, destination)

        def alter_manifest(destination: Path, alter: Callable[[Dict[str, Any]], None]) -> None:
            manifest_path = destination / "snapshot_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            alter(manifest)
            # Deliberately retain the old digest: the immutable signed manifest must reject it.
            manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

        alter_manifest(altered_candidate_order, lambda value: value["metadata"].__setitem__("candidate_order", list(reversed(value["metadata"]["candidate_order"]))))
        alter_manifest(altered_candidate_id, lambda value: value["metadata"]["candidate_ids"][0].__setitem__("candidate_id", "TAMPERED"))

        def alter_binary(destination: Path, key: str, replacement: torch.Tensor) -> None:
            archive = destination / "actor_input_snapshot.pt"
            value = torch.load(archive, map_location="cpu", weights_only=False)
            value["tensors"][key] = replacement
            torch.save(value, archive)

        alter_binary(altered_pair_index, "pair_agent_index", torch.tensor([[2, 1, 0]], dtype=torch.long))
        alter_binary(altered_safe_mask, "safe_mask", torch.tensor([[True, False, True]], dtype=torch.bool))
        alter_binary(altered_agent_mask, "agent_mask", torch.tensor([[True, False, True]], dtype=torch.bool))
        wrong_checkpoint = temporary / "wrong_checkpoint.pt"
        torch.save({"actor": replay_actor.state_dict(), "meta": {"fixture_only": True}}, wrong_checkpoint)

        duplicate_writer = FPS.SnapshotCollectionWriter(root=temporary / "duplicate", collection_binding=binding)
        duplicate_writer.capture(
            decision_id="duplicate", window_id="fixture-window", seed=1, decision_index=0, time_band="night",
            actor_inputs=packed, agent_ids=agent_ids, candidate_ids=candidate_ids, selectable_pair_count=3,
            candidate_support_digest="e" * 64, no_assign_option="NO_ASSIGN_KEEP_CURRENT_PLANS",
            actor_config_value=config, feature_contract=feature_contract, frozen_authority_hashes=authorities,
            source_commit="fixture-only", captured_device=str(device))

        adversarial = {
            "missing_tensor": _expect_error("SNAPSHOT_REQUIRED_TENSOR_MISSING", lambda: FPS._validate_payload(missing)),
            "wrong_shape": _expect_error("SNAPSHOT_CANDIDATE_FEATURE_DIMENSION_MISMATCH", lambda: FPS._validate_payload(wrong_shape)),
            "wrong_dtype": _expect_error("SNAPSHOT_PAIR_AGENT_INDEX_DTYPE_INVALID", lambda: FPS._validate_payload(wrong_dtype)),
            "candidate_order": _expect_error("SNAPSHOT_MANIFEST_DIGEST_MISMATCH", lambda: FPS.load_snapshot(altered_candidate_order)),
            "candidate_id": _expect_error("SNAPSHOT_MANIFEST_DIGEST_MISMATCH", lambda: FPS.load_snapshot(altered_candidate_id)),
            "pair_agent_index": _expect_error("SNAPSHOT_TENSOR_BINARY_DIGEST_MISMATCH", lambda: FPS.load_snapshot(altered_pair_index)),
            "safe_mask": _expect_error("SNAPSHOT_TENSOR_BINARY_DIGEST_MISMATCH", lambda: FPS.load_snapshot(altered_safe_mask)),
            "agent_mask": _expect_error("SNAPSHOT_TENSOR_BINARY_DIGEST_MISMATCH", lambda: FPS.load_snapshot(altered_agent_mask)),
            "no_assign_removal": _expect_error("SNAPSHOT_NO_ASSIGN_MISSING", lambda: FPS._validate_payload(no_assign_removed)),
            "snapshot_digest": _expect_error("SNAPSHOT_DIGEST_MISMATCH", lambda: FPS._validate_payload(digest_tamper)),
            "wrong_checkpoint": _expect_error("SNAPSHOT_REPLAY_CHECKPOINT_MISMATCH", lambda: FPS.replay_snapshot(
                snapshot_root=snapshot_root, collection_manifest_path=temporary / "frozen_policy_snapshots" / "collection_manifest.json",
                checkpoint_path=wrong_checkpoint, actor=H.MultiAgentCandidateAssignmentHead(global_dim=8, demand_dim=6, agent_dim=4, candidate_dim=8),
                actor_config_value=config, feature_contract=feature_contract, frozen_authority_hashes=authorities, expected_device=str(device))),
            "wrong_actor_config": _expect_error("SNAPSHOT_REPLAY_ACTOR_CONFIG_MISMATCH", lambda: FPS.replay_snapshot(
                snapshot_root=snapshot_root, collection_manifest_path=temporary / "frozen_policy_snapshots" / "collection_manifest.json",
                checkpoint_path=checkpoint, actor=H.MultiAgentCandidateAssignmentHead(global_dim=8, demand_dim=6, agent_dim=4, candidate_dim=8),
                actor_config_value={**config, "hidden": 64}, feature_contract=feature_contract,
                frozen_authority_hashes=authorities, expected_device=str(device))),
            "wrong_feature_contract": _expect_error("SNAPSHOT_REPLAY_FEATURE_CONTRACT_MISMATCH", lambda: FPS.replay_snapshot(
                snapshot_root=snapshot_root, collection_manifest_path=temporary / "frozen_policy_snapshots" / "collection_manifest.json",
                checkpoint_path=checkpoint, actor=H.MultiAgentCandidateAssignmentHead(global_dim=8, demand_dim=6, agent_dim=4, candidate_dim=8),
                actor_config_value=config, feature_contract={**feature_contract, "feature_contract_id": "TAMPERED"},
                frozen_authority_hashes=authorities, expected_device=str(device))),
            "duplicate_decision_identity": _expect_error("SNAPSHOT_DUPLICATE_DECISION_IDENTITY", lambda: duplicate_writer.capture(
                decision_id="duplicate", window_id="fixture-window", seed=1, decision_index=0, time_band="night",
                actor_inputs=packed, agent_ids=agent_ids, candidate_ids=candidate_ids, selectable_pair_count=3,
                candidate_support_digest="e" * 64, no_assign_option="NO_ASSIGN_KEEP_CURRENT_PLANS",
                actor_config_value=config, feature_contract=feature_contract, frozen_authority_hashes=authorities,
                source_commit="fixture-only", captured_device=str(device))),
        }
        result = {
            "fixture_device": str(device), "capture_source_tensors_unchanged": source_unchanged,
            "capture_rng_unchanged": rng_unchanged,
            "capture_output_exact": bool(torch.equal(before_logits, after_logits) and torch.equal(before_no_assign, after_no_assign)
                                         and torch.equal(before_probabilities, after_probabilities) and before_argmax == after_argmax),
            "serialized_dtype_and_shape_preserved": dtype_shape_preserved,
            "serialized_tensors_device": sorted({str(value.device) for value in restored["tensors"].values()}),
            "serialization_device_neutral": all(value.device.type == "cpu" for value in restored["tensors"].values()),
            "replay_reproducible_exact": bool(torch.equal(before_logits, replay.pair_logits)
                                              and torch.equal(before_no_assign, replay.no_assign_logit)
                                              and torch.equal(before_probabilities, replay.probabilities)
                                              and before_argmax == replay.argmax_index),
            "support_identity_exact": replay.snapshot_digest == entry["snapshot_digest"] and replay.selected_pair ==
                                      (None if before_argmax == 3 else candidate_ids[before_argmax]),
            "argmax_exact": before_argmax == replay.argmax_index,
            "max_logit_delta": float((before_logits - replay.pair_logits).abs().max().detach().cpu()),
            "max_probability_delta": float((before_probabilities - replay.probabilities).abs().max().detach().cpu()),
            "snapshot_count": collection["snapshot_count"], "adversarial": adversarial,
            "adversarial_passed": sum(bool(value) for value in adversarial.values()),
            "adversarial_total": len(adversarial), "optimizer_steps": 0, "causal_simulator_rollouts": 0,
            "training_visits": 0, "new_training_checkpoint": 0,
        }
    result["passed"] = all([result["capture_source_tensors_unchanged"], result["capture_rng_unchanged"],
                             result["capture_output_exact"], result["serialized_dtype_and_shape_preserved"],
                             result["serialization_device_neutral"], result["replay_reproducible_exact"],
                             result["support_identity_exact"], result["argmax_exact"],
                             result["adversarial_passed"] == result["adversarial_total"]])
    return result


if __name__ == "__main__":
    result = run_fixture_validation()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)
