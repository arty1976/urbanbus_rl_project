#!/usr/bin/env python3
"""BT7-R0: select and freeze the lossless Joint Actor replay-evidence contract.

This runner never calls Local Search, Zero-Loss, a simulator, or an optimizer.
Its only executable fixture is a newly instantiated Joint Actor fed synthetic
non-causal tensors in order to validate serialization and pure replay code.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Sequence
from zoneinfo import ZoneInfo


STAGE = "H4M-AE-R9.8-LS3-BT7-R0"
PASS_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT7_R0_FROZEN_POLICY_SNAPSHOT_PRESERVATION_CONTRACT_COMPLETE"
PASS_CLASS = "A_SUSEONG_LS3_FROZEN_POLICY_REPLAY_EVIDENCE_CONTRACT_READY_FOR_SEPARATE_BT6_RERUN"
BLOCK = "BLOCKED_SNAPSHOT_CONTRACT_NOT_REPLAY_SUFFICIENT"
BT7_SOURCE = "dd5a5107377a31ff8fc661e23f6f1f7ac9be6fee"
BT6_SOURCE = "b404014ac35747ad9c95936fd8ad3fd142a619e6"
BT6_S0_SOURCE = "34742b1c11f3d7ea437c990d6d549c7e45372e6d"
BT5R_SOURCE = "f14fab7dbc72f42968a1aea5dcbbb652dddbeb48"

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
SOURCE_FILES = {
    "05_training/run_h4m_ae_ls3_bt6_postrepair_r2_training.py",
    "05_training/joint_assignment_frozen_policy_snapshot.py",
    "05_training/test_h4m_ae_ls3_bt7_r0_snapshot_preservation.py",
    "05_training/run_h4m_ae_ls3_bt7_r0_snapshot_preservation.py",
}
BT7 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt7_frozen_behavior_review_20260822_193137+09:00"
BT6 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt6_postrepair_r2_training_20260822_160245+09:00"
BT6_S0 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt6_s0_postrepair_scale_redesign_20260822_152944+09:00"
BT5R = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt5r_candidate_support_repair_selection_20260822_151517+09:00"
FROZEN = {
    "gatv2_operational_actor_critic": "run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py",
    "reward_v2": "rewards/mappo_reward_v1.py", "zero_loss": "simulator/zero_loss_admission_adapter.py",
    "local_search_authority": "local_search_contract.py", "candidate_support_deconfounding": "joint_candidate_support_snapshot.py",
    "causal_bridge": "causal_kpi_bridge.py", "r9_8_authorization": "simulator_authorization.py",
    "credit_contract": "joint_assignment_credit_contract.py", "joint_assignment_learning": "joint_assignment_learning.py",
    "r9_7_gate": "run_h4m_ae_r9_7_gate.py", "r9_8_gate": "run_h4m_ae_r9_8_gate.py",
}
LOCKS = {"training_allowed": False, "simulator_execution_allowed": False,
         "performance_comparison_allowed": False, "paper_level_claim_allowed": False,
         "causal_performance_claim_allowed": False}


def now() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul")).replace(microsecond=0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def git(args: Sequence[str]) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT, check=True, text=True, capture_output=True).stdout.strip()


def frozen_hashes() -> Dict[str, str]:
    return {name: sha256(ROOT / rel) for name, rel in FROZEN.items()}


def provenance() -> Dict[str, Any]:
    head, parent = git(["rev-parse", "HEAD"]), git(["rev-parse", "HEAD^"])
    changed = [row for row in git(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]).splitlines() if row]
    return {"source_commit": head, "source_parent": parent, "changed_files": changed,
            "source_only_local_commit": set(changed) == SOURCE_FILES, "github_push_performed": False}


def actor_boundary_audit() -> Dict[str, Any]:
    source = (ROOT / "run_h4m_ae_ls3_bt6_postrepair_r2_training.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    forward_fields = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "actor":
            forward_fields.update(keyword.arg for keyword in node.keywords if keyword.arg)
    exact_fields = ["global_feats", "demand_feats", "agent_feats", "agent_mask", "candidate_feats", "pair_agent_index", "safe_mask"]
    capture_marker = "frozen_actor_snapshot = snapshot_writer.capture("
    model_marker = "model_packed = model_ready_pack("
    forward_marker = "with torch.no_grad():\n                        logits, no_assign = actor("
    return {
        "last_authoritative_boundary": "model_ready_pack output after candidate construction and Zero-Loss filtering, immediately before actor(...) forward",
        "actor_forward_tensor_fields": sorted(forward_fields),
        "required_actor_input_tensors": exact_fields,
        "actor_forward_fields_exact": set(forward_fields) == set(exact_fields),
        "capture_uses_model_ready_pack": "actor_inputs=model_packed" in source,
        "capture_is_after_support_finalization": source.find(model_marker) < source.find(capture_marker),
        "capture_is_before_training_time_actor_forward": source.find(capture_marker) < source.find(forward_marker),
        "candidate_identity_order": "packed[\"pair_keys\"]",
        "agent_identity_order": "sorted(joint.agents, key=lambda item: item.agent_id)",
        "no_assign_metadata": ["no_assign_option", "no_assign_index", "selectable_pair_count", "actor_tensor_pair_count", "storage_padding_pair_count"],
        "feature_normalization_binding": "the materialized model_packed tensors are persisted; replay does not recompute features or normalize",
        "empty_support_padding_preserved": True,
    }


def main() -> None:
    started = time.perf_counter()
    sys.path.insert(0, str(ROOT))
    import joint_assignment_frozen_policy_snapshot as FPS
    from test_h4m_ae_ls3_bt7_r0_snapshot_preservation import run_fixture_validation

    root = ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt7_r0_snapshot_preservation_{now().strftime('%Y%m%d_%H%M%S%z')[:-2]}:00"
    if root.exists():
        raise SystemExit("append-only artifact collision")
    before = frozen_hashes()
    source = provenance()
    bt7_gate = json.loads((BT7 / "gate_decision.json").read_text(encoding="utf-8"))
    bt6_gate = json.loads((BT6 / "gate_decision.json").read_text(encoding="utf-8"))
    binding = {
        "bt7_gate": bt7_gate.get("gate") == "BLOCKED_INSUFFICIENT_FROZEN_DECISION_SNAPSHOT_EVIDENCE",
        "bt7_classification": bt7_gate.get("classification") == "BT7_FROZEN_POLICY_INFERENCE_NOT_REPRODUCIBLE_FROM_PRESERVED_EVIDENCE",
        "bt7_source": bt7_gate.get("source_commit") == BT7_SOURCE,
        "bt6_gate": bt6_gate.get("gate") == "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT6_POST_REPAIR_EXTENDED_BOUNDED_TRAINING_AND_BT7_READINESS_COMPLETE",
        "bt6_source": bt6_gate.get("source_commit") == BT6_SOURCE,
        "bt6s0_exists": BT6_S0.is_dir(), "bt5r_exists": BT5R.is_dir(),
        "source_only_local_commit": source["source_only_local_commit"],
    }
    boundary = actor_boundary_audit()
    fixture = run_fixture_validation()
    after = frozen_hashes()
    schema = {
        "snapshot_schema_version": FPS.SNAPSHOT_SCHEMA_VERSION,
        "collection_schema_version": FPS.COLLECTION_SCHEMA_VERSION,
        "actor_tensor_fields": list(FPS.TENSOR_FIELDS),
        "identity_and_support_fields": ["decision_id", "window_id", "seed", "decision_index", "time_band", "agent_ids",
                                        "candidate_ids", "candidate_order", "candidate_support_digest", "no_assign_option",
                                        "no_assign_index", "selectable_pair_count", "actor_tensor_pair_count", "storage_padding_pair_count"],
        "binding_fields": ["actor_config", "actor_config_sha256", "checkpoint_sha256", "feature_contract",
                           "feature_contract_id", "frozen_authority_hashes", "source_commit", "captured_device"],
        "tensor_metadata": ["dtype", "shape", "numel"],
        "future_collection_checkpoint_binding": "append-only collection_manifest.json, finalized only after the new final checkpoint is written",
    }
    serialization = {
        "format": "per-snapshot torch.save binary tensor archive plus canonical JSON manifest and collection manifest",
        "dtype_shape_preserved": fixture["serialized_dtype_and_shape_preserved"],
        "device_neutral": fixture["serialization_device_neutral"],
        "stored_tensor_device": fixture["serialized_tensors_device"],
        "lossy_transformations": [],
        "digest_coverage": "canonical metadata, every required tensor name/dtype/shape/raw bytes, tensor-binary SHA, snapshot-manifest SHA, collection entries and checkpoint SHA",
    }
    replay = {
        "pure": True, "checkpoint_loaded": "fixture-only checkpoint", "actor_eval": True, "no_grad": True,
        "reproducible_exact": fixture["replay_reproducible_exact"], "support_identity_exact": fixture["support_identity_exact"],
        "argmax_exact": fixture["argmax_exact"], "max_logit_delta": fixture["max_logit_delta"],
        "max_probability_delta": fixture["max_probability_delta"], "fixture_device": fixture["fixture_device"],
        "forbidden_replay_operations": ["Local Search regeneration", "Zero-Loss re-evaluation", "candidate rebuilding",
                                        "feature recomputation", "mask reconstruction", "candidate sorting/re-ranking", "simulator access"],
    }
    capture_neutrality = {"source_tensors_unchanged": fixture["capture_source_tensors_unchanged"],
                           "rng_unchanged": fixture["capture_rng_unchanged"],
                           "actor_output_exact": fixture["capture_output_exact"],
                           "selected_action_exact": fixture["argmax_exact"],
                           "maximum_logit_delta": fixture["max_logit_delta"],
                           "maximum_probability_delta": fixture["max_probability_delta"]}
    selection = {
        "S1_metadata_only_extension": {"sufficient": False, "reason": "cannot supply actor tensors, legal masks, or pair-to-agent mapping"},
        "S2_full_actor_input_tensor_mask_mapping": {"sufficient": False, "reason": "does not independently bind immutable bytes to actor architecture/config and final checkpoint or provide a fail-closed verifier"},
        "S3_full_snapshot_digest_binding_replay_verifier": {"sufficient": True,
            "reason": "lossless tensors plus identities/support metadata, canonical digest, architecture/frozen-authority/feature bindings, final-checkpoint collection binding, and pure verifier"},
        "selected": "S3",
    }
    hard = []
    if not all(binding.values()): hard.append("AUTHORITATIVE_BINDING_FAILURE")
    if not all(boundary[key] for key in ("actor_forward_fields_exact", "capture_uses_model_ready_pack",
                                          "capture_is_after_support_finalization", "capture_is_before_training_time_actor_forward")):
        hard.append("ACTOR_INPUT_BOUNDARY_NOT_PROVEN")
    if not fixture["passed"]: hard.append("SNAPSHOT_SERIALIZATION_OR_REPLAY_VALIDATION_FAILED")
    if before != after: hard.append("FROZEN_AUTHORITY_MUTATION")
    gate, classification = (PASS_GATE, PASS_CLASS) if not hard else (BLOCK, "SNAPSHOT_CONTRACT_NOT_REPLAY_SUFFICIENT")
    lineage = {"BT5_R": BT5R_SOURCE, "BT6_S0": BT6_S0_SOURCE, "BT6": BT6_SOURCE, "BT7": BT7_SOURCE,
               "bt5r_manifest_sha256": sha256(BT5R / "manifest.json"), "bt6s0_manifest_sha256": sha256(BT6_S0 / "manifest.json"),
               "bt6_manifest_sha256": sha256(BT6 / "manifest.json"), "bt7_manifest_sha256": sha256(BT7 / "manifest.json")}
    root.mkdir(parents=True)
    outputs = {
        "bt7r0_actor_input_boundary_audit.json": boundary,
        "bt7r0_snapshot_schema.json": schema,
        "bt7r0_snapshot_serialization_audit.json": serialization,
        "bt7r0_replay_verifier_audit.json": replay,
        "bt7r0_capture_neutrality_audit.json": capture_neutrality,
        "bt7r0_adversarial_tests.json": {"passed": fixture["adversarial_passed"], "total": fixture["adversarial_total"],
                                           "cases": fixture["adversarial"], "all_passed": fixture["adversarial_passed"] == fixture["adversarial_total"]},
        "bt7r0_repair_selection.json": selection,
        "frozen_hash_before_after.json": {"before": before, "after": after, "all_unchanged": before == after},
        "test_results.json": {"binding": binding, "fixture": fixture, "optimizer_steps": 0, "causal_simulator_rollouts": 0,
                              "training_visits": 0, "new_training_checkpoint": 0, "performance_comparison": 0,
                              "TEST6_access": 0, "github_push_performed": False, "hard_failures": hard, "warnings": []},
        "gate_decision.json": {"gate": gate, "classification": classification, "source_commit": source["source_commit"],
                               "lineage": lineage, "hard_failures": hard, "warnings": [], "global_locks": LOCKS,
                               "next_step": "separate authorization for exact R2 BT6 rerun with snapshot preservation enabled" if not hard else "STOP"},
    }
    for name, payload in outputs.items(): dump(root / name, payload)
    (root / "final_report.md").write_text(
        f"# {STAGE} — Frozen-policy snapshot preservation contract\n\n"
        f"gate = {gate}\nclassification = {classification}\nsource_commit = {source['source_commit']}\n\n"
        "The exact actor boundary is the post-Zero-Loss `model_ready_pack` tensor bundle immediately before `actor(...)`. "
        "S3 was selected: lossless device-neutral tensor snapshots, canonical digests, immutable bindings, and a pure replay verifier. "
        f"Fixture replay was exact (max logit/probability delta {fixture['max_logit_delta']}/{fixture['max_probability_delta']}); "
        f"adversarial tests passed {fixture['adversarial_passed']}/{fixture['adversarial_total']}.\n\n"
        "No BT6 rerun, causal rollout, optimizer step, training visit, new training checkpoint, performance comparison, or TEST6 access occurred. "
        "The historical BT6 snapshots were not reconstructed.\n", encoding="utf-8")
    manifest = {path.relative_to(root).as_posix(): sha256(path) for path in root.rglob("*") if path.is_file()}
    dump(root / "manifest.json", {"stage": STAGE, "gate": gate, "classification": classification,
                                   "source_commit": source["source_commit"], "lineage": lineage,
                                   "file_sha256": manifest, "elapsed_seconds": round(time.perf_counter() - started, 3),
                                   "github_push_performed": False})
    (root / ("_SUCCESS.lock" if not hard else "_BLOCKED.lock")).write_text(gate + "\n", encoding="utf-8")
    print(f"[{'PASS' if not hard else 'BLOCKED'}] {gate}")
    print(f"artifact: {root.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
