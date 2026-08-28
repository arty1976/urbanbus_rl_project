#!/usr/bin/env python3
"""R18-R10A source-binding guard validation.

This stage is deliberately no-training.  It validates the minimum guard update
needed after R18-R9: exact R18-R3 trace replay must require the validated
R18-R9 selector source SHA as current, must not require the old R16 selector
SHA as current, and must leave every other frozen source binding unchanged.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


STAGE = "H4M-AE-R9.8-LS3-BT8-R18-R10A"
PASS_GATE = (
    "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_R10A_"
    "SOURCE_BINDING_GUARD_UPDATE_AND_EQUIVALENCE_VALIDATION_COMPLETE"
)
CLASSIFICATION = "A_R18_R9_SELECTOR_SOURCE_GUARD_READY_FOR_FRESH_R18_R10B_EXACT_TRACE_RERUN"
BLOCK = "BLOCKED_R18R10A_SOURCE_BINDING_GUARD_VALIDATION_FAILURE"

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
R18_R9_ROOT = (
    ARTIFACTS
    / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r9_provenance_sampling_identity_decoupling_validation_20260828_084535+09:00"
)
R18_R9_GATE = (
    "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_R9_"
    "PROVENANCE_SAMPLING_IDENTITY_DECOUPLING_AND_EXACT_REPLAY_VALIDATION_COMPLETE"
)


class R18R10AError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise R18R10AError(BLOCK, detail)


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


def dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    require(path.is_file(), f"missing={path}")
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_verified(root: Path) -> dict[str, Any]:
    manifest = load_json(root / "manifest.json")
    hashes = dict(manifest.get("file_sha256", {}))
    mismatches = [name for name, digest in hashes.items()
                  if not (root / name).is_file() or sha256(root / name) != str(digest)]
    require(not mismatches, f"manifest_mismatch={mismatches}")
    return manifest


def selector_auth_payload(R18: Any, *, selector_sha: str | None = None) -> dict[str, Any]:
    return {
        "authorization": R18.R18R10_AUTHORIZATION,
        "selector_source_binding": {
            "mode": R18.R18R10_SELECTOR_BINDING_MODE,
            "r18_r9_selector_sha256": selector_sha or R18.R18R9_SELECTOR_SOURCE_SHA256,
            "old_r16_selector_sha256": R18.R16_SELECTOR_SOURCE_SHA256,
            "require_old_r16_selector_as_current": False,
            "reject_selector_mutation": True,
        },
    }


def source_binding_guard_audit(R18: Any, R17: Any) -> dict[str, Any]:
    r16_path = R17.ARTIFACTS / R17.R16_NAME
    r18r9_identity = load_json(R18_R9_ROOT / "identity_lineage_audit.json")
    r18r9_selector_sha = str(dict(r18r9_identity["source_files"])["joint_assignment_frozen_policy_selector.py"])
    actual_selector_sha = sha256(ROOT / "joint_assignment_frozen_policy_selector.py")
    require(actual_selector_sha == r18r9_selector_sha == R18.R18R9_SELECTOR_SOURCE_SHA256,
            "r18r9_selector_sha_required")
    require(actual_selector_sha != R18.R16_SELECTOR_SOURCE_SHA256, "old_r16_selector_was_still_required_as_current")

    binding = R18.source_hash_binding_for_authorization(
        auth=selector_auth_payload(R18), R17=R17, r16_path=r16_path)
    tampered_blocked = False
    tampered_detail = ""
    try:
        R18.source_hash_binding_for_authorization(
            auth=selector_auth_payload(R18, selector_sha="0" * 64), R17=R17, r16_path=r16_path)
    except R18.R18Error as exc:
        tampered_blocked = True
        tampered_detail = str(exc)
    require(tampered_blocked, "selector_tamper_not_blocked")

    other_frozen_keys = [key for key in binding["expected"] if key != "implementation:selector"]
    unchanged_other = {key: binding["actual"][key] == binding["expected"][key] for key in other_frozen_keys}
    require(all(unchanged_other.values()), "other_frozen_source_sha_changed")
    return {
        "old_r16_selector_required_as_current": False,
        "old_r16_selector_sha256": R18.R16_SELECTOR_SOURCE_SHA256,
        "r18_r9_selector_sha256_required": R18.R18R9_SELECTOR_SOURCE_SHA256,
        "actual_selector_sha256": actual_selector_sha,
        "actual_selector_equals_r18_r9": actual_selector_sha == R18.R18R9_SELECTOR_SOURCE_SHA256,
        "actual_selector_equals_old_r16": actual_selector_sha == R18.R16_SELECTOR_SOURCE_SHA256,
        "selector_one_character_or_sha_mutation_blocks": tampered_blocked,
        "selector_mutation_block_detail": tampered_detail,
        "other_frozen_source_sha_unchanged": True,
        "other_frozen_source_checks": unchanged_other,
        "binding": binding,
    }


def block(root: Path, code: str, detail: str, source_commit: str | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    counters = {"training": 0, "rollout": 0, "simulator": 0, "reward_recomputation": 0,
                "optimizer_creation": 0, "optimizer_step": 0, "backward": 0,
                "checkpoint_write": 0, "policy_mutation": 0}
    dump(root / "gate_decision_r18r10a.json", {"stage": STAGE, "gate": code, "classification": "BLOCKED",
                                               "source_commit": source_commit, "hard_failures": [detail],
                                               "execution_counters": counters, "next_step": "STOP"})
    (root / "r18r10a_final_report.md").write_text(
        f"# R18-R10A BLOCKED\n\n- gate: `{code}`\n- detail: `{detail}`\n", encoding="utf-8")
    manifest = {item.relative_to(root).as_posix(): sha256(item)
                for item in root.rglob("*") if item.is_file() and item.name != "manifest.json"}
    dump(root / "manifest.json", {"stage": STAGE, "gate": code, "source_commit": source_commit,
                                  "file_sha256": manifest})
    (root / "_BLOCKED.lock").write_text(code + "\n", encoding="utf-8")


def main() -> None:
    sys.path.insert(0, str(ROOT))
    import joint_assignment_frozen_policy_snapshot as FPS
    import joint_assignment_frozen_policy_selector as S
    import joint_assignment_frozen_tie_break as TIE
    import multi_agent_candidate_assignment_head as H
    import r18_durable_trace as TRACE
    import run_h4m_ae_ls3_bt8_r17_e1_bounded_training_authorization as R17
    import run_h4m_ae_ls3_bt8_r18_e1_bounded_training as R18
    import run_h4m_ae_ls3_bt8_r18_r9_provenance_sampling_identity_repair_validation as R9

    root = ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r10a_source_binding_guard_validation_{kst_now()}"
    source_commit = git(["rev-parse", "HEAD"])
    try:
        require(git(["status", "--porcelain=v1"]) == "", "dirty_worktree")
        r18r9_manifest = manifest_verified(R18_R9_ROOT)
        r18r9_gate = load_json(R18_R9_ROOT / "gate_decision_r18r9.json")
        require(r18r9_manifest.get("gate") == R18_R9_GATE and r18r9_gate.get("gate") == R18_R9_GATE,
                "r18r9_upstream_not_pass")
        root.mkdir(parents=True, exist_ok=False)

        source_files = {
            "joint_assignment_frozen_policy_selector.py": sha256(ROOT / "joint_assignment_frozen_policy_selector.py"),
            "joint_assignment_frozen_policy_snapshot.py": sha256(ROOT / "joint_assignment_frozen_policy_snapshot.py"),
            "r18_durable_trace.py": sha256(ROOT / "r18_durable_trace.py"),
            "run_h4m_ae_ls3_bt8_r18_e1_bounded_training.py": sha256(ROOT / "run_h4m_ae_ls3_bt8_r18_e1_bounded_training.py"),
            "run_h4m_ae_ls3_bt8_r18_r10a_source_binding_guard_validation.py": sha256(Path(__file__)),
        }
        guard = source_binding_guard_audit(R18, R17)
        source_commit_fixture = R9.source_commit_invariance_fixture(FPS, S)
        semantic_fixture = R9.semantic_mutation_fixture(FPS)
        order_fixture = R9.order_invariance_fixture(FPS, S)
        replay = R9.exact_selector_replay(FPS, S, H)
        instrumentation = R9.instrumentation_equivalence_fixture(FPS, H, TIE, TRACE)
        equivalence = {
            "source_commit_no_longer_changes_policy_sampling_identity": not source_commit_fixture["policy_sampling_identity_differs"],
            "source_commit_still_changes_evidence_digest": source_commit_fixture["evidence_snapshot_digest_differs"],
            "real_state_mutation_changes_sampling_identity": semantic_fixture[
                "real_state_mutation_changes_policy_sampling_identity"],
            "support_mutation_changes_sampling_identity": semantic_fixture[
                "candidate_support_mutation_changes_policy_sampling_identity"],
            "zero_loss_or_legality_mutation_changes_sampling_identity": semantic_fixture[
                "legal_or_zero_loss_support_mutation_changes_policy_sampling_identity"],
            "order_invariance_failures": {
                "candidate_permutation_failure": order_fixture["candidate_permutation_failure"],
                "agent_permutation_failure": order_fixture["agent_permutation_failure"],
                "combined_permutation_failure": order_fixture["combined_permutation_failure"],
            },
            "r18r3_selector_replay_exact": replay["matched_rows"] == 48 and replay["mismatch_count"] == 0,
            "frozen_inference_exact": (
                instrumentation["actor_logits_delta"] == 0.0
                and instrumentation["actor_probabilities_delta"] == 0.0
                and instrumentation["T1_selection_mismatch"] == 0
                and instrumentation["support_mismatch"] == 0
            ),
            "durable_trace_instrumentation_semantics_unchanged": all(
                instrumentation[key] for key in (
                    "gae_semantics_unchanged",
                    "normalization_semantics_unchanged",
                    "ppo_ratio_semantics_unchanged",
                    "clip_semantics_unchanged",
                    "loss_semantics_unchanged",
                )
            ),
        }
        require(all(value is True for key, value in equivalence.items()
                    if key not in {"order_invariance_failures"}), "equivalence_validation_failed")
        require(not any(equivalence["order_invariance_failures"].values()), "order_invariance_regression")

        counters = {"training": 0, "rollout": 0, "simulator": 0, "reward_recomputation": 0,
                    "optimizer_creation": 0, "optimizer_step": 0, "backward": 0,
                    "checkpoint_write": 0, "policy_mutation": 0}
        selector_contract = {
            "stage": STAGE,
            "selector_binding_mode": R18.R18R10_SELECTOR_BINDING_MODE,
            "old_r16_selector_sha256_not_required_as_current": R18.R16_SELECTOR_SOURCE_SHA256,
            "r18_r9_selector_sha256_required_as_current": R18.R18R9_SELECTOR_SOURCE_SHA256,
            "selector_file_mutation_policy": "BLOCK",
            "other_frozen_source_sha_policy": "UNCHANGED",
            "training_authorized": False,
            "bounded_rerun_authorized": False,
        }
        dump(root / "source_binding_guard_audit.json", guard)
        dump(root / "selector_source_binding_contract.json", selector_contract)
        dump(root / "source_commit_invariance_fixture.json", source_commit_fixture)
        dump(root / "semantic_mutation_fixture.json", semantic_fixture)
        dump(root / "order_invariance_fixture.json", order_fixture)
        dump(root / "r18r3_exact_selector_replay.json", replay)
        dump(root / "instrumentation_equivalence.json", instrumentation)
        dump(root / "equivalence_validation.json", equivalence)
        dump(root / "source_file_audit.json", {
            "source_commit": source_commit,
            "source_files": source_files,
            "r18r9_upstream_source_commit": r18r9_gate.get("source_commit"),
        })
        gate = {"stage": STAGE, "gate": PASS_GATE, "classification": CLASSIFICATION,
                "source_commit": source_commit, "execution_counters": counters,
                "hard_failures": [], "warnings": [],
                "next_step": "fresh R18-R10B source-bound one-shot authorization"}
        dump(root / "gate_decision_r18r10a.json", gate)
        (root / "r18r10a_final_report.md").write_text(
            f"# R18-R10A final report\n\n- gate: `{PASS_GATE}`\n"
            f"- classification: `{CLASSIFICATION}`\n- source commit: `{source_commit}`\n"
            "- old R16 selector SHA required as current: `false`\n"
            f"- R18-R9 selector SHA required: `{R18.R18R9_SELECTOR_SOURCE_SHA256}`\n"
            "- selector mutation policy: `BLOCK`\n"
            "- R18-R3 selector replay: `48/48`\n"
            "- frozen inference exact: `true`\n"
            "- durable trace instrumentation semantics unchanged: `true`\n"
            "- training/rollout/simulator/optimizer/backward/checkpoint: `0`\n",
            encoding="utf-8")
        manifest = {item.relative_to(root).as_posix(): sha256(item)
                    for item in root.rglob("*") if item.is_file() and item.name != "manifest.json"}
        dump(root / "manifest.json", {"stage": STAGE, "gate": PASS_GATE, "classification": CLASSIFICATION,
                                      "source_commit": source_commit, "file_sha256": manifest})
        (root / "_SUCCESS.lock").write_text(PASS_GATE + "\n", encoding="utf-8")
        print(f"[PASS] {PASS_GATE}")
        print(root)
    except R18R10AError as exc:
        block(root, exc.code, str(exc), source_commit)
        print(f"[BLOCKED] {exc.code}")
        print(root)
    except Exception as exc:  # noqa: BLE001
        block(root, BLOCK, f"{type(exc).__name__}:{exc}", source_commit)
        print(f"[BLOCKED] {BLOCK}")
        print(root)


if __name__ == "__main__":
    main()
