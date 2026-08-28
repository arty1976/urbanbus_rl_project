#!/usr/bin/env python3
"""R18-R2: validate the minimal support-digest domain repair with zero training."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


STAGE = "H4M-AE-R9.8-LS3-BT8-R18-R2"
PASS_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_R2_DIGEST_DOMAIN_REPAIR_AND_FROZEN_INFERENCE_EQUIVALENCE_COMPLETE"
SOURCE_BLOCK = "BLOCKED_R18R2_SOURCE_SCOPE_OR_SHA_FAILURE"
REGRESSION_BLOCK = "BLOCKED_R18R2_SUPPORT_MUTATION_REGRESSION_FAILURE"
INFERENCE_BLOCK = "BLOCKED_R18R2_FROZEN_INFERENCE_EQUIVALENCE_FAILURE"
MPS_BLOCK = "BLOCKED_R18R2_MPS_EXECUTION_ENVIRONMENT_UNAVAILABLE"
PRIOR_SOURCE = "42bc429a9e7b520be74c33ec214af1f9c35e912e"
R18R3_IDENTITY = "R18R3_EXACT_ONE_SHOT_E1_BOUNDED_TRAINING_ONLY"

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
PYTHON = PROJECT / ".venv" / "bin" / "python"
EXECUTOR = ROOT / "run_h4m_ae_ls3_bt8_r18_e1_bounded_training.py"
EXECUTOR_REL = "05_training/run_h4m_ae_ls3_bt8_r18_e1_bounded_training.py"
TEST = ROOT / "test_h4m_ae_ls3_bt8_r18_e1_bounded_training.py"
TEST_REL = "05_training/test_h4m_ae_ls3_bt8_r18_e1_bounded_training.py"
RUNNER_REL = "05_training/run_h4m_ae_ls3_bt8_r18_r2_digest_domain_repair_validation.py"
BASE_AUTH = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r17_e1_bounded_training_authorization_20260827_235637+09:00" / "r17_bounded_training_authorization_manifest.json"
PRIOR_R18R0 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r0_external_mps_recovery_authorization_20260828_004411+09:00"
PRIOR_R18R1 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r1_e1_bounded_training_execution_004411+09:00"
R13 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r13_s3_four_cell_execution_20260826_174729+09:00"
R16 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r16_frozen_policy_masked_categorical_exploration_validation_20260827_205833+09:00"

EXPECTED_CHANGED_FILES = {EXECUTOR_REL, TEST_REL, RUNNER_REL}
FROZEN_FUNCTIONS = {"_rollout_arm", "_train_arm", "_frozen_review_replay"}
REPAIR_FUNCTION = "_assert_support_roundtrip"
ZERO_COUNTERS = {
    "training": 0, "mps_training": 0, "simulator_rollout": 0, "causal_transition": 0,
    "reward_settlement": 0, "backward": 0, "optimizer_creation": 0, "optimizer_step": 0,
    "checkpoint_write": 0, "policy_mutation": 0, "source_mutation_during_R18R2": 0,
    "test6_access": 0, "github_push": 0,
}


class R18R2Error(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code


def require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise R18R2Error(code, detail)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n", encoding="utf-8")


def git(args: Sequence[str]) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT, text=True, capture_output=True, check=True).stdout.strip()


def stamp() -> str:
    return datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M%S+09:00")


def manifest_audit(root: Path) -> dict[str, Any]:
    manifest = load(root / "manifest.json")
    mismatches = []
    for relative, expected in dict(manifest.get("file_sha256", {})).items():
        path = root / relative
        actual = sha256(path) if path.is_file() else None
        if actual != expected:
            mismatches.append({"path": relative, "expected": expected, "actual": actual})
    return {"path": str(root), "declared_file_count": len(dict(manifest.get("file_sha256", {}))),
            "mismatches": mismatches, "all_match": not mismatches}


def function_hashes(source: str) -> dict[str, str]:
    lines = source.splitlines(True)
    tree = ast.parse(source)
    return {node.name: hashlib.sha256("".join(lines[node.lineno - 1:node.end_lineno]).encode("utf-8")).hexdigest()
            for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def source_scope_audit() -> dict[str, Any]:
    current_commit = git(["rev-parse", "HEAD"])
    require(not git(["status", "--porcelain=v1"]), SOURCE_BLOCK, "dirty_worktree")
    changed_files = set(filter(None, git(["diff", "--name-only", PRIOR_SOURCE, current_commit]).splitlines()))
    require(changed_files == EXPECTED_CHANGED_FILES, SOURCE_BLOCK, f"changed_files={sorted(changed_files)}")
    prior_source = git(["show", f"{PRIOR_SOURCE}:{EXECUTOR_REL}"])
    current_source = EXECUTOR.read_text(encoding="utf-8")
    before, after = function_hashes(prior_source), function_hashes(current_source)
    changed_functions = sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))
    require(changed_functions == [REPAIR_FUNCTION], SOURCE_BLOCK, f"changed_executor_functions={changed_functions}")
    frozen = {name: {"before": before[name], "after": after[name], "unchanged": before[name] == after[name]}
              for name in sorted(FROZEN_FUNCTIONS)}
    require(all(row["unchanged"] for row in frozen.values()), SOURCE_BLOCK, "frozen_execution_function_changed")
    return {
        "prior_source_commit": PRIOR_SOURCE, "source_commit": current_commit,
        "changed_files": sorted(changed_files), "changed_executor_functions": changed_functions,
        "repair_function_before_sha256": before[REPAIR_FUNCTION],
        "repair_function_after_sha256": after[REPAIR_FUNCTION],
        "frozen_function_hashes": frozen,
        "source_files": {EXECUTOR_REL: sha256(EXECUTOR), TEST_REL: sha256(TEST), RUNNER_REL: sha256(Path(__file__))},
        "diff": git(["diff", "--unified=2", PRIOR_SOURCE, current_commit, "--", EXECUTOR_REL]),
    }


def dynamic_support_regression() -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    import joint_assignment_credit_contract as CC
    import run_h4m_ae_ls3_bt8_r18_e1_bounded_training as R18
    import test_h4m_ae_ls3_bt8_r18_e1_bounded_training as T

    baseline = T._support_roundtrip_row()
    transition = baseline["t"]
    metadata = baseline["loaded"]["metadata"]
    result = R18._assert_support_roundtrip(rows=[baseline], CC=CC)
    cases = []
    for mutation in ("pair_identity", "snapshot_digest", "pair_count", "no_assign_index"):
        row = copy.deepcopy(T._support_roundtrip_row())
        mutated = row["loaded"]["metadata"]
        if mutation == "pair_identity":
            mutated["candidate_ids"][0]["candidate_id"] = "MUTATED_CANDIDATE"
        elif mutation == "snapshot_digest":
            mutated["candidate_support_digest"] = "b" * 64
        elif mutation == "pair_count":
            mutated["selectable_pair_count"] = 1
        else:
            mutated["no_assign_index"] = 1
        try:
            R18._assert_support_roundtrip(rows=[row], CC=CC)
            caught, detail = False, None
        except R18.R18Error as exc:
            caught, detail = "ACTION_SUPPORT_MUTATED_BETWEEN_ROLLOUT_AND_UPDATE" in str(exc), str(exc)
        cases.append({"mutation": mutation, "failed_closed": caught, "detail": detail})
    passed = (metadata["candidate_support_digest"] != transition.action_support_digest
              and result.get("behavior_equals_update_support") is True
              and all(row["failed_closed"] for row in cases))
    require(passed, REGRESSION_BLOCK, "dynamic_mutation_cases")
    return {
        "passed": passed, "legitimate_distinct_digest_domains_accepted": True,
        "snapshot_support_digest": metadata["candidate_support_digest"],
        "action_support_digest": transition.action_support_digest,
        "baseline_result": result, "mutation_cases": cases,
        "training": 0, "optimizer_creation": 0, "optimizer_step": 0,
    }


def frozen_inference_equivalence() -> dict[str, Any]:
    require(bool(torch.backends.mps.is_built()) and bool(torch.backends.mps.is_available()), MPS_BLOCK, "mps_unavailable")
    device = torch.device("mps:0")
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "simulator"))
    import joint_assignment_frozen_policy_snapshot as FPS
    import joint_assignment_frozen_tie_break as TIE
    import joint_assignment_learning as JL
    import multi_agent_assignment_contract as MC
    import multi_agent_candidate_assignment_head as H
    import run_h4m_ae_ls3_bt8_f1_bounded_training as F1MOD
    import run_h4m_ae_ls3_bt8_r18_e1_bounded_training as R18

    auth = load(BASE_AUTH)
    entries = [dict(row) for row in auth["upstream"]["review_snapshot_binding"]["entries"] if int(row["seed"]) == 20260822]
    expected_rows = load(R13 / "bt8r13_review_binding.json")["initial_replays"]
    expected_by_cell = {cell: {str(row["snapshot_digest"]): row for row in expected_rows if row["cell_id"] == cell}
                        for cell in ("AC-R1", "BD-R1")}
    checkpoints = auth["checkpoint_contract"]["initial_inputs"]
    rows = []
    actor_digests = {}
    for cell, key in (("AC-R1", "initial:AC-R1"), ("BD-R1", "initial:BD-R1")):
        checkpoint = Path(checkpoints[key]["path"])
        require(sha256(checkpoint) == checkpoints[key]["sha256"], INFERENCE_BLOCK, f"checkpoint={cell}")
        actor = F1MOD.strict_load(checkpoint, H, JL, MC, device)
        before = R18._module_digest(actor)
        actual_rows = R18._frozen_review_replay(actor=actor, entries=entries, FPS=FPS, F1MOD=F1MOD,
                                                H=H, TIE=TIE, device=device)
        after = R18._module_digest(actor)
        require(before == after, INFERENCE_BLOCK, f"actor_mutation={cell}")
        actor_digests[cell] = {"before": before, "after": after, "unchanged": True}
        for actual in actual_rows:
            expected = expected_by_cell[cell][str(actual["snapshot_digest"])]
            exact = all(actual[field] == expected[field] for field in
                        ("pair_logits", "no_assign_logit", "probabilities", "selected_identity",
                         "exact_tie", "tie_count", "selector", "tolerance", "entropy", "top_score_margin"))
            require(exact and actual["finite"], INFERENCE_BLOCK, f"row={cell}:{actual['snapshot_digest']}")
            rows.append({"cell": cell, "snapshot_digest": actual["snapshot_digest"],
                         "logit_delta": 0.0, "probability_delta": 0.0, "selection_equal": True,
                         "t1_exact_tie_equal": True, "actor_mutation": False})
    torch.mps.synchronize()
    require(len(rows) == 6, INFERENCE_BLOCK, f"row_count={len(rows)}")
    return {"passed": True, "device": str(device), "comparison_rows": rows, "comparison_row_count": len(rows),
            "max_abs_logit_delta": 0.0, "max_abs_probability_delta": 0.0,
            "selection_mismatch_count": 0, "t1_exact_tie_mismatch_count": 0,
            "actor_tensor_digests": actor_digests, "optimizer_exposure": 0, "training": 0}


def focused_tests() -> dict[str, Any]:
    command = [str(PYTHON), "-m", "pytest", "-q", str(TEST),
               str(ROOT / "test_joint_assignment_frozen_policy_selector.py"),
               str(ROOT / "test_h4m_ae_ls3_joint_assignment_credit.py")]
    completed = subprocess.run(command, cwd=PROJECT, text=True, capture_output=True)
    return {"command": command, "returncode": completed.returncode, "stdout": completed.stdout,
            "stderr": completed.stderr, "passed": completed.returncode == 0}


def checkpoint_audit(auth: Mapping[str, Any]) -> dict[str, Any]:
    entries = {**auth["checkpoint_contract"]["initial_inputs"], **auth["checkpoint_contract"]["final_evidence_only"]}
    rows = {key: {"path": value["path"], "expected": value["sha256"],
                  "actual": sha256(Path(value["path"])), "unchanged": sha256(Path(value["path"])) == value["sha256"]}
            for key, value in entries.items()}
    return {"entries": rows, "all_unchanged": all(row["unchanged"] for row in rows.values())}


def issue_r18r3_authorization(*, root: Path, source: Mapping[str, Any], evidence_sha256: str) -> dict[str, Any]:
    base = load(BASE_AUTH)
    base_actual = canonical_sha256({key: value for key, value in base.items() if key != "authorization_sha256"})
    require(base.get("authorization_sha256") == base_actual, SOURCE_BLOCK, "base_authorization_sha256")
    output_root = ARTIFACTS / root.name.replace("r18_r2_digest_domain_repair_validation", "r18_r3_e1_bounded_training_execution")
    require(not output_root.exists(), SOURCE_BLOCK, "r18r3_output_root_exists")
    authorization = copy.deepcopy(base)
    authorization["source_commit"] = source["source_commit"]
    authorization["upstream"]["r18_executor"]["sha256"] = sha256(EXECUTOR)
    authorization["checkpoint_contract"]["R18_output_root"] = str(output_root)
    authorization["reauthorization_identity"] = R18R3_IDENTITY
    authorization["reauthorization"] = {
        "one_shot": True, "authorized_at": datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds"),
        "reason": "R18-R2 minimal digest-domain repair validation passed with zero training and exact frozen inference equivalence.",
        "prior_authorization_sha256": base_actual, "prior_blocked_output_root": str(PRIOR_R18R1),
        "prior_blocking_gate": "BLOCKED_R18_EXECUTION_INTEGRITY_FAILURE",
        "r18r2_artifact_root": str(root), "r18r2_gate": PASS_GATE,
        "r18r2_evidence_sha256": evidence_sha256, "source_commit": source["source_commit"],
        "executor_sha256": sha256(EXECUTOR), "identical_r17_envelope": True,
        "new_unused_output_root": str(output_root),
    }
    authorization["authorization_sha256"] = canonical_sha256({key: value for key, value in authorization.items() if key != "authorization_sha256"})
    path = root / "r18r3_bounded_training_authorization_manifest.json"
    dump(path, authorization)
    dry = subprocess.run([str(PYTHON), str(EXECUTOR), "--authorization-manifest", str(path),
                          "--authorization-sha256", authorization["authorization_sha256"], "--dry-run"],
                         cwd=PROJECT, text=True, capture_output=True)
    require(dry.returncode == 0 and load_json_text(dry.stdout).get("authorization_valid") is True,
            SOURCE_BLOCK, f"r18r3_dry_run={dry.stderr or dry.stdout}")
    command = [str(PYTHON), str(EXECUTOR), "--authorization-manifest", str(path),
               "--authorization-sha256", authorization["authorization_sha256"], "--execute-exact-r17-envelope"]
    (root / "r18r3_exact_execution_command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    return {"generated": True, "authorization_identity": R18R3_IDENTITY, "manifest": str(path),
            "authorization_sha256": authorization["authorization_sha256"], "output_root": str(output_root),
            "dry_run_passed": True, "dry_run_stdout": dry.stdout.strip(), "execute_command": command}


def load_json_text(value: str) -> dict[str, Any]:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}


def write_manifest(root: Path, gate: str, source_commit: str) -> None:
    hashes = {item.relative_to(root).as_posix(): sha256(item) for item in root.rglob("*")
              if item.is_file() and item.name != "manifest.json"}
    dump(root / "manifest.json", {"stage": STAGE, "gate": gate, "source_commit": source_commit, "file_sha256": hashes})


def main() -> None:
    root = ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r2_digest_domain_repair_validation_{stamp()}"
    require(not root.exists(), SOURCE_BLOCK, "append_only_artifact_collision")
    root.mkdir(parents=True)
    source_commit = git(["rev-parse", "HEAD"])
    try:
        source = source_scope_audit()
        upstream = {"r18r0": manifest_audit(PRIOR_R18R0), "r18r1": manifest_audit(PRIOR_R18R1),
                    "r13": manifest_audit(R13), "r16": manifest_audit(R16)}
        require(all(row["all_match"] for row in upstream.values()), SOURCE_BLOCK, "upstream_manifest")
        prior_gate = load(PRIOR_R18R1 / "gate_decision.json")
        prior_counters = load(PRIOR_R18R1 / "test_results.json")["execution_counters"]
        require(prior_gate["gate"] == "BLOCKED_R18_EXECUTION_INTEGRITY_FAILURE"
                and all(int(prior_counters[key]) == 0 for key in
                        ("training", "mps_training", "actor_optimizer_step", "critic_optimizer_step", "raw_optimizer_step", "checkpoint_write")),
                SOURCE_BLOCK, "prior_consumed_attempt")
        tests = focused_tests()
        require(tests["passed"], REGRESSION_BLOCK, tests["stdout"] + tests["stderr"])
        support = dynamic_support_regression()
        inference = frozen_inference_equivalence()
        checkpoint = checkpoint_audit(load(BASE_AUTH))
        require(checkpoint["all_unchanged"], SOURCE_BLOCK, "checkpoint_hash")
        evidence_sha256 = canonical_sha256({"source": source, "upstream": upstream, "tests": tests,
                                            "support": support, "inference": inference, "checkpoint": checkpoint,
                                            "execution_counters": ZERO_COUNTERS})
        authorization = issue_r18r3_authorization(root=root, source=source, evidence_sha256=evidence_sha256)
        outputs = {
            "source_scope_and_sha_audit.json": source,
            "upstream_binding_audit.json": {"artifacts": upstream, "prior_r18r1_gate": prior_gate,
                                               "prior_r18r1_execution_counters": prior_counters},
            "dynamic_support_mutation_regression.json": support,
            "frozen_inference_equivalence.json": inference,
            "checkpoint_hash_audit.json": checkpoint,
            "execution_counters.json": ZERO_COUNTERS,
            "test_results.json": {"focused_tests": tests, "hard_failures": [], "training": 0,
                                  "optimizer_creation": 0, "optimizer_step": 0, "checkpoint_write": 0},
            "r18r3_authorization_summary.json": authorization,
            "gate_decision.json": {"stage": STAGE, "gate": PASS_GATE, "classification": "MINIMAL_DIGEST_DOMAIN_REPAIR_VALIDATED",
                                   "source_commit": source["source_commit"], "hard_failures": [],
                                   "training": 0, "r18r3_one_shot_authorized": True,
                                   "next_step": "R18-R3 exact same 6-window / 48-decision / 96-transition bounded training only"},
        }
        for name, value in outputs.items():
            dump(root / name, value)
        (root / "final_report.md").write_text(
            f"# {STAGE}\n\n- gate: `{PASS_GATE}`\n- source commit: `{source['source_commit']}`\n"
            "- executor change: `_assert_support_roundtrip` only\n- dynamic support mutation regression: PASS\n"
            "- frozen inference equivalence: 6/6 exact, zero actor mutation\n- training/optimizer/checkpoint writes: 0\n"
            "- next: R18-R3 exact one-shot bounded training\n", encoding="utf-8")
        write_manifest(root, PASS_GATE, source["source_commit"])
        (root / "_SUCCESS.lock").write_text(PASS_GATE + "\n", encoding="utf-8")
        print(json.dumps({"gate": PASS_GATE, "artifact": str(root), "source_commit": source["source_commit"],
                          "r18r3_authorization": authorization}, ensure_ascii=False, sort_keys=True))
    except R18R2Error as exc:
        dump(root / "execution_counters.json", ZERO_COUNTERS)
        dump(root / "r18r3_authorization_summary.json", {"generated": False, "blocking_gate": exc.code})
        dump(root / "gate_decision.json", {"stage": STAGE, "gate": exc.code, "classification": "BLOCKED",
                                           "source_commit": source_commit, "hard_failures": [str(exc)],
                                           "training": 0, "r18r3_one_shot_authorized": False, "next_step": "STOP"})
        (root / "final_report.md").write_text(f"# {STAGE} blocked\n\n- gate: `{exc.code}`\n- detail: `{exc}`\n- training: `0`\n", encoding="utf-8")
        write_manifest(root, exc.code, source_commit)
        print(json.dumps({"gate": exc.code, "artifact": str(root), "error": str(exc)}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
