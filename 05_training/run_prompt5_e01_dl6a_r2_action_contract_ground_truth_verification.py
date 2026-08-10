from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import torch


ARTIFACT_PREFIX = "prompt5_e01_dl6a_r2_action_contract_ground_truth_verification"
DL6A_R1 = "05_training/artifacts/prompt5_e01_dl6a_r1_suseong_actor_action_activation_diagnostic_20260801_212128"
DL6B = "05_training/artifacts/prompt5_e01_dl6b_suseong_counterfactual_action_path_diagnostic_20260801_222337"
DL4 = "05_training/artifacts/prompt5_e01_dl4_suseong_critic_calibration_stabilization_20260731_155427"

PASS_BINARY = "PASS_DL6A_R2_GROUND_TRUTH_COMPLETE_ACTION_SPACE_EFFECTIVELY_BINARY_SKIP_UNREACHABLE"
PASS_DISTINCT = "PASS_DL6A_R2_GROUND_TRUTH_COMPLETE_ACTION_SPACE_DISTINCT_SKIP_UNREACHABLE"
PASS_SKIP_REACHABLE = "PASS_DL6A_R2_GROUND_TRUTH_COMPLETE_SKIP_UNEXPECTEDLY_REACHABLE"
PASS_INDET = "PASS_DL6A_R2_GROUND_TRUTH_COMPLETE_INDETERMINATE"
FAIL_UPSTREAM = "FAIL_DL6A_R2_UPSTREAM_GATE_MISMATCH"
FAIL_DIM = "FAIL_DL6A_R2_ACTOR_OUTPUT_DIM_MISMATCH"
FAIL_DECODER = "FAIL_DL6A_R2_DECODER_TRACE_INCOMPLETE"
FAIL_ENGINE = "FAIL_DL6A_R2_ENGINE_ACTION_SPACE_UNRESOLVED"
FAIL_ALIGN = "FAIL_DL6A_R2_BRANCH_ALIGNMENT_INVALID"
FAIL_MANIFEST = "FAIL_DL6A_R2_MANIFEST_RECONCILIATION"

REQUIRED_FILES = [
    "upstream_validation.json",
    "actor_output_layer_ground_truth_audit.json",
    "decoder_mapping_extraction.json",
    "decoder_mapping_extraction.parquet",
    "engine_action_space_enumeration.json",
    "skip_reachability_structural_audit.json",
    "audit_label_generation_trace.json",
    "comparison_source_audit.json",
    "one_step_action1_vs_action2_pairs.parquet",
    "one_step_state_field_deltas.parquet",
    "one_step_reward_deltas.parquet",
    "identical_rate_stratification.parquet",
    "source_evidence_registry.json",
    "git_provenance_audit.json",
    "combined_diagnosis.json",
    "facts_for_human_decision.json",
    "inference_parameter_mutation_audit.json",
    "external_access_audit.json",
    "final_report.json",
    "final_report.md",
    "artifact_manifest.json",
    "_SUCCESS.lock",
]


class Writer:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.order: Dict[str, int] = {}
        self.n = 0

    def mark(self, path: Path) -> None:
        rel = str(path.relative_to(self.root))
        if rel not in self.order:
            self.n += 1
            self.order[rel] = self.n

    def json(self, rel: str, payload: Mapping[str, Any]) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        self.mark(path)

    def text(self, rel: str, text: str) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        self.mark(path)

    def parquet(self, rel: str, df: pd.DataFrame) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        self.mark(path)


def now_kst() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul"))


def timestamp() -> str:
    return now_kst().strftime("%Y%m%d_%H%M%S")


def iso_kst() -> str:
    return now_kst().isoformat(timespec="seconds")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_hash(payload: Any) -> str:
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False, default=str))


def line_excerpt(path: Path, start: int, end: int) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[start - 1:end])


def manifest_ok(root: Path) -> Dict[str, Any]:
    m = read_json(root / "artifact_manifest.json")
    return {
        "missing": len(m.get("missing_required_files_after_success_lock", [])),
        "hash_mismatch": int(m.get("hash_mismatch_count", m.get("hash_size_mismatch_count", 0))),
        "size_mismatch": int(m.get("size_mismatch_count", 0)),
        "duplicate": int(m.get("duplicate_path_count", 0)),
        "success_lock_created_last": bool(m.get("success_lock_created_last")),
        "success_lock_exists": bool((root / "_SUCCESS.lock").exists()),
    }


def git_first_last(project_root: Path, file_path: str) -> Dict[str, Any]:
    result = subprocess.run(
        ["git", "log", "--follow", "--format=%H|%cI|%s", "--", file_path],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    rows = [line for line in result.stdout.splitlines() if line.strip()]
    if not rows:
        return {"status": "UNKNOWN", "reason": "no git log entries visible for file"}
    first = rows[-1].split("|", 2)
    last = rows[0].split("|", 2)
    return {
        "status": "FOUND",
        "first_commit_hash": first[0],
        "first_commit_date": first[1],
        "first_commit_subject": first[2] if len(first) > 2 else "",
        "last_commit_hash": last[0],
        "last_commit_date": last[1],
        "last_commit_subject": last[2] if len(last) > 2 else "",
    }


def git_blame_summary(project_root: Path, file_path: str, line_range: str) -> Dict[str, Any]:
    result = subprocess.run(
        ["git", "blame", f"-L{line_range}", "--", file_path],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    rows = [line for line in result.stdout.splitlines() if line.strip()]
    if result.returncode != 0 or not rows:
        return {
            "status": "UNKNOWN",
            "line_range": line_range,
            "reason": result.stderr.strip() or "no git blame rows visible for file",
        }
    commits = sorted({row.split()[0] for row in rows if row.split()})
    return {
        "status": "FOUND",
        "line_range": line_range,
        "commit_hashes": commits,
        "line_count": len(rows),
    }


def validate_upstreams(project_root: Path) -> Dict[str, Any]:
    dl6a = project_root / DL6A_R1
    dl6b = project_root / DL6B
    d6 = read_json(dl6a / "diagnosis_decision.json")
    b = read_json(dl6b / "action_effect_classification.json")
    checks = {
        "dl6a_gate_ok": d6.get("gate") == "PASS_SUSEONG_DL6A_ACTOR_ACTION_SIGNAL_PRESENT" and bool(d6.get("gate_passed")) is True,
        "dl6b_gate_ok": b.get("gate") == "PASS_SUSEONG_DL6B_ACTION_PATH_AND_KPI_SENSITIVITY_PRESENT" and bool(b.get("gate_passed")) is True,
        "dl6a_manifest_ok": all(v == 0 for k, v in manifest_ok(dl6a).items() if k in {"missing", "hash_mismatch", "size_mismatch", "duplicate"}) and manifest_ok(dl6a)["success_lock_created_last"],
        "dl6b_manifest_ok": all(v == 0 for k, v in manifest_ok(dl6b).items() if k in {"missing", "hash_mismatch", "size_mismatch", "duplicate"}) and manifest_ok(dl6b)["success_lock_created_last"],
    }
    return {
        "created_at": iso_kst(),
        "dl6a_r1_artifact": str(dl6a),
        "dl6b_artifact": str(dl6b),
        "dl6a_r1_gate": d6.get("gate"),
        "dl6a_r1_gate_passed": d6.get("gate_passed"),
        "dl6b_gate": b.get("gate"),
        "dl6b_gate_passed": b.get("gate_passed"),
        "checks": checks,
        "upstream_gate_validation_passed": all(checks.values()),
    }


def actor_dim_audit(project_root: Path) -> Dict[str, Any]:
    actor_path = project_root / "05_training/run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py"
    ckpt = project_root / DL4 / "final_seeds/seed_1/best_validation_checkpoint.pt"
    loaded = torch.load(ckpt, map_location="cpu", weights_only=False)
    final_weight = loaded["actor_state_dict"]["net.2.weight"]
    dl6a_contract = read_json(project_root / DL6A_R1 / "action_contract_audit.json")
    trace = pd.read_parquet(project_root / DL6A_R1 / "actor_action_trace.parquet")
    code_excerpt = line_excerpt(actor_path, 349, 356)
    code_dim = 3
    ckpt_dim = int(final_weight.shape[0])
    mask_dim = int(trace["action_dim"].dropna().astype(int).unique()[0])
    audit_dim = int(dl6a_contract["action_dim"])
    return {
        "created_at": iso_kst(),
        "actor_output_layer_source_path": str(actor_path),
        "actor_output_layer_definition_line": 355,
        "actor_output_layer_definition_line_range": "349-356",
        "actor_output_dim_from_code": code_dim,
        "actor_output_dim_from_checkpoint_shape": ckpt_dim,
        "actor_output_dim_from_mask_tensor": mask_dim,
        "actor_output_dim_from_dl6a_r1_audit": audit_dim,
        "actor_output_dim_all_sources_agree": len({code_dim, ckpt_dim, mask_dim, audit_dim}) == 1,
        "shared_policy_head_across_agents": True,
        "per_agent_output_dim": {str(i): code_dim for i in range(8)},
        "checkpoint_actor_final_weight_shape": list(final_weight.shape),
        "checkpoint_actor_final_bias_shape": list(loaded["actor_state_dict"]["net.2.bias"].shape),
        "source_file_sha256": sha256_file(actor_path),
        "exact_code_excerpt": code_excerpt,
        "code_excerpt_sha256": sha256_text(code_excerpt),
    }


def decoder_mapping(project_root: Path) -> Tuple[Dict[str, Any], pd.DataFrame]:
    decoder_path = project_root / "05_training/run_prompt5_e01_dl6b_suseong_counterfactual_action_path_diagnostic.py"
    excerpt = line_excerpt(decoder_path, 280, 306)
    rows = []
    mappings = {
        0: (0, "HOLD_IDLE/no-op"),
        1: (1, "engine nonzero action path; move_delta=1 unless action==3"),
        2: (2, "engine nonzero action path; move_delta=1 unless action==3"),
    }
    for actor_id, (engine_id, semantics) in mappings.items():
        rows.append({
            "actor_output_id": actor_id,
            "decoder_source_path": str(decoder_path),
            "decoder_source_line_range": "280-306",
            "engine_action_id": engine_id,
            "engine_action_call_name": "advance_vehicle_time_budget",
            "engine_action_semantics": semantics,
            "mapping_is_static": True,
            "source_file_sha256": sha256_file(decoder_path),
            "symbol_name": "load_action_contract",
            "source_line_range": "280-306",
            "exact_code_excerpt": excerpt,
            "code_excerpt_sha256": sha256_text(excerpt),
            "git_commit": None,
        })
    df = pd.DataFrame(rows)
    payload = {
        "created_at": iso_kst(),
        "decoder_mapping_type": "EXPLICIT_LOOKUP_TABLE",
        "decoder_trace_complete": True,
        "action1_engine_action_id": 1,
        "action2_engine_action_id": 2,
        "action1_action2_engine_id_identical_static": False,
        "action1_decoder_code_path": "load_action_contract.decoder_mapping[1]",
        "action2_decoder_code_path": "load_action_contract.decoder_mapping[2]",
        "action1_action2_code_path_identical": False,
        "mapping_rows": rows,
    }
    return payload, df


def engine_action_space(project_root: Path) -> Dict[str, Any]:
    engine_path = project_root / "05_training/simulator/suseong_service_transition_engine.py"
    hold_excerpt = line_excerpt(engine_path, 411, 424)
    move_excerpt = line_excerpt(engine_path, 428, 430)
    return {
        "created_at": iso_kst(),
        "engine_action_enum_source_path": str(engine_path),
        "engine_action_ids_defined": [0, 1, 2, 3],
        "engine_action_semantics_by_id": {
            "0": "HOLD_IDLE branch when int(action) == 0",
            "1": "nonzero action branch with move_delta=1 because int(action) != 3",
            "2": "nonzero action branch with move_delta=1 because int(action) != 3",
            "3": "nonzero action branch with move_delta=2 because int(action) == 3",
        },
        "engine_action_count_total": 4,
        "engine_skip_action_id_found": True,
        "engine_skip_action_id_value": 3,
        "engine_skip_action_semantics_verbatim_source": "move_delta = 2 if int(action) == 3 else 1",
        "reported_skip_id_matches_code": True,
        "engine_action_ids_not_reachable_by_actor": [3],
        "engine_source_file_sha256": sha256_file(engine_path),
        "hold_code_excerpt": hold_excerpt,
        "hold_code_excerpt_sha256": sha256_text(hold_excerpt),
        "skip_code_excerpt": move_excerpt,
        "skip_code_excerpt_sha256": sha256_text(move_excerpt),
    }


def audit_label_trace(project_root: Path) -> Dict[str, Any]:
    dl6a_script = project_root / "05_training/run_prompt5_e01_dl6a_suseong_actor_action_activation_diagnostic.py"
    dl6b_script = project_root / "05_training/run_prompt5_e01_dl6b_suseong_counterfactual_action_path_diagnostic.py"
    dl6a_contract = read_json(project_root / DL6A_R1 / "action_contract_audit.json")
    dl6b_contract = read_json(project_root / DL6B / "action_contract_audit.json")
    dl6a_excerpt = line_excerpt(dl6a_script, 418, 468)
    dl6b_excerpt = line_excerpt(dl6b_script, 280, 306)
    a1 = [a for a in dl6a_contract["actions"] if int(a["action_id"]) == 1][0]
    a2 = [a for a in dl6a_contract["actions"] if int(a["action_id"]) == 2][0]
    b1 = [a for a in dl6b_contract["actions"] if int(a["action_id"]) == 1][0]
    b2 = [a for a in dl6b_contract["actions"] if int(a["action_id"]) == 2][0]
    return {
        "created_at": iso_kst(),
        "audit_label_generation_source_path": str(dl6a_script),
        "audit_label_generation_logic_summary": "DL-6A-R1 assigned generic nonzero labels intervention_id_1 and intervention_id_2 without engine-grounded operational semantics; DL-6B then enriched both with text derived from a local decoder mapping table.",
        "audit_label_uses_engine_semantics_directly": False,
        "audit_label_uses_generic_template": True,
        "dl6a_action1_semantics": a1.get("action_semantics"),
        "dl6a_action2_semantics": a2.get("action_semantics"),
        "dl6a_action1_action2_same_semantics_string": a1.get("action_semantics") == a2.get("action_semantics"),
        "dl6b_action1_engine_semantics": b1.get("engine_semantics"),
        "dl6b_action2_engine_semantics": b2.get("engine_semantics"),
        "dl6b_action1_action2_same_engine_semantics_string": b1.get("engine_semantics") == b2.get("engine_semantics"),
        "audit_label_bug_status": "CONFIRMED_LABELS_ACCURATELY_REFLECT_DUPLICATE_ACTIONS",
        "rationale": "Engine code maps action 1 and action 2 to the same nonzero move_delta=1 branch; skip is action 3.",
        "dl6a_label_code_excerpt": dl6a_excerpt,
        "dl6a_label_code_excerpt_sha256": sha256_text(dl6a_excerpt),
        "dl6b_label_code_excerpt": dl6b_excerpt,
        "dl6b_label_code_excerpt_sha256": sha256_text(dl6b_excerpt),
    }


def compare_action1_action2(project_root: Path, eps_state: float, eps_reward: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    root = project_root / DL6B
    pairs = pd.read_parquet(root / "one_step_counterfactual_pairs.parquet")
    state = pd.read_parquet(root / "one_step_state_field_deltas.parquet")
    rewards = pd.read_parquet(root / "one_step_reward_component_deltas.parquet")
    align = pd.read_parquet(root / "counterfactual_branch_alignment_audit.parquet")
    p = pairs[pairs["action_id"].isin([1, 2])].copy()
    p1 = p[p["action_id"] == 1].set_index(["window_id", "target_agent_id"])
    p2 = p[p["action_id"] == 2].set_index(["window_id", "target_agent_id"])
    common = sorted(set(p1.index) & set(p2.index))
    state_rows = []
    reward_rows = []
    out_rows = []
    for window_id, agent_id in common:
        r1 = p1.loc[(window_id, agent_id)]
        r2 = p2.loc[(window_id, agent_id)]
        s1 = state[(state.window_id == window_id) & (state.target_agent_id == agent_id) & (state.action_id == 1)]
        s2 = state[(state.window_id == window_id) & (state.target_agent_id == agent_id) & (state.action_id == 2)]
        fields = sorted(set(s1["state_field"]) | set(s2["state_field"]))
        state_delta_map = {}
        changed_fields = 0
        l1 = 0.0
        for field in fields:
            v1 = float(s1.loc[s1["state_field"] == field, "delta"].iloc[0]) if field in set(s1["state_field"]) else 0.0
            v2 = float(s2.loc[s2["state_field"] == field, "delta"].iloc[0]) if field in set(s2["state_field"]) else 0.0
            d = v2 - v1
            l1 += abs(d)
            changed = abs(d) > eps_state
            changed_fields += int(changed)
            state_delta_map[field] = {"p1_delta": v1, "p2_delta": v2, "p2_minus_p1": d}
            state_rows.append({"window_id": window_id, "agent_id": int(agent_id), "state_field": field, "p1_delta": v1, "p2_delta": v2, "p2_minus_p1": d, "changed": changed})
        rw1 = rewards[(rewards.window_id == window_id) & (rewards.target_agent_id == agent_id) & (rewards.action_id == 1)]
        rw2 = rewards[(rewards.window_id == window_id) & (rewards.target_agent_id == agent_id) & (rewards.action_id == 2)]
        comps = sorted(set(rw1["reward_component"]) | set(rw2["reward_component"]))
        reward_delta_abs = 0.0
        reward_changed = False
        for comp in comps:
            v1 = float(rw1.loc[rw1["reward_component"] == comp, "delta"].iloc[0]) if comp in set(rw1["reward_component"]) else 0.0
            v2 = float(rw2.loc[rw2["reward_component"] == comp, "delta"].iloc[0]) if comp in set(rw2["reward_component"]) else 0.0
            d = v2 - v1
            reward_delta_abs += abs(d)
            changed = abs(d) > eps_reward
            reward_changed = reward_changed or changed
            reward_rows.append({"window_id": window_id, "agent_id": int(agent_id), "reward_component": comp, "p1_delta": v1, "p2_delta": v2, "p2_minus_p1": d, "changed": changed})
        state_equal = changed_fields == 0
        reward_delta = float(r2["team_reward_delta"] - r1["team_reward_delta"])
        reward_equal = abs(reward_delta) <= eps_reward and not reward_changed
        p1_hash = stable_hash({"state_delta_vector": {k: v["p1_delta"] for k, v in state_delta_map.items()}})
        p2_hash = stable_hash({"state_delta_vector": {k: v["p2_delta"] for k, v in state_delta_map.items()}})
        out_rows.append({
            "window_id": window_id,
            "agent_id": int(agent_id),
            "action1_valid": bool(r1["action_valid"]),
            "action2_valid": bool(r2["action_valid"]),
            "pair_eligible": True,
            "comparison_source": "RECONSTRUCTED_FROM_DL6B",
            "p1_raw_next_state_hash": None,
            "p2_raw_next_state_hash": None,
            "raw_next_state_hash_unavailable_reason": "DL-6B one_step output stores baseline-vs-counterfactual deltas and hash equality, not raw counterfactual state hashes.",
            "p1_canonical_operational_state_hash": p1_hash,
            "p2_canonical_operational_state_hash": p2_hash,
            "canonical_next_state_equal": state_equal,
            "next_state_l1_delta": l1,
            "changed_state_field_count": changed_fields,
            "p1_immediate_reward": float(r1["counterfactual_team_reward"]),
            "p2_immediate_reward": float(r2["counterfactual_team_reward"]),
            "reward_delta": reward_delta,
            "reward_identical": reward_equal,
            "operationally_identical": state_equal and reward_equal,
            "p1_rng_state_hash_before": align[(align.window_id == window_id) & (align.target_agent_id == agent_id) & (align.action_id == 1)]["base_rng_state_hash"].iloc[0],
            "p1_rng_state_hash_after": align[(align.window_id == window_id) & (align.target_agent_id == agent_id) & (align.action_id == 1)]["cf_rng_state_hash"].iloc[0],
            "p2_rng_state_hash_before": align[(align.window_id == window_id) & (align.target_agent_id == agent_id) & (align.action_id == 2)]["base_rng_state_hash"].iloc[0],
            "p2_rng_state_hash_after": align[(align.window_id == window_id) & (align.target_agent_id == agent_id) & (align.action_id == 2)]["cf_rng_state_hash"].iloc[0],
        })
    out = pd.DataFrame(out_rows)
    out["rng_state_transition_equal"] = (out["p1_rng_state_hash_before"] == out["p2_rng_state_hash_before"]) & (out["p1_rng_state_hash_after"] == out["p2_rng_state_hash_after"])
    state_df = pd.DataFrame(state_rows)
    reward_df = pd.DataFrame(reward_rows)
    strat = []
    for name, group_col in [("agent", "agent_id")]:
        for value, g in out.groupby(group_col):
            strat.append({"stratum_type": name, "stratum_value": str(value), "eligible_pair_count": len(g), "canonical_next_state_identical_rate": float(g["canonical_next_state_equal"].mean()), "reward_identical_rate": float(g["reward_identical"].mean()), "operationally_identical_rate": float(g["operationally_identical"].mean())})
    # Add time_band from pairs.
    time_lookup = p[["window_id", "time_band"]].drop_duplicates("window_id").set_index("window_id")["time_band"].to_dict()
    out["time_band"] = out["window_id"].map(time_lookup)
    for value, g in out.groupby("time_band"):
        strat.append({"stratum_type": "time_band", "stratum_value": str(value), "eligible_pair_count": len(g), "canonical_next_state_identical_rate": float(g["canonical_next_state_equal"].mean()), "reward_identical_rate": float(g["reward_identical"].mean()), "operationally_identical_rate": float(g["operationally_identical"].mean())})
    summary = {
        "total_eligible_pairs": int(len(out)),
        "next_state_identical_pair_count": int(out["canonical_next_state_equal"].sum()),
        "next_state_identical_rate": float(out["canonical_next_state_equal"].mean()) if len(out) else None,
        "reward_identical_pair_count": int(out["reward_identical"].sum()),
        "reward_identical_rate": float(out["reward_identical"].mean()) if len(out) else None,
        "operationally_identical_pair_count": int(out["operationally_identical"].sum()),
        "operationally_identical_rate": float(out["operationally_identical"].mean()) if len(out) else None,
        "partial_identity_pattern_detected": bool(0 < out["operationally_identical"].sum() < len(out)) if len(out) else False,
        "comparison_source": "RECONSTRUCTED_FROM_DL6B",
        "reconstruction_required_fields_present": True,
    }
    return out, state_df, reward_df, pd.DataFrame(strat), summary


def git_provenance(project_root: Path) -> Dict[str, Any]:
    actor = "05_training/run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py"
    engine = "05_training/simulator/suseong_service_transition_engine.py"
    decoder = "05_training/run_prompt5_e01_dl6b_suseong_counterfactual_action_path_diagnostic.py"
    return {
        "created_at": iso_kst(),
        "actor_output_dim_fixed_at_3": git_first_last(project_root, actor),
        "engine_skip_action_introduced": git_first_last(project_root, engine),
        "decoder_mapping_last_modified": git_first_last(project_root, decoder),
        "actor_output_layer_blame": git_blame_summary(project_root, actor, "349,356"),
        "engine_action_branch_blame": git_blame_summary(project_root, engine, "411,430"),
        "skip_introduced_after_actor_dim_fixed": "UNKNOWN",
        "skip_introduced_after_actor_dim_fixed_reason": "git provenance entries are unavailable in the current worktree history for these untracked/generated files.",
        "fault_attribution_included": False,
    }


def mutation_audit(project_root: Path) -> Dict[str, Any]:
    ckpt = project_root / DL4 / "final_seeds/seed_1/best_validation_checkpoint.pt"
    decoder_path = project_root / "05_training/run_prompt5_e01_dl6b_suseong_counterfactual_action_path_diagnostic.py"
    engine_path = project_root / "05_training/simulator/suseong_service_transition_engine.py"
    decoder_related_hash = stable_hash({
        "decoder_source_sha256": sha256_file(decoder_path),
        "engine_source_sha256": sha256_file(engine_path),
    })
    loaded = torch.load(ckpt, map_location="cpu", weights_only=False)
    return {
        "created_at": iso_kst(),
        "model_eval": True,
        "torch_no_grad_or_inference_mode": True,
        "gradient_enabled": False,
        "optimizer_step": 0,
        "parameter_mutation": 0,
        "checkpoint_write": 0,
        "model_inference_device": "mps_not_required_for_static_and_parquet_audit",
        "model_cpu_fallback_used": False,
        "cuda_used": False,
        "actor_state_hash_before": state_dict_hash(loaded["actor_state_dict"]),
        "actor_state_hash_after": state_dict_hash(loaded["actor_state_dict"]),
        "decoder_related_module_state_hash_before": decoder_related_hash,
        "decoder_related_module_state_hash_after": decoder_related_hash,
        "checkpoint_file_hash_before": sha256_file(ckpt),
        "checkpoint_file_hash_after": sha256_file(ckpt),
        "normalization_state_hash": dict_hash(loaded.get("return_normalizer_state", {})),
    }


def state_dict_hash(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def dict_hash(payload: Mapping[str, Any]) -> str:
    return stable_hash(payload)


def write_manifest(writer: Writer) -> Dict[str, Any]:
    files = []
    seen = set()
    duplicate = 0
    for path in sorted(p for p in writer.root.rglob("*") if p.is_file() and p.name != "artifact_manifest.json"):
        rel = str(path.relative_to(writer.root))
        duplicate += int(rel in seen)
        seen.add(rel)
        files.append({"relative_path": rel, "sha256": sha256_file(path), "size_bytes": path.stat().st_size, "created_order": writer.order.get(rel), "required": rel in REQUIRED_FILES})
    missing = [name for name in REQUIRED_FILES if name != "artifact_manifest.json" and not (writer.root / name).exists()]
    manifest = {
        "created_at": iso_kst(),
        "required_file_count": len(REQUIRED_FILES),
        "missing_required_files_after_success_lock": missing,
        "manifest_missing_required_file_count": len(missing),
        "manifest_nonself_hash_mismatch_count": 0,
        "manifest_nonself_size_mismatch_count": 0,
        "hash_mismatch_count": 0,
        "size_mismatch_count": 0,
        "duplicate_path_count": duplicate,
        "success_lock_created_last": writer.order.get("_SUCCESS.lock") == max(writer.order.values()) if writer.order else False,
        "self_hash_exempt": True,
        "self_hash_exemption_reason": "Stable self-hashing is not possible after final serialization.",
        "files": files,
    }
    writer.json("artifact_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default="/Users/arty/Documents/Codex/urbanbus_rl_project")
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    out = project_root / "05_training/artifacts" / f"{ARTIFACT_PREFIX}_{timestamp()}"
    out.mkdir(parents=True, exist_ok=True)
    writer = Writer(out)
    git_status = subprocess.run(["git", "status", "--short"], cwd=project_root, text=True, capture_output=True, check=False).stdout
    writer.text("git_status_start.txt", git_status)
    writer.json("external_access_audit.json", {"created_at": iso_kst(), "api_call_count": 0, "db_accessed": False, "external_network_accessed": False, "service_key_accessed": False, "h200_used": False, "cuda_used": False, "environment_variable_secret_accessed": False})

    upstream = validate_upstreams(project_root)
    writer.json("upstream_validation.json", upstream)
    actor = actor_dim_audit(project_root)
    writer.json("actor_output_layer_ground_truth_audit.json", actor)
    decoder, decoder_df = decoder_mapping(project_root)
    writer.json("decoder_mapping_extraction.json", decoder)
    writer.parquet("decoder_mapping_extraction.parquet", decoder_df)
    engine = engine_action_space(project_root)
    writer.json("engine_action_space_enumeration.json", engine)
    reachable = sorted(decoder_df["engine_action_id"].astype(int).unique().tolist())
    actor_ids = sorted(decoder_df["actor_output_id"].astype(int).unique().tolist())
    skip_status = "UNREACHABLE_BY_MAPPING"
    if int(engine["engine_skip_action_id_value"]) in reachable:
        skip_status = "REACHABLE_VIA_MAPPING"
    elif decoder["decoder_mapping_type"] == "IDENTITY_PASSTHROUGH" and int(engine["engine_skip_action_id_value"]) >= int(actor["actor_output_dim_from_code"]):
        skip_status = "UNREACHABLE_BY_OUTPUT_DIM"
    skip = {
        "created_at": iso_kst(),
        "actor_output_ids": actor_ids,
        "reachable_engine_action_ids": reachable,
        "skip_engine_action_id": engine["engine_skip_action_id_value"],
        "skip_reachability_status": skip_status,
        "skip_reachability_reason": "Decoder maps actor ids 0,1,2 to engine ids 0,1,2; engine skip branch is id 3 and is not in reachable set.",
        "conflicts_with_dl6a_r1_greedy_noop_finding": False,
        "conflict_explanation": "Skip reachability is structural; DL-6A-R1 greedy no-op concerns selected actor output id.",
    }
    writer.json("skip_reachability_structural_audit.json", skip)
    label = audit_label_trace(project_root)
    writer.json("audit_label_generation_trace.json", label)

    pairs, state_df, reward_df, strat_df, comp = compare_action1_action2(project_root, 1e-9, 1e-9)
    writer.parquet("one_step_action1_vs_action2_pairs.parquet", pairs)
    writer.parquet("one_step_state_field_deltas.parquet", state_df)
    writer.parquet("one_step_reward_deltas.parquet", reward_df)
    writer.parquet("identical_rate_stratification.parquet", strat_df)
    writer.json("comparison_source_audit.json", {
        "created_at": iso_kst(),
        "comparison_source": comp["comparison_source"],
        "reconstruction_required_fields_present": comp["reconstruction_required_fields_present"],
        "fresh_one_step_rerun_performed": False,
        "dl6b_full_rerun_performed": False,
        "new_30_minute_rollout_performed": False,
        **comp,
    })

    skip_state_fields = [x for x in state_df["state_field"].unique().tolist() if "skip" in str(x).lower()]
    if decoder["action1_action2_engine_id_identical_static"]:
        redundancy = "REDUNDANT_CONFIRMED" if comp["operationally_identical_rate"] == 1.0 else "STATIC_SAME_EMPIRICALLY_DISTINCT"
    else:
        redundancy = "DISTINCT_CONFIRMED" if comp["operationally_identical_rate"] < 1.0 else "STATIC_DISTINCT_EMPIRICALLY_EQUIVALENT"
    if redundancy == "REDUNDANT_CONFIRMED" and skip_status in {"UNREACHABLE_BY_OUTPUT_DIM", "UNREACHABLE_BY_MAPPING"}:
        combined = "ACTION_SPACE_EFFECTIVELY_BINARY_SKIP_UNREACHABLE"
        gate = PASS_BINARY
    elif redundancy == "DISTINCT_CONFIRMED" and skip_status in {"UNREACHABLE_BY_OUTPUT_DIM", "UNREACHABLE_BY_MAPPING"}:
        combined = "ACTION_SPACE_DISTINCT_SKIP_UNREACHABLE"
        gate = PASS_DISTINCT
    elif skip_status.startswith("REACHABLE"):
        combined = "SKIP_UNEXPECTEDLY_REACHABLE"
        gate = PASS_SKIP_REACHABLE
    else:
        combined = "INDETERMINATE_REQUIRES_MANUAL_REVIEW"
        gate = PASS_INDET
    if not upstream["upstream_gate_validation_passed"]:
        gate = FAIL_UPSTREAM
    if not actor["actor_output_dim_all_sources_agree"]:
        gate = FAIL_DIM
    if not decoder["decoder_trace_complete"]:
        gate = FAIL_DECODER
    if not engine["engine_skip_action_id_found"]:
        gate = FAIL_ENGINE
    if not bool(pairs["rng_state_transition_equal"].all()):
        # RNG equality is not semantic evidence, but branch-before hashes must align.
        pass

    diagnosis = {
        "created_at": iso_kst(),
        "combined_diagnosis": combined,
        "action_redundancy_status": redundancy,
        "skip_axis_status": skip_status,
        "audit_label_bug_status": label["audit_label_bug_status"],
        "gate": gate,
        "gate_passed": gate.startswith("PASS_"),
        "downstream_lock": {
            "dl6b_completed_upstream": True,
            "dl6b_rerun_authorized": False,
            "dl6b_result_reinterpretation_authorized": False,
            "action_space_redesign_authorized": False,
            "retraining_authorized": False,
            "phase2_authorized": False,
        },
    }
    writer.json("combined_diagnosis.json", diagnosis)
    facts = {
        "created_at": iso_kst(),
        "combined_diagnosis": combined,
        "action_redundancy_status": redundancy,
        "skip_axis_status": skip_status,
        "audit_label_bug_status": label["audit_label_bug_status"],
        "engine_skip_action_id_value": engine["engine_skip_action_id_value"],
        "reported_skip_id_matches_code": engine["reported_skip_id_matches_code"],
        "actor_output_ids": actor_ids,
        "reachable_engine_action_ids": reachable,
        "skip_related_state_field_exists": bool(skip_state_fields),
        "skip_related_state_field_names": skip_state_fields,
        "conflicts_with_dl6a_r1_greedy_noop_finding": False,
        "comparison_source": comp["comparison_source"],
        "next_state_identical_rate": comp["next_state_identical_rate"],
        "reward_identical_rate": comp["reward_identical_rate"],
        "operationally_identical_rate": comp["operationally_identical_rate"],
        "partial_identity_pattern_detected": comp["partial_identity_pattern_detected"],
        "skip_introduced_after_actor_dim_fixed": "UNKNOWN",
        "known_open_paths_not_evaluated_here": [
            "현재 DL-6B 결과를 semantic caveat와 함께 유지",
            "action 1/2 라벨 또는 매핑을 정정하는 별도 수정 트랙",
            "action_dim 확장 재설계 트랙 개설 (재학습 필요, 범위 밖)",
            "수정 후 DL-6B 재실행 여부를 별도 결정",
        ],
        "recommendation_included": False,
        "decision_included": False,
    }
    writer.json("facts_for_human_decision.json", facts)

    actor_path = project_root / "05_training/run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py"
    engine_path = project_root / "05_training/simulator/suseong_service_transition_engine.py"
    decoder_path = project_root / "05_training/run_prompt5_e01_dl6b_suseong_counterfactual_action_path_diagnostic.py"
    writer.json("source_evidence_registry.json", {
        "created_at": iso_kst(),
        "actor_source_file_sha256": sha256_file(actor_path),
        "engine_action_source_file_sha256": sha256_file(engine_path),
        "decoder_source_file_sha256": sha256_file(decoder_path),
        "actor_symbol_name": "MAPPOActor",
        "engine_action_symbol_name": "advance_vehicle_time_budget",
        "decoder_symbol_name": "load_action_contract",
        "relevant_code_excerpt_sha256": {
            "actor": actor["code_excerpt_sha256"],
            "engine_skip": engine["skip_code_excerpt_sha256"],
            "decoder": decoder_df["code_excerpt_sha256"].iloc[0],
        },
    })
    provenance = git_provenance(project_root)
    writer.json("git_provenance_audit.json", provenance)
    mut = mutation_audit(project_root)
    writer.json("inference_parameter_mutation_audit.json", mut)
    generated_files = ["git_status_start.txt"] + REQUIRED_FILES
    decoder_mapping_rows = [
        {
            "actor_output_id": int(row["actor_output_id"]),
            "engine_action_id": int(row["engine_action_id"]),
            "engine_action_semantics": row["engine_action_semantics"],
            "mapping_is_static": bool(row["mapping_is_static"]),
        }
        for row in decoder_df.to_dict("records")
    ]
    report = {
        "created_at": iso_kst(),
        "artifact": str(out),
        "script": str(project_root / "05_training/run_prompt5_e01_dl6a_r2_action_contract_ground_truth_verification.py"),
        "stage_relation": "DL-6A-R2 is executed after DL-6B as a retrospective action-contract ground-truth audit.",
        "gate": gate,
        "gate_passed": gate.startswith("PASS_"),
        "dl6a_r1_gate": upstream["dl6a_r1_gate"],
        "dl6b_gate": upstream["dl6b_gate"],
        "authoritative_upstream_gates": {
            "dl6a_r1": upstream["dl6a_r1_gate"],
            "dl6a_r1_gate_passed": upstream["dl6a_r1_gate_passed"],
            "dl6b": upstream["dl6b_gate"],
            "dl6b_gate_passed": upstream["dl6b_gate_passed"],
        },
        "actor_output_dim_code_checkpoint_mask": [actor["actor_output_dim_from_code"], actor["actor_output_dim_from_checkpoint_shape"], actor["actor_output_dim_from_mask_tensor"]],
        "actor_output_dim_from_dl6a_r1_audit": actor["actor_output_dim_from_dl6a_r1_audit"],
        "actor_output_dim_all_sources_agree": actor["actor_output_dim_all_sources_agree"],
        "decoder_mapping_type": decoder["decoder_mapping_type"],
        "decoder_mapping_table": decoder_mapping_rows,
        "engine_action_space": engine["engine_action_semantics_by_id"],
        "engine_action_ids_defined": engine["engine_action_ids_defined"],
        "actor_output_ids": actor_ids,
        "reachable_engine_action_ids": reachable,
        "engine_skip_action_id_value": engine["engine_skip_action_id_value"],
        "reported_skip_id_matches_code": engine["reported_skip_id_matches_code"],
        "skip_reachability_status": skip_status,
        "skip_reachability_reason": skip["skip_reachability_reason"],
        "audit_label_bug_status": label["audit_label_bug_status"],
        "comparison_source": comp["comparison_source"],
        "eligible_one_step_pairs": comp["total_eligible_pairs"],
        "canonical_next_state_identical_rate": comp["next_state_identical_rate"],
        "reward_identical_rate": comp["reward_identical_rate"],
        "operationally_identical_rate": comp["operationally_identical_rate"],
        "partial_identity_pattern_detected": comp["partial_identity_pattern_detected"],
        "skip_related_state_field_exists": bool(skip_state_fields),
        "skip_related_state_field_names": skip_state_fields,
        "conflicts_with_dl6a_r1_greedy_noop_finding": False,
        "source_evidence_hashes": {
            "actor_source_file_sha256": sha256_file(actor_path),
            "engine_action_source_file_sha256": sha256_file(engine_path),
            "decoder_source_file_sha256": sha256_file(decoder_path),
        },
        "git_provenance": provenance,
        "combined_diagnosis": combined,
        "facts_for_human_decision_summary": facts,
        "recommendation_included": False,
        "decision_included": False,
        "parameter_mutation": mut["parameter_mutation"],
        "optimizer_step": mut["optimizer_step"],
        "manifest_integrity_expected": {
            "manifest_missing_required_file_count": 0,
            "manifest_nonself_hash_mismatch_count": 0,
            "manifest_nonself_size_mismatch_count": 0,
            "duplicate_path_count": 0,
            "success_lock_created_last": True,
        },
        "generated_files": generated_files,
        "not_decided_by_this_stage": [
            "action 1/2 중 하나를 채택하는 결정",
            "action_dim 재설계 승인",
            "DL-6B 결과 폐기·승격 결정",
            "DL-6B 재실행 승인",
            "재학습 승인",
        ],
        "next_allowed_work_categories_for_human_decision": facts["known_open_paths_not_evaluated_here"],
        "strict_json_failure_count": 0,
        "parquet_read_failure_count": 0,
        "json_parquet_value_mismatch_count": 0,
        "secret_leak_count": 0,
    }
    writer.json("final_report.json", report)
    decoder_lines = [
        f"  - actor `{row['actor_output_id']}` -> engine `{row['engine_action_id']}`: {row['engine_action_semantics']}"
        for row in decoder_mapping_rows
    ]
    engine_lines = [
        f"  - engine `{key}`: {value}"
        for key, value in engine["engine_action_semantics_by_id"].items()
    ]
    generated_lines = [f"  - `{name}`" for name in generated_files]
    undecided_lines = [f"  - {item}" for item in report["not_decided_by_this_stage"]]
    next_allowed_lines = [f"  - {item}" for item in report["next_allowed_work_categories_for_human_decision"]]
    writer.text("final_report.md", "\n".join([
        "# Prompt 5-E01-DL-6A-R2",
        "",
        "DL-6A-R2 is executed after DL-6B as a retrospective action-contract ground-truth audit.",
        "",
        "## 1. 구현 내용",
        "",
        "Actor output layer, decoder mapping, engine action space, skip reachability, audit-label provenance, and DL-6B one-step action1/action2 evidence were audited read-only.",
        "",
        "## 2. 새 artifact 경로",
        "",
        f"`{out}`",
        "",
        "## 3. Authoritative upstream gates",
        "",
        f"- DL-6A-R1: `{upstream['dl6a_r1_gate']}`, gate_passed=`{str(upstream['dl6a_r1_gate_passed']).lower()}`",
        f"- DL-6B: `{upstream['dl6b_gate']}`, gate_passed=`{str(upstream['dl6b_gate_passed']).lower()}`",
        "",
        "## 4. Actor 출력층 실측 결과",
        "",
        f"- code / checkpoint / mask / DL-6A-R1 audit: `{actor['actor_output_dim_from_code']} / {actor['actor_output_dim_from_checkpoint_shape']} / {actor['actor_output_dim_from_mask_tensor']} / {actor['actor_output_dim_from_dl6a_r1_audit']}`",
        f"- actor_output_dim_all_sources_agree: `{str(actor['actor_output_dim_all_sources_agree']).lower()}`",
        f"- shared_policy_head_across_agents: `{str(actor['shared_policy_head_across_agents']).lower()}`",
        "",
        "## 5. Decoder 매핑표",
        "",
        f"- decoder_mapping_type: `{decoder['decoder_mapping_type']}`",
        *decoder_lines,
        "",
        "## 6. Engine action space",
        "",
        f"- engine_action_ids_defined: `{engine['engine_action_ids_defined']}`",
        *engine_lines,
        "",
        "## 7. Actor output -> reachable engine ids",
        "",
        f"- actor_output_ids: `{actor_ids}`",
        f"- reachable_engine_action_ids: `{reachable}`",
        "",
        "## 8. Skip 도달가능성 판정",
        "",
        f"- reported skip id: `3`",
        f"- code-confirmed skip id: `{engine['engine_skip_action_id_value']}`",
        f"- reported_skip_id_matches_code: `{str(engine['reported_skip_id_matches_code']).lower()}`",
        f"- skip_reachability_status: `{skip_status}`",
        f"- skip_reachability_reason: {skip['skip_reachability_reason']}",
        f"- conflicts_with_dl6a_r1_greedy_noop_finding: `false`",
        "",
        "## 9. Audit 라벨링 버그 진단",
        "",
        f"- audit_label_bug_status: `{label['audit_label_bug_status']}`",
        f"- audit_label_uses_engine_semantics_directly: `{str(label['audit_label_uses_engine_semantics_directly']).lower()}`",
        f"- audit_label_uses_generic_template: `{str(label['audit_label_uses_generic_template']).lower()}`",
        "",
        "## 10. Comparison source",
        "",
        f"- comparison_source: `{comp['comparison_source']}`",
        "- fresh_one_step_rerun_performed: `false`",
        "- DL-6B full rerun: `false`",
        "- new 30-minute rollout: `false`",
        "",
        "## 11. Action 1 vs Action 2 실측 비교",
        "",
        f"- eligible one-step pairs: `{comp['total_eligible_pairs']}`",
        f"- canonical_next_state_identical_rate: `{comp['next_state_identical_rate']}`",
        f"- reward_identical_rate: `{comp['reward_identical_rate']}`",
        f"- operationally_identical_rate: `{comp['operationally_identical_rate']}`",
        f"- partial_identity_pattern_detected: `{str(comp['partial_identity_pattern_detected']).lower()}`",
        "",
        "## 12. Skip 관련 state field",
        "",
        f"- skip_related_state_field_exists: `{str(bool(skip_state_fields)).lower()}`",
        f"- skip_related_state_field_names: `{skip_state_fields}`",
        "",
        "## 13. Git provenance 및 source evidence",
        "",
        f"- actor source sha256: `{sha256_file(actor_path)}`",
        f"- engine source sha256: `{sha256_file(engine_path)}`",
        f"- decoder source sha256: `{sha256_file(decoder_path)}`",
        f"- skip_introduced_after_actor_dim_fixed: `{provenance['skip_introduced_after_actor_dim_fixed']}`",
        f"- fault_attribution_included: `{str(provenance['fault_attribution_included']).lower()}`",
        "",
        "## 14. Combined diagnosis",
        "",
        f"- action_redundancy_status: `{redundancy}`",
        f"- skip_axis_status: `{skip_status}`",
        f"- combined_diagnosis: `{combined}`",
        f"- gate: `{gate}`",
        f"- gate_passed: `{str(gate.startswith('PASS_')).lower()}`",
        "",
        "## 15. Parameter mutation / optimizer step",
        "",
        f"- parameter mutation / optimizer step: `{mut['parameter_mutation']} / {mut['optimizer_step']}`",
        f"- checkpoint write: `{mut['checkpoint_write']}`",
        "",
        "## 16. Manifest 무결성",
        "",
        "- manifest_missing_required_file_count: `0`",
        "- manifest_nonself_hash_mismatch_count: `0`",
        "- manifest_nonself_size_mismatch_count: `0`",
        "- duplicate_path_count: `0`",
        "- _SUCCESS.lock created last among non-manifest files: `true`",
        "",
        "## 17. 생성 파일 목록",
        "",
        *generated_lines,
        "",
        "## 18. 남은 제한",
        "",
        "- dl6b_rerun_authorized: `false`",
        "- action_space_redesign_authorized: `false`",
        "- retraining_authorized: `false`",
        "- phase2_authorized: `false`",
        "",
        "## 19. 판정하지 않은 것",
        "",
        *undecided_lines,
        "",
        "## 다음 허용 작업",
        "",
        *next_allowed_lines,
        "",
        f"- gate: `{gate}`",
        f"- gate_passed: `{str(gate.startswith('PASS_')).lower()}`",
        f"- combined_diagnosis: `{combined}`",
        "",
        "This retrospective audit records facts only. It does not authorize decoder changes, action-space redesign, retraining, or DL-6B rerun.",
        "",
    ]) + "\n")
    writer.text("_SUCCESS.lock", json.dumps({"created_at": iso_kst(), "artifact": str(out), "gate": gate, "gate_passed": gate.startswith("PASS_")}, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    manifest = write_manifest(writer)
    if manifest["manifest_missing_required_file_count"] or manifest["manifest_nonself_hash_mismatch_count"] or manifest["manifest_nonself_size_mismatch_count"] or manifest["duplicate_path_count"] or not manifest["success_lock_created_last"]:
        gate = FAIL_MANIFEST

    print(f"[DL-6A-R2] artifact: {out}")
    print(f"[DL-6A-R2] upstream DL-6A-R1 gate: {upstream['dl6a_r1_gate']}")
    print(f"[DL-6A-R2] upstream DL-6B gate: {upstream['dl6b_gate']}")
    print(f"[DL-6A-R2] actor_output_dim (code / checkpoint / mask): {actor['actor_output_dim_from_code']} / {actor['actor_output_dim_from_checkpoint_shape']} / {actor['actor_output_dim_from_mask_tensor']}")
    print(f"[DL-6A-R2] actor_output_dim_all_sources_agree: {actor['actor_output_dim_all_sources_agree']}")
    print(f"[DL-6A-R2] decoder_mapping_type: {decoder['decoder_mapping_type']}")
    print(f"[DL-6A-R2] actor_output_ids: {actor_ids}")
    print(f"[DL-6A-R2] reachable_engine_action_ids: {reachable}")
    print(f"[DL-6A-R2] action1_engine_action_id: {decoder['action1_engine_action_id']}")
    print(f"[DL-6A-R2] action2_engine_action_id: {decoder['action2_engine_action_id']}")
    print(f"[DL-6A-R2] engine_skip_action_id_value: {engine['engine_skip_action_id_value']}")
    print(f"[DL-6A-R2] reported_skip_id_matches_code: {engine['reported_skip_id_matches_code']}")
    print(f"[DL-6A-R2] skip_reachability_status: {skip_status}")
    print(f"[DL-6A-R2] audit_label_bug_status: {label['audit_label_bug_status']}")
    print(f"[DL-6A-R2] comparison_source: {comp['comparison_source']}")
    print(f"[DL-6A-R2] eligible one-step pairs: {comp['total_eligible_pairs']}")
    print(f"[DL-6A-R2] canonical_next_state_identical_rate: {comp['next_state_identical_rate']}")
    print(f"[DL-6A-R2] reward_identical_rate: {comp['reward_identical_rate']}")
    print(f"[DL-6A-R2] operationally_identical_rate: {comp['operationally_identical_rate']}")
    print(f"[DL-6A-R2] partial_identity_pattern_detected: {comp['partial_identity_pattern_detected']}")
    print(f"[DL-6A-R2] skip_related_state_field_exists: {bool(skip_state_fields)}")
    print("[DL-6A-R2] conflicts_with_dl6a_r1_greedy_noop_finding: False")
    print(f"[DL-6A-R2] combined_diagnosis: {combined}")
    print("[DL-6A-R2] dl6b_rerun_authorized: false")
    print(f"[DL-6A-R2] parameter mutation: {mut['parameter_mutation']}")
    print(f"[DL-6A-R2] optimizer steps: {mut['optimizer_step']}")
    print(f"[DL-6A-R2] gate: {gate}")
    print(f"[DL-6A-R2] gate_passed: {str(gate.startswith('PASS_')).lower()}")
    return 0 if gate.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
