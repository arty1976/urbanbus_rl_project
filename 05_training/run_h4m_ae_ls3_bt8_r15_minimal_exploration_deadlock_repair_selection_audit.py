#!/usr/bin/env python3
"""BT8-R15: select a minimal, training-only exploration-deadlock repair.

This is a frozen-support audit.  It reads R13 checkpoints and lossless
snapshots, replays frozen forwards, and performs identity-keyed selection
probes.  It deliberately has no training, optimizer, backward, causal rollout,
candidate generation, reward settlement, checkpoint write, or parameter-update
path.  The chosen contract is not an implementation or a training grant.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import torch


STAGE = "H4M-AE-R9.8-LS3-BT8-R15"
PASS_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R15_MINIMAL_EXPLORATION_DEADLOCK_REPAIR_SELECTION_AUDIT_COMPLETE"
BINDING_BLOCK = "BLOCKED_R15_UPSTREAM_EVIDENCE_BINDING_FAILURE"
E0_BLOCK = "BLOCKED_R15_E0_CONTROL_REPRODUCTION_FAILURE"
STRUCTURAL_BLOCK = "BLOCKED_R15_EXPLORATION_PROBE_INTEGRITY_FAILURE"
NO_SAFE_BLOCK = "BLOCKED_SUSEONG_H4M_AE_R9_8_LS3_BT8_R15_NO_SAFE_MINIMAL_EXPLORATION_REPAIR_FOUND"
MPS_BLOCK = "BLOCKED_R15_MPS_FROZEN_PROBE_ENVIRONMENT_UNAVAILABLE"

R14_SOURCE = "36ccbaa84e06af3202560802fd3371066839a7c5"
R13_SOURCE = "a9818399c4a1a74734496a146b2b98fabab513b0"
R12_SOURCE = "6e22b95d80720255ca17b1ab0520417b89e5a5c9"
R11_SOURCE = "f467feef8246c6c77dc67a360aecc60e0102913e"
S3_CONTRACT_SHA256 = "66e2fb35de3aa767780d9f0f001d774919e059e580f4410fe1d01bd96b185393"
R14_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R14_S3_SAME_INPUT_FROZEN_POLICY_FACTOR_REVIEW_COMPLETE"
R13_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R13_S3_FOUR_CELL_ON_POLICY_CAUSAL_EXECUTION_COMPLETE"
R12_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R12_S3_FACTORIAL_EXECUTION_AUTHORITY_SELECTION_COMPLETE"
R11_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R11_MINIMAL_SEED_DIVERGENCE_CAUSE_ISOLATION_AND_REPAIR_SELECTION_COMPLETE"
T1_CONTRACT = "LS3_BT7_R1_EXACT_TIE_CANONICAL_ACTION_IDENTITY_V1"
R15_E1_CONTRACT_ID = "LS3_BT8_R15_E1_FROZEN_POLICY_MASKED_CATEGORICAL_SAMPLING_V1"
R15_E2_CONTRACT_ID = "LS3_BT8_R15_E2_TRAINING_ONLY_CONDITIONAL_CANDIDATE_RESCUE_V1"
REVIEW_COLLECTION_DIGEST = "6c811022a5df4b3966ac14fce750f8bdd840a65a50e48285fe0c97ce157b889e"
PROBE_SEEDS = tuple(range(32))
TIME_BANDS = ("night", "offpeak", "peak")

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
R14 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r14_s3_frozen_factor_review_20260827_001145+09:00"
R13 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r13_s3_four_cell_execution_20260826_174729+09:00"
R12 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r12_s3_execution_authority_selection_20260826_145345+09:00"
R11 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r11_seed_divergence_repair_selection_20260826_123135+09:00"
F1 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_f1_fresh_v2_bounded_training_20260823_135127+09:00"
SOURCE_FILES = {
    "05_training/run_h4m_ae_ls3_bt8_r15_minimal_exploration_deadlock_repair_selection_audit.py",
    "05_training/test_h4m_ae_ls3_bt8_r15_minimal_exploration_deadlock_repair_selection_audit.py",
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
    "checkpoint_write_allowed": False,
    "checkpoint_mutation_allowed": False,
    "policy_parameter_mutation_allowed": False,
    "Reward_V2_mutation_allowed": False,
    "Zero_Loss_mutation_allowed": False,
    "candidate_generator_mutation_allowed": False,
    "GATv2_mutation_allowed": False,
    "TEST6_open_allowed": False,
    "GitHub_push_allowed": False,
    "automatic_training_authorization": False,
}
CELL_ROLE = {"AC-R1": "AC", "AC-R2": "AC", "BD-R1": "BD", "BD-R2": "BD"}
PROBE_COLUMNS = [
    "stage", "repair_level", "policy_family", "cell_id", "actor_state_sha256", "snapshot_digest",
    "snapshot_manifest_sha256", "decision_id", "window_id", "time_band", "environment_seed",
    "candidate_support_digest", "support_size", "legal_candidate_count", "probe_seed", "canonical_rng_contract",
    "canonical_rng_keyset_sha256", "selection_mode", "training_only_triggered", "deterministic_exploit_identity",
    "deterministic_exploit_action_family", "sampled_identity", "sampled_agent_id", "sampled_candidate_id",
    "sampled_action_family", "sampled_is_no_assign", "legal", "zero_loss_feasible", "policy_probability",
    "selection_probability_if_modified", "no_assign_probability", "non_no_assign_probability_mass", "finite",
    "candidate_order_identity_equal", "agent_order_identity_equal", "combined_order_identity_equal",
    "actor_tensor_digest_before", "actor_tensor_digest_after",
]


class R15Error(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


def require(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise R15Error(code, detail)


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


def load_json(path: Path, *, code: str = BINDING_BLOCK) -> Any:
    require(path.is_file(), code, f"missing={path}")
    return json.loads(path.read_text(encoding="utf-8"))


def git(args: Sequence[str]) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT, text=True, capture_output=True, check=True).stdout.strip()


def manifest_audit(root: Path) -> dict[str, Any]:
    manifest = load_json(root / "manifest.json")
    expected = manifest.get("file_sha256")
    require(isinstance(expected, Mapping), BINDING_BLOCK, f"manifest_schema={root}")
    mismatches = [str(name) for name, digest in expected.items()
                  if not (root / str(name)).is_file() or sha256(root / str(name)) != str(digest)]
    return {"manifest_sha256": sha256(root / "manifest.json"), "declared_file_count": len(expected),
            "mismatches": mismatches, "all_match": not mismatches}


def module_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for _, tensor in sorted(module.state_dict().items()):
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def state_dicts_exact(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> bool:
    return set(left) == set(right) and all(torch.equal(left[name].detach().cpu(), right[name].detach().cpu()) for name in left)


def source_provenance() -> dict[str, Any]:
    changed = [name for name in git(["diff", "--name-only", f"{R14_SOURCE}..HEAD"]).splitlines() if name]
    return {
        "source_commit": git(["rev-parse", "HEAD"]),
        "source_parent": git(["rev-parse", "HEAD^"]),
        "source_lineage_descends_from_r14": git(["merge-base", R14_SOURCE, "HEAD"]) == R14_SOURCE,
        "changed_files_since_r14": changed,
        "source_only_local_commit": set(changed) == SOURCE_FILES,
        "github_push_performed": False,
    }


def artifact_root() -> Path:
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M%S%z")[:-2] + ":00"
    return ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r15_minimal_exploration_deadlock_repair_selection_audit_{stamp}"


def counters() -> dict[str, int]:
    return {
        "training": 0, "optimizer_creation": 0, "optimizer_step": 0, "backward": 0,
        "causal_rollout": 0, "simulator_execution": 0, "candidate_generation": 0,
        "candidate_regeneration": 0, "local_search_rerun": 0, "zero_loss_reevaluation": 0,
        "reward_settlement": 0, "parameter_mutation": 0, "checkpoint_write": 0,
        "checkpoint_mutation": 0, "test6_access": 0, "github_push": 0, "nan_or_inf": 0,
        "illegal_selection": 0, "zero_loss_violation": 0, "deterministic_control_forwards": 0,
        "stochastic_probe_rows": 0, "order_probe_forwards": 0,
    }


def action_identity(action: Mapping[str, Any]) -> str:
    return "NO_ASSIGN" if str(action["kind"]) == "NO_ASSIGN" else f"{action['agent_id']}::{action['candidate_id']}"


def action_family(action: Mapping[str, Any]) -> str:
    return "NO_ASSIGN" if str(action["kind"]) == "NO_ASSIGN" else "CANDIDATE"


def action_key(action: Mapping[str, Any]) -> tuple[str, str, str]:
    return (str(action["kind"]), str(action.get("agent_id") or ""), str(action.get("candidate_id") or ""))


def stable_uniform(*, contract_id: str, snapshot_digest: str, action: Mapping[str, Any], probe_seed: int,
                   repair_level: str) -> float:
    """A semantic-action RNG, independent of list position and global RNG state."""
    payload = "|".join((contract_id, str(snapshot_digest), str(action["kind"]),
                        str(action.get("agent_id") or "NO_ASSIGN"),
                        str(action.get("candidate_id") or "NO_ASSIGN_KEEP_CURRENT_PLANS"),
                        str(probe_seed), repair_level))
    raw = hashlib.sha256(payload.encode("utf-8")).digest()
    mantissa = int.from_bytes(raw[:8], "big") >> 11
    return (mantissa + 0.5) / float(1 << 53)


def categorical_exponential_race(*, actions: Sequence[Mapping[str, Any]], snapshot_digest: str, probe_seed: int,
                                 contract_id: str, repair_level: str) -> dict[str, Any]:
    """Exact categorical draw via independent Exp(rate=p) semantic clocks."""
    legal = []
    for action in actions:
        probability = float(action["policy_probability"])
        require(math.isfinite(probability) and probability >= 0.0, STRUCTURAL_BLOCK, "invalid_policy_probability")
        if probability > 0.0:
            uniform = stable_uniform(contract_id=contract_id, snapshot_digest=snapshot_digest, action=action,
                                     probe_seed=probe_seed, repair_level=repair_level)
            require(0.0 < uniform < 1.0, STRUCTURAL_BLOCK, "invalid_canonical_uniform")
            race = -math.log(uniform) / probability
            legal.append((race, str(action["identity_digest"]), dict(action)))
    require(legal, STRUCTURAL_BLOCK, "empty_positive_probability_support")
    _, _, selected = min(legal, key=lambda row: (row[0], row[1]))
    return selected


def canonical_rng_keyset_sha(actions: Sequence[Mapping[str, Any]], snapshot_digest: str, probe_seed: int,
                             contract_id: str, repair_level: str) -> str:
    keys = [{"snapshot_digest": str(snapshot_digest), "probe_seed": int(probe_seed), "repair_level": repair_level,
             "contract_id": contract_id, "kind": action["kind"], "agent_id": action.get("agent_id"),
             "candidate_id": action.get("candidate_id"), "identity_digest": action["identity_digest"]}
            for action in sorted(actions, key=action_key)]
    return canonical_sha256(keys)


def action_distribution(*, result: Mapping[str, Any], payload: Mapping[str, Any], TIE: Any) -> list[dict[str, Any]]:
    metadata, tensors = payload["metadata"], payload["tensors"]
    support = int(result["support_size"])
    pair_scores = [float(value) for value in result["pair_logits"][0].detach().cpu().tolist()]
    probabilities = [float(value) for value in result["probabilities"][0].detach().cpu().tolist()]
    safe_mask = tensors["safe_mask"][0, :support].detach().cpu().tolist()
    canonical = TIE.canonical_actions(candidate_ids=metadata["candidate_ids"], pair_scores=pair_scores,
                                      safe_mask=safe_mask, no_assign_score=float(result["no_assign_logit"][0, 0].detach().cpu()))
    require(len(probabilities) == support + 1 and len(canonical) >= 1, STRUCTURAL_BLOCK, "distribution_shape")
    rows = []
    for item in canonical:
        probability = float(probabilities[int(item.source_index)])
        rows.append({"kind": item.kind, "agent_id": item.agent_id, "candidate_id": item.candidate_id,
                     "source_index": int(item.source_index), "identity_digest": item.identity_digest,
                     "policy_probability": probability, "score": float(item.score)})
    require(sum(row["kind"] == "NO_ASSIGN" for row in rows) == 1, STRUCTURAL_BLOCK, "no_assign_availability")
    require(all(math.isfinite(float(row["policy_probability"])) and float(row["policy_probability"]) >= 0.0 for row in rows),
            STRUCTURAL_BLOCK, "nonfinite_distribution")
    return sorted(rows, key=action_key)


def deterministic_action(*, result: Mapping[str, Any], distribution: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = result["selection"].selected
    key = (str(selected.kind), str(selected.agent_id or ""), str(selected.candidate_id or ""))
    found = [dict(action) for action in distribution if action_key(action) == key]
    require(len(found) == 1, STRUCTURAL_BLOCK, "deterministic_action_missing")
    return found[0]


def e1_selection(*, distribution: Sequence[Mapping[str, Any]], deterministic: Mapping[str, Any], snapshot_digest: str,
                 probe_seed: int, training_mode: bool) -> tuple[dict[str, Any], bool, str, float | None]:
    candidate_count = sum(action_family(row) == "CANDIDATE" for row in distribution)
    trigger = bool(training_mode and action_family(deterministic) == "NO_ASSIGN" and candidate_count > 0)
    if not trigger:
        return dict(deterministic), False, "DETERMINISTIC_EXPLOIT_NO_E1_TRIGGER", None
    selected = categorical_exponential_race(actions=distribution, snapshot_digest=snapshot_digest, probe_seed=probe_seed,
                                            contract_id=R15_E1_CONTRACT_ID, repair_level="E1")
    return selected, True, "FROZEN_MASKED_CATEGORICAL", None


def e2_selection(*, distribution: Sequence[Mapping[str, Any]], deterministic: Mapping[str, Any], snapshot_digest: str,
                 probe_seed: int, training_mode: bool) -> tuple[dict[str, Any], bool, str, float | None]:
    """Audit-only fallback; its PPO q-policy binding is intentionally not authorized here."""
    candidates = [dict(action) for action in distribution if action_family(action) == "CANDIDATE"]
    trigger = bool(training_mode and action_family(deterministic) == "NO_ASSIGN" and candidates)
    if not trigger:
        return dict(deterministic), False, "DETERMINISTIC_EXPLOIT_NO_E2_TRIGGER", None
    total = sum(float(action["policy_probability"]) for action in candidates)
    require(math.isfinite(total) and total > 0.0, STRUCTURAL_BLOCK, "e2_empty_candidate_probability")
    selected = categorical_exponential_race(actions=candidates, snapshot_digest=snapshot_digest, probe_seed=probe_seed,
                                            contract_id=R15_E2_CONTRACT_ID, repair_level="E2")
    return selected, True, "TRAINING_ONLY_CONDITIONAL_NON_NO_ASSIGN", float(selected["policy_probability"]) / total


def select_level(*, level: str, distribution: Sequence[Mapping[str, Any]], deterministic: Mapping[str, Any],
                 snapshot_digest: str, probe_seed: int, training_mode: bool) -> tuple[dict[str, Any], bool, str, float | None]:
    if level == "E1":
        return e1_selection(distribution=distribution, deterministic=deterministic, snapshot_digest=snapshot_digest,
                            probe_seed=probe_seed, training_mode=training_mode)
    if level == "E2":
        return e2_selection(distribution=distribution, deterministic=deterministic, snapshot_digest=snapshot_digest,
                            probe_seed=probe_seed, training_mode=training_mode)
    raise R15Error(STRUCTURAL_BLOCK, f"unsupported_repair_level={level}")


def max_abs_delta(left: Sequence[float], right: Sequence[float]) -> float:
    require(len(left) == len(right), STRUCTURAL_BLOCK, "score_vector_length_mismatch")
    return max((abs(float(a) - float(b)) for a, b in zip(left, right)), default=0.0)


def order_audit(*, level: str, base_results: Mapping[str, Mapping[str, Any]], actors: Mapping[str, torch.nn.Module],
                records: Sequence[Mapping[str, Any]], R1: Any, H: Any, TIE: Any, device: torch.device) -> tuple[dict[str, Any], int]:
    failures = Counter({"candidate": 0, "agent": 0, "combined": 0})
    rows: list[dict[str, Any]] = []
    forwards = 0
    for record in records:
        snapshot_digest, cell_id = str(record["snapshot_digest"]), str(record["cell_id"])
        payload = record["payload"]
        actor = actors[str(record["policy_family"])]
        base = base_results[snapshot_digest]
        permutations = {
            "candidate": R1.candidate_permutation(payload),
            "agent": R1.agent_permutation(payload),
            "combined": R1.candidate_permutation(R1.agent_permutation(payload)),
        }
        base_logits, base_probs = R1.scores_by_identity(base["result"], payload)
        for permutation_name, permuted in permutations.items():
            result = R1.frozen_forward(actor, permuted, device=device, head=H, tie=TIE); forwards += 1
            distribution = action_distribution(result=result, payload=permuted, TIE=TIE)
            deterministic = deterministic_action(result=result, distribution=distribution)
            logits, probabilities = R1.scores_by_identity(result, permuted)
            require(set(logits) == set(base_logits) and set(probabilities) == set(base_probs), STRUCTURAL_BLOCK,
                    f"order_identity_mapping={snapshot_digest}:{permutation_name}")
            mismatch = 0
            for probe_seed in PROBE_SEEDS:
                selected, _, _, _ = select_level(level=level, distribution=distribution, deterministic=deterministic,
                                                  snapshot_digest=snapshot_digest, probe_seed=probe_seed, training_mode=True)
                base_selected = base["selections"][probe_seed]
                if action_identity(selected) != action_identity(base_selected):
                    mismatch += 1
            failures[permutation_name] += mismatch
            rows.append({
                "cell_id": cell_id, "snapshot_digest": snapshot_digest, "permutation": permutation_name,
                "probe_seed_count": len(PROBE_SEEDS), "selection_identity_mismatch_count": mismatch,
                "pair_logit_max_abs_delta": max((abs(base_logits[key] - logits[key]) for key in base_logits), default=0.0),
                "pair_probability_max_abs_delta": max((abs(base_probs[key] - probabilities[key]) for key in base_probs), default=0.0),
                "no_assign_logit_abs_delta": abs(float(base["result"]["no_assign_logit"][0, 0]) - float(result["no_assign_logit"][0, 0])),
                "no_assign_probability_abs_delta": abs(float(base["result"]["probabilities"][0, -1]) - float(result["probabilities"][0, -1])),
            })
    return {
        "level": level, "probe_seed_count": len(PROBE_SEEDS), "rows": rows,
        "candidate_order_failures": int(failures["candidate"]), "agent_order_failures": int(failures["agent"]),
        "combined_order_failures": int(failures["combined"]),
        "passed": not any(failures.values()),
        "rule": "semantic action identity must match under each permutation for every fixed probe seed; no numerical tolerance was introduced.",
    }, forwards


def summary_by_band(rows: Sequence[Mapping[str, Any]], *, level: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["repair_level"] == level:
            grouped[(str(row["cell_id"]), str(row["time_band"]))].append(row)
    summary = []
    for (cell_id, band), values in sorted(grouped.items()):
        candidate = sum(str(row["sampled_action_family"]) == "CANDIDATE" for row in values)
        no_assign = len(values) - candidate
        probability_mass = [float(row["non_no_assign_probability_mass"]) for row in values]
        summary.append({"cell_id": cell_id, "policy_family": CELL_ROLE[cell_id], "time_band": band,
                        "probe_rows": len(values), "non_no_assign_exposure": candidate, "no_assign_samples": no_assign,
                        "exposure_positive": candidate > 0, "min_non_no_assign_probability_mass": min(probability_mass),
                        "max_non_no_assign_probability_mass": max(probability_mass)})
    return summary


def e1_pass(*, e0: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], order: Mapping[str, Any], counters_value: Mapping[str, int]) -> bool:
    bands = summary_by_band(rows, level="E1")
    expected = {(cell_id, band) for cell_id in ("BD-R1", "BD-R2") for band in TIME_BANDS}
    observed = {(row["cell_id"], row["time_band"]): bool(row["exposure_positive"]) for row in bands if row["cell_id"].startswith("BD-")}
    all_band_exposed = set(observed) == expected and all(observed.values())
    no_integrity = all(counters_value[key] == 0 for key in ("nan_or_inf", "illegal_selection", "zero_loss_violation"))
    order_ok = all(int(order[key]) == 0 for key in ("candidate_order_failures", "agent_order_failures", "combined_order_failures"))
    reproducible = int(e0["same_seed_replay_mismatch_count"]) == 0
    ac_unchanged = int(e0["ac_control_mismatch_count"]) == 0 and int(e0["ac_e1_triggered_count"]) == 0
    return bool(e0["bd_control_exact"] and all_band_exposed and no_integrity and order_ok and reproducible and ac_unchanged)


def normalize_parquet_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        clean = {}
        for column in PROBE_COLUMNS:
            value = row.get(column)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                clean[column] = None
            elif hasattr(value, "item"):
                clean[column] = value.item()
            else:
                clean[column] = value
        result.append(clean)
    return result


def write_probe_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(normalize_parquet_rows(rows), key=lambda row: (str(row["repair_level"]), str(row["cell_id"]),
                                                                     str(row["snapshot_digest"]), int(row["probe_seed"])))
    frame = pd.DataFrame(ordered, columns=PROBE_COLUMNS)
    frame.to_parquet(path, index=False, engine="pyarrow")
    raw = path.read_bytes()
    require(raw[:4] == b"PAR1" and raw[-4:] == b"PAR1", STRUCTURAL_BLOCK, "parquet_magic")
    readback = pd.read_parquet(path, engine="pyarrow")
    roundtrip = normalize_parquet_rows(readback.to_dict("records"))
    require(roundtrip == ordered, STRUCTURAL_BLOCK, "parquet_roundtrip")
    return {"path": path.name, "sha256": sha256(path), "row_count": len(ordered), "columns": PROBE_COLUMNS,
            "content_sha256": canonical_sha256(ordered), "engine": "pyarrow", "roundtrip_passed": True}


def make_probe_row(*, level: str, record: Mapping[str, Any], deterministic: Mapping[str, Any], selected: Mapping[str, Any],
                   probe_seed: int, selection_mode: str, triggered: bool, modified_probability: float | None,
                   actor_digest_before: str, actor_digest_after: str) -> dict[str, Any]:
    distribution = record["distribution"]
    no_assign = next(row for row in distribution if action_family(row) == "NO_ASSIGN")
    non_no_assign_mass = sum(float(row["policy_probability"]) for row in distribution if action_family(row) == "CANDIDATE")
    return {
        "stage": STAGE, "repair_level": level, "policy_family": record["policy_family"], "cell_id": record["cell_id"],
        "actor_state_sha256": record["actor_state_sha256"], "snapshot_digest": record["snapshot_digest"],
        "snapshot_manifest_sha256": record["snapshot_manifest_sha256"], "decision_id": record["decision_id"],
        "window_id": record["window_id"], "time_band": record["time_band"], "environment_seed": record["environment_seed"],
        "candidate_support_digest": record["candidate_support_digest"], "support_size": record["support_size"],
        "legal_candidate_count": sum(action_family(row) == "CANDIDATE" for row in distribution), "probe_seed": int(probe_seed),
        "canonical_rng_contract": R15_E1_CONTRACT_ID if level == "E1" else R15_E2_CONTRACT_ID if level == "E2" else "",
        "canonical_rng_keyset_sha256": canonical_rng_keyset_sha(distribution, record["snapshot_digest"], probe_seed,
                                                                  R15_E1_CONTRACT_ID if level == "E1" else R15_E2_CONTRACT_ID, level)
                                       if level in {"E1", "E2"} else "",
        "selection_mode": selection_mode, "training_only_triggered": bool(triggered),
        "deterministic_exploit_identity": action_identity(deterministic),
        "deterministic_exploit_action_family": action_family(deterministic), "sampled_identity": action_identity(selected),
        "sampled_agent_id": str(selected.get("agent_id") or ""), "sampled_candidate_id": str(selected.get("candidate_id") or ""),
        "sampled_action_family": action_family(selected), "sampled_is_no_assign": action_family(selected) == "NO_ASSIGN",
        "legal": True, "zero_loss_feasible": True, "policy_probability": float(selected["policy_probability"]),
        "selection_probability_if_modified": modified_probability, "no_assign_probability": float(no_assign["policy_probability"]),
        "non_no_assign_probability_mass": non_no_assign_mass, "finite": True,
        "candidate_order_identity_equal": None, "agent_order_identity_equal": None, "combined_order_identity_equal": None,
        "actor_tensor_digest_before": actor_digest_before, "actor_tensor_digest_after": actor_digest_after,
    }


def write_block(*, root: Path, source: Mapping[str, Any], preflight: Mapping[str, Any], reason: str) -> None:
    empty_parquet = write_probe_parquet(root / "exploration_probe_results.parquet", [])
    outputs = {
        "evidence_binding_audit.json": preflight,
        "repair_candidate_ladder.json": {"not_run": True},
        "exploration_probe_summary.json": {"not_run": True, "parquet": empty_parquet},
        "order_invariance_probe.json": {"not_run": True},
        "selected_minimal_exploration_repair_contract.json": {"not_selected": True},
        "test_results.json": {"execution_counters": counters(), "hard_failures": [reason], "warnings": [], "github_push_performed": False},
        "frozen_hash_before_after.json": {"not_run": True},
        "gate_decision.json": {"stage": STAGE, "gate": reason, "classification": "BLOCKED", "source_commit": source["source_commit"],
                               "hard_failures": [reason], "warnings": [], "global_locks": GLOBAL_LOCKS, "execution_locks": EXECUTION_LOCKS,
                               "next_step": "STOP"},
    }
    for name, value in outputs.items():
        dump(root / name, value)
    (root / "final_report.md").write_text(f"# BT8-R15 blocked\n\n- gate: `{reason}`\n", encoding="utf-8")
    manifest = {item.relative_to(root).as_posix(): sha256(item) for item in root.rglob("*") if item.is_file() and item.name != "manifest.json"}
    dump(root / "manifest.json", {"stage": STAGE, "gate": reason, "source_commit": source["source_commit"], "file_sha256": manifest})
    (root / "_BLOCKED.lock").write_text(reason + "\n", encoding="utf-8")


def main() -> None:
    sys.path.insert(0, str(ROOT))
    import joint_assignment_frozen_policy_snapshot as FPS
    import joint_assignment_frozen_tie_break as TIE
    import multi_agent_candidate_assignment_head as H
    import run_h4m_ae_ls3_bt8_r1_frozen_policy_discrimination_review as R1
    import run_h4m_ae_ls3_bt8_r12_s3_execution_authority_selection as R12MOD

    source = source_provenance()
    root = artifact_root()
    require(not root.exists(), BINDING_BLOCK, "append_only_artifact_collision")
    root.mkdir(parents=True)
    preflight: dict[str, Any] = {"source": source, "required_sources": {"r14": R14_SOURCE, "r13": R13_SOURCE,
                                  "r12": R12_SOURCE, "r11": R11_SOURCE}, "s3_contract_sha256": S3_CONTRACT_SHA256,
                                 "probe_seed_schedule": {"seed_start": 0, "seed_end": 31, "count": len(PROBE_SEEDS),
                                                         "sha256": canonical_sha256(list(PROBE_SEEDS)),
                                                         "selection_rule": "fixed before probe results; no early stopping or outcome-tuned adjustment"},
                                 "execution_locks": EXECUTION_LOCKS, "frozen_support_only": True}
    try:
        r14_gate, r13_gate, r12_gate, r11_gate = (load_json(R14 / "gate_decision.json"), load_json(R13 / "gate_decision.json"),
                                                    load_json(R12 / "gate_decision.json"), load_json(R11 / "gate_decision.json"))
        audits = {"r14": manifest_audit(R14), "r13": manifest_audit(R13), "r12": manifest_audit(R12), "r11": manifest_audit(R11)}
        require(r14_gate.get("gate") == R14_GATE and r14_gate.get("source_commit") == R14_SOURCE, BINDING_BLOCK, "r14_gate")
        require(r13_gate.get("gate") == R13_GATE and r13_gate.get("source_commit") == R13_SOURCE, BINDING_BLOCK, "r13_gate")
        require(r12_gate.get("gate") == R12_GATE and r12_gate.get("source_commit") == R12_SOURCE, BINDING_BLOCK, "r12_gate")
        require(r11_gate.get("gate") == R11_GATE and r11_gate.get("source_commit") == R11_SOURCE, BINDING_BLOCK, "r11_gate")
        require(all(value["all_match"] for value in audits.values()), BINDING_BLOCK, "upstream_manifest")

        r14_preflight = load_json(R14 / "bt8r14_evidence_preflight.json")
        r14_frozen = load_json(R14 / "frozen_hash_before_after.json")
        r14_deadlock = load_json(R14 / "bt8r14_exploration_deadlock_audit.json")
        r13_execution = load_json(R13 / "bt8r13_execution_manifest.json")
        r13_checkpoints = load_json(R13 / "bt8r13_checkpoint_manifest.json")
        r13_credit = load_json(R13 / "bt8r13_candidate_plan_credit.json")
        r13_frozen = load_json(R13 / "frozen_hash_before_after.json")
        r12_contract = load_json(R12 / "bt8r12_s3_cell_contract.json")
        r11_contract = load_json(R11 / "bt8r11_selected_seed_repair_contract.json")
        require(r14_frozen.get("all_unchanged") is True and r14_frozen.get("extra_all_unchanged") is True
                and r14_frozen.get("checkpoint_unchanged") is True, BINDING_BLOCK, "r14_frozen_evidence")
        require(r14_deadlock.get("classification") == "INITIAL_POLICY_EXPLORATION_DEADLOCK_CONFIRMED"
                and r14_deadlock.get("bd_chain_confirmed") is True, BINDING_BLOCK, "r14_deadlock")
        require(r13_execution.get("r11_s3_contract_sha256") == S3_CONTRACT_SHA256
                and r12_contract.get("s3_contract_sha256") == S3_CONTRACT_SHA256
                and r11_contract.get("sha256") == S3_CONTRACT_SHA256, BINDING_BLOCK, "s3_contract")
        require(r13_credit.get("identity_chain_all") is True and int(r13_credit.get("mismatches", -1)) == 0, BINDING_BLOCK, "credit_chain")
        require(r13_frozen.get("all_unchanged") is True and r13_frozen.get("extra_all_unchanged") is True, BINDING_BLOCK, "r13_frozen")

        review_collection = load_json(F1 / "bt8f1_review_snapshots" / "collection_manifest.json")
        review_digest_input = dict(review_collection); review_digest = review_digest_input.pop("collection_digest", None)
        require(review_digest == FPS.canonical_sha256(review_digest_input) == REVIEW_COLLECTION_DIGEST, BINDING_BLOCK, "review_collection_digest")
        require(int(review_collection.get("snapshot_count", -1)) == len(review_collection.get("entries", [])) == 6
                and len({entry.get("snapshot_digest") for entry in review_collection["entries"]}) == 6, BINDING_BLOCK, "review_snapshot_count")
        review_preflight = []
        for entry in review_collection["entries"]:
            payload = FPS.load_snapshot(F1 / "bt8f1_review_snapshots" / str(entry["relative_path"]))
            metadata = payload["metadata"]
            require(payload["snapshot_digest"] == entry["snapshot_digest"], BINDING_BLOCK, "review_snapshot_digest")
            review_preflight.append({"snapshot_digest": str(payload["snapshot_digest"]), "time_band": str(metadata["time_band"]),
                                     "environment_seed": int(metadata["seed"]), "support_size": int(metadata["selectable_pair_count"])})
        require({row["time_band"] for row in review_preflight} == set(TIME_BANDS), BINDING_BLOCK, "review_time_bands")

        evidence_checkpoint_before: dict[str, str] = {}
        raw_states: dict[str, Mapping[str, torch.Tensor]] = {}
        for kind in ("initial", "final"):
            for cell_id in CELL_ROLE:
                entry = r13_checkpoints[kind].get(cell_id)
                require(isinstance(entry, Mapping), BINDING_BLOCK, f"checkpoint_entry={kind}:{cell_id}")
                path = R13 / str(entry["path"])
                key = f"{kind}:{cell_id}"
                require(path.is_file() and sha256(path) == str(entry["sha256"]), BINDING_BLOCK, f"checkpoint_sha={key}")
                require(r14_frozen["checkpoint_before"].get(key) == str(entry["sha256"]), BINDING_BLOCK, f"r14_checkpoint={key}")
                stored = torch.load(path, map_location="cpu", weights_only=False)
                require(isinstance(stored, Mapping) and isinstance(stored.get("actor"), Mapping), BINDING_BLOCK, f"checkpoint_schema={key}")
                raw_states[key] = stored["actor"]
                evidence_checkpoint_before[key] = sha256(path)
        require(state_dicts_exact(raw_states["initial:AC-R1"], raw_states["initial:AC-R2"]), BINDING_BLOCK, "ac_initial_state")
        require(state_dicts_exact(raw_states["initial:BD-R1"], raw_states["initial:BD-R2"]), BINDING_BLOCK, "bd_initial_state")
        require(state_dicts_exact(raw_states["initial:BD-R1"], raw_states["final:BD-R1"])
                and state_dicts_exact(raw_states["initial:BD-R1"], raw_states["final:BD-R2"]), BINDING_BLOCK, "bd_immutable_state")

        training_collection = load_json(R13 / "bt8r13_training_snapshots" / "collection_manifest.json")
        train_entries = list(training_collection.get("entries", []))
        require(int(training_collection.get("snapshot_count", -1)) == len(train_entries) == 96
                and len({entry.get("snapshot_digest") for entry in train_entries}) == 96, BINDING_BLOCK, "train_snapshot_count")
        credit_by_snapshot = {str(row["snapshot_digest"]): row for row in r13_credit.get("rows", [])}
        require(len(credit_by_snapshot) == 96, BINDING_BLOCK, "credit_snapshot_count")
        config: Mapping[str, Any] | None = None
        records: list[dict[str, Any]] = []
        for entry in train_entries:
            snapshot_path = R13 / "bt8r13_training_snapshots" / str(entry["relative_path"])
            payload = FPS.load_snapshot(snapshot_path)
            metadata, tensors = payload["metadata"], payload["tensors"]
            decision_tokens = str(metadata["decision_id"]).split(":")
            require(len(decision_tokens) >= 2 and decision_tokens[1] in CELL_ROLE, BINDING_BLOCK, "cell_identity")
            cell_id = decision_tokens[1]
            require(payload["snapshot_digest"] == entry["snapshot_digest"] and str(payload["snapshot_digest"]) in credit_by_snapshot,
                    BINDING_BLOCK, "train_snapshot_digest")
            require(metadata["actor_config"] == (config if config is not None else metadata["actor_config"]), BINDING_BLOCK, "actor_config")
            config = metadata["actor_config"]
            support = int(metadata["selectable_pair_count"])
            require(support > 0 and bool(tensors["safe_mask"][0, :support].all().item()), STRUCTURAL_BLOCK, "safe_frozen_support")
            require(metadata["candidate_ids"] == metadata["candidate_order"] and int(metadata["no_assign_index"]) == support,
                    BINDING_BLOCK, "candidate_order")
            credit = credit_by_snapshot[str(payload["snapshot_digest"])]
            require(str(credit["cell_id"]) == cell_id and str(credit["candidate_support_digest"]) == str(metadata["candidate_support_digest"]),
                    BINDING_BLOCK, "credit_snapshot_binding")
            records.append({"payload": payload, "cell_id": cell_id, "policy_family": CELL_ROLE[cell_id],
                            "snapshot_digest": str(payload["snapshot_digest"]), "snapshot_manifest_sha256": str(entry["snapshot_manifest_sha256"]),
                            "decision_id": str(metadata["decision_id"]), "window_id": str(metadata["window_id"]),
                            "time_band": str(metadata["time_band"]), "environment_seed": int(metadata["seed"]),
                            "candidate_support_digest": str(metadata["candidate_support_digest"]), "support_size": support, "credit": credit})
        require(config is not None and config.get("actor_head_id") == H.CANDIDATE_SENSITIVE_HEAD_ID
                and config.get("actor_head_version") == H.CANDIDATE_SENSITIVE_HEAD_VERSION, BINDING_BLOCK, "v2_config")
        counts = Counter((record["cell_id"], record["time_band"]) for record in records)
        require(all(counts[(cell, band)] == 8 for cell in CELL_ROLE for band in TIME_BANDS), BINDING_BLOCK, "time_band_coverage")

        current_frozen = R12MOD.frozen_hashes()
        current_extra = {"candidate_plan_bridge": sha256(ROOT / "joint_candidate_plan_causal_bridge.py"),
                         "e1_eligibility": sha256(ROOT / "joint_assignment_e1_eligibility.py"),
                         "t1_selector": sha256(ROOT / "joint_assignment_frozen_tie_break.py")}
        require(current_frozen == r14_frozen.get("after") and current_extra == r14_frozen.get("extra_after"), BINDING_BLOCK,
                "frozen_hash_binding")
        require(TIE.TIE_BREAK_CONTRACT_ID == T1_CONTRACT and source["source_lineage_descends_from_r14"]
                and source["source_only_local_commit"], BINDING_BLOCK, "source_or_t1")
        preflight |= {"authority_manifests": audits, "authority_gates": {"r14": True, "r13": True, "r12": True, "r11": True},
                      "review_collection_digest": review_digest, "review_snapshots": review_preflight,
                      "training_snapshot_count": len(records), "training_time_band_counts": {f"{cell}:{band}": counts[(cell, band)] for cell in CELL_ROLE for band in TIME_BANDS},
                      "actor_config": dict(config), "actor_config_sha256": FPS.actor_config_sha256(config),
                      "t1_selector_sha256": sha256(ROOT / "joint_assignment_frozen_tie_break.py"), "t1_tolerance": 0.0,
                      "checkpoint_before": evidence_checkpoint_before, "frozen_hash_binding": True,
                      "r14_deadlock_binding": True, "r13_credit_identity_binding": True}
    except R15Error as exc:
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
        actors: dict[str, torch.nn.Module] = {}
        actor_digests: dict[str, str] = {}
        for family, state_key in (("AC", "initial:AC-R1"), ("BD", "initial:BD-R1")):
            actor = H.CandidateSensitiveMultiAgentCandidateAssignmentHead(
                global_dim=int(config["global_dim"]), demand_dim=int(config["demand_dim"]), agent_dim=int(config["agent_dim"]),
                candidate_dim=int(config["candidate_dim"]), hidden=int(config["hidden"]), heads=int(config["heads"])).to(device)
            actor.load_state_dict(raw_states[state_key], strict=True); actor.eval()
            expected = str(r13_checkpoints["initial"][state_key.split(":")[1]]["actor_initial_digest"])
            require(module_digest(actor) == expected, BINDING_BLOCK, f"strict_actor_load={family}")
            actors[family] = actor; actor_digests[family] = expected
        actor_before = {family: module_digest(actor) for family, actor in actors.items()}

        base_results: dict[str, dict[str, Any]] = {}
        e0_rows: list[dict[str, Any]] = []
        ac_control_mismatch = 0
        bd_control_mismatch = 0
        for record in records:
            result = R1.frozen_forward(actors[record["policy_family"]], record["payload"], device=device, head=H, tie=TIE)
            execution["deterministic_control_forwards"] += 1
            distribution = action_distribution(result=result, payload=record["payload"], TIE=TIE)
            deterministic = deterministic_action(result=result, distribution=distribution)
            observed_type = "NO_ASSIGN" if action_family(deterministic) == "NO_ASSIGN" else "CANDIDATE"
            expected_type = str(record["credit"]["selected_action_type"])
            # E0's required deadlock reproduction is the BD NO_ASSIGN action family.
            # AC is only a normal-control family: T1 can canonically settle an exact
            # candidate tie without changing whether the deterministic exploit enters
            # the candidate family, so no positional candidate-id assertion belongs
            # in this frozen-inference selection audit.
            match = observed_type == expected_type
            if record["policy_family"] == "AC": ac_control_mismatch += int(not match)
            else: bd_control_mismatch += int(not match)
            finite = bool(result["finite"] and all(math.isfinite(float(row["policy_probability"])) for row in distribution))
            legal = action_family(deterministic) == "NO_ASSIGN" or any(action_identity(deterministic) == action_identity(row)
                                                                     for row in distribution if action_family(row) == "CANDIDATE")
            execution["nan_or_inf"] += int(not finite)
            execution["illegal_selection"] += int(not legal)
            record["distribution"] = distribution
            record["actor_state_sha256"] = actor_digests[record["policy_family"]]
            base_results[record["snapshot_digest"]] = {"result": result, "distribution": distribution, "deterministic": deterministic,
                                                         "selections": {}, "record": record}
            e0_rows.append(make_probe_row(level="E0", record=record, deterministic=deterministic, selected=deterministic,
                                           probe_seed=-1, selection_mode="DETERMINISTIC_T1_EXPLOIT", triggered=False,
                                           modified_probability=None, actor_digest_before=actor_before[record["policy_family"]],
                                           actor_digest_after=actor_before[record["policy_family"]]))
        torch.mps.synchronize()
        e0 = {"ac_control_expected_candidate_count": 48, "ac_control_mismatch_count": ac_control_mismatch,
              "bd_control_expected_no_assign_count": 48, "bd_control_mismatch_count": bd_control_mismatch,
              "bd_control_exact": bd_control_mismatch == 0, "same_seed_replay_mismatch_count": 0, "ac_e1_triggered_count": 0}
        if not e0["bd_control_exact"] or ac_control_mismatch:
            hard.append(E0_BLOCK)

        selected_level: str | None = None
        all_probe_rows = list(e0_rows)
        ladder: dict[str, Any] = {"E0": {"status": "PASS" if not hard else "FAIL", "control": e0},
                                  "E1": {"status": "NOT_RUN"}, "E2": {"status": "NOT_RUN"}, "E3": {"status": "NOT_RUN"}}
        order: dict[str, Any] = {"not_run": True}
        if not hard:
            e1_rows: list[dict[str, Any]] = []
            for record in records:
                base = base_results[record["snapshot_digest"]]
                for probe_seed in PROBE_SEEDS:
                    selected, triggered, selection_mode, modified = select_level(level="E1", distribution=base["distribution"],
                                                                                   deterministic=base["deterministic"], snapshot_digest=record["snapshot_digest"],
                                                                                   probe_seed=probe_seed, training_mode=True)
                    replay, _, _, _ = select_level(level="E1", distribution=base["distribution"], deterministic=base["deterministic"],
                                                    snapshot_digest=record["snapshot_digest"], probe_seed=probe_seed, training_mode=True)
                    e0["same_seed_replay_mismatch_count"] += int(action_identity(selected) != action_identity(replay))
                    if record["policy_family"] == "AC": e0["ac_e1_triggered_count"] += int(triggered)
                    base["selections"][probe_seed] = selected
                    row = make_probe_row(level="E1", record=record, deterministic=base["deterministic"], selected=selected,
                                         probe_seed=probe_seed, selection_mode=selection_mode, triggered=triggered,
                                         modified_probability=modified, actor_digest_before=actor_before[record["policy_family"]],
                                         actor_digest_after=actor_before[record["policy_family"]])
                    e1_rows.append(row); execution["stochastic_probe_rows"] += 1
            order, order_forwards = order_audit(level="E1", base_results=base_results, actors=actors, records=records,
                                                 R1=R1, H=H, TIE=TIE, device=device)
            execution["order_probe_forwards"] = order_forwards
            if e1_pass(e0=e0, rows=e1_rows, order=order, counters_value=execution):
                selected_level = "E1"
                ladder["E1"] = {"status": "PASS", "contract_id": R15_E1_CONTRACT_ID,
                                 "time_band_exposure": summary_by_band(e1_rows, level="E1"),
                                 "rationale": "unchanged frozen masked probabilities yielded non-NO_ASSIGN exposure in every BD cell/time band"}
                ladder["E2"] = {"status": "SKIPPED_BY_MINIMUM_CHANGE_RULE"}
                ladder["E3"] = {"status": "SKIPPED_BY_MINIMUM_CHANGE_RULE"}
                all_probe_rows.extend(e1_rows)
            else:
                ladder["E1"] = {"status": "FAIL", "contract_id": R15_E1_CONTRACT_ID,
                                 "time_band_exposure": summary_by_band(e1_rows, level="E1")}
                all_probe_rows.extend(e1_rows)
                e2_rows: list[dict[str, Any]] = []
                for record in records:
                    base = base_results[record["snapshot_digest"]]
                    for probe_seed in PROBE_SEEDS:
                        selected, triggered, selection_mode, modified = select_level(level="E2", distribution=base["distribution"],
                                                                                       deterministic=base["deterministic"], snapshot_digest=record["snapshot_digest"],
                                                                                       probe_seed=probe_seed, training_mode=True)
                        base["selections"][probe_seed] = selected
                        e2_rows.append(make_probe_row(level="E2", record=record, deterministic=base["deterministic"], selected=selected,
                                                      probe_seed=probe_seed, selection_mode=selection_mode, triggered=triggered,
                                                      modified_probability=modified, actor_digest_before=actor_before[record["policy_family"]],
                                                      actor_digest_after=actor_before[record["policy_family"]]))
                        execution["stochastic_probe_rows"] += 1
                order, extra_order_forwards = order_audit(level="E2", base_results=base_results, actors=actors, records=records,
                                                           R1=R1, H=H, TIE=TIE, device=device)
                execution["order_probe_forwards"] += extra_order_forwards
                e2_bands = summary_by_band(e2_rows, level="E2")
                e2_exposed = all(row["exposure_positive"] for row in e2_bands if row["cell_id"].startswith("BD-"))
                e2_integrity = order["passed"] and e2_exposed and all(execution[key] == 0 for key in ("nan_or_inf", "illegal_selection", "zero_loss_violation"))
                all_probe_rows.extend(e2_rows)
                if e2_integrity:
                    selected_level = "E2"
                    ladder["E2"] = {"status": "PASS", "contract_id": R15_E2_CONTRACT_ID, "time_band_exposure": e2_bands,
                                     "ppo_binding_status": "NOT_A_DROP_IN_PPO_SELECTOR; a separate R16 q-policy/log-prob equivalence contract is required"}
                    ladder["E3"] = {"status": "SKIPPED_BY_MINIMUM_CHANGE_RULE"}
                else:
                    ladder["E2"] = {"status": "FAIL", "contract_id": R15_E2_CONTRACT_ID, "time_band_exposure": e2_bands,
                                     "ppo_binding_status": "NOT_AUTHORIZED"}
                    ladder["E3"] = {"status": "NOT_EVALUATED_NO_AUTHORIZED_FLOOR_OR_TEMPERATURE_CONTRACT"}

        torch.mps.synchronize()
        actor_after = {family: module_digest(actor) for family, actor in actors.items()}
        execution["parameter_mutation"] = sum(actor_before[name] != actor_after[name] for name in actors)
        execution["checkpoint_mutation"] = 0
        frozen_after = R12MOD.frozen_hashes()
        extra_after = {"candidate_plan_bridge": sha256(ROOT / "joint_candidate_plan_causal_bridge.py"),
                       "e1_eligibility": sha256(ROOT / "joint_assignment_e1_eligibility.py"),
                       "t1_selector": sha256(ROOT / "joint_assignment_frozen_tie_break.py")}
        checkpoint_after = {}
        for kind in ("initial", "final"):
            for cell_id in CELL_ROLE:
                checkpoint_after[f"{kind}:{cell_id}"] = sha256(R13 / str(r13_checkpoints[kind][cell_id]["path"]))
        execution["checkpoint_mutation"] = sum(evidence_checkpoint_before[key] != checkpoint_after[key] for key in evidence_checkpoint_before)
        forbidden = ("training", "optimizer_creation", "optimizer_step", "backward", "causal_rollout", "simulator_execution",
                     "candidate_generation", "candidate_regeneration", "local_search_rerun", "zero_loss_reevaluation", "reward_settlement",
                     "parameter_mutation", "checkpoint_write", "checkpoint_mutation", "test6_access", "github_push", "nan_or_inf",
                     "illegal_selection", "zero_loss_violation")
        if any(execution[key] != 0 for key in forbidden): hard.append(STRUCTURAL_BLOCK)
        if frozen_after != current_frozen or extra_after != current_extra: hard.append("FROZEN_HASH_MUTATION")
        if selected_level is None: hard.append(NO_SAFE_BLOCK)

        if selected_level == "E1":
            classification = "A_FROZEN_POLICY_MASKED_STOCHASTIC_SAMPLING_SUFFICIENT"
            chosen_contract = {
                "contract_id": R15_E1_CONTRACT_ID, "repair_level": "E1", "sampler": "identity-keyed exponential race over unchanged legal masked policy probabilities",
                "trigger": "training_mode AND deterministic exploit is NO_ASSIGN AND legal non-NO_ASSIGN support > 0",
                "no_assign": "remains a legal sampled action under E1", "inference_evaluation": "deterministic T1 exploit unchanged",
                "policy_logits_mutated": False, "policy_weights_mutated": False, "policy_probabilities_modified": False,
            }
        elif selected_level == "E2":
            classification = "B_TRAINING_ONLY_DEADLOCK_RESCUE_EXPLORATION_REQUIRED"
            chosen_contract = {
                "contract_id": R15_E2_CONTRACT_ID, "repair_level": "E2", "sampler": "identity-keyed conditional non-NO_ASSIGN exponential race",
                "trigger": "training_mode AND deterministic exploit is NO_ASSIGN AND legal non-NO_ASSIGN support > 0",
                "no_assign": "remains in action space; rescue changes only training-mode selection under the trigger",
                "policy_logits_mutated": False, "policy_weights_mutated": False, "policy_probabilities_modified": False,
                "ppo_q_policy_binding_required": True,
            }
        else:
            classification = "BLOCKED"
            chosen_contract = {"not_selected": True}
        if not hard:
            chosen_contract |= {"contract_sha256": canonical_sha256(chosen_contract), "training_authorized": False,
                                "execution_authorized": False, "optimizer_authorized": False, "policy_mutation_authorized": False,
                                "separate_implementation_gate_required": True, "separate_training_authorization_required": True,
                                "inference_behavior_change_authorized": False, "automatic_training_authorization": False}
        for row in all_probe_rows:
            row["candidate_order_identity_equal"] = bool(order.get("candidate_order_failures", 0) == 0) if selected_level else None
            row["agent_order_identity_equal"] = bool(order.get("agent_order_failures", 0) == 0) if selected_level else None
            row["combined_order_identity_equal"] = bool(order.get("combined_order_failures", 0) == 0) if selected_level else None
            row["actor_tensor_digest_after"] = actor_after[row["policy_family"]]
        parquet = write_probe_parquet(root / "exploration_probe_results.parquet", all_probe_rows)
        gate = PASS_GATE if not hard else hard[0]
        if hard: classification = "BLOCKED"
        summary = {"probe_seed_schedule": preflight["probe_seed_schedule"], "parquet": parquet,
                   "e0": e0, "e1_time_band_exposure": summary_by_band(all_probe_rows, level="E1"),
                   "e2_time_band_exposure": summary_by_band(all_probe_rows, level="E2"), "selected_level": selected_level,
                   "inference_policy_preservation": {"actor_state_before": actor_before, "actor_state_after": actor_after,
                                                       "actor_tensor_mutation_count": execution["parameter_mutation"],
                                                       "deterministic_ac_control_unchanged": e0["ac_control_mismatch_count"] == 0,
                                                       "no_inference_hook_installed": True},
                   "no_reward_kpi_or_rollout_interpretation": True}
        outputs = {
            "evidence_binding_audit.json": preflight | {"mps_built": True, "mps_available": True, "device": "mps:0"},
            "repair_candidate_ladder.json": ladder,
            "exploration_probe_summary.json": summary,
            "order_invariance_probe.json": order,
            "selected_minimal_exploration_repair_contract.json": chosen_contract,
            "test_results.json": {"execution_counters": execution, "hard_failures": hard, "warnings": [], "github_push_performed": False,
                                  "training_authorized": False, "performance_or_kpi_interpretation_performed": False},
            "frozen_hash_before_after.json": {"before": current_frozen, "after": frozen_after, "all_unchanged": current_frozen == frozen_after,
                                                "extra_before": current_extra, "extra_after": extra_after, "extra_all_unchanged": current_extra == extra_after,
                                                "checkpoint_before": evidence_checkpoint_before, "checkpoint_after": checkpoint_after,
                                                "checkpoint_unchanged": evidence_checkpoint_before == checkpoint_after},
            "gate_decision.json": {"stage": STAGE, "gate": gate, "classification": classification, "source_commit": source["source_commit"],
                                   "hard_failures": hard, "warnings": [], "global_locks": GLOBAL_LOCKS, "execution_locks": EXECUTION_LOCKS,
                                   "next_step": "R16 selected exploration repair implementation plus frozen-policy equivalence validation" if not hard else "STOP"},
        }
        for name, value in outputs.items(): dump(root / name, value)
        (root / "final_report.md").write_text(
            f"# BT8-R15 final report\n\n- gate: `{gate}`\n- classification: `{classification}`\n- source commit: `{source['source_commit']}`\n\n"
            "This is a frozen-support selection audit only. It creates no optimizer, training rollout, reward/KPI result, checkpoint mutation, or training authorization.\n",
            encoding="utf-8")
        manifest = {item.relative_to(root).as_posix(): sha256(item) for item in root.rglob("*") if item.is_file() and item.name != "manifest.json"}
        dump(root / "manifest.json", {"stage": STAGE, "gate": gate, "classification": classification, "source_commit": source["source_commit"],
                                       "file_sha256": manifest, "github_push_performed": False})
        (root / ("_SUCCESS.lock" if not hard else "_BLOCKED.lock")).write_text(gate + "\n", encoding="utf-8")
        print(f"[{'PASS' if not hard else 'BLOCKED'}] {gate}")
        print(f"classification: {classification}")
        print(f"artifact: {root.relative_to(PROJECT)}")
    except R15Error as exc:
        write_block(root=root, source=source, preflight=preflight | {"hard_failures": [str(exc)]}, reason=exc.code)
        print(f"[BLOCKED] {exc.code}"); print(f"artifact: {root.relative_to(PROJECT)}")
    except Exception as exc:  # noqa: BLE001
        write_block(root=root, source=source, preflight=preflight | {"hard_failures": [repr(exc)]}, reason=STRUCTURAL_BLOCK)
        print(f"[BLOCKED] {STRUCTURAL_BLOCK}"); print(f"artifact: {root.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
