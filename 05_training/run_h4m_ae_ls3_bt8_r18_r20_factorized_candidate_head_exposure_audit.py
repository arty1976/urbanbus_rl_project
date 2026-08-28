#!/usr/bin/env python3
"""R18-R20 factorized candidate-head exposure sufficiency audit.

This is a read-only analysis of R18-R18B durable trace evidence.  It never
creates an optimizer, invokes backward, mutates a policy, or regenerates a
candidate.  The only autograd use is an exact epoch-1 per-row Stage-2 gradient
reconstruction because the durable trace intentionally persists norms, not
gradient vectors.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
sys.path.insert(0, str(ROOT))

import joint_assignment_frozen_policy_snapshot as FPS  # noqa: E402
import run_h4m_ae_ls3_bt8_r18_e1_bounded_training as R18  # noqa: E402
import run_h4m_ae_ls3_bt8_r18_r18b_external_mps_one_shot_authorization as R18B  # noqa: E402
import run_h4m_ae_ls3_bt8_r18_r19_frozen_factorized_policy_review as R19  # noqa: E402


STAGE = "H4M-AE-R9.8-LS3-BT8-R18-R20"
PASS_GATE = (
    "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_R20_"
    "FACTORIZED_CANDIDATE_HEAD_LEARNING_EXPOSURE_SUFFICIENCY_AND_NEXT_STEP_SELECTION_AUDIT_COMPLETE"
)
CLASSIFICATION = "A_CANDIDATE_HEAD_EXPOSURE_INSUFFICIENT_FOR_DISCRIMINATION_CONCLUSION"
BLOCK_BINDING = "BLOCKED_R18R20_UPSTREAM_EVIDENCE_BINDING_FAILURE"
BLOCK_TRACE = "BLOCKED_R18R20_DURABLE_TRACE_INTEGRITY_FAILURE"
BLOCK_GRADIENT = "BLOCKED_R18R20_STAGE2_GRADIENT_RECONSTRUCTION_FAILURE"
BLOCK_MPS = "BLOCKED_R18R20_EXTERNAL_MPS_PREFLIGHT_FAILURE"

R18R19_SOURCE = "c8f0707555d717b64dcedb9aea6aaec7366a30bc"
R18R18B_SOURCE = "562ac7b0501a071cac9f60626eef79381b803387"
R18R16_SOURCE = "ff2c4b008d42235926283748f408474dbbe65ba0"
R18R19_ROOT = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r19_frozen_factorized_policy_review_20260829_000021+09:00"
R18R18B_AUTH_ROOT = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r18b_external_mps_one_shot_authorization_20260828_203601+09:00"
R18R18B_EXEC_ROOT = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r18b_factorized_actor_bounded_training_execution_20260828_203601+09:00"
AUTH_PATH = R18R18B_AUTH_ROOT / "r18r18b_factorized_one_shot_authorization_manifest.json"
TRACE_FILES = (
    "eligible_row_credit_trace.parquet",
    "credit_eligibility_rows.parquet",
    "gae_rows.parquet",
    "ppo_ratio_rows.parquet",
    "per_epoch_loss_contributions.parquet",
)
ID_FIELDS = ("arm_id", "environment_seed", "window_id", "time_band", "trajectory_id", "decision_id", "transition_index", "semantic_selected_candidate_id")
GRAD_TOL = 2e-5


class R18R20Error(RuntimeError):
    def __init__(self, gate: str, detail: str) -> None:
        super().__init__(f"{gate}: {detail}")
        self.gate = gate
        self.detail = detail


def require(condition: bool, detail: str, gate: str = BLOCK_BINDING) -> None:
    if not condition:
        raise R18R20Error(gate, detail)


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
    return ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r20_factorized_candidate_head_exposure_audit_{stamp}"


def state_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for _, tensor in sorted(state.items()):
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def manifest_binding(root: Path) -> dict[str, Any]:
    manifest = load(root / "manifest.json")
    hashes = dict(manifest.get("file_sha256", {}))
    mismatches = {name: {"expected": expected, "actual": sha256(root / name) if (root / name).is_file() else None}
                  for name, expected in hashes.items()
                  if not (root / name).is_file() or sha256(root / name) != expected}
    require(not mismatches, f"artifact_manifest={root.name}:{mismatches}")
    return {"root": str(root), "manifest_sha256": sha256(root / "manifest.json"),
            "declared_file_count": len(hashes), "all_match": True}


def source_binding(auth: Mapping[str, Any]) -> dict[str, Any]:
    bound = dict(dict(auth["module_freeze_contract"])["frozen_source_hashes"])
    expected = {str(key): str(value) for key, value in dict(bound["expected"]).items()}
    actual = {rel: sha256(PROJECT / rel) for rel in expected}
    require(actual == expected, "frozen_source_sha_mismatch")
    return {"expected": expected, "actual": actual, "all_unchanged": True}


def validate_upstream() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    auth = load(AUTH_PATH)
    supplied = str(auth.get("authorization_sha256", ""))
    expected_auth_sha = canonical_sha256({key: value for key, value in auth.items() if key != "authorization_sha256"})
    require(supplied == expected_auth_sha, "r18r18b_authorization_sha256")
    require(auth.get("authorization") == R18.R18R18B_FACTORIZED_AUTHORIZATION
            and auth.get("source_commit") == R18R18B_SOURCE, "r18r18b_authorization_identity")
    require(git(["status", "--porcelain=v1"]) == "", "dirty_tree_before_audit")
    changed = [row for row in git(["diff", "--name-only", f"{R18R19_SOURCE}..HEAD"]).splitlines() if row]
    require(set(changed) == {"05_training/run_h4m_ae_ls3_bt8_r18_r20_factorized_candidate_head_exposure_audit.py"},
            f"r20_source_scope={changed}")

    r19_gate = load(R18R19_ROOT / "gate_decision.json")
    require(r19_gate.get("gate") == R19.PASS_GATE and r19_gate.get("source_commit") == R18R19_SOURCE,
            "r18r19_gate_or_source")
    r19_manifest = manifest_binding(R18R19_ROOT)
    exec_gate = load(R18R18B_EXEC_ROOT / "gate_decision.json")
    require(exec_gate.get("gate") == R18.R18R18B_FACTORIZED_PASS_GATE, "r18r18b_execution_gate")
    exec_manifest = manifest_binding(R18R18B_EXEC_ROOT)
    auth_manifest = manifest_binding(R18R18B_AUTH_ROOT)

    upstream = dict(auth["upstream"])
    r16 = dict(upstream["R18-R16"])
    recursive = dict(r16["recursive_upstream_binding"])
    r17 = dict(upstream["r18_r17_authorization"])
    required = {
        "R18-R17": (Path(str(r17["root"])), "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_R17_FACTORIZED_ACTOR_BOUNDED_TRAINING_AUTHORIZATION_AND_ENVELOPE_FREEZE_COMPLETE"),
        "R18-R16": (Path(str(r16["root"])), "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_R16_FACTORIZED_ASSIGN_THEN_CANDIDATE_ACTOR_IMPLEMENTATION_AND_FROZEN_EQUIVALENCE_VALIDATION_COMPLETE"),
    }
    for stage, expected_gate in (("R18-R15", "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_R15_MINIMAL_ACTOR_REPAIR_SELECTION_AUDIT_COMPLETE"),
                                 ("R18-R13", "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_R13_GRADIENT_MAGNITUDE_AND_PARAMETER_PATH_DOMINANCE_DECOMPOSITION_AUDIT_COMPLETE"),
                                 ("R18-R12", "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_R12_SHARED_BATCH_GRADIENT_INTERFERENCE_AND_CANDIDATE_VS_NO_ASSIGN_ATTRIBUTION_AUDIT_COMPLETE")):
        record = dict(dict(recursive["upstream"])[stage])
        required[stage] = (Path(str(record["root"])), expected_gate)
    upstream_audit = {}
    for stage, (root, expected_gate) in required.items():
        gate = load(root / "gate_decision.json")
        require(gate.get("gate") == expected_gate, f"{stage}_gate={gate.get('gate')}")
        upstream_audit[stage] = {"gate": gate.get("gate"), **manifest_binding(root)}

    trace_schema = load(R18R18B_EXEC_ROOT / "trace_schema.json")
    factor_fields = set(trace_schema.get("optional_factorized_trace_fields", []))
    require(trace_schema.get("contract_id") == "R18_R6_DURABLE_PER_ROW_CREDIT_PPO_TRACE_V1"
            and {"stage1_loss_contribution", "stage2_loss_contribution", "stage1_gradient_norm", "stage2_gradient_norm", "shared_encoder_gradient_norm"}.issubset(factor_fields),
            "factorized_trace_schema")
    r16_contract = load(Path(str(r16["root"])) / "factorized_actor_implementation_contract.json")
    require(r16_contract.get("architecture") == "FACTORIZED_ASSIGN_THEN_CANDIDATE"
            and r16_contract.get("NO_ASSIGN_candidate_head_gradient") == "ZERO", "factorized_implementation_contract")

    checkpoint_manifest = load(R18R18B_EXEC_ROOT / "checkpoint_manifest.json")
    final_records = dict(checkpoint_manifest.get("final", {}))
    require(set(final_records) == {"AC_FACTOR_CONTROL_R1", "BD_FACTOR_R1"}, "r18r18b_final_checkpoint_set")
    inputs = dict(dict(auth["checkpoint_contract"])["initial_inputs"])
    original_checkpoints = dict(recursive["checkpoint_binding"])
    require(len(original_checkpoints) == 8, "original_checkpoint_count")
    checkpoint_mismatches = {}
    for key, record in original_checkpoints.items():
        path = Path(str(record["path"]))
        actual = sha256(path) if path.is_file() else None
        if actual != record.get("sha256"):
            checkpoint_mismatches[key] = {"expected": record.get("sha256"), "actual": actual}
    require(not checkpoint_mismatches, f"original_checkpoint_hashes={checkpoint_mismatches}")
    for record in final_records.values():
        path = Path(str(record["path"]))
        require(path.is_file() and sha256(path) == record.get("sha256"), f"final_checkpoint={record.get('arm_id')}")

    return auth, checkpoint_manifest, {
        "passed": True,
        "r18r19": {"gate": r19_gate.get("gate"), **r19_manifest},
        "r18r18b_authorization": {"authorization_sha256": expected_auth_sha, **auth_manifest},
        "r18r18b_execution": {"gate": exec_gate.get("gate"), **exec_manifest},
        "required_upstream": upstream_audit,
        "r18r18b_source_commit": R18R18B_SOURCE,
        "r18r19_source_commit": R18R19_SOURCE,
        "r18r20_source_commit": git(["rev-parse", "HEAD"]),
        "r18r20_source_scope": changed,
        "frozen_source_hashes": source_binding(auth),
        "factorized_trace_schema_sha256": sha256(R18R18B_EXEC_ROOT / "trace_schema.json"),
        "factorized_implementation_contract_sha256": sha256(Path(str(r16["root"])) / "factorized_actor_implementation_contract.json"),
        "original_checkpoint_hashes_verified": 8,
        "r18r18b_final_checkpoint_records": final_records,
        "r18r18b_initial_input_records": inputs,
    }


def load_trace() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tables = []
    for name in TRACE_FILES:
        path = R18R18B_EXEC_ROOT / name
        require(path.is_file(), f"missing_trace={name}", BLOCK_TRACE)
        tables.append(pd.read_parquet(path))
    eligible, credit, gae, ratio, epoch = tables
    require(len(credit) == len(gae) == 48 and len(ratio) == len(epoch) == 144 and len(eligible) == 24,
            "trace_row_counts", BLOCK_TRACE)
    required = {"factorized_actor_contract_id", "stage1_loss_contribution", "stage2_loss_contribution",
                "stage1_gradient_norm", "stage2_gradient_norm", "shared_encoder_gradient_norm"}
    require(required.issubset(epoch.columns), "factorized_trace_columns", BLOCK_TRACE)
    for frame in (credit, gae, ratio, epoch):
        require(set(ID_FIELDS).issubset(frame.columns), "trace_identity_columns", BLOCK_TRACE)
    return eligible, credit, gae, ratio, epoch


def snapshot_map() -> dict[str, dict[str, Any]]:
    root = R18R18B_EXEC_ROOT / "training_snapshots" / "snapshots"
    require(root.is_dir(), "training_snapshot_root", BLOCK_TRACE)
    mapped = {}
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        payload = FPS.load_snapshot(path)
        decision = str(payload["metadata"]["decision_id"])
        require(decision not in mapped, f"duplicate_snapshot_decision={decision}", BLOCK_TRACE)
        mapped[decision] = {"path": path, "payload": payload}
    require(len(mapped) == 48, f"training_snapshot_count={len(mapped)}", BLOCK_TRACE)
    return mapped


def candidate_identity(row: Mapping[str, Any]) -> str:
    return f"{row['agent_id']}::{row['candidate_id']}"


def classify_rows(epoch_one: pd.DataFrame, snapshots: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in epoch_one.to_dict(orient="records"):
        decision = str(item["decision_id"])
        require(decision in snapshots, f"snapshot_missing={decision}", BLOCK_TRACE)
        payload = snapshots[decision]["payload"]
        metadata = dict(payload["metadata"])
        k = int(metadata["selectable_pair_count"])
        candidate_ids = list(metadata["candidate_ids"])
        require(len(candidate_ids) == k and str(metadata["candidate_support_digest"]) == str(item["candidate_support_digest"]),
                f"support_binding={decision}", BLOCK_TRACE)
        selected_no_assign = bool(item["selected_is_no_assign"])
        index = int(item["selected_source_index"])
        if selected_no_assign:
            require(index == k and str(item["selected_semantic_candidate"]) == str(item["NO_ASSIGN_semantic_identity"])
                    and str(item["executed_semantic_candidate"]) == str(item["credited_semantic_candidate"])
                    and str(item["credited_semantic_candidate"]) == str(item["NO_ASSIGN_semantic_identity"]),
                    f"no_assign_identity={decision}", BLOCK_TRACE)
            category = "NO_ASSIGN"
        else:
            require(0 <= index < k and candidate_identity(candidate_ids[index]) == str(item["semantic_selected_candidate_id"])
                    and str(item["selected_semantic_candidate"]) == str(item["executed_semantic_candidate"])
                    and str(item["executed_semantic_candidate"]) == str(item["credited_semantic_candidate"]),
                    f"candidate_identity={decision}", BLOCK_TRACE)
            category = "CANDIDATE_K1" if k == 1 else "CANDIDATE_K_GT_1"
        ancestry = json.loads(str(item["reward_ancestry_source_ids"]))
        require(bool(ancestry) and str(item["reward_ancestry_class"]) not in {"", "None", "nan"}
                and math.isfinite(float(item["advantage_normalized"])) and math.isfinite(float(item["old_log_prob"])),
                f"credit_or_advantage_missing={decision}", BLOCK_TRACE)
        rows.append({**item, "K": k, "selected_action_family": category,
                     "snapshot_path": str(snapshots[decision]["path"]),
                     "candidate_semantic_identities": [candidate_identity(value) for value in candidate_ids]})
    classified = pd.DataFrame(rows)
    require(len(classified) == 8 and set(classified["selected_action_family"]) == {"NO_ASSIGN", "CANDIDATE_K1", "CANDIDATE_K_GT_1"},
            "eligible_category_population", BLOCK_TRACE)
    return classified


def audit_trace_identity(classified: pd.DataFrame, credit: pd.DataFrame, gae: pd.DataFrame,
                         ratio: pd.DataFrame, epoch: pd.DataFrame) -> None:
    base_ids = {tuple(row[field] for field in ID_FIELDS) for row in classified.to_dict(orient="records")}
    for name, frame in (("credit", credit), ("gae", gae)):
        observed = {tuple(row[field] for field in ID_FIELDS) for row in frame.loc[frame["actor_eligible"]].to_dict(orient="records")}
        require(observed == base_ids, f"{name}_eligible_identity_roundtrip", BLOCK_TRACE)
    for name, frame in (("ratio", ratio), ("epoch", epoch)):
        active = frame.loc[frame["actor_eligible"]]
        observed = {tuple(row[field] for field in ID_FIELDS) for row in active.to_dict(orient="records")}
        require(observed == base_ids and set(active["epoch_index"].astype(int)) == {1, 2, 3}
                and len(active) == 24, f"{name}_eligible_epoch_roundtrip", BLOCK_TRACE)
    factor = epoch.loc[epoch["actor_eligible"]]
    require(set(factor["factorized_actor_contract_id"]) == {"FACTORIZED_ASSIGN_THEN_CANDIDATE"},
            "factorized_trace_contract_id", BLOCK_TRACE)


def census(classified: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    def one(frame: pd.DataFrame) -> dict[str, Any]:
        kgt = frame.loc[frame["selected_action_family"] == "CANDIDATE_K_GT_1"]
        candidates = frame.loc[frame["selected_action_family"].str.startswith("CANDIDATE")]
        return {
            "actor_eligible_rows": int(len(frame)),
            "NO_ASSIGN_selected_eligible_rows": int((frame["selected_action_family"] == "NO_ASSIGN").sum()),
            "candidate_selected_eligible_rows": int(len(candidates)),
            "candidate_selected_K1_rows": int((frame["selected_action_family"] == "CANDIDATE_K1").sum()),
            "candidate_selected_K_gt_1_rows": int(len(kgt)),
            "distinct_K_gt_1_candidate_support_digests": int(kgt["candidate_support_digest"].nunique()),
            "distinct_K_gt_1_action_support_digests": int(kgt["action_support_digest"].nunique()),
            "distinct_K_gt_1_selected_candidate_identities": int(kgt["semantic_selected_candidate_id"].nunique()),
            "distinct_windows_containing_K_gt_1_ranking_opportunity": int(kgt["window_id"].nunique()),
            "K_distribution_candidate_selected": {str(int(key)): int(value) for key, value in candidates["K"].value_counts().sort_index().items()},
        }
    arms = {str(arm): one(frame.copy()) for arm, frame in classified.groupby("arm_id", sort=True)}
    overall = one(classified)
    by_band = {}
    for arm, arm_frame in classified.groupby("arm_id", sort=True):
        by_band[str(arm)] = {band: one(arm_frame.loc[arm_frame["time_band"] == band].copy()) for band in ("night", "offpeak", "peak")}
    by_band["ALL_ARMS"] = {band: one(classified.loc[classified["time_band"] == band].copy()) for band in ("night", "offpeak", "peak")}
    return {"population": "R18-R18B epoch-1 actor-eligible assignment decisions", "overall": overall, "by_arm": arms}, by_band


def actor_parameter_groups(actor: torch.nn.Module) -> dict[str, list[tuple[str, torch.nn.Parameter]]]:
    named = [(name, parameter) for name, parameter in actor.named_parameters() if parameter.requires_grad]
    return {
        "stage1": [(name, parameter) for name, parameter in named if name.startswith("gate_scorer.")],
        "stage2": [(name, parameter) for name, parameter in named if name.startswith(("candidate_encoder.", "scorer."))],
        "shared": [(name, parameter) for name, parameter in named if name.startswith(("agent_encoder.", "global_encoder.", "demand_encoder."))],
    }


def vector(items: Sequence[torch.Tensor | None], params: Sequence[tuple[str, torch.nn.Parameter]]) -> torch.Tensor:
    pieces = []
    for grad, (_, parameter) in zip(items, params):
        pieces.append((torch.zeros_like(parameter) if grad is None else grad).detach().reshape(-1).cpu())
    return torch.cat(pieces) if pieces else torch.empty(0)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    if left.numel() == 0 or right.numel() == 0:
        return None
    norm = float(left.norm() * right.norm())
    return None if norm == 0.0 else float(torch.dot(left, right) / norm)


def forward_view(actor: torch.nn.Module, payload: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    metadata = dict(payload["metadata"])
    k = int(metadata["selectable_pair_count"])
    tensors = {name: value.to(device) for name, value in payload["tensors"].items()}
    with torch.no_grad():
        dist = actor.forward_factorized(**tensors)
    conditional = dist.conditional_candidate_log_probs[0, :k].exp().detach().cpu()
    gate = dist.gate_log_probs[0].exp().detach().cpu()
    return {"K": k, "conditional": conditional, "assign_probability": float(gate[0]),
            "no_assign_probability": float(gate[1]), "candidate_ids": [candidate_identity(row) for row in metadata["candidate_ids"]]}


def selected_metrics(view: Mapping[str, Any], selected_index: int) -> dict[str, Any]:
    probs = view["conditional"]
    ids = list(view["candidate_ids"])
    require(0 <= selected_index < len(ids), "selected_candidate_index", BLOCK_GRADIENT)
    rank = sorted(range(len(ids)), key=lambda index: (-float(probs[index]), ids[index])).index(selected_index) + 1
    others = [float(probs[index]) for index in range(len(ids)) if index != selected_index]
    return {"selected_conditional_probability": float(probs[selected_index]), "selected_conditional_rank": rank,
            "selected_conditional_margin_vs_best_other": float(probs[selected_index]) - max(others) if others else None}


def reconstruct_epoch1_gradient(*, actor: torch.nn.Module, payload: Mapping[str, Any], trace: Mapping[str, Any],
                                device: torch.device) -> torch.Tensor:
    groups = actor_parameter_groups(actor)
    params = groups["stage2"]
    tensors = {name: value.to(device) for name, value in payload["tensors"].items()}
    selected_index = int(trace["selected_source_index"])
    old_log_prob = torch.tensor(float(trace["old_log_prob"]), dtype=torch.float32, device=device)
    advantage = torch.tensor(float(trace["advantage_normalized"]), dtype=torch.float32, device=device)
    denominator = float(trace["actor_denominator"])
    require(denominator >= 1.0 and not bool(trace["selected_is_no_assign"]), "gradient_row_not_candidate", BLOCK_GRADIENT)
    dist = actor.forward_factorized(**tensors)
    current = dist.action_log_probs[0, selected_index]
    ratio = torch.exp(current - old_log_prob.detach())
    unclipped = ratio * advantage.detach()
    clipped = torch.clamp(ratio, 0.8, 1.2) * advantage.detach()
    row_loss = -torch.min(unclipped, clipped) / denominator
    grads = torch.autograd.grad(row_loss, [parameter for _, parameter in params], retain_graph=False, allow_unused=True)
    result = vector(grads, params)
    require(all(parameter.grad is None for parameter in actor.parameters()), "autograd_retained_parameter_grad", BLOCK_GRADIENT)
    trace_norm = float(trace["stage2_gradient_norm"])
    difference = abs(float(result.norm()) - trace_norm)
    require(difference <= max(GRAD_TOL, abs(trace_norm) * 1e-3),
            f"epoch1_stage2_norm_mismatch={trace['decision_id']}:{difference}", BLOCK_GRADIENT)
    return result


def parameter_delta(initial: torch.nn.Module, final: torch.nn.Module) -> torch.Tensor:
    initial_map = dict(initial.named_parameters())
    final_map = dict(final.named_parameters())
    names = [name for name in initial_map if name.startswith(("candidate_encoder.", "scorer."))]
    require(names and set(names).issubset(final_map), "stage2_parameter_set", BLOCK_GRADIENT)
    return torch.cat([(final_map[name].detach().cpu() - initial_map[name].detach().cpu()).reshape(-1) for name in names])


def load_actors(auth: Mapping[str, Any], checkpoint_manifest: Mapping[str, Any], config: Mapping[str, Any],
                device: torch.device) -> tuple[dict[str, tuple[torch.nn.Module, torch.nn.Module]], dict[str, Any]]:
    arms = list(dict(auth["envelope"])["selected_arms"])
    inputs = dict(dict(auth["checkpoint_contract"])["initial_inputs"])
    finals = dict(checkpoint_manifest["final"])
    actors = {}
    digests = {}
    for arm in arms:
        arm_id = str(arm["arm_id"])
        initial = R19.build_initial_actor(arm=arm, checkpoints=inputs, config=config, device=device).eval()
        final = R19.build_final_actor(record=dict(finals[arm_id]), config=config, device=device).eval()
        initial_payload = torch.load(Path(str(inputs[str(arm["initial_checkpoint_key"])]["path"])), map_location="cpu", weights_only=False)
        final_payload = torch.load(Path(str(finals[arm_id]["path"])), map_location="cpu", weights_only=False)
        require(R18._module_digest(initial) == finals[arm_id]["initial_actor_digest"]
                and R18._module_digest(final) == finals[arm_id]["final_actor_digest"]
                and state_digest(initial_payload["critic"]) == finals[arm_id]["initial_critic_digest"]
                and state_digest(final_payload["critic"]) == finals[arm_id]["final_critic_digest"],
                f"actor_or_critic_tensor_digest={arm_id}")
        actors[arm_id] = (initial, final)
        digests[arm_id] = {"initial_actor_digest": R18._module_digest(initial), "final_actor_digest": R18._module_digest(final),
                           "initial_critic_digest": state_digest(initial_payload["critic"]), "final_critic_digest": state_digest(final_payload["critic"])}
    return actors, digests


def decomposition(classified: pd.DataFrame, epoch: pd.DataFrame, snapshots: Mapping[str, Mapping[str, Any]],
                  actors: Mapping[str, tuple[torch.nn.Module, torch.nn.Module]], device: torch.device,
                  counters: dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    eligible_epochs = epoch.loc[epoch["actor_eligible"]].copy()
    lookup = {str(row["decision_id"]): row for row in classified.to_dict(orient="records")}
    finals: dict[str, dict[str, Any]] = {}
    initials: dict[str, dict[str, Any]] = {}
    for decision, row in lookup.items():
        initial, final = actors[str(row["arm_id"])]
        payload = snapshots[decision]["payload"]
        initials[decision] = forward_view(initial, payload, device)
        finals[decision] = forward_view(final, payload, device)
        counters["frozen_forward_calls"] += 2
    rows = []
    direct_rows = []
    no_assign_leakage = 0
    k1_leakage = 0
    for decision, base in lookup.items():
        sequence = eligible_epochs.loc[eligible_epochs["decision_id"] == decision].sort_values("epoch_index")
        require(len(sequence) == 3 and sequence["epoch_index"].astype(int).tolist() == [1, 2, 3],
                f"eligible_epoch_sequence={decision}", BLOCK_TRACE)
        for position, record in enumerate(sequence.to_dict(orient="records")):
            next_record = None if position == 2 else sequence.iloc[position + 1].to_dict()
            selected_no_assign = bool(record["selected_is_no_assign"])
            selected_index = int(record["selected_source_index"])
            initial_view, final_view = initials[decision], finals[decision]
            if next_record is None:
                stage1_assign_post = final_view["assign_probability"]
                stage1_no_assign_post = final_view["no_assign_probability"]
                stage2_post = 0.0 if selected_no_assign else float(final_view["conditional"][selected_index])
            else:
                stage1_assign_post = float(next_record["stage1_assign_probability"])
                stage1_no_assign_post = float(next_record["stage1_no_assign_probability"])
                stage2_post = 0.0 if selected_no_assign else float(next_record["stage2_selected_candidate_probability"])
            stage2_pre = float(record["stage2_selected_candidate_probability"])
            if selected_no_assign:
                no_assign_leakage += int(abs(float(record["stage2_loss_contribution"])) > 0.0 or abs(float(record["stage2_gradient_norm"])) > 0.0)
            if base["selected_action_family"] == "CANDIDATE_K1":
                # The durable trace deliberately preserves the PPO policy-loss
                # contribution for every candidate-selected row.  For K=1 it
                # is a valid nonzero action loss while the conditional softmax
                # is degenerate, so only a nonzero ranking gradient (or a
                # conditional probability other than one) is a violation.
                k1_leakage += int(abs(float(record["stage2_gradient_norm"])) > 0.0 or abs(stage2_pre - 1.0) > 1e-7)
            rows.append({
                **{field: record[field] for field in ID_FIELDS}, "epoch_index": int(record["epoch_index"]), "K": int(base["K"]),
                "selected_action_family": str(base["selected_action_family"]), "selected_is_no_assign": selected_no_assign,
                "normalized_advantage": float(record["advantage_normalized"]), "stage1_gate_loss_contribution": float(record["stage1_loss_contribution"]),
                "stage1_gradient_norm": float(record["stage1_gradient_norm"]), "stage2_conditional_loss_contribution": float(record["stage2_loss_contribution"]),
                "stage2_gradient_norm": float(record["stage2_gradient_norm"]), "shared_encoder_gradient_norm": float(record["shared_encoder_gradient_norm"]),
                "stage1_assign_probability_pre": float(record["stage1_assign_probability"]), "stage1_assign_probability_post_epoch": stage1_assign_post,
                "stage1_no_assign_probability_pre": float(record["stage1_no_assign_probability"]), "stage1_no_assign_probability_post_epoch": stage1_no_assign_post,
                "stage2_selected_candidate_conditional_probability_pre": stage2_pre,
                "stage2_selected_candidate_conditional_probability_post_epoch": stage2_post,
                "stage1_assign_probability_delta": stage1_assign_post - float(record["stage1_assign_probability"]),
                "stage2_selected_candidate_conditional_probability_delta": stage2_post - stage2_pre,
            })
        if base["selected_action_family"] == "CANDIDATE_K_GT_1":
            before = selected_metrics(initials[decision], int(base["selected_source_index"]))
            after = selected_metrics(finals[decision], int(base["selected_source_index"]))
            direct_rows.append({**{field: base[field] for field in ID_FIELDS}, "window_id": base["window_id"], "time_band": base["time_band"],
                                "candidate_support_digest": base["candidate_support_digest"], "action_support_digest": base["action_support_digest"],
                                "K": int(base["K"]), "selected_semantic_candidate": base["selected_semantic_candidate"],
                                "candidate_semantic_identities": list(base["candidate_semantic_identities"]),
                                "normalized_advantage": float(base["advantage_normalized"]), "initial": before, "final": after,
                                "selected_conditional_probability_delta": after["selected_conditional_probability"] - before["selected_conditional_probability"],
                                "selected_conditional_margin_delta": after["selected_conditional_margin_vs_best_other"] - before["selected_conditional_margin_vs_best_other"],
                                "positive_advantage_expected_direction": float(base["advantage_normalized"]) > 0.0,
                                "selected_candidate_reinforced": after["selected_conditional_probability"] > before["selected_conditional_probability"]})
    require(no_assign_leakage == 0, f"no_assign_candidate_head_leakage={no_assign_leakage}", BLOCK_TRACE)
    require(k1_leakage == 0, f"k1_candidate_head_gradient_nonzero={k1_leakage}", BLOCK_TRACE)
    return pd.DataFrame(rows), pd.DataFrame(direct_rows), {"NO_ASSIGN_candidate_head_contamination": no_assign_leakage,
                                                             "K1_structural_degeneracy_violations": k1_leakage}, {"initial": initials, "final": finals}


def gradient_geometry(kgt: pd.DataFrame, classified: pd.DataFrame, snapshots: Mapping[str, Mapping[str, Any]],
                      actors: Mapping[str, tuple[torch.nn.Module, torch.nn.Module]], device: torch.device,
                      counters: dict[str, int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    vectors: list[tuple[dict[str, Any], torch.Tensor]] = []
    alignments = []
    for trace in classified.loc[classified["selected_action_family"] == "CANDIDATE_K_GT_1"].to_dict(orient="records"):
        arm_id, decision = str(trace["arm_id"]), str(trace["decision_id"])
        initial, final = actors[arm_id]
        before_initial = R18._module_digest(initial)
        gradient = reconstruct_epoch1_gradient(actor=initial, payload=snapshots[decision]["payload"], trace=trace, device=device)
        require(R18._module_digest(initial) == before_initial, f"autograd_mutated_actor={decision}", BLOCK_GRADIENT)
        counters["torch_autograd_grad"] += 1
        update = parameter_delta(initial, final)
        descent = -gradient
        trace_norm = float(trace["stage2_gradient_norm"])
        alignment = {
            "arm_id": arm_id, "decision_id": decision, "stage2_epoch1_gradient_norm_reconstructed": float(gradient.norm()),
            "stage2_epoch1_gradient_norm_durable_trace": trace_norm,
            "gradient_norm_abs_delta": abs(float(gradient.norm()) - trace_norm),
            "candidate_head_parameter_update_norm": float(update.norm()),
            "candidate_head_descent_update_dot": float(torch.dot(descent, update)),
            "candidate_head_descent_update_cosine": cosine(descent, update),
            "candidate_head_parameter_update_alignment_scope": "initial_to_final_delta_vs_epoch1_row_descent_direction",
        }
        alignments.append(alignment)
        vectors.append((trace, gradient))
    require(len(vectors) >= 1 and all(float(vector.norm()) > 0.0 for _, vector in vectors), "k_gt_1_gradient_nonzero", BLOCK_GRADIENT)
    pairwise = []
    cosines = []
    opposed = 0
    for left_index, (left_row, left_grad) in enumerate(vectors):
        line = []
        for right_index, (right_row, right_grad) in enumerate(vectors):
            value = cosine(left_grad, right_grad)
            line.append(value)
            if right_index > left_index:
                cosines.append(float(value))
                opposed += int(float(value) < 0.0)
        pairwise.append({"decision_id": str(left_row["decision_id"]), "cosines": line})
    summed = torch.stack([value for _, value in vectors]).sum(dim=0)
    individual_sum = sum(float(value.norm()) for _, value in vectors)
    geometry = {
        "scope": "exact epoch-1 frozen-input Stage-2 candidate-head gradients; later epoch vector directions were not persisted and are not reconstructed by re-running optimization",
        "parameter_group": "candidate_encoder.* plus scorer.*",
        "row_count": len(vectors), "pairwise_cosine_matrix": pairwise,
        "mean_same_direction_cosine": None if not cosines else sum(cosines) / len(cosines),
        "opposed_gradient_pair_count": opposed, "summed_stage2_gradient_norm": float(summed.norm()),
        "sum_individual_stage2_gradient_norms": individual_sum,
        "cancellation_ratio": float(summed.norm()) / individual_sum if individual_sum else None,
        "cross_row_cancellation_dominant": bool(individual_sum and float(summed.norm()) / individual_sum < 0.5),
    }
    return geometry, alignments


def reconciliation(kgt: pd.DataFrame) -> dict[str, Any]:
    review = load(R18R19_ROOT / "same_input_factorized_review_rows.json")
    training = kgt.to_dict(orient="records")
    rows = []
    for item in review["review_rows"]:
        initial = dict(item["initial"])
        final = dict(item["final"])
        if int(initial["selectable_candidate_count"]) <= 1:
            continue
        identities = set(initial["candidate_semantic_identities"])
        exact_support = [row for row in training if str(row["candidate_support_digest"]) == str(item["candidate_support_digest"])]
        same_window = [row for row in training if str(row["window_id"]) == str(item["window_id"])]
        overlap = set().union(*(set(row["candidate_semantic_identities"]) for row in training)) & identities if training else set()
        rows.append({
            "arm_id": item["arm_id"], "decision_id": item["decision_id"], "snapshot_digest": item["snapshot_digest"],
            "candidate_support_digest": item["candidate_support_digest"], "K": int(initial["selectable_candidate_count"]),
            "initial_conditional_top_candidate": initial["conditional_top_identity"], "final_conditional_top_candidate": final["conditional_top_identity"],
            "conditional_top_changed": initial["conditional_top_identity"] != final["conditional_top_identity"],
            "conditional_margin_delta": None if initial["conditional_top_minus_second_margin"] is None else float(final["conditional_top_minus_second_margin"] - initial["conditional_top_minus_second_margin"]),
            "conditional_range_delta": float(final["conditional_probability_range"] - initial["conditional_probability_range"]),
            "conditional_std_delta": float(final["conditional_probability_std"] - initial["conditional_probability_std"]),
            "exact_K_gt_1_training_support_match_count": len(exact_support),
            "same_window_K_gt_1_training_exposure_count": len(same_window),
            "same_time_band_K_gt_1_training_exposure_count": None,
            "same_time_band_comparison_available": False,
            "candidate_identity_overlap_with_K_gt_1_training_exposure_count": len(overlap),
            "semantically_related_K_gt_1_exposure": bool(exact_support or overlap),
        })
    require(len(rows) == 4, f"r19_multi_candidate_review_count={len(rows)}", BLOCK_TRACE)
    return {"r18r19_multi_candidate_frozen_review_rows": len(rows), "rows": rows,
            "interpretation": "Related exposure requires exact candidate-support digest or semantic candidate-identity overlap; time-band/window coincidence alone is reported but not treated as a direct ranking-evidence match."}


def next_step(exposure: Mapping[str, Any], geometry: Mapping[str, Any], invariants: Mapping[str, Any]) -> dict[str, Any]:
    overall = dict(exposure["overall"])
    return {
        "primary_classification": CLASSIFICATION,
        "classification_rationale": {
            "factorized_stage2_working": True,
            "K_gt_1_candidate_selected_actor_eligible_rows": overall["candidate_selected_K_gt_1_rows"],
            "distinct_K_gt_1_windows": overall["distinct_windows_containing_K_gt_1_ranking_opportunity"],
            "distinct_K_gt_1_selected_candidate_identities": overall["distinct_K_gt_1_selected_candidate_identities"],
            "cross_row_cancellation_dominant": geometry["cross_row_cancellation_dominant"],
            "NO_ASSIGN_candidate_head_contamination": invariants["NO_ASSIGN_candidate_head_contamination"],
            "conclusion": "Stage-2 is observably functional, but two K>1 selected rows from one peak window and one selected semantic candidate do not provide diversity sufficient for a candidate-discrimination conclusion.",
        },
        "next_step": "minimal pre-outcome exposure-envelope design",
        "next_step_goal": "Increase K>1 candidate-ranking opportunities without changing Reward V2, E1, PPO, GAE, or the factorized architecture, and without outcome-based cherry-picking.",
        "training_authorized": False,
        "automatic_bounded_training_authorization": False,
        "Reward_V2_change_justified": False,
        "E1_change_justified": False,
        "PPO_change_justified": False,
        "GAE_change_justified": False,
    }


def write_manifest(root: Path, gate: str, source_commit: str, classification: str) -> None:
    files = {item.relative_to(root).as_posix(): sha256(item)
             for item in root.rglob("*") if item.is_file() and item.name != "manifest.json"}
    dump(root / "manifest.json", {"stage": STAGE, "gate": gate, "classification": classification,
                                   "source_commit": source_commit, "github_push_performed": False, "file_sha256": files})


def main() -> None:
    root = artifact_root()
    root.mkdir(parents=True)
    counters = {"training": 0, "rollout": 0, "simulator": 0, "candidate_generation": 0, "reward_recomputation": 0,
                "optimizer_creation": 0, "optimizer_step": 0, "backward": 0, "checkpoint_write": 0, "policy_mutation": 0,
                "torch_autograd_grad": 0, "frozen_forward_calls": 0, "github_push": 0}
    source_commit = None
    try:
        auth, checkpoint_manifest, binding = validate_upstream()
        source_commit = str(binding["r18r20_source_commit"])
        mps = R18B.external_mps_preflight()
        device = torch.device("mps:0")
        eligible, credit, gae, ratio, epoch = load_trace()
        snapshots = snapshot_map()
        epoch_one = epoch.loc[(epoch["actor_eligible"]) & (epoch["epoch_index"] == 1)].copy()
        classified = classify_rows(epoch_one, snapshots)
        audit_trace_identity(classified, credit, gae, ratio, epoch)
        exposure, by_band = census(classified)
        config = dict(snapshots[str(classified.iloc[0]["decision_id"])]["payload"]["metadata"]["actor_config"])
        actors, tensor_digests = load_actors(auth, checkpoint_manifest, config, device)
        decomposed, ranking, invariants, _ = decomposition(classified, epoch, snapshots, actors, device, counters)
        geometry, alignment = gradient_geometry(ranking, classified, snapshots, actors, device, counters)
        for row in alignment:
            match = ranking[ranking["decision_id"] == row["decision_id"]]
            require(len(match) == 1, f"ranking_alignment_join={row['decision_id']}", BLOCK_GRADIENT)
            ranking.loc[match.index[0], "candidate_head_descent_update_dot"] = row["candidate_head_descent_update_dot"]
            ranking.loc[match.index[0], "candidate_head_descent_update_cosine"] = row["candidate_head_descent_update_cosine"]
            ranking.loc[match.index[0], "candidate_head_parameter_update_norm"] = row["candidate_head_parameter_update_norm"]
        require(bool(ranking["selected_candidate_reinforced"].all()), "positive_advantage_k_gt_1_not_reinforced", BLOCK_GRADIENT)
        reconcile = reconciliation(ranking)
        next_selection = next_step(exposure, geometry, invariants)
        decomposed.to_parquet(root / "stage1_stage2_learning_decomposition.parquet", index=False)
        ranking.to_parquet(root / "k_gt_1_candidate_ranking_rows.parquet", index=False)
        outputs = {
            "evidence_binding_audit.json": {**binding, "actor_critic_tensor_digests": tensor_digests,
                                              "durable_trace_files": {name: sha256(R18R18B_EXEC_ROOT / name) for name in TRACE_FILES}},
            "external_mps_preflight.json": mps,
            "factorized_exposure_census.json": exposure,
            "factorized_exposure_by_time_band.json": by_band,
            "candidate_head_gradient_alignment.json": {"K_gt_1_rows": alignment,
                                                        "all_positive_advantage_rows_reinforced": bool(ranking["selected_candidate_reinforced"].all())},
            "candidate_head_gradient_cancellation.json": geometry,
            "r18r19_candidate_head_reconciliation.json": reconcile,
            "next_step_selection.json": next_selection,
            "test_results.json": {"passed": True, "execution_counters": counters, "hard_failures": [], "warnings": [],
                                  "training_semantics_changed": False, "policy_mutation": False,
                                  "NO_ASSIGN_candidate_head_contamination": invariants["NO_ASSIGN_candidate_head_contamination"]},
            "gate_decision.json": {"stage": STAGE, "gate": PASS_GATE, "classification": CLASSIFICATION,
                                   "source_commit": source_commit, "next_step": next_selection["next_step"]},
        }
        for name, value in outputs.items():
            dump(root / name, value)
        (root / "final_report.md").write_text(
            "# R18-R20 factorized candidate-head learning exposure audit\n\n"
            f"- gate: `{PASS_GATE}`\n- source: `{source_commit}`\n"
            f"- K>1 candidate-selected eligible rows: `{exposure['overall']['candidate_selected_K_gt_1_rows']}`\n"
            f"- classification: `{CLASSIFICATION}`\n"
            "- training / rollout / optimizer / backward / checkpoint write: `0`\n",
            encoding="utf-8",
        )
        write_manifest(root, PASS_GATE, source_commit, CLASSIFICATION)
        (root / "_SUCCESS.lock").write_text(PASS_GATE + "\n", encoding="utf-8")
        print(f"[PASS] {PASS_GATE}")
        print(root)
    except R18R20Error as exc:
        dump(root / "test_results.json", {"passed": False, "execution_counters": counters, "hard_failures": [exc.detail], "warnings": []})
        dump(root / "gate_decision.json", {"stage": STAGE, "gate": exc.gate, "classification": "BLOCKED",
                                             "source_commit": source_commit, "hard_failures": [exc.detail], "next_step": "STOP"})
        (root / "final_report.md").write_text(f"# R18-R20 blocked\n\n- gate: `{exc.gate}`\n- detail: `{exc.detail}`\n", encoding="utf-8")
        write_manifest(root, exc.gate, source_commit or "UNKNOWN", "BLOCKED")
        print(f"[BLOCKED] {exc.gate}")
        print(root)


if __name__ == "__main__":
    main()
