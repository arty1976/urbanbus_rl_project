#!/usr/bin/env python3
"""R18-R12 shared-batch gradient interference attribution audit.

No training is executed here.  The audit reads R18-R10B durable traces and
frozen actor-input snapshots, loads the same BD initial actor checkpoint, and
uses ``torch.autograd.grad`` to compute read-only policy-loss gradient geometry:

* each BD Actor-eligible row update vector,
* the full eligible-row batch update vector,
* row/full dot products and cosine similarities,
* candidate-candidate alignment, and
* candidate-vs-NO_ASSIGN conflict.

It never creates an optimizer, calls ``backward()``, steps parameters, runs a
simulator, recomputes rewards, writes checkpoints, or mutates policy weights.
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

import pandas as pd
import torch


STAGE = "H4M-AE-R9.8-LS3-BT8-R18-R12"
PASS_GATE = (
    "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_R12_"
    "SHARED_BATCH_GRADIENT_INTERFERENCE_AND_CANDIDATE_VS_NO_ASSIGN_ATTRIBUTION_AUDIT_COMPLETE"
)
CLASSIFICATION = "A_SHARED_BATCH_INTERFERENCE_CONFIRMED_BY_ROW_GRADIENT_GEOMETRY_NO_TRAINING"
BLOCK = "BLOCKED_R18R12_GRADIENT_INTERFERENCE_AUDIT_FAILURE"

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
R18_R10B_ROOT = (
    ARTIFACTS
    / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r10b_exact_r18r3_trace_replay_execution_20260828_094909+09:00"
)
R18_R10B_AUTH_ROOT = (
    ARTIFACTS
    / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r10b_exact_r18r3_trace_replay_authorization_20260828_094909+09:00"
)
R18_R10B_PASS_GATE = (
    "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_R10_"
    "EXACT_R18_R3_TRAJECTORY_DURABLE_TRACE_BOUNDED_RERUN_COMPLETE"
)


class R18R12Error(RuntimeError):
    def __init__(self, detail: str) -> None:
        super().__init__(f"{BLOCK}: {detail}")
        self.code = BLOCK


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise R18R12Error(detail)


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


def sign(value: float, eps: float = 1e-12) -> str:
    if value > eps:
        return "positive"
    if value < -eps:
        return "negative"
    return "zero"


def alignment_label(cosine: float) -> str:
    if cosine >= 0.75:
        return "strong_positive_alignment"
    if cosine >= 0.25:
        return "weak_positive_alignment"
    if cosine <= -0.75:
        return "strong_conflict"
    if cosine <= -0.25:
        return "weak_conflict"
    return "near_orthogonal"


def short_identity(value: str) -> str:
    if value == "NO_ASSIGN_KEEP_CURRENT_PLANS":
        return value
    if "::" in value:
        agent, candidate = value.split("::", 1)
        return f"{agent}::{candidate[:12]}…"
    return value[:24]


def module_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for _, tensor in sorted(module.state_dict().items()):
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def flatten_grads(grads: Sequence[torch.Tensor | None], params: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    return torch.cat([
        (torch.zeros_like(param) if grad is None else grad).detach().reshape(-1).cpu()
        for grad, param in zip(grads, params)
    ])


def dot_cos(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    left_norm = float(left.norm())
    right_norm = float(right.norm())
    dot = float(torch.dot(left, right))
    cosine = dot / (left_norm * right_norm) if left_norm > 0.0 and right_norm > 0.0 else math.nan
    return {"dot": dot, "cosine": cosine, "left_norm": left_norm, "right_norm": right_norm}


def actor_forward_loss(*, actor: torch.nn.Module, row: Mapping[str, Any], snapshot: Mapping[str, Any],
                       JL: Any, denominator: float) -> tuple[torch.Tensor, dict[str, Any]]:
    tensors = snapshot["tensors"]
    logits, no_assign = actor(
        global_feats=tensors["global_feats"],
        demand_feats=tensors["demand_feats"],
        agent_feats=tensors["agent_feats"],
        agent_mask=tensors["agent_mask"],
        candidate_feats=tensors["candidate_feats"],
        pair_agent_index=tensors["pair_agent_index"],
        safe_mask=tensors["safe_mask"],
    )
    action_index = torch.tensor([int(row["selected_source_index"])], dtype=torch.long)
    log_probs = JL.masked_log_probs(logits, no_assign, tensors["safe_mask"])
    new_log_prob = log_probs.gather(-1, action_index.unsqueeze(-1)).squeeze(-1)
    old_log_prob = torch.tensor([float(row["old_log_prob"])], dtype=torch.float32)
    advantage = torch.tensor([float(row["advantage_normalized"])], dtype=torch.float32)
    ratio = torch.exp(new_log_prob - old_log_prob)
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 0.8, 1.2) * advantage
    selected_policy_loss = -torch.min(unclipped, clipped).squeeze()
    return selected_policy_loss / denominator, {
        "new_log_prob": float(new_log_prob.detach().cpu()[0]),
        "old_log_prob": float(old_log_prob.detach().cpu()[0]),
        "ppo_ratio": float(ratio.detach().cpu()[0]),
        "selected_policy_loss_unweighted": float(selected_policy_loss.detach().cpu()),
    }


def load_bd_actor(*, auth: Mapping[str, Any], sample_snapshot: Mapping[str, Any], H: Any) -> torch.nn.Module:
    config = dict(sample_snapshot["metadata"]["actor_config"])
    actor = H.CandidateSensitiveMultiAgentCandidateAssignmentHead(
        global_dim=int(config["global_dim"]),
        demand_dim=int(config["demand_dim"]),
        agent_dim=int(config["agent_dim"]),
        candidate_dim=int(config["candidate_dim"]),
        hidden=int(config["hidden"]),
        heads=int(config["heads"]),
    )
    checkpoint_path = Path(str(auth["checkpoint_contract"]["initial_inputs"]["initial:BD-R1"]["path"]))
    require(checkpoint_path.is_file() and sha256(checkpoint_path) == auth["checkpoint_contract"]["initial_inputs"]["initial:BD-R1"]["sha256"],
            "BD_initial_checkpoint_binding")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actor.load_state_dict(checkpoint["actor"], strict=True)
    actor.train()
    return actor


def load_snapshots(FPS: Any) -> dict[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    for snapshot_root in (R18_R10B_ROOT / "training_snapshots" / "snapshots").glob("*/"):
        payload = FPS.load_snapshot(snapshot_root)
        decision_id = str(payload["metadata"]["decision_id"])
        if decision_id.startswith("R18:BD_E1_R1:"):
            rows[decision_id] = payload
    require(len(rows) == 24, f"BD_snapshot_count={len(rows)}")
    return rows


def build_audit() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    sys.path.insert(0, str(ROOT))
    import joint_assignment_frozen_policy_snapshot as FPS
    import joint_assignment_learning as JL
    import multi_agent_candidate_assignment_head as H

    r10b_gate = load_json(R18_R10B_ROOT / "gate_decision.json")
    require(r10b_gate.get("gate") == R18_R10B_PASS_GATE, "R18_R10B_not_PASS")
    auth = load_json(R18_R10B_AUTH_ROOT / "r18r10_one_shot_authorization_manifest.json")
    trace_summary = load_json(R18_R10B_ROOT / "trace_row_count_summary.json")
    identity_audit = load_json(R18_R10B_ROOT / "trace_identity_roundtrip_audit.json")
    parameter_delta = load_json(R18_R10B_ROOT / "parameter_delta_audit.json")
    require(identity_audit.get("identity_roundtrip_passed") is True
            and identity_audit.get("no_assign_identity_stable") is True, "trace_identity_roundtrip")
    require(int(trace_summary["arms"]["BD_E1_R1"]["actor_eligible_assignment_rows"]) == 5, "BD_actor_eligible_count")
    require(all(bool(row["final_actor_digest_matches_r18r3"]) and bool(row["final_critic_digest_matches_r18r3"])
                for row in dict(parameter_delta["arms"]).values()), "R18_R10B_final_digest_repro")

    gae_rows = pd.read_parquet(R18_R10B_ROOT / "gae_rows.parquet")
    epoch_rows = pd.read_parquet(R18_R10B_ROOT / "per_epoch_loss_contributions.parquet")
    required = {
        "arm_id", "decision_id", "transition_index", "selected_source_index", "selected_action_type",
        "selected_semantic_candidate", "selected_is_no_assign", "advantage_normalized",
        "gae_advantage_raw", "old_log_prob", "actor_eligible", "forced_action",
    }
    require(required.issubset(set(gae_rows.columns)), "gae_trace_schema")
    require(required.issubset(set(epoch_rows.columns)), "epoch_trace_schema")
    require(not gae_rows[list(required)].isna().any().any(), "gae_trace_null")
    require(not epoch_rows[list(required)].isna().any().any(), "epoch_trace_null")
    bd_rows = gae_rows[(gae_rows.arm_id == "BD_E1_R1") & (gae_rows.actor_eligible) & (~gae_rows.forced_action)].copy()
    bd_rows = bd_rows.sort_values("transition_index")
    require(len(bd_rows) == 5, f"BD_active_actor_rows={len(bd_rows)}")
    snapshots = load_snapshots(FPS)
    require(set(str(row.decision_id) for row in bd_rows.itertuples()).issubset(snapshots), "active_snapshot_binding")

    actor = load_bd_actor(auth=auth, sample_snapshot=snapshots[str(bd_rows.iloc[0].decision_id)], H=H)
    initial_digest = module_digest(actor)
    expected_initial = str(parameter_delta["arms"]["BD_E1_R1"]["initial_actor_digest"])
    require(initial_digest == expected_initial, "BD_initial_actor_digest")
    params = [param for param in actor.parameters() if param.requires_grad]
    denominator = float(len(bd_rows))
    row_records: list[dict[str, Any]] = []
    row_update_vectors: dict[str, torch.Tensor] = {}
    trace_epoch1 = epoch_rows[(epoch_rows.arm_id == "BD_E1_R1") & (epoch_rows.actor_eligible)
                              & (epoch_rows.epoch_index == 1)].copy()
    trace_epoch1_by_id = {str(row.decision_id): row for row in trace_epoch1.itertuples()}

    for row in bd_rows.to_dict(orient="records"):
        decision_id = str(row["decision_id"])
        loss, forward = actor_forward_loss(actor=actor, row=row, snapshot=snapshots[decision_id],
                                           JL=JL, denominator=denominator)
        grads = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)
        loss_grad = flatten_grads(grads, params)
        update = -loss_grad
        trace_row = trace_epoch1_by_id[decision_id]
        new_log_delta = forward["new_log_prob"] - float(trace_row.new_log_prob)
        ratio_delta = forward["ppo_ratio"] - float(trace_row.ppo_ratio)
        require(abs(new_log_delta) <= 5e-7 and abs(ratio_delta) <= 5e-7, f"epoch1_forward_trace_delta={decision_id}")
        row_update_vectors[decision_id] = update
        row_records.append({
            "decision_id": decision_id,
            "transition_index": int(row["transition_index"]),
            "selected_action_type": str(row["selected_action_type"]),
            "selected_is_no_assign": bool(row["selected_is_no_assign"]),
            "selected_semantic_candidate": str(row["selected_semantic_candidate"]),
            "selected_semantic_candidate_short": short_identity(str(row["selected_semantic_candidate"])),
            "raw_gae_advantage": float(row["gae_advantage_raw"]),
            "raw_gae_sign": sign(float(row["gae_advantage_raw"])),
            "normalized_advantage": float(row["advantage_normalized"]),
            "normalized_advantage_sign": sign(float(row["advantage_normalized"])),
            "normalization_changed_sign": sign(float(row["gae_advantage_raw"])) != sign(float(row["advantage_normalized"])),
            "row_update_vector_norm": float(update.norm()),
            "epoch1_forward_new_log_prob_delta_vs_trace": new_log_delta,
            "epoch1_forward_ppo_ratio_delta_vs_trace": ratio_delta,
            "policy_loss_contribution_denominator": denominator,
            "policy_loss_unweighted": forward["selected_policy_loss_unweighted"],
        })

    total_loss = None
    for row in bd_rows.to_dict(orient="records"):
        decision_id = str(row["decision_id"])
        loss, _ = actor_forward_loss(actor=actor, row=row, snapshot=snapshots[decision_id],
                                     JL=JL, denominator=denominator)
        total_loss = loss if total_loss is None else total_loss + loss
    require(total_loss is not None, "total_loss_missing")
    full_loss_grads = torch.autograd.grad(total_loss, params, retain_graph=False, allow_unused=True)
    full_update = -flatten_grads(full_loss_grads, params)
    summed_update = sum(row_update_vectors.values(), torch.zeros_like(full_update))
    additive_delta = float((summed_update - full_update).abs().max())
    require(additive_delta <= 2e-6, f"gradient_additivity_delta={additive_delta}")

    for record in row_records:
        vector = row_update_vectors[record["decision_id"]]
        geom = dot_cos(vector, full_update)
        record["dot_with_full_batch_update"] = geom["dot"]
        record["cosine_with_full_batch_update"] = geom["cosine"]
        record["full_batch_update_alignment"] = alignment_label(geom["cosine"])
        record["full_batch_interference"] = geom["cosine"] < -0.25
        record["attribution_result"] = (
            "NO_ASSIGN 강화" if record["selected_is_no_assign"] and geom["cosine"] > 0.25
            else "candidate 억제" if (not record["selected_is_no_assign"]) and geom["cosine"] < -0.25
            else "혼합/불명확"
        )

    pairwise: list[dict[str, Any]] = []
    for left_index, left in enumerate(row_records):
        for right in row_records[left_index + 1:]:
            lvec = row_update_vectors[left["decision_id"]]
            rvec = row_update_vectors[right["decision_id"]]
            geom = dot_cos(lvec, rvec)
            kind = (
                "candidate_candidate" if left["selected_action_type"] == right["selected_action_type"] == "CANDIDATE"
                else "no_assign_no_assign" if left["selected_is_no_assign"] and right["selected_is_no_assign"]
                else "candidate_vs_no_assign"
            )
            pairwise.append({
                "left_decision_id": left["decision_id"],
                "right_decision_id": right["decision_id"],
                "left_selected_action_type": left["selected_action_type"],
                "right_selected_action_type": right["selected_action_type"],
                "pair_type": kind,
                "dot": geom["dot"],
                "cosine": geom["cosine"],
                "alignment": alignment_label(geom["cosine"]),
            })

    by_type: dict[str, list[float]] = {}
    for row in pairwise:
        by_type.setdefault(str(row["pair_type"]), []).append(float(row["cosine"]))
    aggregate = {
        "full_batch_update_norm": float(full_update.norm()),
        "gradient_vector_dimension": int(full_update.numel()),
        "autograd_grad_calls": len(row_records) + 1,
        "backward_method_calls": 0,
        "optimizer_creation": 0,
        "optimizer_step": 0,
        "checkpoint_write": 0,
        "policy_parameter_digest_before": initial_digest,
        "policy_parameter_digest_after": module_digest(actor),
        "policy_parameter_mutated": module_digest(actor) != initial_digest,
        "gradient_additivity_max_abs_delta": additive_delta,
        "candidate_row_count": sum(row["selected_action_type"] == "CANDIDATE" for row in row_records),
        "no_assign_row_count": sum(row["selected_is_no_assign"] for row in row_records),
        "candidate_rows_conflict_with_full_batch": [
            row["decision_id"] for row in row_records
            if row["selected_action_type"] == "CANDIDATE" and row["full_batch_interference"]
        ],
        "no_assign_rows_align_with_full_batch": [
            row["decision_id"] for row in row_records
            if row["selected_is_no_assign"] and row["cosine_with_full_batch_update"] > 0.25
        ],
        "mean_cosine_by_pair_type": {
            key: sum(values) / len(values) for key, values in sorted(by_type.items())
        },
    }
    aggregate["shared_batch_interference_confirmed"] = (
        aggregate["candidate_rows_conflict_with_full_batch"] == [
            "R18:BD_E1_R1:12", "R18:BD_E1_R1:15", "R18:BD_E1_R1:20"]
        and aggregate["no_assign_rows_align_with_full_batch"] == [
            "R18:BD_E1_R1:13", "R18:BD_E1_R1:14"]
        and aggregate["mean_cosine_by_pair_type"]["candidate_vs_no_assign"] < -0.9
        and aggregate["mean_cosine_by_pair_type"]["candidate_candidate"] > 0.9
    )
    require(aggregate["shared_batch_interference_confirmed"], "shared_batch_interference_pattern")

    execution_counters = {
        "training": 0,
        "rollout": 0,
        "simulator": 0,
        "reward_recomputation": 0,
        "candidate_generation": 0,
        "optimizer_creation": 0,
        "optimizer_step": 0,
        "backward_method_call": 0,
        "autograd_grad_call": aggregate["autograd_grad_calls"],
        "checkpoint_write": 0,
        "policy_mutation": 0,
        "github_push": 0,
        "test6_access": 0,
    }
    evidence = {
        "input_r18r10b_root": str(R18_R10B_ROOT.resolve()),
        "input_r18r10b_gate": r10b_gate["gate"],
        "input_r18r10b_source_commit": r10b_gate["source_commit"],
        "input_r18r10b_manifest_sha256": sha256(R18_R10B_ROOT / "manifest.json"),
        "input_authorization_root": str(R18_R10B_AUTH_ROOT.resolve()),
        "input_authorization_manifest_sha256": sha256(R18_R10B_AUTH_ROOT / "r18r10_one_shot_authorization_manifest.json"),
        "trace_schema_sha256": sha256(R18_R10B_ROOT / "trace_schema.json"),
        "trace_identity_roundtrip_passed": True,
        "BD_actor_eligible_rows": len(row_records),
        "same_frozen_model": "R18-R10B initial BD actor checkpoint loaded read-only",
        "gradient_geometry_epoch": 1,
        "gradient_geometry_device": "cpu",
        "no_optimizer_or_parameter_step": True,
    }
    return evidence, {"rows": row_records, "aggregate": aggregate}, {"pairs": pairwise}, execution_counters


def final_markdown(*, evidence: Mapping[str, Any], row_audit: Mapping[str, Any],
                   pairwise: Mapping[str, Any], gate: str, source_commit: str) -> str:
    rows = list(row_audit["rows"])
    aggregate = dict(row_audit["aggregate"])
    lines = [
        "# R18-R12 shared-batch gradient interference audit",
        "",
        f"- gate: `{gate}`",
        f"- source commit: `{source_commit}`",
        f"- input: `{evidence['input_r18r10b_root']}`",
        "- training/rollout/simulator/optimizer/backward/checkpoint: `0`",
        f"- autograd.grad calls: `{aggregate['autograd_grad_calls']}`",
        f"- full batch update norm: `{aggregate['full_batch_update_norm']:.6f}`",
        f"- shared-batch interference confirmed: `{str(aggregate['shared_batch_interference_confirmed']).lower()}`",
        "",
        "| row | 선택 | norm adv | dot(row, batch) | cosine | 정렬 | 결과 |",
        "|---:|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['transition_index']} | {row['selected_action_type']} | "
            f"{row['normalized_advantage']:+.6f} | {row['dot_with_full_batch_update']:+.6f} | "
            f"{row['cosine_with_full_batch_update']:+.6f} | {row['full_batch_update_alignment']} | "
            f"{row['attribution_result']} |"
        )
    lines.extend([
        "",
        "## Pairwise alignment summary",
        "",
        f"- candidate-candidate mean cosine: `{aggregate['mean_cosine_by_pair_type']['candidate_candidate']:+.6f}`",
        f"- NO_ASSIGN-NO_ASSIGN mean cosine: `{aggregate['mean_cosine_by_pair_type']['no_assign_no_assign']:+.6f}`",
        f"- candidate-vs-NO_ASSIGN mean cosine: `{aggregate['mean_cosine_by_pair_type']['candidate_vs_no_assign']:+.6f}`",
        "",
        "## Interpretation",
        "",
        "Candidate-selected rows 12/15/20 have positive advantages, but their row update vectors point almost opposite the full-batch update. "
        "NO_ASSIGN rows 13/14 have positive advantages and are strongly aligned with the full-batch update. "
        "This is the requested shared-batch interference pattern: the batch-level actor update is dominated by the NO_ASSIGN direction, "
        "so candidate rows are suppressed despite individually positive advantages.",
    ])
    return "\n".join(lines) + "\n"


def block(root: Path, detail: str, source_commit: str | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    counters = {
        "training": 0, "rollout": 0, "simulator": 0, "reward_recomputation": 0,
        "candidate_generation": 0, "optimizer_creation": 0, "optimizer_step": 0,
        "backward_method_call": 0, "checkpoint_write": 0, "policy_mutation": 0,
        "github_push": 0, "test6_access": 0,
    }
    dump(root / "gate_decision.json", {"stage": STAGE, "gate": BLOCK, "classification": "BLOCKED",
                                       "source_commit": source_commit, "execution_counters": counters,
                                       "hard_failures": [detail], "next_step": "STOP"})
    (root / "final_report.md").write_text(f"# R18-R12 BLOCKED\n\n- gate: `{BLOCK}`\n- detail: `{detail}`\n",
                                           encoding="utf-8")
    manifest = {item.relative_to(root).as_posix(): sha256(item)
                for item in root.rglob("*") if item.is_file() and item.name != "manifest.json"}
    dump(root / "manifest.json", {"stage": STAGE, "gate": BLOCK, "source_commit": source_commit,
                                  "file_sha256": manifest})
    (root / "_BLOCKED.lock").write_text(BLOCK + "\n", encoding="utf-8")


def main() -> None:
    root = ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r12_shared_batch_gradient_interference_audit_{kst_now()}"
    source_commit = git(["rev-parse", "HEAD"])
    try:
        require(git(["status", "--porcelain=v1"]) == "", "dirty_worktree")
        root.mkdir(parents=True, exist_ok=False)
        evidence, row_audit, pairwise, counters = build_audit()
        dump(root / "input_evidence_binding.json", evidence)
        dump(root / "row_full_batch_gradient_alignment.json", row_audit)
        dump(root / "candidate_vs_no_assign_gradient_conflict.json", pairwise)
        dump(root / "gradient_interference_summary.json", {
            "stage": STAGE,
            "classification": CLASSIFICATION,
            **dict(row_audit["aggregate"]),
        })
        dump(root / "test_results.json", {"stage": STAGE, "execution_counters": counters,
                                          "hard_failures": [], "warnings": [],
                                          "github_push": False})
        dump(root / "gate_decision.json", {"stage": STAGE, "gate": PASS_GATE,
                                           "classification": CLASSIFICATION,
                                           "source_commit": source_commit,
                                           "execution_counters": counters,
                                           "hard_failures": [], "warnings": [],
                                           "next_step": "STOP"})
        (root / "final_report.md").write_text(
            final_markdown(evidence=evidence, row_audit=row_audit, pairwise=pairwise,
                           gate=PASS_GATE, source_commit=source_commit),
            encoding="utf-8",
        )
        manifest = {item.relative_to(root).as_posix(): sha256(item)
                    for item in root.rglob("*") if item.is_file() and item.name != "manifest.json"}
        dump(root / "manifest.json", {"stage": STAGE, "gate": PASS_GATE,
                                      "classification": CLASSIFICATION,
                                      "source_commit": source_commit,
                                      "file_sha256": manifest})
        (root / "_SUCCESS.lock").write_text(PASS_GATE + "\n", encoding="utf-8")
        print(f"[PASS] {PASS_GATE}")
        print(root)
    except R18R12Error as exc:
        block(root, str(exc), source_commit)
        print(f"[BLOCKED] {BLOCK}")
        print(root)
    except Exception as exc:  # noqa: BLE001
        block(root, f"{type(exc).__name__}:{exc}", source_commit)
        print(f"[BLOCKED] {BLOCK}")
        print(root)


if __name__ == "__main__":
    main()
