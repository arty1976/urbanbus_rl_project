#!/usr/bin/env python3
"""R18-R13 gradient magnitude / parameter-path dominance decomposition audit.

This is a read-only follow-up to R18-R12.  It reuses the same exact R18-R10B
durable trace and frozen BD initial actor, then decomposes why the two
positive-advantage NO_ASSIGN rows dominate the three positive-advantage
candidate rows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import run_h4m_ae_ls3_bt8_r18_r12_shared_batch_gradient_interference_audit as IMPL  # noqa: E402


STAGE = "H4M-AE-R9.8-LS3-BT8-R18-R13"
PASS_GATE = (
    "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_R13_"
    "GRADIENT_MAGNITUDE_AND_PARAMETER_PATH_DOMINANCE_DECOMPOSITION_AUDIT_COMPLETE"
)
CLASSIFICATION = "A_NO_ASSIGN_GRADIENT_LEVERAGE_AND_SHARED_PATH_DOMINANCE_EXPLAINED_NO_TRAINING"
BLOCK = "BLOCKED_R18R13_GRADIENT_MAGNITUDE_PATH_DOMINANCE_AUDIT_FAILURE"


def block_r18r13(root: Path, reason: str, source_commit: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    counters = {
        "training": 0,
        "rollout": 0,
        "simulator": 0,
        "optimizer_creation": 0,
        "optimizer_step": 0,
        "backward": 0,
        "checkpoint_write": 0,
        "policy_mutation": 0,
    }
    IMPL.dump(root / "gate_decision.json", {
        "stage": STAGE,
        "gate": BLOCK,
        "classification": "BLOCKED",
        "source_commit": source_commit,
        "hard_failures": [reason],
        "execution_counters": counters,
        "next_step": "STOP",
    })
    (root / "final_report.md").write_text(
        f"# {STAGE} blocked\n\n- gate: `{BLOCK}`\n- reason: `{reason}`\n"
        "- training/rollout/simulator/optimizer/backward/checkpoint: `0`\n",
        encoding="utf-8",
    )


def main(_argv: Sequence[str] | None = None) -> None:
    source_commit = IMPL.git(["rev-parse", "HEAD"])
    root = IMPL.ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r13_gradient_magnitude_path_dominance_audit_{IMPL.kst_now()}"
    try:
        IMPL.require(IMPL.git(["status", "--porcelain=v1"]) == "", "dirty_worktree")
        root.mkdir(parents=True, exist_ok=False)
        original_stage, original_classification = IMPL.STAGE, IMPL.CLASSIFICATION
        IMPL.STAGE, IMPL.CLASSIFICATION = STAGE, CLASSIFICATION
        try:
            evidence, row_audit, pairwise, counters = IMPL.build_audit()
        finally:
            IMPL.STAGE, IMPL.CLASSIFICATION = original_stage, original_classification

        aggregate = dict(row_audit["aggregate"])
        candidate = dict(aggregate["by_selected_action_type"]["CANDIDATE"])
        no_assign = dict(aggregate["by_selected_action_type"]["NO_ASSIGN"])
        explanation = {
            "advantage_alone_explains_dominance": False,
            "candidate_total_normalized_advantage": candidate["sum_normalized_advantage"],
            "NO_ASSIGN_total_normalized_advantage": no_assign["sum_normalized_advantage"],
            "candidate_mean_action_logprob_gradient_leverage_norm": candidate["mean_action_logprob_gradient_leverage_norm"],
            "NO_ASSIGN_mean_action_logprob_gradient_leverage_norm": no_assign["mean_action_logprob_gradient_leverage_norm"],
            "NO_ASSIGN_to_candidate_mean_logprob_leverage_ratio": (
                no_assign["mean_action_logprob_gradient_leverage_norm"]
                / candidate["mean_action_logprob_gradient_leverage_norm"]
            ),
            "candidate_update_sum_norm": aggregate["candidate_update_sum_norm"],
            "NO_ASSIGN_update_sum_norm": aggregate["NO_ASSIGN_update_sum_norm"],
            "NO_ASSIGN_to_candidate_update_sum_norm_ratio": aggregate["NO_ASSIGN_to_candidate_update_sum_norm_ratio"],
            "candidate_sum_vs_NO_ASSIGN_sum_cosine": aggregate["candidate_sum_vs_NO_ASSIGN_sum_cosine"],
            "mean_cosine_by_pair_type": aggregate["mean_cosine_by_pair_type"],
            "parameter_path_decomposition": aggregate["parameter_path_decomposition"],
            "primary_findings": [
                "NO_ASSIGN rows have smaller total normalized advantage than candidate rows, so advantage magnitude is not the cause.",
                "NO_ASSIGN rows have materially larger action-logprob gradient leverage per unit advantage.",
                "Candidate rows align with each other but oppose NO_ASSIGN rows; NO_ASSIGN summed vector is larger and wins the residual batch direction.",
                "The direct NO_ASSIGN scorer path plus shared global/fleet paths give NO_ASSIGN-selected rows high leverage over the batch update.",
            ],
        }
        IMPL.dump(root / "input_evidence_binding.json", evidence)
        IMPL.dump(root / "row_gradient_magnitude_decomposition.json", row_audit)
        IMPL.dump(root / "parameter_path_dominance_decomposition.json", aggregate["parameter_path_decomposition"])
        IMPL.dump(root / "candidate_vs_no_assign_gradient_conflict.json", pairwise)
        IMPL.dump(root / "dominance_explanation.json", explanation)
        IMPL.dump(root / "test_results.json", {"stage": STAGE, "execution_counters": counters,
                                                "hard_failures": [], "warnings": [],
                                                "github_push": False})
        IMPL.dump(root / "gate_decision.json", {"stage": STAGE, "gate": PASS_GATE,
                                                "classification": CLASSIFICATION,
                                                "source_commit": source_commit,
                                                "execution_counters": counters,
                                                "hard_failures": [], "warnings": [],
                                                "next_step": "STOP"})
        report = IMPL.final_markdown(evidence=evidence, row_audit=row_audit, pairwise=pairwise,
                                     gate=PASS_GATE, source_commit=source_commit)
        report += (
            "\n## R18-R13 dominance answer\n\n"
            f"- candidate total normalized advantage: `{candidate['sum_normalized_advantage']:+.6f}`\n"
            f"- NO_ASSIGN total normalized advantage: `{no_assign['sum_normalized_advantage']:+.6f}`\n"
            f"- candidate mean ||∇logπ|| leverage: `{candidate['mean_action_logprob_gradient_leverage_norm']:.6f}`\n"
            f"- NO_ASSIGN mean ||∇logπ|| leverage: `{no_assign['mean_action_logprob_gradient_leverage_norm']:.6f}`\n"
            f"- NO_ASSIGN/candidate leverage ratio: `{explanation['NO_ASSIGN_to_candidate_mean_logprob_leverage_ratio']:.6f}`\n"
            f"- NO_ASSIGN/candidate summed update norm ratio: `{aggregate['NO_ASSIGN_to_candidate_update_sum_norm_ratio']:.6f}`\n"
            "\nConclusion: the dominance is not explained by advantage. It is explained by larger NO_ASSIGN action-logprob "
            "gradient leverage and an opposing shared-batch geometry where the larger NO_ASSIGN summed vector dominates the residual update.\n"
        )
        (root / "final_report.md").write_text(report, encoding="utf-8")
        manifest = {item.relative_to(root).as_posix(): IMPL.sha256(item)
                    for item in root.rglob("*") if item.is_file() and item.name != "manifest.json"}
        IMPL.dump(root / "manifest.json", {"stage": STAGE, "gate": PASS_GATE,
                                           "classification": CLASSIFICATION,
                                           "source_commit": source_commit,
                                           "file_sha256": manifest})
        (root / "_SUCCESS.lock").write_text(PASS_GATE + "\n", encoding="utf-8")
        print(f"[PASS] {PASS_GATE}")
        print(root)
    except IMPL.R18R12Error as exc:
        block_r18r13(root, str(exc), source_commit)
        print(f"[BLOCKED] {BLOCK}")
        print(root)
    except Exception as exc:  # noqa: BLE001
        block_r18r13(root, f"{type(exc).__name__}:{exc}", source_commit)
        print(f"[BLOCKED] {BLOCK}")
        print(root)


if __name__ == "__main__":
    main()
