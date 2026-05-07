from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


STEP_ID = "step173"
ARTIFACT_VERSION = "zero_loss_patent_claim_drafting_packet_v1_step173"
BUNDLE_STATUS = "ZERO_LOSS_PATENT_CLAIM_DRAFTING_PACKET_READY_NONCLAIM"

GUARD_FLAGS: Dict[str, bool] = {
    "paper_level_claim_allowed": False,
    "causal_performance_claim_allowed": False,
    "actual_operational_claim_allowed": False,
    "actual_evidence_execution_allowed": False,
    "command_execution_allowed": False,
    "train_allowed": False,
    "legal_novelty_opinion_provided": False,
    "filing_ready_without_attorney_review": False,
}

PIPELINE_STEPS = [
    {"step": "160", "name": "Zero-Loss Pickup Evidence Reporter", "role": "Counts pickup attempts and zero-loss successes; writes evidence bundle."},
    {"step": "161", "name": "Route-Aware Pickup Attempt Event Writer", "role": "Writes pickup attempt, ETA counterfactual, and attention input tables."},
    {"step": "162", "name": "Route-Aware Rollout Adapter", "role": "Adapts raw route-aware rollout output into Step 161-compatible normalized events."},
    {"step": "163", "name": "GATv2 Real Attention Extractor", "role": "Extracts real GATv2Conv attention weights through return_attention_weights."},
    {"step": "164", "name": "Real Attention Pipeline Connector", "role": "Connects real attention output to the zero-loss evidence pipeline."},
    {"step": "165", "name": "Attempt-Specific Route/Path Attention Filter", "role": "Filters attention to attempt-related route/path edge or node overlap."},
    {"step": "166", "name": "Patent Evidence Report v2", "role": "Combines zero-loss ETA results and filtered real attention into a patent-style report."},
    {"step": "167", "name": "Actual Route-Aware Rollout Evidence Runbook", "role": "Defines actual-like evidence run order."},
    {"step": "168", "name": "Project Log Update", "role": "Records the non-claim evidence pipeline in project_log.md."},
    {"step": "169", "name": "Actual Evidence Input Readiness Checklist", "role": "Checks raw_events, window_rollup, route_stop_sequence, manifest, and model readiness."},
    {"step": "170", "name": "Actual Evidence Command Packet", "role": "Generates dry-run commands for the evidence pipeline."},
    {"step": "171", "name": "Actual Evidence Execution Release Checklist", "role": "Requires operator approval before execution release."},
    {"step": "172", "name": "Final Zero-Loss Patent Evidence Handoff Index", "role": "Indexes the Step 160~171 evidence package."},
]

CLAIM_ELEMENTS = [
    {
        "claim_family": "Independent method claim",
        "element_id": "M1",
        "element_name": "receive_state_and_candidate_pickup",
        "technical_content": "Receive current vehicle state, already-boarded passenger destination/ETA state, and candidate new passenger pickup/dropoff request.",
        "supporting_pipeline_steps": "161,162,169",
        "evidence_artifact": "pickup_attempt_events.*; normalized_route_aware_rollout_events.*",
    },
    {
        "claim_family": "Independent method claim",
        "element_id": "M2",
        "element_name": "counterfactual_eta_without_new_pickup",
        "technical_content": "Compute ETA of already-boarded passenger(s) under the route plan without accepting the new pickup.",
        "supporting_pipeline_steps": "160,161,162",
        "evidence_artifact": "eta_counterfactual.*",
    },
    {
        "claim_family": "Independent method claim",
        "element_id": "M3",
        "element_name": "counterfactual_eta_with_new_pickup",
        "technical_content": "Compute ETA of already-boarded passenger(s) under the candidate route plan with the new pickup inserted.",
        "supporting_pipeline_steps": "160,161,162",
        "evidence_artifact": "eta_counterfactual.*",
    },
    {
        "claim_family": "Independent method claim",
        "element_id": "M4",
        "element_name": "zero_loss_threshold_decision",
        "technical_content": "Approve the candidate pickup only when ETA_with_new_pickup minus ETA_without_new_pickup is zero or within a configured simulator-resolution epsilon.",
        "supporting_pipeline_steps": "160,166",
        "evidence_artifact": "zero_loss_summary.json; patent_evidence_summary_v2.json",
    },
    {
        "claim_family": "Independent method claim",
        "element_id": "M5",
        "element_name": "gatv2_attention_extraction",
        "technical_content": "Extract attention weights from a GATv2 encoder applied to a transport graph corresponding to the route-aware state.",
        "supporting_pipeline_steps": "163,164",
        "evidence_artifact": "gatv2_real_attention_edges.*; gatv2_attention.*",
    },
    {
        "claim_family": "Independent method claim",
        "element_id": "M6",
        "element_name": "attempt_specific_route_path_attention_filter",
        "technical_content": "Filter extracted attention weights according to route/path elements associated with the candidate pickup, including current-to-pickup, pickup-to-dropoff, and existing passenger path segments.",
        "supporting_pipeline_steps": "165,166",
        "evidence_artifact": "attention_mass_by_attempt_path_v2.csv; top_filtered_attention_edges_v2.csv",
    },
    {
        "claim_family": "Independent method claim",
        "element_id": "M7",
        "element_name": "evidence_bundle_generation",
        "technical_content": "Generate a reproducible evidence bundle containing pickup attempt counts, zero-loss success counts, ETA deltas, route/path attention mass, and manifest guard flags.",
        "supporting_pipeline_steps": "160,166,167,172",
        "evidence_artifact": "zero_loss_evidence_manifest.json; patent_evidence_report_v2_manifest.json; final handoff index",
    },
]

DEPENDENT_CLAIMS = [
    {"claim_no": "2", "depends_on": "1", "draft": "The method of claim 1, wherein the zero-loss threshold is set to zero seconds or to a simulator time-resolution epsilon."},
    {"claim_no": "3", "depends_on": "1", "draft": "The method of claim 1, wherein the ETA delta is computed for each already-boarded passenger and the candidate pickup is rejected if any such passenger has a positive ETA loss beyond the threshold."},
    {"claim_no": "4", "depends_on": "1", "draft": "The method of claim 1, wherein the transport graph includes stop nodes and route/path edges including distance, travel-time, generalized-cost, and long-edge attributes."},
    {"claim_no": "5", "depends_on": "1", "draft": "The method of claim 1, wherein the attention evidence is filtered by route/path overlap for current-to-pickup, pickup-to-dropoff, and existing-passenger path segments."},
    {"claim_no": "6", "depends_on": "1", "draft": "The method of claim 1, wherein the evidence bundle includes an attention-mass distribution grouped by path segment and by zero-loss success status."},
    {"claim_no": "7", "depends_on": "1", "draft": "The method of claim 1, wherein a manifest records checkpoint hash, dataset identifier, configuration hash, simulator version, and non-claim guard status."},
    {"claim_no": "8", "depends_on": "1", "draft": "The method of claim 1, wherein the dispatch command is withheld unless an input readiness checklist and an execution release checklist are satisfied."},
    {"claim_no": "9", "depends_on": "1", "draft": "The method of claim 1, wherein the system outputs a report containing total pickup attempts, zero-loss success count, zero-loss success rate, and top filtered attention edges for each attempt."},
]

PRIOR_ART_DIFF = [
    {
        "topic": "Delay-threshold ride sharing",
        "known_risk": "Existing systems may reject pickups when expected delay exceeds a user-defined maximum detour or delay threshold.",
        "drafting_differentiator": "Draft claims around zero-loss acceptance for already-boarded passengers plus reproducible GATv2 route/path attention evidence, not merely around a maximum allowed delay.",
        "claim_elements": "M2,M3,M4,M5,M6,M7",
    },
    {
        "topic": "Zero-detour insertion",
        "known_risk": "Research and routing systems may attempt zero-detour pickup/dropoff insertion on existing route plans.",
        "drafting_differentiator": "Distinguish zero-detour geometry from zero ETA loss for already-boarded passengers and require counterfactual ETA evidence plus graph-attention explanation.",
        "claim_elements": "M4,M5,M6",
    },
    {
        "topic": "General ML dispatch / RL dispatch",
        "known_risk": "ML/RL systems for dispatch may already optimize reward or travel time.",
        "drafting_differentiator": "Use deterministic zero-loss gate and evidence bundle as a safety/verification layer around dispatch, not only a learned reward objective.",
        "claim_elements": "M4,M7",
    },
    {
        "topic": "Generic MLOps artifact governance",
        "known_risk": "Hash manifests and model governance are common in MLOps.",
        "drafting_differentiator": "Tie manifest locks to public-transit pickup decision evidence, ETA counterfactuals, route/path attention filtering, and command-release gates.",
        "claim_elements": "M7",
    },
]

EVIDENCE_TO_CLAIM = [
    {"evidence_file": "pickup_attempt_events.*", "claim_support": "M1", "notes": "Identifies candidate pickup attempts and passenger/request metadata."},
    {"evidence_file": "eta_counterfactual.*", "claim_support": "M2,M3,M4", "notes": "Provides ETA without/with candidate pickup and delta-based zero-loss decision."},
    {"evidence_file": "gatv2_real_attention_edges.*", "claim_support": "M5", "notes": "Records raw real GATv2Conv attention rows at graph snapshot level."},
    {"evidence_file": "gatv2_attention.*", "claim_support": "M5,M6", "notes": "Records attempt-level attention evidence used by zero-loss report."},
    {"evidence_file": "attention_mass_by_attempt_path_v2.csv", "claim_support": "M6,M7", "notes": "Shows attention mass grouped by attempt and path segment."},
    {"evidence_file": "patent_evidence_summary_v2.json", "claim_support": "M4,M7", "notes": "Summarizes pickup attempt count, zero-loss count, rate, and non-claim flags."},
    {"evidence_file": "final_zero_loss_patent_evidence_handoff_index_step172.*", "claim_support": "M7", "notes": "Indexes full evidence pipeline and guards."},
]


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stable_hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def render_markdown(manifest: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Zero-Loss Pickup Threshold Patent Claim Drafting Packet — Step 173")
    lines.append("")
    lines.append("## Status")
    lines.append("")
    lines.append(f"- audit_status: `{manifest['audit_status']}`")
    lines.append(f"- bundle_status: `{manifest['bundle_status']}`")
    lines.append("- legal status: technical drafting aid only; attorney review required")
    lines.append("")
    lines.append("## Proposed invention summary")
    lines.append("")
    lines.append(
        "A route-aware dispatch method approves a candidate shared pickup only when the "
        "counterfactual ETA loss of already-boarded passenger(s) is zero or within a configured "
        "simulator-resolution epsilon, and generates a reproducible evidence bundle containing "
        "real GATv2 attention weights filtered by attempt-specific route/path overlap."
    )
    lines.append("")
    lines.append("## Independent method claim draft — technical skeleton")
    lines.append("")
    lines.append("1. A computer-implemented method comprising:")
    for e in CLAIM_ELEMENTS:
        lines.append(f"   - ({e['element_id']}) {e['technical_content']}")
    lines.append("")
    lines.append("## Dependent claim candidates")
    lines.append("")
    for d in DEPENDENT_CLAIMS:
        lines.append(f"- Claim {d['claim_no']} depending on claim {d['depends_on']}: {d['draft']}")
    lines.append("")
    lines.append("## Evidence-to-claim mapping")
    lines.append("")
    lines.append("| Evidence file | Claim support | Notes |")
    lines.append("|---|---|---|")
    for row in EVIDENCE_TO_CLAIM:
        lines.append(f"| `{row['evidence_file']}` | {row['claim_support']} | {row['notes']} |")
    lines.append("")
    lines.append("## Prior-art differentiation drafting notes")
    lines.append("")
    lines.append("| Topic | Known risk | Drafting differentiator |")
    lines.append("|---|---|---|")
    for row in PRIOR_ART_DIFF:
        lines.append(f"| {row['topic']} | {row['known_risk']} | {row['drafting_differentiator']} |")
    lines.append("")
    lines.append("## Guard flags")
    lines.append("")
    for key, value in GUARD_FLAGS.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## Required next legal actions")
    lines.append("")
    lines.append("- Patent attorney claim chart review")
    lines.append("- KIPRIS / Google Patents / WIPO / USPTO / EPO prior-art search")
    lines.append("- Novelty and inventive-step assessment")
    lines.append("- Jurisdiction-specific claim language revision")
    lines.append("- Filing strategy for method/system/computer-readable-medium claims")
    lines.append("")
    return "\n".join(lines) + "\n"


def generate_packet(output_root: Path, title: str = "Zero-Loss Pickup Threshold") -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    file_paths = {
        "packet_md": output_root / "zero_loss_patent_claim_drafting_packet_step173.md",
        "packet_json": output_root / "zero_loss_patent_claim_drafting_packet_step173.json",
        "claim_element_matrix": output_root / "zero_loss_claim_element_matrix_step173.csv",
        "dependent_claim_candidates": output_root / "zero_loss_dependent_claim_candidates_step173.csv",
        "evidence_to_claim_mapping": output_root / "zero_loss_evidence_to_claim_mapping_step173.csv",
        "prior_art_differentiation_matrix": output_root / "zero_loss_prior_art_differentiation_matrix_step173.csv",
        "manifest": output_root / "zero_loss_claim_drafting_manifest_step173.json",
    }

    payload = {
        "artifact_version": ARTIFACT_VERSION,
        "step_id": STEP_ID,
        "created_at_utc": now,
        "title": title,
        "purpose": "technical patent claim drafting packet for Zero-Loss Pickup Threshold evidence pipeline",
        "not_legal_opinion": True,
        "attorney_review_required": True,
        "pipeline_steps": PIPELINE_STEPS,
        "claim_elements": CLAIM_ELEMENTS,
        "dependent_claim_candidates": DEPENDENT_CLAIMS,
        "evidence_to_claim_mapping": EVIDENCE_TO_CLAIM,
        "prior_art_differentiation_matrix": PRIOR_ART_DIFF,
        **GUARD_FLAGS,
    }

    write_csv(file_paths["claim_element_matrix"], CLAIM_ELEMENTS)
    write_csv(file_paths["dependent_claim_candidates"], DEPENDENT_CLAIMS)
    write_csv(file_paths["evidence_to_claim_mapping"], EVIDENCE_TO_CLAIM)
    write_csv(file_paths["prior_art_differentiation_matrix"], PRIOR_ART_DIFF)
    write_json(file_paths["packet_json"], payload)

    provisional_manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "step_id": STEP_ID,
        "audit_status": "PASS",
        "bundle_status": BUNDLE_STATUS,
        "created_at_utc": now,
        "output_root": str(output_root),
        "title": title,
        "not_legal_opinion": True,
        "attorney_review_required": True,
        "pipeline_step_count": len(PIPELINE_STEPS),
        "claim_element_count": len(CLAIM_ELEMENTS),
        "dependent_claim_candidate_count": len(DEPENDENT_CLAIMS),
        "prior_art_differentiation_count": len(PRIOR_ART_DIFF),
        "evidence_mapping_count": len(EVIDENCE_TO_CLAIM),
        **GUARD_FLAGS,
        "outputs": {k: str(v) for k, v in file_paths.items() if k != "manifest"},
    }

    file_paths["packet_md"].write_text(render_markdown(provisional_manifest), encoding="utf-8")

    file_hashes = {k: stable_hash_file(v) for k, v in file_paths.items() if k != "manifest"}
    manifest = {**provisional_manifest, "file_sha256": file_hashes, "manifest": str(file_paths["manifest"])}
    write_json(file_paths["manifest"], manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 173 Zero-Loss patent claim drafting packet")
    parser.add_argument("--output-root", default="artifacts/patent_evidence/zero_loss_patent_claim_drafting_packet_step173_selftest")
    parser.add_argument("--title", default="Zero-Loss Pickup Threshold")
    args = parser.parse_args()

    manifest = generate_packet(Path(args.output_root), title=args.title)
    print("[OK] Step 173 zero-loss patent claim drafting packet generated")
    print(f"[OK] audit_status              : {manifest['audit_status']}")
    print(f"[OK] bundle_status             : {manifest['bundle_status']}")
    print(f"[OK] pipeline_step_count       : {manifest['pipeline_step_count']}")
    print(f"[OK] claim_element_count       : {manifest['claim_element_count']}")
    print(f"[OK] dependent_claim_candidate_count : {manifest['dependent_claim_candidate_count']}")
    print(f"[OK] prior_art_differentiation_count : {manifest['prior_art_differentiation_count']}")
    print(f"[OK] not_legal_opinion         : {manifest['not_legal_opinion']}")
    print(f"[OK] attorney_review_required  : {manifest['attorney_review_required']}")
    print(f"[OK] paper_level_claim_allowed : {manifest['paper_level_claim_allowed']}")
    print(f"[OK] causal_performance_claim_allowed : {manifest['causal_performance_claim_allowed']}")
    print(f"[OK] filing_ready_without_attorney_review : {manifest['filing_ready_without_attorney_review']}")
    print(f"[OK] output_root               : {manifest['output_root']}")
    print(f"[OK] manifest                  : {manifest['manifest']}")


if __name__ == "__main__":
    main()
