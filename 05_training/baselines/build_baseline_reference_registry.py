from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


KPI_12 = [
    "cv_headway",
    "avg_wait_seconds",
    "bunching_rate",
    "on_time_rate",
    "intervention_rate",
    "energy_proxy",
    "passenger_demand_generated",
    "passenger_served_count",
    "passenger_service_rate",
    "passenger_wait_p95_seconds",
    "energy_proxy_per_passenger",
    "fleet_reduction_ratio",
]


BASELINE_SPECS = {
    "B0R_historical_12kpi_compat": {
        "condition_id": "B0R",
        "family": "historical_replay_compat",
        "role": "Historical replay / compatibility reference baseline",
        "root": "artifacts/baseline_v1/B0R_historical_12kpi_compat",
        "overall_path": "artifacts/baseline_v1/B0R_historical_12kpi_compat/kpi_overall.json",
        "manifest_path": "artifacts/baseline_v1/B0R_historical_12kpi_compat/aggregation_manifest.json",
        "comparison_scope": "historical_reference",
        "release_status": "REFERENCE_AVAILABLE_COMPAT_GUARDED",
        "allowed_use": [
            "Use as historical reference baseline.",
            "Use in baseline reference tables with null/compatibility caveats.",
        ],
        "prohibited_use": [
            "Do not treat null compatibility fields as observed zeros.",
            "Do not use as causal performance proof.",
        ],
    },
    "B1_noop": {
        "condition_id": "B1",
        "family": "noncausal_replay_noop",
        "role": "No-op replay baseline",
        "root": "artifacts/baseline_v1/B1_noop/canonical_eval",
        "overall_path": "artifacts/baseline_v1/B1_noop/canonical_eval/kpi_overall.json",
        "manifest_path": "artifacts/baseline_v1/B1_noop/canonical_eval/aggregation_manifest.json",
        "comparison_scope": "noncausal_noop_reference",
        "release_status": "REFERENCE_AVAILABLE_NONCAUSAL_GUARDED",
        "allowed_use": [
            "Use as no-intervention replay baseline.",
            "Use for scale comparison against B2 and B0C.",
        ],
        "prohibited_use": [
            "Do not use as causal performance proof.",
        ],
    },
    "B2_rulebased_calibrated": {
        "condition_id": "B2",
        "family": "noncausal_replay_rulebased_calibrated",
        "role": "Calibrated rule-based replay baseline",
        "root": "artifacts/baseline_v1/B2_rulebased_calibrated/canonical_eval",
        "overall_path": "artifacts/baseline_v1/B2_rulebased_calibrated/canonical_eval/kpi_overall.json",
        "manifest_path": "artifacts/baseline_v1/B2_rulebased_calibrated/canonical_eval/aggregation_manifest.json",
        "extra_validation_path": "artifacts/baseline_v1/B2_rulebased_calibrated/b2_intervention_rate_validation.json",
        "comparison_scope": "noncausal_rulebased_reference",
        "release_status": "REFERENCE_AVAILABLE_NONCAUSAL_GUARDED",
        "allowed_use": [
            "Use as calibrated simple rule-based baseline.",
            "Use to verify that rule-based baseline is not no-op leakage.",
        ],
        "prohibited_use": [
            "Do not use as causal performance proof.",
            "Do not treat replay-backed improvements as real intervention effects.",
        ],
    },
    "B0C_causal_shadow_v1": {
        "condition_id": "B0C",
        "family": "synthetic_causal_shadow_scaffold",
        "role": "Synthetic causal-shadow baseline, simulator sanity validated, still locked",
        "root": "artifacts/baseline_v1/B0C_causal_shadow_v1/canonical_eval",
        "overall_path": "artifacts/baseline_v1/B0C_causal_shadow_v1/canonical_eval/kpi_overall.json",
        "manifest_path": "artifacts/baseline_v1/B0C_causal_shadow_v1/canonical_eval/aggregation_manifest.json",
        "extra_validation_path": "artifacts/baseline_v1/B0C_causal_shadow_v1/simulator_validation/b0c_simulator_validation_report.json",
        "promotion_manifest_path": "artifacts/baseline_v1/B0C_causal_shadow_v1/promotion_candidate/b0c_validation_promotion_candidate_manifest.json",
        "comparison_scope": "phase2_simulator_internal_candidate_still_locked",
        "release_status": "PROMOTION_CANDIDATE_STILL_LOCKED",
        "allowed_use": [
            "Use as simulator sanity validation candidate.",
            "Use for internal Phase 2 planning.",
            "Use as locked reference candidate only.",
        ],
        "prohibited_use": [
            "Do not claim real-world operational improvement.",
            "Do not set causal_comparison_allowed=true automatically.",
            "Do not use as paper-level causal result evidence.",
        ],
    },
}


def load_json_optional(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def kpi_summary(overall: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not overall:
        return {
            "kpi_schema_present": False,
            "available_kpis": [],
            "missing_kpis": KPI_12,
        }

    kpis = overall.get("kpis", {})
    available = [k for k in KPI_12 if k in kpis]
    missing = [k for k in KPI_12 if k not in kpis]

    return {
        "kpi_schema_present": len(missing) == 0,
        "available_kpis": available,
        "missing_kpis": missing,
        "kpi_means": {
            k: kpis.get(k, {}).get("mean")
            for k in KPI_12
            if k in kpis
        },
    }


def build_markdown(payload: Dict[str, Any]) -> str:
    lines: List[str] = []

    lines.append("# Baseline Reference Registry")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("This registry locks the baseline reference set used before later A-family actual MAPPO comparison.")
    lines.append("")
    lines.append("This registry does not release causal or paper-level claims.")
    lines.append("")
    lines.append("## Global Claim Guard")
    lines.append("")
    lines.append("| Guard | Value |")
    lines.append("|---|---|")
    for key, value in payload["global_claim_guards"].items():
        lines.append(f"| {key} | {value} |")

    lines.append("")
    lines.append("## Baseline Registry")
    lines.append("")
    lines.append("| baseline_id | condition_id | family | release_status | causal_allowed | windows |")
    lines.append("|---|---|---|---|---:|---:|")

    for baseline_id, item in payload["baselines"].items():
        summary = item.get("overall_summary", {})
        lines.append(
            f"| {baseline_id} | {item['condition_id']} | {item['family']} | "
            f"{item['release_status']} | {summary.get('causal_comparison_allowed')} | "
            f"{summary.get('n_windows_total')} |"
        )

    lines.append("")
    lines.append("## Recommended Use")
    lines.append("")
    lines.append("- B0R, B1, and B2_calibrated are baseline reference artifacts for later A-family comparison.")
    lines.append("- B0C is a simulator validation candidate and remains locked.")
    lines.append("- Actual A-family MAPPO results are not included in this registry yet.")
    lines.append("- Winner selection and paper-level claims remain prohibited.")
    lines.append("")

    lines.append("## Baseline Details")
    lines.append("")

    for baseline_id, item in payload["baselines"].items():
        lines.append(f"### {baseline_id}")
        lines.append("")
        lines.append(f"- role: {item['role']}")
        lines.append(f"- root: `{item['root']}`")
        lines.append(f"- release_status: `{item['release_status']}`")
        lines.append(f"- comparison_scope: `{item['comparison_scope']}`")
        lines.append("")
        lines.append("Allowed use:")
        for x in item["allowed_use"]:
            lines.append(f"- {x}")
        lines.append("")
        lines.append("Prohibited use:")
        for x in item["prohibited_use"]:
            lines.append(f"- {x}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="artifacts/baseline_v1/baseline_reference_registry",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    hard_failures: List[str] = []
    warnings: List[str] = []
    baselines: Dict[str, Any] = {}

    for baseline_id, spec in BASELINE_SPECS.items():
        overall_path = Path(spec["overall_path"])
        manifest_path = Path(spec["manifest_path"])

        overall = load_json_optional(overall_path)
        manifest = load_json_optional(manifest_path)

        if overall is None:
            hard_failures.append(f"{baseline_id}: missing kpi_overall.json at {overall_path}")

        if manifest is None:
            warnings.append(f"{baseline_id}: missing aggregation_manifest.json at {manifest_path}")

        extra_validation = None
        extra_validation_path = spec.get("extra_validation_path")
        if extra_validation_path:
            extra_validation = load_json_optional(Path(extra_validation_path))
            if extra_validation is None:
                warnings.append(f"{baseline_id}: missing extra validation at {extra_validation_path}")

        promotion_manifest = None
        promotion_manifest_path = spec.get("promotion_manifest_path")
        if promotion_manifest_path:
            promotion_manifest = load_json_optional(Path(promotion_manifest_path))
            if promotion_manifest is None:
                warnings.append(f"{baseline_id}: missing promotion manifest at {promotion_manifest_path}")

        causal_allowed = bool(overall.get("causal_comparison_allowed", False)) if overall else None

        # Registry-level guard: all reference artifacts must remain non-claim at this stage.
        if causal_allowed is True:
            hard_failures.append(f"{baseline_id}: causal_comparison_allowed must be false in registry stage")

        kpi_info = kpi_summary(overall)

        if baseline_id != "B0R_historical_12kpi_compat" and not kpi_info["kpi_schema_present"]:
            warnings.append(f"{baseline_id}: not all 12 KPI fields present")

        if baseline_id == "B2_rulebased_calibrated" and extra_validation is not None:
            if extra_validation.get("audit_status") != "PASS":
                hard_failures.append("B2_rulebased_calibrated: intervention validation must be PASS")
            intervention_rate = extra_validation.get("intervention_rate")
            if intervention_rate is None or float(intervention_rate) <= 0:
                hard_failures.append("B2_rulebased_calibrated: intervention_rate must be positive")

        if baseline_id == "B0C_causal_shadow_v1":
            if extra_validation is not None:
                if extra_validation.get("audit_status") != "PASS":
                    hard_failures.append("B0C_causal_shadow_v1: simulator validation report must be PASS")
                guards = extra_validation.get("claim_guards", {})
                if guards.get("causal_comparison_allowed") is not False:
                    hard_failures.append("B0C_causal_shadow_v1: validation report causal guard must be false")

            if promotion_manifest is not None:
                if promotion_manifest.get("promotion_candidate") is not True:
                    hard_failures.append("B0C_causal_shadow_v1: promotion_candidate must be true")
                if promotion_manifest.get("promotion_released") is not False:
                    hard_failures.append("B0C_causal_shadow_v1: promotion_released must remain false")

        baselines[baseline_id] = {
            "condition_id": spec["condition_id"],
            "family": spec["family"],
            "role": spec["role"],
            "root": spec["root"],
            "overall_path": str(overall_path),
            "manifest_path": str(manifest_path),
            "comparison_scope": spec["comparison_scope"],
            "release_status": spec["release_status"],
            "allowed_use": spec["allowed_use"],
            "prohibited_use": spec["prohibited_use"],
            "overall_summary": {
                "artifact_version": overall.get("artifact_version") if overall else None,
                "mode": overall.get("mode") if overall else None,
                "condition_ids": overall.get("condition_ids") if overall else None,
                "n_condition_seed_pairs": overall.get("n_condition_seed_pairs") if overall else None,
                "n_windows_total": overall.get("n_windows_total") if overall else None,
                "causal_comparison_allowed": causal_allowed,
            },
            "kpi_summary": kpi_info,
            "manifest_summary": {
                "window_rollup_file_count": manifest.get("window_rollup_file_count") if manifest else None,
                "row_counts": manifest.get("row_counts") if manifest else None,
                "validation_summary": manifest.get("validation_summary") if manifest else None,
                "warnings": manifest.get("warnings") if manifest else None,
            },
            "extra_validation": extra_validation,
            "promotion_manifest": promotion_manifest,
        }

    audit_status = "FAIL" if hard_failures else ("PASS_WITH_WARNINGS" if warnings else "PASS")

    payload: Dict[str, Any] = {
        "artifact_version": "baseline_reference_registry_v1",
        "audit_status": audit_status,
        "registry_status": "LOCKED_REFERENCE_SET" if audit_status != "FAIL" else "BLOCKED",
        "baselines": baselines,
        "global_claim_guards": {
            "causal_comparison_allowed": False,
            "paper_level_claim_allowed": False,
            "actual_results": False,
            "winner_selected": False,
            "trainable_reward_promoted": False,
            "auto_promotion_allowed": False,
        },
        "comparison_readiness": {
            "baseline_side_ready": audit_status != "FAIL",
            "actual_policy_side_ready": False,
            "requires_h200_actual_a_family_results": True,
        },
        "hard_failures": hard_failures,
        "warnings": warnings,
        "non_claim_statement": (
            "This registry locks baseline references only. It does not contain H200 actual MAPPO results "
            "and does not allow causal or paper-level performance claims."
        ),
    }

    json_path = output_root / "baseline_reference_registry.json"
    md_path = output_root / "baseline_reference_registry.md"

    dump_json(json_path, payload)
    md_path.write_text(build_markdown(payload), encoding="utf-8")

    print("[OK] Baseline reference registry completed")
    print("[OK] audit_status:", audit_status)
    print("[OK] registry_status:", payload["registry_status"])
    print("[OK] baseline_count:", len(baselines))
    print("[OK] hard_failures:", len(hard_failures))
    print("[OK] warnings:", len(warnings))
    print("[OK] json:", json_path)
    print("[OK] markdown:", md_path)

    if hard_failures:
        raise SystemExit(f"[FAIL] {hard_failures}")


if __name__ == "__main__":
    main()
