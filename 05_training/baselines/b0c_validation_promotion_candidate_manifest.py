from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    lines: List[str] = []

    lines.append("# B0C Validation Promotion Candidate Manifest")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(f"| baseline_id | {payload['baseline_id']} |")
    lines.append(f"| condition_id | {payload['condition_id']} |")
    lines.append(f"| manifest_status | {payload['manifest_status']} |")
    lines.append(f"| promotion_candidate | {payload['promotion_candidate']} |")
    lines.append(f"| promotion_released | {payload['promotion_released']} |")
    lines.append("")
    lines.append("## Validation Source")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(f"| validation_report | {payload['validation_source']['validation_report']} |")
    lines.append(f"| validation_audit_status | {payload['validation_source']['audit_status']} |")
    lines.append(f"| hard_failures | {payload['validation_source']['hard_failure_count']} |")
    lines.append(f"| warnings | {payload['validation_source']['warning_count']} |")
    lines.append("")
    lines.append("## Claim Guards")
    lines.append("")
    lines.append("| Guard | Value |")
    lines.append("|---|---|")
    for key, value in payload["claim_guards"].items():
        lines.append(f"| {key} | {value} |")
    lines.append("")
    lines.append("## Required Release Gates")
    lines.append("")
    for item in payload["required_release_gates"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Non-claim Statement")
    lines.append("")
    lines.append(payload["non_claim_statement"])
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation-report",
        default="artifacts/baseline_v1/B0C_causal_shadow_v1/simulator_validation/b0c_simulator_validation_report.json",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/baseline_v1/B0C_causal_shadow_v1/promotion_candidate",
    )
    args = parser.parse_args()

    validation_report_path = Path(args.validation_report)
    output_root = Path(args.output_root)

    if not validation_report_path.exists():
        raise SystemExit(f"[FAIL] missing validation report: {validation_report_path}")

    report = load_json(validation_report_path)

    hard_failures = report.get("hard_failures", [])
    warnings = report.get("warnings", [])
    audit_status = report.get("audit_status")

    promotion_candidate = audit_status in {"PASS", "PASS_WITH_WARNINGS"} and len(hard_failures) == 0

    if not promotion_candidate:
        manifest_status = "BLOCKED_BY_VALIDATION_REPORT"
    elif audit_status == "PASS_WITH_WARNINGS":
        manifest_status = "PROMOTION_CANDIDATE_STILL_LOCKED_WITH_WARNINGS"
    else:
        manifest_status = "PROMOTION_CANDIDATE_STILL_LOCKED"

    report_guards = report.get("claim_guards", {})

    # Safety: even a PASS validation report must not auto-release claims.
    if report_guards.get("causal_comparison_allowed") is not False:
        raise SystemExit("[FAIL] validation report has causal_comparison_allowed != false")

    if report_guards.get("paper_level_claim_allowed") is not False:
        raise SystemExit("[FAIL] validation report has paper_level_claim_allowed != false")

    if report_guards.get("auto_promotion_allowed") is not False:
        raise SystemExit("[FAIL] validation report has auto_promotion_allowed != false")

    payload: Dict[str, Any] = {
        "artifact_version": "b0c_validation_promotion_candidate_manifest_v1",
        "manifest_status": manifest_status,
        "baseline_id": "B0C_causal_shadow_v1",
        "condition_id": "B0C",
        "promotion_candidate": bool(promotion_candidate),
        "promotion_released": False,
        "validation_source": {
            "validation_report": str(validation_report_path),
            "audit_status": audit_status,
            "hard_failure_count": len(hard_failures),
            "warning_count": len(warnings),
            "hard_failures": hard_failures,
            "warnings": warnings,
        },
        "claim_guards": {
            "causal_comparison_allowed": False,
            "paper_level_claim_allowed": False,
            "actual_results": False,
            "winner_selected": False,
            "trainable_reward_promoted": False,
            "auto_promotion_allowed": False,
            "operator_approval_required": True,
            "explicit_release_gate_required": True,
        },
        "required_release_gates": [
            "B0C-6 simulator validation report PASS or PASS_WITH_WARNINGS",
            "B0C-7 promotion candidate manifest validation PASS",
            "operator approval decision recorded",
            "operator_approval_granted = true",
            "final explicit release manifest PASS",
            "separate simulator validation manifest archived",
            "paper-level claim review still separate",
        ],
        "allowed_current_use": [
            "Use B0C as a simulator sanity validation candidate.",
            "Use B0C for internal calibration discussion.",
            "Use B0C as a locked reference candidate for later Phase 2 planning.",
        ],
        "prohibited_current_use": [
            "Do not claim real-world operational improvement.",
            "Do not set causal_comparison_allowed=true automatically.",
            "Do not use as paper-level result table evidence.",
            "Do not select a winner policy based on B0C scaffold alone.",
        ],
        "non_claim_statement": (
            "B0C passed simulator sanity validation, but this manifest is still locked. "
            "It is a promotion candidate only, not an automatic causal-comparison release."
        ),
    }

    output_root.mkdir(parents=True, exist_ok=True)

    json_path = output_root / "b0c_validation_promotion_candidate_manifest.json"
    md_path = output_root / "b0c_validation_promotion_candidate_manifest.md"

    dump_json(json_path, payload)
    write_markdown(md_path, payload)

    print("[OK] B0C validation promotion candidate manifest completed")
    print("[OK] manifest_status:", manifest_status)
    print("[OK] promotion_candidate:", promotion_candidate)
    print("[OK] promotion_released:", False)
    print("[OK] claim_guard: false")
    print("[OK] json:", json_path)
    print("[OK] markdown:", md_path)

    if not promotion_candidate:
        raise SystemExit("[FAIL] B0C is not a promotion candidate")


if __name__ == "__main__":
    main()
