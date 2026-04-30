from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


STEP_ID = 141
ARTIFACT_VERSION = "baseline_reference_manifest_step141_v1"

REQUIRED_CANONICAL_FILES = [
    "kpi_by_window.parquet",
    "kpi_by_seed.parquet",
    "kpi_by_time_band.parquet",
    "kpi_overall.json",
    "aggregation_manifest.json",
]

BASELINE_SPECS = [
    {
        "baseline_id": "B0R",
        "legacy_condition_id": "B0",
        "display_name": "historical replay baseline",
        "reference_role": "historical_replay_reference",
        "root_candidates": [
            "artifacts/baseline_v1/B0_historical/canonical_eval",
            "artifacts/baseline_v1/B0_historical",
        ],
    },
    {
        "baseline_id": "B1",
        "legacy_condition_id": "B1",
        "display_name": "no-op baseline",
        "reference_role": "no_op_reference",
        "root_candidates": [
            "artifacts/baseline_v1/B1_noop/canonical_eval",
            "artifacts/baseline_v1/B1_noop",
        ],
    },
    {
        "baseline_id": "B2",
        "legacy_condition_id": "B2",
        "display_name": "rule-based baseline",
        "reference_role": "rule_based_reference",
        "root_candidates": [
            "artifacts/baseline_v1/B2_rulebased/canonical_eval",
            "artifacts/baseline_v1/B2_rulebased",
        ],
    },
]

FORBIDDEN_TRUE_KEYS = [
    "causal_comparison_allowed",
    "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
    "best_reward_claim_allowed",
    "train_with_this_reward_allowed",
    "actual_training_allowed",
    "winner_selected",
    "trainable_reward_promoted",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json_any(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except Exception:
            continue
    raise RuntimeError(f"failed_to_read_json: {path}")


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def dump_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def find_root(project_root: Path, candidates: List[str]) -> Optional[Path]:
    for raw in candidates:
        p = project_root / raw
        if p.exists():
            return p
    return None


def recursive_forbidden_true(obj: Any, prefix: str = "") -> List[str]:
    violations: List[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in FORBIDDEN_TRUE_KEYS and bool(value):
                violations.append(f"forbidden_true: {path}")
            if isinstance(value, (dict, list)):
                violations.extend(recursive_forbidden_true(value, path))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            path = f"{prefix}[{idx}]"
            if isinstance(value, (dict, list)):
                violations.extend(recursive_forbidden_true(value, path))
    return violations


def read_json_if_exists(path: Path, warnings: List[str], label: str) -> Dict[str, Any]:
    if not path.exists():
        warnings.append(f"{label}_missing")
        return {}
    try:
        return load_json_any(path)
    except Exception as exc:
        warnings.append(f"{label}_unreadable: {exc}")
        return {}


def inspect_baseline(project_root: Path, spec: Dict[str, Any]) -> Dict[str, Any]:
    root = find_root(project_root, spec["root_candidates"])
    warnings: List[str] = []
    missing: List[str] = []

    result: Dict[str, Any] = {
        "baseline_id": spec["baseline_id"],
        "legacy_condition_id": spec["legacy_condition_id"],
        "display_name": spec["display_name"],
        "reference_role": spec["reference_role"],
        "root": str(root) if root else "",
        "root_exists": bool(root),
        "canonical_files": {},
        "missing_canonical_files": missing,
        "kpi_overall_summary": {},
        "aggregation_manifest_summary": {},
        "reference_status": "MISSING",
        "causal_comparison_allowed": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "best_reward_claim_allowed": False,
        "train_with_this_reward_allowed": False,
        "actual_training_allowed": False,
        "winner_selected": False,
        "trainable_reward_promoted": False,
        "warnings": warnings,
    }

    if root is None:
        missing.extend(REQUIRED_CANONICAL_FILES)
        warnings.append("baseline_root_missing")
        return result

    for fname in REQUIRED_CANONICAL_FILES:
        p = root / fname
        exists = p.exists()
        result["canonical_files"][fname] = {
            "path": str(p),
            "exists": exists,
            "size_bytes": int(p.stat().st_size) if exists else 0,
        }
        if not exists:
            missing.append(fname)

    overall = read_json_if_exists(root / "kpi_overall.json", warnings, "kpi_overall")
    if overall:
        result["kpi_overall_summary"] = {
            "mode": overall.get("mode"),
            "condition_ids": overall.get("condition_ids", []),
            "n_windows_total": overall.get("n_windows_total"),
            "n_condition_seed_pairs": overall.get("n_condition_seed_pairs"),
            "causal_comparison_allowed": bool(overall.get("causal_comparison_allowed", False)),
            "kpi_names": sorted(list(overall.get("kpis", {}).keys())) if isinstance(overall.get("kpis"), dict) else [],
        }
        if bool(overall.get("causal_comparison_allowed", False)):
            warnings.append("kpi_overall_has_causal_comparison_allowed_true")

    agg = read_json_if_exists(root / "aggregation_manifest.json", warnings, "aggregation_manifest")
    if agg:
        result["aggregation_manifest_summary"] = {
            "mode": agg.get("mode"),
            "row_counts": agg.get("row_counts", {}),
            "strict_canonical": agg.get("strict_canonical"),
            "causal_comparison_allowed": bool(agg.get("causal_comparison_allowed", False)),
            "warnings": agg.get("warnings", []),
        }
        if bool(agg.get("causal_comparison_allowed", False)):
            warnings.append("aggregation_manifest_has_causal_comparison_allowed_true")

    if not missing:
        result["reference_status"] = "READY_AS_NONCAUSAL_BASELINE_REFERENCE"
    elif (root / "kpi_by_window.parquet").exists():
        result["reference_status"] = "PARTIAL_REFERENCE_HAS_WINDOW_KPI"
    else:
        result["reference_status"] = "INCOMPLETE_REFERENCE_MISSING_CANONICAL_OUTPUTS"

    return result


def build_markdown(payload: Dict[str, Any]) -> str:
    lines = [
        "# Step 141 B-group baseline reference manifest",
        "",
        "This step records B0R/B1/B2 as local baseline references for later comparison with A-family MAPPO outputs.",
        "",
        "This step does not execute MAPPO and does not permit causal or paper-level claims.",
        "",
        "## Status",
        "",
        f"- audit_status: `{payload['audit_status']}`",
        f"- baseline_reference_status: `{payload['baseline_reference_status']}`",
        f"- ready_baseline_count: `{payload['ready_baseline_count']}`",
        f"- required_baseline_count: `{payload['required_baseline_count']}`",
        f"- causal_comparison_allowed: `{payload['causal_comparison_allowed']}`",
        f"- paper_level_claim_allowed: `{payload['paper_level_claim_allowed']}`",
        "",
        "## Baseline references",
        "",
        "| baseline | role | status | root |",
        "|---|---|---|---|",
    ]
    for ref in payload["baseline_references"]:
        lines.append(
            f"| {ref['baseline_id']} | {ref['reference_role']} | {ref['reference_status']} | `{ref['root']}` |"
        )
    lines.extend([
        "",
        "## Guard",
        "",
        "- B0R/B1/B2 are reference baselines, not reward candidates.",
        "- A-family MAPPO reward ablation remains separate.",
        "- Historical/replay/stub outputs remain non-causal references unless rerun through a validated causal simulator.",
        "",
    ])
    return "\n".join(lines)


def run_generator(project_root: Path, output_root: Path) -> Dict[str, Any]:
    project_root = project_root.resolve()
    output_root = output_root.resolve()

    refs = [inspect_baseline(project_root, spec) for spec in BASELINE_SPECS]
    ready_count = sum(
        1 for ref in refs
        if ref["reference_status"] == "READY_AS_NONCAUSAL_BASELINE_REFERENCE"
    )
    required_count = len(BASELINE_SPECS)

    blocking_reasons: List[str] = []
    warnings: List[str] = []

    for ref in refs:
        baseline_id = ref["baseline_id"]
        for w in ref.get("warnings", []):
            warnings.append(f"{baseline_id}: {w}")
        if ref["reference_status"] != "READY_AS_NONCAUSAL_BASELINE_REFERENCE":
            blocking_reasons.append(f"{baseline_id}_not_ready: {ref['reference_status']}")
        for v in recursive_forbidden_true(ref):
            blocking_reasons.append(f"{baseline_id}: {v}")

    audit_status = "PASS" if not blocking_reasons else "PASS_WITH_INCOMPLETE_BASELINES"
    baseline_reference_status = (
        "B_GROUP_BASELINE_REFERENCE_READY"
        if ready_count == required_count
        else "B_GROUP_BASELINE_REFERENCE_INCOMPLETE"
    )

    manifest_path = output_root / "baseline_reference_manifest_step141.json"
    md_path = output_root / "baseline_reference_manifest_step141.md"

    payload: Dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "step": STEP_ID,
        "created_at_utc": utc_now(),
        "project_root": str(project_root),
        "output_root": str(output_root),
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "audit_status": audit_status,
        "baseline_reference_status": baseline_reference_status,
        "ready_baseline_count": ready_count,
        "required_baseline_count": required_count,
        "blocking_reasons": blocking_reasons,
        "warnings": warnings,
        "baseline_references": refs,
        "comparison_role": {
            "B0R": "historical replay reference",
            "B1": "no-op reference",
            "B2": "rule-based reference",
            "A_family": "future A/A90/A80/A70 MAPPO reward ablation outputs",
        },
        "causal_comparison_allowed": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "best_reward_claim_allowed": False,
        "train_with_this_reward_allowed": False,
        "actual_training_allowed": False,
        "winner_selected": False,
        "trainable_reward_promoted": False,
        "scope_note": (
            "Step 141 creates a B-group baseline reference manifest only. "
            "It does not execute MAPPO, select a reward winner, or permit causal claims."
        ),
        "manifest_path": str(manifest_path),
        "markdown_path": str(md_path),
    }

    dump_json(manifest_path, payload)
    dump_text(md_path, build_markdown(payload))
    dump_text(project_root / "05_training" / "rewards" / "baseline_reference_manifest_step141.md", build_markdown(payload))
    dump_json(project_root / "05_training" / "rewards" / "baseline_reference_manifest_step141.latest.json", {
        "manifest_path": str(manifest_path),
        "audit_status": audit_status,
        "baseline_reference_status": baseline_reference_status,
        "ready_baseline_count": ready_count,
        "required_baseline_count": required_count,
        "causal_comparison_allowed": False,
        "paper_level_claim_allowed": False,
        "created_at_utc": payload["created_at_utc"],
    })

    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-root", default="artifacts/rewards/baseline_reference_manifest_step141")
    args = parser.parse_args()

    payload = run_generator(Path(args.project_root), Path(args.project_root) / args.output_root)

    print("[OK] Step 141 B-group baseline reference manifest generator completed")
    print(f"[OK] audit_status                 : {payload['audit_status']}")
    print(f"[OK] baseline_reference_status    : {payload['baseline_reference_status']}")
    print(f"[OK] ready_baseline_count         : {payload['ready_baseline_count']}")
    print(f"[OK] required_baseline_count      : {payload['required_baseline_count']}")
    print(f"[OK] causal_comparison_allowed    : {payload['causal_comparison_allowed']}")
    print(f"[OK] paper_level_claim_allowed    : {payload['paper_level_claim_allowed']}")
    print(f"[OK] manifest                     : {payload['manifest_path']}")

    for ref in payload["baseline_references"]:
        print(f"[OK] {ref['baseline_id']} status: {ref['reference_status']}")

    if payload["warnings"]:
        for warning in payload["warnings"]:
            print(f"[WARN] {warning}")

    if payload["blocking_reasons"]:
        for reason in payload["blocking_reasons"]:
            print(f"[INFO] incomplete_baseline_reason: {reason}")


if __name__ == "__main__":
    main()
