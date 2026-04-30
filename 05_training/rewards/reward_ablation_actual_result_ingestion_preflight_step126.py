from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve(project_root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else project_root / p


def materialize(spec: Dict[str, Any], output_root: Path) -> Dict[str, Any]:
    checklist: List[Dict[str, Any]] = []
    for name in spec["required_bundle_files"]:
        checklist.append({
            "check_id": f"required_file_{name}",
            "category": "bundle_file",
            "required_item": name,
            "current_status": "not_ingested",
            "passed": False,
        })
    for name in spec["required_detail_columns"]:
        checklist.append({
            "check_id": f"required_column_{name}",
            "category": "detail_column",
            "required_item": name,
            "current_status": "not_ingested",
            "passed": False,
        })

    output_root.mkdir(parents=True, exist_ok=True)
    checklist_path = output_root / "actual_result_ingestion_preflight_checklist_step126.csv"
    bundle_template_path = output_root / "actual_result_bundle_manifest_template_step126.json"
    manifest_path = output_root / "reward_ablation_actual_result_ingestion_preflight_manifest_step126.json"

    write_csv(checklist_path, checklist, ["check_id", "category", "required_item", "current_status", "passed"])

    bundle_template = {
        "artifact_version": "actual_result_bundle_manifest_template_step126_v1",
        "created_at_utc": utc_now(),
        "actual_results": True,
        "template_only": False,
        "noop_only": False,
        "dry_run_only": False,
        "smoke_only": False,
        "winner_selected": False,
        "train_with_this_reward_allowed": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "required_row_count": int(spec["expected_actual_result_rows"]),
        "bundle_files": {name: "" for name in spec["required_bundle_files"]},
        "note": "Template only. It describes a future actual result bundle.",
    }
    dump_json(bundle_template_path, bundle_template)

    manifest = {
        "artifact_version": "reward_ablation_actual_result_ingestion_preflight_manifest_step126_v1",
        "created_at_utc": utc_now(),
        "preflight_status": spec["preflight_status"],
        "candidate_count": len(spec["candidate_ids"]),
        "condition_count": len(spec["condition_ids"]),
        "seed_count": len(spec["seeds"]),
        "expected_actual_result_rows": int(spec["expected_actual_result_rows"]),
        "required_bundle_file_count": len(spec["required_bundle_files"]),
        "required_detail_column_count": len(spec["required_detail_columns"]),
        "checklist_rows": len(checklist),
        "ingestion_allowed_now": False,
        "actual_results_ingested": False,
        "winner_selected": False,
        "train_with_this_reward_allowed": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "best_reward_claim_allowed": False,
        "output_root": str(output_root),
        "output_files": {
            "preflight_checklist_csv": str(checklist_path),
            "actual_result_bundle_manifest_template_json": str(bundle_template_path),
            "manifest": str(manifest_path),
        },
        "source_spec_snapshot": spec,
    }
    dump_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="05_training/rewards/reward_ablation_actual_result_ingestion_preflight_step126.json")
    parser.add_argument("--output-root", default="artifacts/rewards/reward_ablation_actual_result_ingestion_preflight_step126")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    spec = load_json(resolve(project_root, args.spec))
    output_root = resolve(project_root, args.output_root)
    manifest = materialize(spec, output_root)

    print("[OK] Step 126 reward ablation actual result ingestion preflight materialized")
    print(f"[OK] preflight_status: {manifest['preflight_status']}")
    print(f"[OK] expected_rows   : {manifest['expected_actual_result_rows']}")
    print(f"[OK] checklist_rows  : {manifest['checklist_rows']}")
    print(f"[OK] ingestion_allowed: {manifest['ingestion_allowed_now']}")
    print(f"[OK] actual_ingested : {manifest['actual_results_ingested']}")
    print(f"[OK] winner_selected: {manifest['winner_selected']}")
    print(f"[OK] train_allowed  : {manifest['train_with_this_reward_allowed']}")
    print(f"[OK] manifest       : {manifest['output_files']['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
