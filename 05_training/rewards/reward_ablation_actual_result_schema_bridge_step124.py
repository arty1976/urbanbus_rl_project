from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json_any_encoding(path: Path) -> Dict[str, Any]:
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


def stable_run_id(candidate_id: str, condition_id: str, seed: int) -> str:
    raw = f"{candidate_id}|{condition_id}|{seed}"
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"step124_{candidate_id}_{condition_id}_seed{int(seed):03d}_{h}"


def materialize(spec: Dict[str, Any], output_root: Path) -> Dict[str, Any]:
    candidate_ids = list(spec["candidate_ids"])
    condition_ids = list(spec["condition_ids"])
    seeds = [int(x) for x in spec["seeds"]]
    fieldnames = list(spec["required_actual_detail_columns"])

    detail_rows: List[Dict[str, Any]] = []
    for candidate_id in candidate_ids:
        for condition_id in condition_ids:
            for seed in seeds:
                row = {k: "" for k in fieldnames}
                row.update({
                    "run_id": stable_run_id(candidate_id, condition_id, seed),
                    "candidate_id": candidate_id,
                    "condition_id": condition_id,
                    "seed": int(seed),
                    "reward_candidate_version": "step115_candidate_matrix_v1",
                    "execution_status": "WAITING_FOR_ACTUAL_RESULTS",
                    "actual_result": False,
                    "trained_model": False,
                    "checkpoint_path": "",
                    "checkpoint_sha256": "",
                    "source_mode": "not_executed",
                    "adapter_mode": "not_executed",
                    "canonical_eval_path": "",
                    "reward_total_mean": "",
                    "reward_total_std": "",
                    "constraint_violation_count": "",
                    "hard_constraint_passed": False,
                    "paper_level_claim_allowed": False,
                    "causal_performance_claim_allowed": False,
                    "winner_eligible": False,
                })
                for kpi in spec["canonical_12_kpis"]:
                    row[f"{kpi}_mean"] = ""
                detail_rows.append(row)

    summary_fieldnames = [
        "candidate_id",
        "planned_condition_count",
        "planned_seed_count",
        "planned_run_count",
        "actual_result_count",
        "winner_eligible_count",
        "winner_selected",
        "train_with_this_reward_allowed",
    ]
    summary_rows = []
    for candidate_id in candidate_ids:
        summary_rows.append({
            "candidate_id": candidate_id,
            "planned_condition_count": len(condition_ids),
            "planned_seed_count": len(seeds),
            "planned_run_count": len(condition_ids) * len(seeds),
            "actual_result_count": 0,
            "winner_eligible_count": 0,
            "winner_selected": False,
            "train_with_this_reward_allowed": False,
        })

    output_root.mkdir(parents=True, exist_ok=True)
    detail_path = output_root / "reward_ablation_actual_result_detail_template_step124.csv"
    summary_path = output_root / "reward_ablation_actual_result_summary_template_step124.csv"
    selection_path = output_root / "reward_ablation_actual_result_selection_input_template_step124.json"
    manifest_path = output_root / "reward_ablation_actual_result_schema_bridge_manifest_step124.json"

    write_csv(detail_path, detail_rows, fieldnames)
    write_csv(summary_path, summary_rows, summary_fieldnames)

    selection_input = {
        "artifact_version": "reward_ablation_actual_result_selection_input_template_step124_v1",
        "created_at_utc": utc_now(),
        "actual_results": False,
        "winner_selected": False,
        "winner_eligible_candidates": [],
        "selection_ready": False,
        "reason": "Template only. No actual ablation result has been ingested.",
    }
    dump_json(selection_path, selection_input)

    manifest = {
        "artifact_version": "reward_ablation_actual_result_schema_bridge_manifest_step124_v1",
        "created_at_utc": utc_now(),
        "bridge_status": spec["bridge_status"],
        "candidate_count": len(candidate_ids),
        "condition_count": len(condition_ids),
        "seed_count": len(seeds),
        "detail_rows": len(detail_rows),
        "expected_actual_detail_rows": int(spec["expected_actual_detail_rows"]),
        "actual_results": False,
        "winner_selected": False,
        "winner_eligible": False,
        "train_with_this_reward_allowed": False,
        "actual_training_allowed": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "best_reward_claim_allowed": False,
        "output_root": str(output_root),
        "output_files": {
            "actual_detail_template_csv": str(detail_path),
            "actual_summary_template_csv": str(summary_path),
            "selection_input_template_json": str(selection_path),
            "manifest": str(manifest_path),
        },
        "source_spec_snapshot": spec,
    }
    dump_json(manifest_path, manifest)
    return manifest


def resolve_path(project_root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return project_root / p


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="05_training/rewards/reward_ablation_actual_result_schema_bridge_step124.json")
    parser.add_argument("--output-root", default="artifacts/rewards/reward_ablation_actual_result_schema_bridge_step124")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    spec_path = resolve_path(project_root, args.spec)
    output_root = resolve_path(project_root, args.output_root)

    spec = load_json_any_encoding(spec_path)
    manifest = materialize(spec, output_root)

    print("[OK] Step 124 reward ablation actual result schema bridge materialized")
    print(f"[OK] bridge_status : {manifest['bridge_status']}")
    print(f"[OK] detail_rows   : {manifest['detail_rows']}")
    print(f"[OK] actual_results: {manifest['actual_results']}")
    print(f"[OK] winner_selected: {manifest['winner_selected']}")
    print(f"[OK] train_allowed : {manifest['train_with_this_reward_allowed']}")
    print(f"[OK] output_root   : {output_root}")
    print(f"[OK] manifest      : {manifest['output_files']['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
