from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


ARTIFACT_VERSION = "reward_ablation_result_writer_guard_step118_v1"

EXPECTED_CANDIDATES = ["R0", "R1", "R2", "R3", "R4", "R5"]
EXPECTED_CONDITIONS = ["A", "A90", "A80", "A70"]
EXPECTED_SEEDS = [1, 2, 3]

CORE_12_KPIS = [
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

REQUIRED_DETAIL_COLUMNS = [
    "candidate_id",
    "condition_id",
    "seed",
    "window_count",
    "evaluation_horizon_minutes",
    "source_mode",
    "actual_results",
    "winner_selected",
    "train_with_this_reward_allowed",
    "hard_constraint_violation_count",
    "hard_constraint_pass",
    *CORE_12_KPIS,
]

REQUIRED_SUMMARY_COLUMNS = [
    "candidate_id",
    "condition_count",
    "seed_count",
    "window_count_total",
    "actual_results",
    "winner_selected",
    "best_claim_allowed",
    "train_with_this_reward_allowed",
    "hard_constraint_violation_count_total",
    "hard_constraint_pass_all",
    "selection_eligible",
    "selection_block_reason",
]

REQUIRED_VIOLATION_COLUMNS = [
    "candidate_id",
    "condition_id",
    "seed",
    "constraint_id",
    "metric",
    "threshold",
    "observed_value",
    "violation",
    "severity",
    "notes",
]


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


def resolve_path(project_root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return project_root / p


def infer_candidates(matrix: Dict[str, Any]) -> List[str]:
    raw = matrix.get("reward_candidates", matrix.get("candidates", []))
    out: List[str] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                cid = item.get("candidate_id") or item.get("id")
            else:
                cid = str(item)
            if cid:
                out.append(str(cid))
    return out or EXPECTED_CANDIDATES


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_detail_template_rows(
    candidates: List[str],
    conditions: List[str],
    seeds: List[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for candidate_id in candidates:
        for condition_id in conditions:
            for seed in seeds:
                row: Dict[str, Any] = {
                    "candidate_id": candidate_id,
                    "condition_id": condition_id,
                    "seed": int(seed),
                    "window_count": 0,
                    "evaluation_horizon_minutes": 30,
                    "source_mode": "template_not_evaluated",
                    "actual_results": False,
                    "winner_selected": False,
                    "train_with_this_reward_allowed": False,
                    "hard_constraint_violation_count": 0,
                    "hard_constraint_pass": False,
                }
                for kpi in CORE_12_KPIS:
                    row[kpi] = ""
                rows.append(row)

    return rows


def build_summary_template_rows(candidates: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for candidate_id in candidates:
        rows.append({
            "candidate_id": candidate_id,
            "condition_count": 0,
            "seed_count": 0,
            "window_count_total": 0,
            "actual_results": False,
            "winner_selected": False,
            "best_claim_allowed": False,
            "train_with_this_reward_allowed": False,
            "hard_constraint_violation_count_total": 0,
            "hard_constraint_pass_all": False,
            "selection_eligible": False,
            "selection_block_reason": "template_only_no_actual_ablation_results",
        })

    return rows


def build_violation_template_rows(candidates: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for candidate_id in candidates:
        rows.append({
            "candidate_id": candidate_id,
            "condition_id": "",
            "seed": "",
            "constraint_id": "template_no_violation_recorded",
            "metric": "",
            "threshold": "",
            "observed_value": "",
            "violation": False,
            "severity": "none",
            "notes": "Template row only. Actual ablation has not been evaluated.",
        })
    return rows


def build_selection_summary(candidates: List[str]) -> Dict[str, Any]:
    return {
        "artifact_version": "reward_ablation_selection_summary_template_step118_v1",
        "created_at_utc": utc_now(),
        "actual_results_available": False,
        "winner_selected": False,
        "selected_candidate_id": None,
        "selection_method": "blocked_until_actual_ablation_results_exist",
        "best_claim_allowed": False,
        "train_with_this_reward_allowed": False,
        "trainable_reward_promoted": False,
        "candidate_ids": candidates,
        "promotion_blockers": [
            "No actual ablation result table has been generated.",
            "No candidate has been evaluated across required A-family conditions.",
            "Hard-constraint pass/fail has not been computed from actual results.",
            "Winner selection is prohibited until actual result rows are present.",
            "MAPPO training remains disallowed by Step 116 promotion gate.",
        ],
    }


def build_manifest(
    matrix_path: Path,
    schema_path: Path,
    output_root: Path,
    candidates: List[str],
    detail_path: Path,
    summary_path: Path,
    violation_path: Path,
    selection_path: Path,
) -> Dict[str, Any]:
    return {
        "artifact_version": ARTIFACT_VERSION,
        "created_at_utc": utc_now(),
        "writer_status": "TEMPLATE_RESULTS_WRITTEN_NOT_EVALUATED",
        "candidate_ids": candidates,
        "condition_ids": EXPECTED_CONDITIONS,
        "seeds": EXPECTED_SEEDS,
        "source_inputs": {
            "reward_ablation_matrix_step115": str(matrix_path),
            "reward_ablation_result_schema_step117": str(schema_path),
        },
        "output_root": str(output_root),
        "output_files": {
            "detail_template_csv": str(detail_path),
            "summary_template_csv": str(summary_path),
            "hard_constraint_violations_template_csv": str(violation_path),
            "selection_summary_template_json": str(selection_path),
        },
        "actual_results": False,
        "winner_selected": False,
        "best_claim_allowed": False,
        "train_with_this_reward_allowed": False,
        "trainable_reward_promoted": False,
        "notes": [
            "Step 118 creates template result files and guard metadata only.",
            "It does not run MAPPO training or reward ablation evaluation.",
            "All result rows are marked actual_results=false.",
            "Winner selection is blocked until actual ablation rows are provided.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        default="05_training/rewards/reward_ablation_matrix_step115.json",
    )
    parser.add_argument(
        "--schema",
        default="05_training/rewards/reward_ablation_result_schema_step117.json",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/rewards/reward_ablation_result_writer_guard_step118",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    matrix_path = resolve_path(project_root, args.matrix)
    schema_path = resolve_path(project_root, args.schema)
    output_root = resolve_path(project_root, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not matrix_path.exists():
        raise SystemExit(f"reward ablation matrix not found: {matrix_path}")
    if not schema_path.exists():
        raise SystemExit(f"reward ablation result schema not found: {schema_path}")

    matrix = load_json_any_encoding(matrix_path)
    _schema = load_json_any_encoding(schema_path)

    candidates = infer_candidates(matrix)

    detail_rows = build_detail_template_rows(candidates, EXPECTED_CONDITIONS, EXPECTED_SEEDS)
    summary_rows = build_summary_template_rows(candidates)
    violation_rows = build_violation_template_rows(candidates)
    selection_summary = build_selection_summary(candidates)

    detail_path = output_root / "reward_ablation_results_by_candidate_seed_condition_template.csv"
    summary_path = output_root / "reward_ablation_results_by_candidate_template.csv"
    violation_path = output_root / "reward_ablation_hard_constraint_violations_template.csv"
    selection_path = output_root / "reward_ablation_selection_summary_template.json"
    manifest_path = output_root / "reward_ablation_result_writer_guard_manifest.json"

    write_csv(detail_path, REQUIRED_DETAIL_COLUMNS, detail_rows)
    write_csv(summary_path, REQUIRED_SUMMARY_COLUMNS, summary_rows)
    write_csv(violation_path, REQUIRED_VIOLATION_COLUMNS, violation_rows)
    dump_json(selection_path, selection_summary)

    manifest = build_manifest(
        matrix_path=matrix_path,
        schema_path=schema_path,
        output_root=output_root,
        candidates=candidates,
        detail_path=detail_path,
        summary_path=summary_path,
        violation_path=violation_path,
        selection_path=selection_path,
    )
    dump_json(manifest_path, manifest)

    print("[OK] Step 118 reward ablation result writer guard completed")
    print("[OK] writer_status : TEMPLATE_RESULTS_WRITTEN_NOT_EVALUATED")
    print(f"[OK] candidate_count: {len(candidates)}")
    print(f"[OK] detail_rows    : {len(detail_rows)}")
    print(f"[OK] summary_rows   : {len(summary_rows)}")
    print(f"[OK] violation_rows : {len(violation_rows)}")
    print("[OK] actual_results : False")
    print("[OK] winner_selected: False")
    print("[OK] train_allowed : False")
    print(f"[OK] output_root   : {output_root}")
    print(f"[OK] manifest      : {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
