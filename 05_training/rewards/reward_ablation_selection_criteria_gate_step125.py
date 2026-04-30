from __future__ import annotations
import argparse, csv, json
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
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)

def materialize(spec: Dict[str, Any], output_root: Path) -> Dict[str, Any]:
    cols = list(spec["scorecard_columns_required_for_future_selection"])
    rows = []
    for candidate_id in spec["candidate_ids"]:
        row = {c: "" for c in cols}
        row.update({
            "candidate_id": candidate_id,
            "eligible": False,
            "actual_result_count": 0,
            "seed_variance_summary": "not_evaluated",
            "selection_rank": "",
            "selection_reason": "No actual reward ablation results have been ingested.",
            "winner_selected": False,
        })
        rows.append(row)
    output_root.mkdir(parents=True, exist_ok=True)
    scorecard = output_root / "reward_ablation_selection_scorecard_template_step125.csv"
    manifest_path = output_root / "reward_ablation_selection_criteria_gate_manifest_step125.json"
    write_csv(scorecard, rows, cols)
    manifest = {
        "artifact_version": "reward_ablation_selection_criteria_gate_manifest_step125_v1",
        "created_at_utc": utc_now(),
        "gate_status": spec["gate_status"],
        "candidate_count": len(spec["candidate_ids"]),
        "condition_count": len(spec["condition_ids"]),
        "seed_count": len(spec["seeds"]),
        "required_actual_result_rows": int(spec["required_actual_result_rows"]),
        "scorecard_rows": len(rows),
        "actual_results": False,
        "winner_selected": False,
        "winner_selection_allowed_now": False,
        "train_with_this_reward_allowed": False,
        "actual_training_allowed": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "best_reward_claim_allowed": False,
        "output_root": str(output_root),
        "output_files": {"scorecard_template_csv": str(scorecard), "manifest": str(manifest_path)},
        "source_spec_snapshot": spec,
    }
    dump_json(manifest_path, manifest)
    return manifest

def resolve_path(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="05_training/rewards/reward_ablation_selection_criteria_gate_step125.json")
    ap.add_argument("--output-root", default="artifacts/rewards/reward_ablation_selection_criteria_gate_step125")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    manifest = materialize(load_json(resolve_path(root, args.spec)), resolve_path(root, args.output_root))
    print("[OK] Step 125 reward ablation selection criteria gate materialized")
    print(f"[OK] gate_status   : {manifest['gate_status']}")
    print(f"[OK] candidate_count: {manifest['candidate_count']}")
    print(f"[OK] scorecard_rows: {manifest['scorecard_rows']}")
    print(f"[OK] actual_results: {manifest['actual_results']}")
    print(f"[OK] winner_selected: {manifest['winner_selected']}")
    print(f"[OK] train_allowed : {manifest['train_with_this_reward_allowed']}")
    print(f"[OK] manifest      : {manifest['output_files']['manifest']}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
