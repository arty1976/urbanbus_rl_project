from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


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


def materialize(spec: Dict[str, Any], output_root: Path) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)

    decision_path = output_root / "trainable_reward_promotion_decision_step127.json"
    evidence_path = output_root / "trainable_reward_promotion_missing_evidence_step127.json"
    manifest_path = output_root / "trainable_reward_promotion_decision_package_manifest_step127.json"

    decision = {
        "artifact_version": "trainable_reward_promotion_decision_step127_v1",
        "created_at_utc": utc_now(),
        "decision_status": spec["decision_status"],
        "decision": spec["decision"],
        "current_evidence_status": spec["current_evidence_status"],
        "guard_flags": spec["guard_flags"],
    }
    dump_json(decision_path, decision)

    missing_evidence = {
        "artifact_version": "trainable_reward_promotion_missing_evidence_step127_v1",
        "created_at_utc": utc_now(),
        "promotion_evidence_complete": False,
        "missing_items": list(spec["decision_inputs_required_before_promotion"]),
        "reason": spec["decision"]["promotion_block_reason"],
    }
    dump_json(evidence_path, missing_evidence)

    manifest = {
        "artifact_version": "trainable_reward_promotion_decision_package_manifest_step127_v1",
        "created_at_utc": utc_now(),
        "decision_status": spec["decision_status"],
        "candidate_count": len(spec["candidate_ids"]),
        "condition_count": len(spec["condition_ids"]),
        "seed_count": len(spec["seeds"]),
        "required_actual_result_rows": int(spec["required_actual_result_rows"]),
        "actual_results": False,
        "actual_results_ingested": False,
        "winner_selected": False,
        "trainable_reward_promoted": False,
        "selected_candidate_id": None,
        "train_with_this_reward_allowed": False,
        "actual_training_allowed": False,
        "final_reward_design_claim_allowed": False,
        "best_reward_claim_allowed": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "output_root": str(output_root),
        "output_files": {
            "decision_json": str(decision_path),
            "missing_evidence_json": str(evidence_path),
            "manifest": str(manifest_path)
        },
        "source_spec_snapshot": spec
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
    parser.add_argument("--spec", default="05_training/rewards/trainable_reward_promotion_decision_package_step127.json")
    parser.add_argument("--output-root", default="artifacts/rewards/trainable_reward_promotion_decision_package_step127")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    spec_path = resolve_path(project_root, args.spec)
    output_root = resolve_path(project_root, args.output_root)

    spec = load_json_any_encoding(spec_path)
    manifest = materialize(spec, output_root)

    print("[OK] Step 127 trainable reward promotion decision package materialized")
    print(f"[OK] decision_status: {manifest['decision_status']}")
    print(f"[OK] promoted       : {manifest['trainable_reward_promoted']}")
    print(f"[OK] selected       : {manifest['selected_candidate_id']}")
    print(f"[OK] train_allowed  : {manifest['train_with_this_reward_allowed']}")
    print(f"[OK] actual_results : {manifest['actual_results']}")
    print(f"[OK] manifest       : {manifest['output_files']['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
