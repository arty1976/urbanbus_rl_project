from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

REQUIRED_SNAPSHOT_ATTENTION_COLUMNS = [
    "state_ts",
    "layer_id",
    "head_id",
    "edge_rank",
    "src_node",
    "dst_node",
    "src_idx",
    "dst_idx",
    "attention_weight",
    "attention_source",
    "real_gatv2conv_attention_extracted",
]

REQUIRED_STEP160_ATTENTION_COLUMNS = [
    "attempt_id",
    "state_ts",
    "layer_id",
    "head_id",
    "src_node",
    "dst_node",
    "attention_weight",
]

REQUIRED_FALSE_FLAGS = [
    "actual_operational_claim_allowed",
    "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
    "trained_model_claim_allowed",
    "train_allowed",
]


def load_json_any_encoding(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    raise RuntimeError(f"unsupported table format: {path}")


def require_columns(df: pd.DataFrame, required: List[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"{label} missing required columns: {missing}")


def bool_from_any(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def validate_manifest(manifest_path: Path, require_real_gatv2conv: bool = False) -> Dict[str, Any]:
    manifest = load_json_any_encoding(manifest_path)
    for flag in REQUIRED_FALSE_FLAGS:
        if bool_from_any(manifest.get(flag, True)):
            raise RuntimeError(f"claim guard must remain false: {flag}")
    if not bool_from_any(manifest.get("simulation_evidence_only", False)):
        raise RuntimeError("simulation_evidence_only must remain true")

    outputs = manifest.get("output_files", {})
    if "gatv2_real_attention_edges" not in outputs:
        raise RuntimeError("manifest missing output_files.gatv2_real_attention_edges")
    snapshot_path = Path(outputs["gatv2_real_attention_edges"])
    if not snapshot_path.exists():
        # Interpret relative to manifest directory if needed.
        alt = manifest_path.parent / snapshot_path.name
        if alt.exists():
            snapshot_path = alt
        else:
            raise RuntimeError(f"snapshot attention output not found: {snapshot_path}")
    snapshot = read_table(snapshot_path)
    require_columns(snapshot, REQUIRED_SNAPSHOT_ATTENTION_COLUMNS, "gatv2_real_attention_edges")
    if snapshot.empty:
        raise RuntimeError("gatv2_real_attention_edges is empty")
    weights = pd.to_numeric(snapshot["attention_weight"], errors="raise")
    if weights.isna().any():
        raise RuntimeError("snapshot attention_weight contains null values")
    if bool((weights < 0).any()):
        raise RuntimeError("snapshot attention_weight must be non-negative")

    summary = manifest.get("summary", {})
    real = bool_from_any(summary.get("real_gatv2conv_attention_extracted", False))
    if require_real_gatv2conv and not real:
        raise RuntimeError("real GATv2Conv attention extraction required but manifest reports false")
    if require_real_gatv2conv:
        source_counts = summary.get("attention_source_counts", {})
        if "gatv2conv_return_attention_weights" not in source_counts:
            raise RuntimeError("required real attention source missing: gatv2conv_return_attention_weights")

    attempt_path_value = outputs.get("gatv2_attention")
    attempt_row_count = 0
    if attempt_path_value:
        attempt_path = Path(attempt_path_value)
        if not attempt_path.exists():
            alt = manifest_path.parent / attempt_path.name
            if alt.exists():
                attempt_path = alt
            else:
                raise RuntimeError(f"attempt-level attention output not found: {attempt_path}")
        attempt = read_table(attempt_path)
        require_columns(attempt, REQUIRED_STEP160_ATTENTION_COLUMNS, "gatv2_attention")
        if attempt.empty:
            raise RuntimeError("gatv2_attention is empty despite being listed")
        aw = pd.to_numeric(attempt["attention_weight"], errors="raise")
        if aw.isna().any() or bool((aw < 0).any()):
            raise RuntimeError("attempt-level attention_weight must be finite and non-negative")
        attempt_row_count = int(len(attempt))

    print("[OK] Step 163 GATv2 real attention extractor validation PASS")
    print(f"[OK] manifest                  : {manifest_path}")
    print(f"[OK] audit_status              : {manifest.get('audit_status')}")
    print(f"[OK] bundle_status             : {manifest.get('bundle_status')}")
    print(f"[OK] real_gatv2conv_attention_extracted : {real}")
    print(f"[OK] snapshot_attention_row_count       : {len(snapshot)}")
    print(f"[OK] attempt_attention_row_count        : {attempt_row_count}")
    print(f"[OK] paper_level_claim_allowed : {manifest.get('paper_level_claim_allowed')}")
    print(f"[OK] causal_performance_claim_allowed : {manifest.get('causal_performance_claim_allowed')}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Step 163 GATv2 real attention extractor bundle")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--require-real-gatv2conv", action="store_true")
    args = parser.parse_args()
    validate_manifest(Path(args.manifest), require_real_gatv2conv=bool(args.require_real_gatv2conv))


if __name__ == "__main__":
    main()
