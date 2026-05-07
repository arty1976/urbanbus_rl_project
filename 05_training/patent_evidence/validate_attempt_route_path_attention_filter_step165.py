from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


EXPECTED_STATUS = "ATTEMPT_ROUTE_PATH_ATTENTION_FILTERED_FOR_ZERO_LOSS_EVIDENCE_NONCLAIM"
EXPECTED_FLAGS = {
    "simulation_evidence_only": True,
    "actual_operational_claim_allowed": False,
    "paper_level_claim_allowed": False,
    "causal_performance_claim_allowed": False,
    "train_allowed": False,
}


def load_json_any_encoding(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


def bool_from_any(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def validate_manifest(path: Path, require_real_attention: bool = False, require_no_fallback: bool = False) -> Dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"manifest not found: {path}")
    m = load_json_any_encoding(path)

    if m.get("audit_status") != "PASS":
        raise RuntimeError(f"audit_status must be PASS, got {m.get('audit_status')}")
    if m.get("bundle_status") != EXPECTED_STATUS:
        raise RuntimeError(f"bundle_status mismatch: {m.get('bundle_status')}")

    for key, expected in EXPECTED_FLAGS.items():
        if bool_from_any(m.get(key)) != expected:
            raise RuntimeError(f"guard flag mismatch: {key}={m.get(key)}, expected={expected}")

    out = m.get("output_files", {})
    required_files = [
        "pickup_attempt_events",
        "eta_counterfactual",
        "gatv2_attention",
        "attempt_route_path_filter_report",
        "attention_path_mass_by_attempt",
        "run_manifest",
    ]
    for key in required_files:
        p = Path(out.get(key, ""))
        if not p.exists():
            # Allow paths relative to manifest parent via output_root.
            p2 = Path(m.get("output_root", ".")) / p.name
            if not p2.exists():
                raise RuntimeError(f"output file missing for {key}: {p}")

    rows = m.get("row_counts", {})
    attempt_count = int(rows.get("pickup_attempt_count", -1))
    filtered_count = int(rows.get("filtered_attention_row_count", -1))
    with_attention = int(rows.get("attempt_count_with_filtered_attention", -1))
    fallback_count = int(rows.get("fallback_attempt_count", -1))
    exact_count = int(rows.get("exact_edge_overlap_count", -1))

    if attempt_count <= 0:
        raise RuntimeError("pickup_attempt_count must be positive")
    if filtered_count <= 0:
        raise RuntimeError("filtered_attention_row_count must be positive")
    if with_attention <= 0:
        raise RuntimeError("attempt_count_with_filtered_attention must be positive")
    if exact_count <= 0:
        raise RuntimeError("exact_edge_overlap_count must be positive for Step 165 sample/validated path")

    quality = m.get("quality", {})
    if require_real_attention and not bool_from_any(quality.get("real_gatv2conv_attention_extracted_all")):
        raise RuntimeError("real_gatv2conv_attention_extracted_all must be true")
    if require_no_fallback and fallback_count != 0:
        raise RuntimeError(f"fallback_attempt_count must be 0, got {fallback_count}")

    attn_path = Path(out["gatv2_attention"])
    if not attn_path.exists():
        attn_path = Path(m.get("output_root", ".")) / attn_path.name
    attn = pd.read_csv(attn_path)
    required_attn_cols = [
        "attempt_id",
        "state_ts",
        "layer_id",
        "head_id",
        "src_node",
        "dst_node",
        "attention_weight",
        "path_segment",
        "filter_match_type",
        "attempt_specific_filter_applied",
    ]
    missing = [c for c in required_attn_cols if c not in attn.columns]
    if missing:
        raise RuntimeError(f"gatv2_attention missing columns: {missing}")

    if require_real_attention:
        if "real_gatv2conv_attention_extracted" not in attn.columns:
            raise RuntimeError("gatv2_attention missing real_gatv2conv_attention_extracted")
        if not attn["real_gatv2conv_attention_extracted"].map(bool_from_any).all():
            raise RuntimeError("not all attention rows are marked real_gatv2conv_attention_extracted")

    print("[OK] Step 165 attempt-specific route/path attention filter validation PASS")
    print(f"[OK] manifest                  : {path}")
    print(f"[OK] pickup_attempt_count      : {attempt_count}")
    print(f"[OK] filtered_attention_row_count : {filtered_count}")
    print(f"[OK] attempt_count_with_filtered_attention : {with_attention}")
    print(f"[OK] fallback_attempt_count    : {fallback_count}")
    print(f"[OK] exact_edge_overlap_count  : {exact_count}")
    print(f"[OK] paper_level_claim_allowed : {m.get('paper_level_claim_allowed')}")
    print(f"[OK] causal_performance_claim_allowed : {m.get('causal_performance_claim_allowed')}")
    return m


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--require-real-attention", action="store_true")
    parser.add_argument("--require-no-fallback", action="store_true")
    args = parser.parse_args()
    validate_manifest(Path(args.manifest), args.require_real_attention, args.require_no_fallback)


if __name__ == "__main__":
    main()
