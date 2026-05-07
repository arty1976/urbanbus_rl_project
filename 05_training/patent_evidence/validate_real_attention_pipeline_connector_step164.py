from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

EXPECTED_FLAGS = {
    "simulation_evidence_only": True,
    "actual_operational_claim_allowed": False,
    "paper_level_claim_allowed": False,
    "causal_performance_claim_allowed": False,
    "trained_model_claim_allowed": False,
    "train_allowed": False,
}
BUNDLE_STATUS = "REAL_GATV2_ATTENTION_CONNECTED_TO_ZERO_LOSS_EVIDENCE_NONCLAIM"
REQUIRED_ATTEMPT_COLUMNS = ["attempt_id","state_ts","condition_id","seed","vehicle_id","existing_passenger_id","new_passenger_id","route_id","direction_id","pickup_stop_id","dropoff_stop_id","decision"]
REQUIRED_ETA_COLUMNS = ["attempt_id","existing_passenger_id","eta_without_new_pickup_sec","eta_with_new_pickup_sec"]
REQUIRED_ATTENTION_COLUMNS = ["attempt_id","state_ts","layer_id","head_id","src_node","dst_node","attention_weight"]

def load_json_any_encoding(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")

def bool_from_any(value: Any) -> bool:
    if isinstance(value, bool): return value
    if isinstance(value, (int, float)): return bool(value)
    return str(value).strip().lower() in {"true","1","yes","y"}

def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet": return pd.read_parquet(path)
    if path.suffix.lower() == ".csv": return pd.read_csv(path)
    if path.suffix.lower() == ".jsonl": return pd.read_json(path, lines=True)
    if path.suffix.lower() == ".json": return pd.read_json(path)
    raise RuntimeError(f"unsupported table format: {path}")

def require_columns(df: pd.DataFrame, required: List[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing: raise RuntimeError(f"{label} missing required columns: {missing}")

def resolve_output(output_files: Dict[str, Any], key: str, manifest_path: Path) -> Path:
    p = Path(str(output_files.get(key, "")))
    if p.exists(): return p
    p2 = manifest_path.parent / p.name
    if p2.exists(): return p2
    raise RuntimeError(f"missing output file {key}: {p}")

def validate_manifest(path: Path, require_real_attention: bool) -> Dict[str, Any]:
    if not path.exists(): raise RuntimeError(f"manifest not found: {path}")
    m = load_json_any_encoding(path)
    if m.get("audit_status") != "PASS": raise RuntimeError(f"audit_status must be PASS, got {m.get('audit_status')}")
    if m.get("bundle_status") != BUNDLE_STATUS: raise RuntimeError(f"bundle_status mismatch: {m.get('bundle_status')}")
    for key, expected in EXPECTED_FLAGS.items():
        if bool_from_any(m.get(key)) != expected:
            raise RuntimeError(f"guard flag mismatch: {key}={m.get(key)}, expected={expected}")
    if not bool_from_any(m.get("real_attention_connected")): raise RuntimeError("real_attention_connected must be true")
    if not bool_from_any(m.get("proxy_attention_replaced")): raise RuntimeError("proxy_attention_replaced must be true")

    files = m.get("output_files", {})
    attempts_path = resolve_output(files, "pickup_attempt_events", path)
    eta_path = resolve_output(files, "eta_counterfactual", path)
    attention_path = resolve_output(files, "gatv2_attention", path)
    run_manifest_path = resolve_output(files, "run_manifest", path)
    quality_path = resolve_output(files, "attention_connection_quality_report", path)

    attempts = read_table(attempts_path); eta = read_table(eta_path); attention = read_table(attention_path)
    require_columns(attempts, REQUIRED_ATTEMPT_COLUMNS, "pickup_attempt_events")
    require_columns(eta, REQUIRED_ETA_COLUMNS, "eta_counterfactual")
    require_columns(attention, REQUIRED_ATTENTION_COLUMNS, "gatv2_attention")
    if attempts.empty or eta.empty or attention.empty: raise RuntimeError("Step 164 output tables must not be empty")
    attempt_ids = set(attempts["attempt_id"].astype(str))
    if missing := sorted(attempt_ids - set(eta["attempt_id"].astype(str))): raise RuntimeError(f"eta missing attempt_ids: {missing[:10]}")
    if missing := sorted(attempt_ids - set(attention["attempt_id"].astype(str))): raise RuntimeError(f"attention missing attempt_ids: {missing[:10]}")
    weights = pd.to_numeric(attention["attention_weight"], errors="raise")
    if weights.isna().any() or bool((weights < 0).any()): raise RuntimeError("attention_weight must be finite and non-negative")
    if require_real_attention:
        if "real_gatv2conv_attention_extracted" not in attention.columns: raise RuntimeError("gatv2_attention missing real_gatv2conv_attention_extracted")
        if not bool(attention["real_gatv2conv_attention_extracted"].map(bool_from_any).all()): raise RuntimeError("not all attention rows are real GATv2Conv rows")
        if "attention_source" not in attention.columns: raise RuntimeError("gatv2_attention missing attention_source")
        sources = set(attention["attention_source"].astype(str).str.lower().unique().tolist())
        if not any("gatv2conv" in s or "return_attention_weights" in s for s in sources): raise RuntimeError(f"attention_source mismatch: {sources}")
    quality = load_json_any_encoding(quality_path)
    if int(quality.get("pickup_attempt_count", -1)) != len(attempts): raise RuntimeError("quality pickup_attempt_count mismatch")
    if int(quality.get("real_attention_row_count", -1)) != len(attention): raise RuntimeError("quality real_attention_row_count mismatch")
    if require_real_attention and not bool_from_any(quality.get("real_gatv2conv_attention_extracted_all")): raise RuntimeError("quality real_gatv2conv_attention_extracted_all must be true")
    run_manifest = load_json_any_encoding(run_manifest_path)
    for key, expected in EXPECTED_FLAGS.items():
        if bool_from_any(run_manifest.get(key)) != expected: raise RuntimeError(f"run_manifest guard mismatch: {key}={run_manifest.get(key)}, expected={expected}")
    if not bool_from_any(run_manifest.get("real_attention_connected")): raise RuntimeError("run_manifest real_attention_connected must be true")
    if not bool_from_any(run_manifest.get("proxy_attention_replaced")): raise RuntimeError("run_manifest proxy_attention_replaced must be true")
    return m

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Step 164 real attention pipeline connector bundle")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--require-real-attention", action="store_true")
    args = parser.parse_args()
    m = validate_manifest(Path(args.manifest), require_real_attention=bool(args.require_real_attention))
    s = m.get("summary", {})
    print("[OK] Step 164 real attention pipeline connector validation PASS")
    print(f"[OK] manifest                  : {args.manifest}")
    print(f"[OK] pickup_attempt_count      : {s.get('pickup_attempt_count')}")
    print(f"[OK] real_attention_row_count  : {s.get('real_attention_row_count')}")
    print(f"[OK] attempt_count_with_real_attention : {s.get('attempt_count_with_real_attention')}")
    print(f"[OK] real_gatv2conv_attention_extracted_all : {s.get('real_gatv2conv_attention_extracted_all')}")
    print("[OK] paper_level_claim_allowed : False")
    print("[OK] causal_performance_claim_allowed : False")

if __name__ == "__main__":
    main()
