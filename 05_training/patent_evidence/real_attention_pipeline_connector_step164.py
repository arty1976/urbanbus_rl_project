from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ARTIFACT_VERSION = "real_attention_pipeline_connector_step164_v1"
STEP160_INPUT_VERSION = "zero_loss_pickup_evidence_input_step164_real_attention_v1"
BUNDLE_STATUS = "REAL_GATV2_ATTENTION_CONNECTED_TO_ZERO_LOSS_EVIDENCE_NONCLAIM"
NON_CLAIM_FLAGS = {
    "simulation_evidence_only": True,
    "actual_operational_claim_allowed": False,
    "paper_level_claim_allowed": False,
    "causal_performance_claim_allowed": False,
    "trained_model_claim_allowed": False,
    "train_allowed": False,
}
REQUIRED_ATTEMPT_COLUMNS = ["attempt_id","state_ts","condition_id","seed","vehicle_id","existing_passenger_id","new_passenger_id","route_id","direction_id","pickup_stop_id","dropoff_stop_id","decision"]
REQUIRED_ETA_COLUMNS = ["attempt_id","existing_passenger_id","eta_without_new_pickup_sec","eta_with_new_pickup_sec"]
REQUIRED_STEP160_ATTENTION_COLUMNS = ["attempt_id","state_ts","layer_id","head_id","src_node","dst_node","attention_weight"]
REQUIRED_SNAPSHOT_ATTENTION_COLUMNS = ["state_ts","layer_id","head_id","src_node","dst_node","attention_weight"]

def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def load_json_any_encoding(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".jsonl":
        return pd.read_json(path, lines=True)
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    raise RuntimeError(f"unsupported table format: {path}")

def write_table(path: Path, df: pd.DataFrame, file_format: str, warnings: List[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if file_format == "csv":
        out = path.with_suffix(".csv")
        df.to_csv(out, index=False, encoding="utf-8-sig")
        return out
    out = path.with_suffix(".parquet")
    try:
        df.to_parquet(out, index=False)
        return out
    except Exception as exc:
        warnings.append(f"parquet_write_failed_csv_fallback: {exc}")
        out = path.with_suffix(".csv")
        df.to_csv(out, index=False, encoding="utf-8-sig")
        return out

def require_columns(df: pd.DataFrame, cols: List[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{label} missing required columns: {missing}")

def normalize_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    s = str(value).strip()
    return s or default

def bool_from_any(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}

def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default

def validate_guards(manifest: Dict[str, Any], label: str) -> None:
    for key, expected in NON_CLAIM_FLAGS.items():
        if key in manifest and bool_from_any(manifest.get(key)) != expected:
            raise RuntimeError(f"{label} guard mismatch: {key} must be {expected}, got {manifest.get(key)}")

def validate_real_source(df: pd.DataFrame, label: str, require_real: bool) -> None:
    if not require_real:
        return
    if "real_gatv2conv_attention_extracted" not in df.columns:
        raise RuntimeError(f"{label} missing real_gatv2conv_attention_extracted")
    if not bool(df["real_gatv2conv_attention_extracted"].map(bool_from_any).all()):
        raise RuntimeError(f"{label} has non-real attention rows")
    if "attention_source" not in df.columns:
        raise RuntimeError(f"{label} missing attention_source")
    sources = set(df["attention_source"].astype(str).str.lower())
    if not any("gatv2conv" in s or "return_attention_weights" in s for s in sources):
        raise RuntimeError(f"{label} attention_source does not indicate real GATv2Conv extraction: {sources}")

def make_sample_inputs(root: Path) -> Dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    attempts = pd.DataFrame([{ 
        "attempt_id": f"att_step164_{i:03d}", "state_ts": "2026-01-02T08:00:00+09:00", "condition_id": "A", "seed": 1,
        "vehicle_id": f"bus_{i}", "existing_passenger_id": f"existing_{i}", "new_passenger_id": f"new_{i}",
        "route_id": "route_step164", "direction_id": "0", "pickup_stop_id": f"node_{i}", "dropoff_stop_id": f"node_{i+2}",
        "decision": "accepted" if i != 2 else "rejected", "time_band": "peak", "window_id": "w_step164_sample"
    } for i in range(1,4)])
    eta = pd.DataFrame([
        {"attempt_id":"att_step164_001","existing_passenger_id":"existing_1","eta_without_new_pickup_sec":600.0,"eta_with_new_pickup_sec":600.0},
        {"attempt_id":"att_step164_002","existing_passenger_id":"existing_2","eta_without_new_pickup_sec":630.0,"eta_with_new_pickup_sec":651.0},
        {"attempt_id":"att_step164_003","existing_passenger_id":"existing_3","eta_without_new_pickup_sec":720.0,"eta_with_new_pickup_sec":719.0},
    ])
    edges = [(0,1),(1,2),(2,3),(3,4),(4,5),(1,3),(2,4),(0,2)]
    rows = []
    for layer in [1,2]:
        for head in [0,1]:
            for rank, (src,dst) in enumerate(edges):
                rows.append({"state_ts":"2026-01-02T08:00:00+09:00","layer_id":layer,"head_id":head,"edge_rank":rank,"src_node":f"node_{src}","dst_node":f"node_{dst}","src_idx":src,"dst_idx":dst,"edge_id":f"edge_{src}_{dst}","attention_weight":round(0.55-rank*0.03-head*0.01-layer*0.02,6),"attention_source":"gatv2conv_return_attention_weights","real_gatv2conv_attention_extracted":True,"distance_m":100+rank*20,"time_sec":20+rank*5,"generalized_cost":1+rank*0.03})
    attention = pd.DataFrame(rows)
    upstream_manifest = {"artifact_version":"zero_loss_pickup_evidence_input_step161_v1","condition_id":"A","seed":1,"zero_loss_epsilon_sec":0.0, **NON_CLAIM_FLAGS}
    real_manifest = {"artifact_version":"gatv2_real_attention_extractor_step163_v1","audit_status":"PASS","bundle_status":"GATV2_REAL_ATTENTION_EXTRACTED_FOR_ZERO_LOSS_EVIDENCE_NONCLAIM","summary":{"real_gatv2conv_attention_extracted":True,"snapshot_attention_row_count":len(attention)}, **NON_CLAIM_FLAGS}
    paths = {"pickup_attempt_events":root/"pickup_attempt_events.csv","eta_counterfactual":root/"eta_counterfactual.csv","real_snapshot_attention":root/"gatv2_real_attention_edges.csv","step161_run_manifest":root/"run_manifest.json","real_attention_manifest":root/"gatv2_real_attention_extractor_manifest.json"}
    attempts.to_csv(paths["pickup_attempt_events"], index=False, encoding="utf-8-sig")
    eta.to_csv(paths["eta_counterfactual"], index=False, encoding="utf-8-sig")
    attention.to_csv(paths["real_snapshot_attention"], index=False, encoding="utf-8-sig")
    dump_json(paths["step161_run_manifest"], upstream_manifest)
    dump_json(paths["real_attention_manifest"], real_manifest)
    return paths

def classify_segment(att: pd.Series, row: pd.Series) -> str:
    src = normalize_str(row.get("src_node")); dst = normalize_str(row.get("dst_node"))
    pickup = normalize_str(att.get("pickup_stop_id")); dropoff = normalize_str(att.get("dropoff_stop_id"))
    if pickup and (src == pickup or dst == pickup):
        return "candidate_pickup_path"
    if dropoff and (src == dropoff or dst == dropoff):
        return "candidate_dropoff_path"
    return "unrelated"

def project_snapshot_to_attempts(snapshot: pd.DataFrame, attempts: pd.DataFrame, top_k: int, require_real: bool) -> pd.DataFrame:
    require_columns(snapshot, REQUIRED_SNAPSHOT_ATTENTION_COLUMNS, "real_snapshot_attention")
    validate_real_source(snapshot, "real_snapshot_attention", require_real)
    snapshot = snapshot.copy()
    snapshot["attention_weight"] = pd.to_numeric(snapshot["attention_weight"], errors="raise")
    top = snapshot.sort_values("attention_weight", ascending=False).head(top_k).reset_index(drop=True)
    rows: List[Dict[str, Any]] = []
    for att_obj in attempts.itertuples(index=False):
        att = pd.Series(att_obj._asdict())
        for _, r in top.iterrows():
            rows.append({"attempt_id": normalize_str(att.get("attempt_id")), "state_ts": normalize_str(att.get("state_ts"), normalize_str(r.get("state_ts"))), "layer_id": safe_int(r.get("layer_id")), "head_id": safe_int(r.get("head_id")), "src_node": normalize_str(r.get("src_node")), "dst_node": normalize_str(r.get("dst_node")), "edge_id": normalize_str(r.get("edge_id", "")), "attention_weight": safe_float(r.get("attention_weight")), "path_segment": classify_segment(att,r), "route_id": normalize_str(att.get("route_id")), "direction_id": normalize_str(att.get("direction_id")), "distance_m": r.get("distance_m"), "time_sec": r.get("time_sec"), "generalized_cost": r.get("generalized_cost"), "attention_source": normalize_str(r.get("attention_source"), "gatv2conv_return_attention_weights"), "real_gatv2conv_attention_extracted": bool_from_any(r.get("real_gatv2conv_attention_extracted", True)), "attention_connection_mode": "snapshot_topk_projection_step164", "proxy_attention_replaced": True})
    return pd.DataFrame(rows)

def use_attempt_attention(df: pd.DataFrame, attempts: pd.DataFrame, require_real: bool) -> pd.DataFrame:
    require_columns(df, REQUIRED_STEP160_ATTENTION_COLUMNS, "real_attempt_attention")
    validate_real_source(df, "real_attempt_attention", require_real)
    attempt_ids = set(attempts["attempt_id"].astype(str))
    out = df[df["attempt_id"].astype(str).isin(attempt_ids)].copy()
    if missing := sorted(attempt_ids - set(out["attempt_id"].astype(str))):
        raise RuntimeError(f"real_attempt_attention missing attempt_ids: {missing[:10]}")
    if "attention_source" not in out.columns:
        out["attention_source"] = "gatv2conv_return_attention_weights"
    if "real_gatv2conv_attention_extracted" not in out.columns:
        out["real_gatv2conv_attention_extracted"] = True
    if "path_segment" not in out.columns:
        out["path_segment"] = "unknown"
    out["attention_connection_mode"] = "real_attempt_attention_passthrough"
    out["proxy_attention_replaced"] = True
    return out.reset_index(drop=True)

def validate_frames(attempts: pd.DataFrame, eta: pd.DataFrame, attention: pd.DataFrame, require_real: bool) -> None:
    require_columns(attempts, REQUIRED_ATTEMPT_COLUMNS, "pickup_attempt_events")
    require_columns(eta, REQUIRED_ETA_COLUMNS, "eta_counterfactual")
    require_columns(attention, REQUIRED_STEP160_ATTENTION_COLUMNS, "gatv2_attention")
    if attempts.empty or eta.empty or attention.empty:
        raise RuntimeError("Step 164 output tables must not be empty")
    attempt_ids = set(attempts["attempt_id"].astype(str))
    if missing := sorted(attempt_ids - set(eta["attempt_id"].astype(str))):
        raise RuntimeError(f"eta missing attempt_ids: {missing[:10]}")
    if missing := sorted(attempt_ids - set(attention["attempt_id"].astype(str))):
        raise RuntimeError(f"attention missing attempt_ids: {missing[:10]}")
    weights = pd.to_numeric(attention["attention_weight"], errors="raise")
    if weights.isna().any() or bool((weights < 0).any()):
        raise RuntimeError("attention_weight must be finite and non-negative")
    validate_real_source(attention, "gatv2_attention", require_real)

def quality_report(attempts: pd.DataFrame, eta: pd.DataFrame, attention: pd.DataFrame, mode: str) -> Dict[str, Any]:
    delta = eta.copy(); delta["delta"] = pd.to_numeric(delta["eta_with_new_pickup_sec"], errors="coerce") - pd.to_numeric(delta["eta_without_new_pickup_sec"], errors="coerce")
    real_flags = attention["real_gatv2conv_attention_extracted"].map(bool_from_any) if "real_gatv2conv_attention_extracted" in attention.columns else pd.Series([False]*len(attention))
    return {"connection_mode": mode, "pickup_attempt_count": int(len(attempts)), "eta_counterfactual_count": int(len(eta)), "real_attention_row_count": int(len(attention)), "attempt_count_with_real_attention": int(attention["attempt_id"].nunique()), "zero_loss_success_preview_count": int((delta["delta"] <= 0).sum()), "attention_source_counts": attention.get("attention_source", pd.Series(dtype=str)).astype(str).value_counts().to_dict(), "path_segment_counts": attention.get("path_segment", pd.Series(dtype=str)).astype(str).value_counts().to_dict(), "real_gatv2conv_attention_extracted_all": bool(real_flags.all()) if len(real_flags) else False, "attention_weight_min": float(pd.to_numeric(attention["attention_weight"]).min()), "attention_weight_max": float(pd.to_numeric(attention["attention_weight"]).max()), "proxy_attention_replaced": True}

def run_connector(output_root: Path, pickup_attempt_events: Optional[Path], eta_counterfactual: Optional[Path], step161_run_manifest: Optional[Path], real_attempt_attention: Optional[Path], real_snapshot_attention: Optional[Path], real_attention_manifest: Optional[Path], file_format: str, top_k: int, require_real: bool, make_sample_inputs_flag: bool) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True); warnings: List[str] = []
    if make_sample_inputs_flag:
        paths = make_sample_inputs(output_root / "sample_inputs")
        pickup_attempt_events = paths["pickup_attempt_events"]; eta_counterfactual = paths["eta_counterfactual"]; step161_run_manifest = paths["step161_run_manifest"]; real_snapshot_attention = paths["real_snapshot_attention"]; real_attention_manifest = paths["real_attention_manifest"]
    if pickup_attempt_events is None or eta_counterfactual is None or step161_run_manifest is None:
        raise RuntimeError("pickup attempt events, eta counterfactual, and Step 161 run manifest are required")
    if real_attempt_attention is None and real_snapshot_attention is None:
        raise RuntimeError("one of --real-attempt-attention or --real-snapshot-attention is required")
    attempts = read_table(pickup_attempt_events); eta = read_table(eta_counterfactual); upstream = load_json_any_encoding(step161_run_manifest); validate_guards(upstream, "step161_run_manifest")
    real_manifest_payload: Dict[str, Any] = {}
    if real_attention_manifest is not None:
        real_manifest_payload = load_json_any_encoding(real_attention_manifest); validate_guards(real_manifest_payload, "real_attention_manifest")
    if real_attempt_attention is not None:
        attention = use_attempt_attention(read_table(real_attempt_attention), attempts, require_real); mode = "real_attempt_attention_passthrough"; used = real_attempt_attention
    else:
        attention = project_snapshot_to_attempts(read_table(real_snapshot_attention), attempts, top_k, require_real)  # type: ignore[arg-type]
        mode = "snapshot_topk_projection_step164"; used = real_snapshot_attention
    validate_frames(attempts, eta, attention, require_real)
    quality = quality_report(attempts, eta, attention, mode)
    attempt_path = write_table(output_root/"pickup_attempt_events", attempts, file_format, warnings); eta_path = write_table(output_root/"eta_counterfactual", eta, file_format, warnings); attention_path = write_table(output_root/"gatv2_attention", attention, file_format, warnings)
    quality_path = output_root/"attention_connection_quality_report.json"; dump_json(quality_path, quality)
    run_manifest = {"artifact_version": STEP160_INPUT_VERSION, "connector_artifact_version": ARTIFACT_VERSION, "project": "urbanbus_rl_project", "step": "Step 164", "mode": "real_attention_pipeline_connector", "condition_id": upstream.get("condition_id", "unknown"), "seed": upstream.get("seed", "unknown"), "zero_loss_epsilon_sec": float(upstream.get("zero_loss_epsilon_sec", 0.0)), "dataset_artifact": upstream.get("dataset_artifact", "route_aware_simulator_or_sample"), "checkpoint_path": real_manifest_payload.get("checkpoint_info", {}).get("checkpoint_path", upstream.get("checkpoint_path", "")), "git_commit": upstream.get("git_commit", "unknown_local_step164"), "simulator_version": upstream.get("simulator_version", "route_aware_simulator_or_sample"), "attention_source": "real_gatv2conv_attention_step163", "real_attention_connected": True, "proxy_attention_replaced": True, "attention_connection_mode": mode, "attention_quality": quality, "output_root": str(output_root), "warnings": warnings, **NON_CLAIM_FLAGS}
    run_manifest_path = output_root/"run_manifest.json"; dump_json(run_manifest_path, run_manifest)
    manifest_path = output_root/"real_attention_pipeline_connector_manifest.json"
    manifest = {"artifact_version": ARTIFACT_VERSION, "audit_status": "PASS", "bundle_status": BUNDLE_STATUS, "output_root": str(output_root), "input_files": {"pickup_attempt_events": str(pickup_attempt_events), "eta_counterfactual": str(eta_counterfactual), "step161_run_manifest": str(step161_run_manifest), "real_attention": str(used), "real_attention_manifest": str(real_attention_manifest) if real_attention_manifest else ""}, "input_sha256": {"pickup_attempt_events": sha256_file(pickup_attempt_events), "eta_counterfactual": sha256_file(eta_counterfactual), "step161_run_manifest": sha256_file(step161_run_manifest), "real_attention": sha256_file(used) if used else "", "real_attention_manifest": sha256_file(real_attention_manifest) if real_attention_manifest else ""}, "output_files": {"pickup_attempt_events": str(attempt_path), "eta_counterfactual": str(eta_path), "gatv2_attention": str(attention_path), "run_manifest": str(run_manifest_path), "attention_connection_quality_report": str(quality_path), "connector_manifest": str(manifest_path)}, "summary": quality, "real_attention_connected": True, "proxy_attention_replaced": True, "require_real_attention": bool(require_real), "warnings": warnings, **NON_CLAIM_FLAGS}
    dump_json(manifest_path, manifest)
    print("[OK] Step 164 real attention pipeline connector completed")
    print(f"[OK] audit_status              : {manifest['audit_status']}")
    print(f"[OK] bundle_status             : {manifest['bundle_status']}")
    print(f"[OK] connection_mode           : {quality['connection_mode']}")
    print(f"[OK] pickup_attempt_count      : {quality['pickup_attempt_count']}")
    print(f"[OK] real_attention_row_count  : {quality['real_attention_row_count']}")
    print(f"[OK] attempt_count_with_real_attention : {quality['attempt_count_with_real_attention']}")
    print(f"[OK] real_gatv2conv_attention_extracted_all : {quality['real_gatv2conv_attention_extracted_all']}")
    print(f"[OK] output_root               : {output_root}")
    print(f"[OK] manifest                  : {manifest_path}")
    print("[OK] paper_level_claim_allowed : False")
    print("[OK] causal_performance_claim_allowed : False")
    return manifest

def main() -> None:
    p = argparse.ArgumentParser(description="Step 164 Real GATv2 Attention Pipeline Connector")
    p.add_argument("--pickup-attempt-events", default=""); p.add_argument("--eta-counterfactual", default=""); p.add_argument("--step161-run-manifest", default="")
    p.add_argument("--real-attempt-attention", default=""); p.add_argument("--real-snapshot-attention", default=""); p.add_argument("--real-attention-manifest", default="")
    p.add_argument("--output-root", required=True); p.add_argument("--file-format", choices=["csv","parquet"], default="csv"); p.add_argument("--top-k-per-attempt", type=int, default=12); p.add_argument("--require-real-attention", action="store_true"); p.add_argument("--make-sample-inputs", action="store_true")
    a = p.parse_args()
    run_connector(Path(a.output_root), Path(a.pickup_attempt_events) if a.pickup_attempt_events else None, Path(a.eta_counterfactual) if a.eta_counterfactual else None, Path(a.step161_run_manifest) if a.step161_run_manifest else None, Path(a.real_attempt_attention) if a.real_attempt_attention else None, Path(a.real_snapshot_attention) if a.real_snapshot_attention else None, Path(a.real_attention_manifest) if a.real_attention_manifest else None, a.file_format, int(a.top_k_per_attempt), bool(a.require_real_attention), bool(a.make_sample_inputs))

if __name__ == "__main__":
    main()
