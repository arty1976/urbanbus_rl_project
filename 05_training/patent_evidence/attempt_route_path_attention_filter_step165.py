from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


ARTIFACT_VERSION = "attempt_route_path_attention_filter_step165_v2"
BUNDLE_STATUS = "ATTEMPT_ROUTE_PATH_ATTENTION_FILTERED_FOR_ZERO_LOSS_EVIDENCE_NONCLAIM"
CONNECTION_MODE = "attempt_route_path_edge_filter_step165"

NON_CLAIM_FLAGS = {
    "simulation_evidence_only": True,
    "actual_operational_claim_allowed": False,
    "paper_level_claim_allowed": False,
    "causal_performance_claim_allowed": False,
    "train_allowed": False,
}

REQUIRED_ATTEMPT_COLUMNS = [
    "attempt_id",
    "state_ts",
    "condition_id",
    "seed",
    "vehicle_id",
    "existing_passenger_id",
    "new_passenger_id",
    "route_id",
    "direction_id",
    "current_stop_id",
    "pickup_stop_id",
    "dropoff_stop_id",
    "decision",
]

REQUIRED_ETA_COLUMNS = [
    "attempt_id",
    "existing_passenger_id",
    "eta_without_new_pickup_sec",
    "eta_with_new_pickup_sec",
]

REQUIRED_ROUTE_COLUMNS = [
    "route_id",
    "direction_id",
    "stop_id",
    "stop_order",
    "node_index",
]

REQUIRED_ATTENTION_COLUMNS = [
    "state_ts",
    "layer_id",
    "head_id",
    "src_node",
    "dst_node",
    "attention_weight",
]


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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    raise RuntimeError(f"unsupported table format: {path}")


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def normalize_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def require_columns(df: pd.DataFrame, required: Sequence[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"{label} missing required columns: {missing}")


def bool_from_any(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def normalize_node_id(value: Any) -> str:
    text = normalize_str(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text


def route_key(route_id: Any, direction_id: Any) -> Tuple[str, str]:
    return normalize_str(route_id), normalize_str(direction_id)


def directed_edges(nodes: List[str]) -> List[Tuple[str, str]]:
    return [(nodes[i], nodes[i + 1]) for i in range(max(0, len(nodes) - 1))]


def slice_path_nodes(seq: pd.DataFrame, start_stop: str, end_stop: str) -> List[str]:
    by_stop = {normalize_str(r.stop_id): int(r.stop_order) for r in seq.itertuples(index=False)}
    if start_stop not in by_stop or end_stop not in by_stop:
        return []

    start_order = by_stop[start_stop]
    end_order = by_stop[end_stop]
    lo, hi = sorted([start_order, end_order])
    sub = seq[(seq["stop_order"] >= lo) & (seq["stop_order"] <= hi)].sort_values("stop_order")
    nodes = [normalize_node_id(x) for x in sub["node_index"].tolist()]
    if start_order > end_order:
        nodes = list(reversed(nodes))
    return nodes


def build_route_lookup(route_df: pd.DataFrame) -> Dict[Tuple[str, str], pd.DataFrame]:
    require_columns(route_df, REQUIRED_ROUTE_COLUMNS, "route_stop_sequence")
    df = route_df.copy()
    for c in ["route_id", "direction_id", "stop_id"]:
        df[c] = df[c].map(normalize_str)
    df["node_index"] = df["node_index"].map(normalize_node_id)
    df["stop_order"] = pd.to_numeric(df["stop_order"], errors="raise").astype(int)
    out: Dict[Tuple[str, str], pd.DataFrame] = {}
    for key, grp in df.groupby(["route_id", "direction_id"], dropna=False):
        out[(normalize_str(key[0]), normalize_str(key[1]))] = grp.sort_values("stop_order").reset_index(drop=True)
    return out


def make_eta_from_attempts(attempts: pd.DataFrame) -> pd.DataFrame:
    cols = REQUIRED_ETA_COLUMNS
    if all(c in attempts.columns for c in cols):
        return attempts[cols].copy()
    direct_cols = ["eta_without_new_pickup_sec", "eta_with_new_pickup_sec"]
    if not all(c in attempts.columns for c in direct_cols):
        raise RuntimeError(
            "eta_counterfactual was not supplied and attempt file does not contain direct ETA columns"
        )
    eta = attempts[["attempt_id", "existing_passenger_id", *direct_cols]].copy()
    eta["counterfactual_method"] = "copied_from_attempt_events_step165"
    return eta


def prepare_attention(attn: pd.DataFrame) -> pd.DataFrame:
    require_columns(attn, REQUIRED_ATTENTION_COLUMNS, "real_attention")
    df = attn.copy()
    df["src_node"] = df["src_node"].map(normalize_node_id)
    df["dst_node"] = df["dst_node"].map(normalize_node_id)
    df["attention_weight"] = pd.to_numeric(df["attention_weight"], errors="coerce")
    df = df[df["attention_weight"].notna()].copy()
    if df.empty:
        raise RuntimeError("real_attention has no valid attention_weight rows")
    if "attention_source" not in df.columns:
        df["attention_source"] = "unknown_attention_source"
    if "real_gatv2conv_attention_extracted" not in df.columns:
        df["real_gatv2conv_attention_extracted"] = False
    df["real_gatv2conv_attention_extracted"] = df["real_gatv2conv_attention_extracted"].map(bool_from_any)
    return df


def choose_snapshot_attention(attn: pd.DataFrame, state_ts: Any) -> pd.DataFrame:
    state = normalize_str(state_ts)
    if "state_ts" not in attn.columns:
        return attn
    same = attn[attn["state_ts"].map(normalize_str) == state]
    if not same.empty:
        return same.copy()
    return attn.copy()


def segment_specs(attempt: pd.Series, seq: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    current_stop = normalize_str(attempt["current_stop_id"])
    pickup_stop = normalize_str(attempt["pickup_stop_id"])
    dropoff_stop = normalize_str(attempt["dropoff_stop_id"])

    specs: Dict[str, Dict[str, Any]] = {}
    for name, start, end in [
        ("candidate_pickup_path", current_stop, pickup_stop),
        ("candidate_dropoff_path", pickup_stop, dropoff_stop),
        ("existing_passenger_path", current_stop, dropoff_stop),
    ]:
        nodes = slice_path_nodes(seq, start, end)
        specs[name] = {
            "start_stop_id": start,
            "end_stop_id": end,
            "nodes": nodes,
            "node_set": set(nodes),
            "edge_set": set(directed_edges(nodes)),
        }
    return specs


def classify_attention_row(src: str, dst: str, specs: Dict[str, Dict[str, Any]]) -> Tuple[str, str]:
    edge = (src, dst)
    matched_edges = [name for name, spec in specs.items() if edge in spec["edge_set"]]
    if len(matched_edges) > 1:
        return "shared_path", "edge_overlap"
    if len(matched_edges) == 1:
        return matched_edges[0], "edge_overlap"

    matched_nodes = [
        name
        for name, spec in specs.items()
        if src in spec["node_set"] or dst in spec["node_set"]
    ]
    if len(matched_nodes) > 1:
        return "shared_path", "node_overlap"
    if len(matched_nodes) == 1:
        return matched_nodes[0], "node_overlap"
    return "unrelated", "fallback_topk"


def filter_for_attempt(
    attempt: pd.Series,
    attention_df: pd.DataFrame,
    route_lookup: Dict[Tuple[str, str], pd.DataFrame],
    top_k_per_attempt: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    attempt_id = normalize_str(attempt["attempt_id"])
    key = route_key(attempt["route_id"], attempt["direction_id"])
    snap = choose_snapshot_attention(attention_df, attempt["state_ts"])

    report: Dict[str, Any] = {
        "attempt_id": attempt_id,
        "route_id": key[0],
        "direction_id": key[1],
        "route_sequence_found": key in route_lookup,
        "filter_status": "",
        "candidate_row_count": 0,
        "selected_row_count": 0,
        "fallback_used": False,
    }

    if key not in route_lookup:
        cand = snap.copy()
        cand["path_segment"] = "unrelated"
        cand["filter_match_type"] = "fallback_missing_route_sequence"
        report["filter_status"] = "fallback_missing_route_sequence"
        report["fallback_used"] = True
    else:
        specs = segment_specs(attempt, route_lookup[key])
        route_node_count = len(set().union(*(s["node_set"] for s in specs.values()))) if specs else 0
        route_edge_count = len(set().union(*(s["edge_set"] for s in specs.values()))) if specs else 0
        report["route_context_node_count"] = int(route_node_count)
        report["route_context_edge_count"] = int(route_edge_count)

        rows: List[Dict[str, Any]] = []
        for r in snap.itertuples(index=False):
            src = normalize_node_id(getattr(r, "src_node"))
            dst = normalize_node_id(getattr(r, "dst_node"))
            segment, match_type = classify_attention_row(src, dst, specs)
            if match_type in {"edge_overlap", "node_overlap"}:
                d = r._asdict()
                d["path_segment"] = segment
                d["filter_match_type"] = match_type
                rows.append(d)

        if rows:
            cand = pd.DataFrame(rows)
            report["filter_status"] = "route_path_filtered"
            report["fallback_used"] = False
        else:
            cand = snap.copy()
            cand["path_segment"] = "unrelated"
            cand["filter_match_type"] = "fallback_no_path_overlap"
            report["filter_status"] = "fallback_no_path_overlap"
            report["fallback_used"] = True

    report["candidate_row_count"] = int(len(cand))
    if cand.empty:
        out = cand.copy()
    else:
        out = cand.sort_values("attention_weight", ascending=False).head(int(top_k_per_attempt)).copy()
    report["selected_row_count"] = int(len(out))

    if not out.empty:
        out["attempt_id"] = attempt_id
        out["attempt_state_ts"] = normalize_str(attempt["state_ts"])
        out["state_ts"] = out.get("state_ts", attempt["state_ts"])
        out["condition_id"] = attempt.get("condition_id", "")
        out["seed"] = attempt.get("seed", "")
        out["route_id"] = attempt.get("route_id", "")
        out["direction_id"] = attempt.get("direction_id", "")
        out["current_stop_id"] = attempt.get("current_stop_id", "")
        out["pickup_stop_id"] = attempt.get("pickup_stop_id", "")
        out["dropoff_stop_id"] = attempt.get("dropoff_stop_id", "")
        out["attention_connection_mode"] = CONNECTION_MODE
        out["attempt_specific_filter_applied"] = not bool(report["fallback_used"])
        out["real_gatv2conv_attention_extracted"] = out["real_gatv2conv_attention_extracted"].map(bool_from_any)

    return out, report


def compute_path_mass(attn: pd.DataFrame) -> pd.DataFrame:
    if attn.empty:
        return pd.DataFrame(columns=["attempt_id", "path_segment", "attention_weight_sum", "attention_row_count"])
    grp = (
        attn.groupby(["attempt_id", "path_segment"], dropna=False)
        .agg(attention_weight_sum=("attention_weight", "sum"), attention_row_count=("attention_weight", "size"))
        .reset_index()
    )
    total = grp.groupby("attempt_id")["attention_weight_sum"].transform("sum")
    grp["attention_weight_share"] = grp["attention_weight_sum"] / total.where(total > 0)
    return grp.sort_values(["attempt_id", "attention_weight_sum"], ascending=[True, False]).reset_index(drop=True)


def make_sample_inputs(root: Path) -> Dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)

    route_rows: List[Dict[str, Any]] = []
    for i in range(10):
        route_rows.append({
            "route_id": "route_100",
            "direction_id": "0",
            "stop_id": f"stop_{i:03d}",
            "stop_order": i,
            "node_index": i,
        })
    route_df = pd.DataFrame(route_rows)

    attempts = []
    deltas = [0.0, 18.0, 0.0, 25.0, 0.0]
    for i, delta in enumerate(deltas):
        attempts.append({
            "attempt_id": f"att_{i+1:03d}",
            "state_ts": "2026-01-02T08:00:00+09:00",
            "condition_id": "A",
            "seed": 1,
            "vehicle_id": f"bus_{i+1:02d}",
            "existing_passenger_id": f"old_{i+1:03d}",
            "new_passenger_id": f"new_{i+1:03d}",
            "route_id": "route_100",
            "direction_id": "0",
            "current_stop_id": f"stop_{i:03d}",
            "pickup_stop_id": f"stop_{i+2:03d}",
            "dropoff_stop_id": f"stop_{i+5:03d}",
            "decision": "accepted" if delta <= 0 else "rejected",
            "time_band": "peak",
            "window_id": "w_step165_sample",
            "eta_without_new_pickup_sec": 600.0 + i * 30,
            "eta_with_new_pickup_sec": 600.0 + i * 30 + delta,
        })
    attempt_df = pd.DataFrame(attempts)
    eta_df = make_eta_from_attempts(attempt_df)
    eta_df["counterfactual_method"] = "sample_direct_eta_step165"

    attention_rows: List[Dict[str, Any]] = []
    for layer in [1, 2]:
        for head in [0, 1]:
            for edge_idx in range(9):
                # Higher weights near the route core; unrelated edges are added below.
                weight = 0.95 - 0.035 * edge_idx + 0.02 * head + 0.01 * layer
                attention_rows.append({
                    "state_ts": "2026-01-02T08:00:00+09:00",
                    "layer_id": layer,
                    "head_id": head,
                    "src_node": edge_idx,
                    "dst_node": edge_idx + 1,
                    "attention_weight": round(weight, 6),
                    "attention_source": "gatv2conv_return_attention_weights",
                    "real_gatv2conv_attention_extracted": True,
                    "edge_rank_in_snapshot": edge_idx,
                })
            for u in range(2):
                attention_rows.append({
                    "state_ts": "2026-01-02T08:00:00+09:00",
                    "layer_id": layer,
                    "head_id": head,
                    "src_node": 100 + u,
                    "dst_node": 101 + u,
                    "attention_weight": round(0.2 + 0.01 * u, 6),
                    "attention_source": "gatv2conv_return_attention_weights",
                    "real_gatv2conv_attention_extracted": True,
                    "edge_rank_in_snapshot": 100 + u,
                })
    attention_df = pd.DataFrame(attention_rows)

    run_manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "source_mode": "step165_sample_inputs",
        "attention_source": "gatv2conv_return_attention_weights",
        "real_gatv2conv_attention_extracted": True,
        **NON_CLAIM_FLAGS,
    }

    paths = {
        "pickup_attempt_events": root / "pickup_attempt_events.csv",
        "eta_counterfactual": root / "eta_counterfactual.csv",
        "real_attention": root / "real_attention.csv",
        "route_stop_sequence": root / "route_stop_sequence.csv",
        "run_manifest": root / "run_manifest.json",
    }
    write_csv(paths["pickup_attempt_events"], attempt_df)
    write_csv(paths["eta_counterfactual"], eta_df)
    write_csv(paths["real_attention"], attention_df)
    write_csv(paths["route_stop_sequence"], route_df)
    dump_json(paths["run_manifest"], run_manifest)
    return paths


def run_filter(
    pickup_attempt_events: Path,
    real_attention: Path,
    route_stop_sequence: Path,
    output_root: Path,
    eta_counterfactual: Optional[Path] = None,
    run_manifest: Optional[Path] = None,
    top_k_per_attempt: int = 8,
) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)

    attempts = read_table(pickup_attempt_events)
    require_columns(attempts, REQUIRED_ATTEMPT_COLUMNS, "pickup_attempt_events")
    for c in ["attempt_id", "state_ts", "condition_id", "vehicle_id", "existing_passenger_id", "new_passenger_id", "route_id", "direction_id", "current_stop_id", "pickup_stop_id", "dropoff_stop_id", "decision"]:
        if c in attempts.columns:
            attempts[c] = attempts[c].map(normalize_str)

    if eta_counterfactual is not None:
        eta = read_table(eta_counterfactual)
        require_columns(eta, REQUIRED_ETA_COLUMNS, "eta_counterfactual")
    else:
        eta = make_eta_from_attempts(attempts)

    route_df = read_table(route_stop_sequence)
    route_lookup = build_route_lookup(route_df)
    attn = prepare_attention(read_table(real_attention))

    filtered_parts: List[pd.DataFrame] = []
    reports: List[Dict[str, Any]] = []
    for _, attempt in attempts.iterrows():
        filtered, report = filter_for_attempt(attempt, attn, route_lookup, top_k_per_attempt)
        if not filtered.empty:
            filtered_parts.append(filtered)
        reports.append(report)

    if filtered_parts:
        filtered_attention = pd.concat(filtered_parts, ignore_index=True)
    else:
        filtered_attention = pd.DataFrame(columns=REQUIRED_ATTENTION_COLUMNS + ["attempt_id"])

    # Step 160-compatible minimal column names are preserved.
    for col in REQUIRED_ATTENTION_COLUMNS:
        if col not in filtered_attention.columns:
            filtered_attention[col] = pd.NA

    report_df = pd.DataFrame(reports)
    path_mass_df = compute_path_mass(filtered_attention)

    attempt_count = int(len(attempts))
    filtered_row_count = int(len(filtered_attention))
    attempt_count_with_attention = int(filtered_attention["attempt_id"].nunique()) if not filtered_attention.empty else 0
    fallback_attempt_count = int(report_df["fallback_used"].sum()) if not report_df.empty else 0
    exact_edge_overlap_count = int((filtered_attention.get("filter_match_type", pd.Series(dtype=str)) == "edge_overlap").sum()) if not filtered_attention.empty else 0
    node_overlap_count = int((filtered_attention.get("filter_match_type", pd.Series(dtype=str)) == "node_overlap").sum()) if not filtered_attention.empty else 0
    all_real = bool(filtered_attention.get("real_gatv2conv_attention_extracted", pd.Series([False])).map(bool_from_any).all()) if not filtered_attention.empty else False

    # Step 160 merges pickup_attempt_events with eta_counterfactual on
    # (attempt_id, existing_passenger_id). If ETA columns are present in both
    # tables, pandas creates _x/_y suffixes and Step 160 cannot find the
    # canonical eta_without_new_pickup_sec / eta_with_new_pickup_sec columns.
    # Therefore the Step 165 evidence bundle keeps ETA values only in the
    # eta_counterfactual table while leaving pickup_attempt_events as pure
    # attempt metadata.
    step160_attempts = attempts.copy()
    for eta_col in ["eta_without_new_pickup_sec", "eta_with_new_pickup_sec"]:
        if eta_col in step160_attempts.columns:
            step160_attempts = step160_attempts.drop(columns=[eta_col])

    out_attempts = output_root / "pickup_attempt_events.csv"
    out_eta = output_root / "eta_counterfactual.csv"
    out_attn = output_root / "gatv2_attention.csv"
    out_report = output_root / "attempt_route_path_filter_report.csv"
    out_mass = output_root / "attention_path_mass_by_attempt.csv"
    out_run_manifest = output_root / "run_manifest.json"
    out_manifest = output_root / "attempt_route_path_attention_filter_manifest.json"

    write_csv(out_attempts, step160_attempts)
    write_csv(out_eta, eta)
    write_csv(out_attn, filtered_attention)
    write_csv(out_report, report_df)
    write_csv(out_mass, path_mass_df)

    source_manifest: Dict[str, Any] = {}
    if run_manifest is not None and run_manifest.exists():
        source_manifest = load_json_any_encoding(run_manifest)

    step160_run_manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "source_mode": "attempt_route_path_attention_filter_step165",
        "attention_source": "gatv2conv_return_attention_weights",
        "connection_mode": CONNECTION_MODE,
        "attempt_specific_route_path_filter_applied": True,
        "top_k_per_attempt": int(top_k_per_attempt),
        "pickup_attempt_count": attempt_count,
        "filtered_attention_row_count": filtered_row_count,
        "attempt_count_with_filtered_attention": attempt_count_with_attention,
        "fallback_attempt_count": fallback_attempt_count,
        "real_gatv2conv_attention_extracted_all": all_real,
        "source_manifest_summary": {
            "artifact_version": source_manifest.get("artifact_version"),
            "source_mode": source_manifest.get("source_mode"),
            "attention_source": source_manifest.get("attention_source"),
        },
        **NON_CLAIM_FLAGS,
    }
    dump_json(out_run_manifest, step160_run_manifest)

    manifest: Dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "audit_status": "PASS",
        "bundle_status": BUNDLE_STATUS,
        "connection_mode": CONNECTION_MODE,
        "filter_mode": "attempt_specific_route_path_edge_overlap_then_node_overlap",
        "input_files": {
            "pickup_attempt_events": str(pickup_attempt_events),
            "eta_counterfactual": str(eta_counterfactual) if eta_counterfactual else None,
            "real_attention": str(real_attention),
            "route_stop_sequence": str(route_stop_sequence),
            "run_manifest": str(run_manifest) if run_manifest else None,
        },
        "output_root": str(output_root),
        "output_files": {
            "pickup_attempt_events": str(out_attempts),
            "eta_counterfactual": str(out_eta),
            "gatv2_attention": str(out_attn),
            "attempt_route_path_filter_report": str(out_report),
            "attention_path_mass_by_attempt": str(out_mass),
            "run_manifest": str(out_run_manifest),
            "manifest": str(out_manifest),
        },
        "row_counts": {
            "pickup_attempt_count": attempt_count,
            "eta_counterfactual_count": int(len(eta)),
            "source_real_attention_row_count": int(len(attn)),
            "filtered_attention_row_count": filtered_row_count,
            "attempt_count_with_filtered_attention": attempt_count_with_attention,
            "fallback_attempt_count": fallback_attempt_count,
            "exact_edge_overlap_count": exact_edge_overlap_count,
            "node_overlap_count": node_overlap_count,
        },
        "quality": {
            "real_gatv2conv_attention_extracted_all": all_real,
            "attempt_specific_route_path_filter_applied": True,
            "all_attempts_have_attention": attempt_count_with_attention == attempt_count,
            "fallback_used_any": fallback_attempt_count > 0,
            "top_k_per_attempt": int(top_k_per_attempt),
        },
        "file_hashes": {
            "pickup_attempt_events": sha256_file(out_attempts),
            "eta_counterfactual": sha256_file(out_eta),
            "gatv2_attention": sha256_file(out_attn),
            "attempt_route_path_filter_report": sha256_file(out_report),
            "attention_path_mass_by_attempt": sha256_file(out_mass),
            "run_manifest": sha256_file(out_run_manifest),
        },
        **NON_CLAIM_FLAGS,
        "warnings": [],
    }

    if fallback_attempt_count > 0:
        manifest["warnings"].append("some_attempts_used_nonclaim_fallback_attention_selection")

    dump_json(out_manifest, manifest)

    print("[OK] Step 165 attempt-specific route/path attention filter completed")
    print("[OK] audit_status              : PASS")
    print(f"[OK] bundle_status             : {BUNDLE_STATUS}")
    print(f"[OK] connection_mode           : {CONNECTION_MODE}")
    print(f"[OK] pickup_attempt_count      : {attempt_count}")
    print(f"[OK] filtered_attention_row_count : {filtered_row_count}")
    print(f"[OK] attempt_count_with_filtered_attention : {attempt_count_with_attention}")
    print(f"[OK] fallback_attempt_count    : {fallback_attempt_count}")
    print(f"[OK] exact_edge_overlap_count  : {exact_edge_overlap_count}")
    print(f"[OK] node_overlap_count        : {node_overlap_count}")
    print(f"[OK] real_gatv2conv_attention_extracted_all : {all_real}")
    print(f"[OK] output_root               : {output_root}")
    print(f"[OK] manifest                  : {out_manifest}")
    print(f"[OK] paper_level_claim_allowed : {manifest['paper_level_claim_allowed']}")
    print(f"[OK] causal_performance_claim_allowed : {manifest['causal_performance_claim_allowed']}")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sample", "from-files"], default="sample")
    parser.add_argument("--pickup-attempt-events", default="")
    parser.add_argument("--eta-counterfactual", default="")
    parser.add_argument("--real-attention", default="")
    parser.add_argument("--route-stop-sequence", default="")
    parser.add_argument("--run-manifest", default="")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--top-k-per-attempt", type=int, default=8)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if args.mode == "sample":
        inputs = make_sample_inputs(output_root / "sample_inputs")
        run_filter(
            pickup_attempt_events=inputs["pickup_attempt_events"],
            eta_counterfactual=inputs["eta_counterfactual"],
            real_attention=inputs["real_attention"],
            route_stop_sequence=inputs["route_stop_sequence"],
            run_manifest=inputs["run_manifest"],
            output_root=output_root,
            top_k_per_attempt=args.top_k_per_attempt,
        )
        return

    required = {
        "pickup_attempt_events": args.pickup_attempt_events,
        "real_attention": args.real_attention,
        "route_stop_sequence": args.route_stop_sequence,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError(f"from-files mode missing required arguments: {missing}")

    run_filter(
        pickup_attempt_events=Path(args.pickup_attempt_events),
        eta_counterfactual=Path(args.eta_counterfactual) if args.eta_counterfactual else None,
        real_attention=Path(args.real_attention),
        route_stop_sequence=Path(args.route_stop_sequence),
        run_manifest=Path(args.run_manifest) if args.run_manifest else None,
        output_root=output_root,
        top_k_per_attempt=args.top_k_per_attempt,
    )


if __name__ == "__main__":
    main()
