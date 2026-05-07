from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


ARTIFACT_VERSION = "patent_evidence_report_v2_step166_v1"
BUNDLE_STATUS = "PATENT_EVIDENCE_REPORT_V2_READY_NONCLAIM"

NON_CLAIM_FLAGS = {
    "simulation_evidence_only": True,
    "actual_operational_claim_allowed": False,
    "paper_level_claim_allowed": False,
    "causal_performance_claim_allowed": False,
    "train_allowed": False,
}

STEP160_STATUS = "ZERO_LOSS_PICKUP_EVIDENCE_BUNDLE_READY_NONCLAIM"
STEP165_STATUS = "ATTEMPT_ROUTE_PATH_ATTENTION_FILTERED_FOR_ZERO_LOSS_EVIDENCE_NONCLAIM"

PATH_SEGMENTS = [
    "existing_passenger_path",
    "candidate_pickup_path",
    "candidate_dropoff_path",
    "shared_path",
    "unrelated",
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


def bool_from_any(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def require_columns(df: pd.DataFrame, required: Sequence[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"{label} missing required columns: {missing}")


def require_nonclaim_flags(payload: Dict[str, Any], label: str) -> None:
    for key, expected in NON_CLAIM_FLAGS.items():
        observed = payload.get(key, expected)
        if bool_from_any(observed) != expected:
            raise RuntimeError(f"{label} guard flag mismatch: {key}={observed}, expected={expected}")


def resolve_path(raw_path: str, manifest_path: Path) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    candidates = [Path.cwd() / p, manifest_path.parent / p]
    for c in candidates:
        if c.exists():
            return c
    return Path.cwd() / p


def dataframe_to_markdown_simple(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_empty_"
    work = df.head(max_rows).copy()
    for col in work.columns:
        work[col] = work[col].map(lambda x: "" if pd.isna(x) else str(x))
    cols = list(work.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in work.itertuples(index=False):
        vals = [str(v).replace("|", "\\|") for v in row]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def validate_step160_manifest(manifest: Dict[str, Any]) -> None:
    if manifest.get("audit_status") != "PASS":
        raise RuntimeError("Step 160 manifest audit_status must be PASS")
    if manifest.get("bundle_status") != STEP160_STATUS:
        raise RuntimeError(f"Step 160 bundle_status mismatch: {manifest.get('bundle_status')}")
    require_nonclaim_flags(manifest, "Step 160 manifest")
    summary = manifest.get("summary", {})
    for key in ["total_pickup_attempts", "zero_loss_success_count", "zero_loss_success_rate"]:
        if key not in summary:
            raise RuntimeError(f"Step 160 summary missing {key}")
    outputs = manifest.get("output_files", {})
    for key in ["eta_delta_by_attempt", "zero_loss_by_condition", "attention_mass_by_segment", "top_attention_edges"]:
        if key not in outputs:
            raise RuntimeError(f"Step 160 output_files missing {key}")


def validate_step165_manifest(manifest: Dict[str, Any]) -> None:
    if manifest.get("audit_status") != "PASS":
        raise RuntimeError("Step 165 manifest audit_status must be PASS")
    if manifest.get("bundle_status") != STEP165_STATUS:
        raise RuntimeError(f"Step 165 bundle_status mismatch: {manifest.get('bundle_status')}")
    require_nonclaim_flags(manifest, "Step 165 manifest")
    outputs = manifest.get("output_files", {})
    for key in ["gatv2_attention", "attention_path_mass_by_attempt", "attempt_route_path_filter_report"]:
        if key not in outputs:
            raise RuntimeError(f"Step 165 output_files missing {key}")
    row_counts = manifest.get("row_counts", {})
    for key in ["pickup_attempt_count", "filtered_attention_row_count", "attempt_count_with_filtered_attention", "fallback_attempt_count", "exact_edge_overlap_count", "node_overlap_count"]:
        if key not in row_counts:
            raise RuntimeError(f"Step 165 row_counts missing {key}")


def load_inputs(step160_manifest_path: Path, step165_manifest_path: Path) -> Dict[str, Any]:
    step160 = load_json_any_encoding(step160_manifest_path)
    step165 = load_json_any_encoding(step165_manifest_path)
    validate_step160_manifest(step160)
    validate_step165_manifest(step165)

    s160_files = step160.get("output_files", {})
    s165_files = step165.get("output_files", {})

    return {
        "step160_manifest": step160,
        "step165_manifest": step165,
        "eta_delta": read_table(resolve_path(s160_files["eta_delta_by_attempt"], step160_manifest_path)),
        "zero_loss_by_condition": read_table(resolve_path(s160_files["zero_loss_by_condition"], step160_manifest_path)),
        "step160_attention_mass_by_segment": read_table(resolve_path(s160_files["attention_mass_by_segment"], step160_manifest_path)),
        "step160_top_attention_edges": read_table(resolve_path(s160_files["top_attention_edges"], step160_manifest_path)),
        "filtered_attention": read_table(resolve_path(s165_files["gatv2_attention"], step165_manifest_path)),
        "path_mass_by_attempt": read_table(resolve_path(s165_files["attention_path_mass_by_attempt"], step165_manifest_path)),
        "filter_report": read_table(resolve_path(s165_files["attempt_route_path_filter_report"], step165_manifest_path)),
    }


def enrich_attempt_attention(eta_delta: pd.DataFrame, filtered_attention: pd.DataFrame, filter_report: pd.DataFrame) -> pd.DataFrame:
    require_columns(eta_delta, ["attempt_id", "delta_eta_existing_passenger_sec", "zero_loss_success"], "eta_delta_by_attempt")
    require_columns(filtered_attention, ["attempt_id", "attention_weight"], "filtered_attention")

    attn = filtered_attention.copy()
    attn["attempt_id"] = attn["attempt_id"].map(normalize_str)
    attn["attention_weight"] = pd.to_numeric(attn["attention_weight"], errors="coerce").fillna(0.0)
    if "path_segment" not in attn.columns:
        attn["path_segment"] = "unknown"
    if "filter_match_type" not in attn.columns:
        attn["filter_match_type"] = "unknown"
    if "real_gatv2conv_attention_extracted" not in attn.columns:
        attn["real_gatv2conv_attention_extracted"] = False
    attn["real_gatv2conv_attention_extracted"] = attn["real_gatv2conv_attention_extracted"].map(bool_from_any)

    base_cols = [
        "attempt_id",
        "condition_id",
        "seed",
        "route_id",
        "direction_id",
        "decision",
        "delta_eta_existing_passenger_sec",
        "zero_loss_success",
        "eta_without_new_pickup_sec",
        "eta_with_new_pickup_sec",
    ]
    keep = [c for c in base_cols if c in eta_delta.columns]
    attempts = eta_delta[keep].copy()
    attempts["attempt_id"] = attempts["attempt_id"].map(normalize_str)

    group = attn.groupby("attempt_id", dropna=False)
    rows: List[Dict[str, Any]] = []
    for attempt_id, grp in group:
        row: Dict[str, Any] = {
            "attempt_id": attempt_id,
            "filtered_attention_row_count": int(len(grp)),
            "filtered_attention_mass_total": float(grp["attention_weight"].sum()),
            "attention_weight_mean": float(grp["attention_weight"].mean()),
            "attention_weight_max": float(grp["attention_weight"].max()),
            "edge_overlap_count": int((grp["filter_match_type"] == "edge_overlap").sum()),
            "node_overlap_count": int((grp["filter_match_type"] == "node_overlap").sum()),
            "fallback_row_count": int(grp["filter_match_type"].astype(str).str.contains("fallback", case=False, na=False).sum()),
            "real_gatv2conv_attention_extracted_all": bool(grp["real_gatv2conv_attention_extracted"].all()),
        }
        for seg in PATH_SEGMENTS:
            sub = grp[grp["path_segment"] == seg]
            row[f"{seg}_attention_mass"] = float(sub["attention_weight"].sum()) if not sub.empty else 0.0
            row[f"{seg}_attention_row_count"] = int(len(sub))
        rows.append(row)

    attn_summary = pd.DataFrame(rows)
    if attn_summary.empty:
        attn_summary = pd.DataFrame(columns=["attempt_id"])

    out = attempts.merge(attn_summary, on="attempt_id", how="left")
    for col in ["filtered_attention_row_count", "edge_overlap_count", "node_overlap_count", "fallback_row_count"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    for col in [c for c in out.columns if c.endswith("_attention_mass") or c in {"filtered_attention_mass_total", "attention_weight_mean", "attention_weight_max"}]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    if "real_gatv2conv_attention_extracted_all" in out.columns:
        out["real_gatv2conv_attention_extracted_all"] = out["real_gatv2conv_attention_extracted_all"].fillna(False).map(bool_from_any)

    if not filter_report.empty and "attempt_id" in filter_report.columns:
        fr = filter_report.copy()
        fr["attempt_id"] = fr["attempt_id"].map(normalize_str)
        keep_fr = [c for c in ["attempt_id", "filter_status", "fallback_used", "route_sequence_found", "candidate_row_count", "selected_row_count"] if c in fr.columns]
        out = out.merge(fr[keep_fr], on="attempt_id", how="left")

    return out.sort_values("attempt_id").reset_index(drop=True)


def aggregate_success_path(filtered_attention: pd.DataFrame, eta_delta: pd.DataFrame) -> pd.DataFrame:
    require_columns(filtered_attention, ["attempt_id", "attention_weight"], "filtered_attention")
    success = eta_delta[["attempt_id", "zero_loss_success"]].copy()
    success["attempt_id"] = success["attempt_id"].map(normalize_str)
    attn = filtered_attention.copy()
    attn["attempt_id"] = attn["attempt_id"].map(normalize_str)
    attn["attention_weight"] = pd.to_numeric(attn["attention_weight"], errors="coerce").fillna(0.0)
    if "path_segment" not in attn.columns:
        attn["path_segment"] = "unknown"
    if "filter_match_type" not in attn.columns:
        attn["filter_match_type"] = "unknown"
    merged = attn.merge(success, on="attempt_id", how="left")
    rows = (
        merged.groupby(["zero_loss_success", "path_segment", "filter_match_type"], dropna=False)
        .agg(attention_mass_sum=("attention_weight", "sum"), attention_row_count=("attention_weight", "size"), attempt_count=("attempt_id", "nunique"))
        .reset_index()
    )
    total = rows.groupby("zero_loss_success")["attention_mass_sum"].transform("sum")
    rows["attention_mass_share_within_success_group"] = rows["attention_mass_sum"] / total.where(total > 0)
    return rows.sort_values(["zero_loss_success", "attention_mass_sum"], ascending=[False, False]).reset_index(drop=True)


def select_top_filtered_edges(filtered_attention: pd.DataFrame, eta_delta: pd.DataFrame, top_k: int) -> pd.DataFrame:
    attn = filtered_attention.copy()
    attn["attempt_id"] = attn["attempt_id"].map(normalize_str)
    attn["attention_weight"] = pd.to_numeric(attn["attention_weight"], errors="coerce").fillna(0.0)
    success_cols = [c for c in ["attempt_id", "zero_loss_success", "delta_eta_existing_passenger_sec"] if c in eta_delta.columns]
    merged = attn.merge(eta_delta[success_cols], on="attempt_id", how="left")
    top = merged.sort_values(["attempt_id", "attention_weight"], ascending=[True, False]).groupby("attempt_id", as_index=False).head(int(top_k)).reset_index(drop=True)
    keep = [
        "attempt_id", "zero_loss_success", "delta_eta_existing_passenger_sec", "state_ts", "layer_id", "head_id", "src_node", "dst_node", "attention_weight", "path_segment", "filter_match_type", "route_id", "direction_id", "current_stop_id", "pickup_stop_id", "dropoff_stop_id", "attention_source", "real_gatv2conv_attention_extracted",
    ]
    keep = [c for c in keep if c in top.columns]
    return top[keep]


def build_summary(step160: Dict[str, Any], step165: Dict[str, Any], attempt_summary: pd.DataFrame, success_path: pd.DataFrame) -> Dict[str, Any]:
    s = step160.get("summary", {})
    rc = step165.get("row_counts", {})
    total = int(s.get("total_pickup_attempts", 0))
    success = int(s.get("zero_loss_success_count", 0))
    filtered_rows = int(rc.get("filtered_attention_row_count", 0))
    edge_overlap = int(rc.get("exact_edge_overlap_count", 0))
    node_overlap = int(rc.get("node_overlap_count", 0))
    fallback_attempts = int(rc.get("fallback_attempt_count", 0))
    return {
        "artifact_version": ARTIFACT_VERSION,
        "step": "Step 166",
        "evidence_type": "zero_loss_pickup_patent_evidence_v2_with_attempt_specific_real_gatv2_attention",
        "total_pickup_attempts": total,
        "zero_loss_success_count": success,
        "zero_loss_failure_count": int(s.get("zero_loss_failure_count", max(0, total - success))),
        "zero_loss_success_rate": float(s.get("zero_loss_success_rate", success / total if total else 0.0)),
        "zero_loss_epsilon_sec": float(s.get("zero_loss_epsilon_sec", 0.0)),
        "filtered_attention_row_count": filtered_rows,
        "attempt_count_with_filtered_attention": int(rc.get("attempt_count_with_filtered_attention", 0)),
        "fallback_attempt_count": fallback_attempts,
        "exact_edge_overlap_count": edge_overlap,
        "node_overlap_count": node_overlap,
        "exact_edge_overlap_share": float(edge_overlap / filtered_rows) if filtered_rows else 0.0,
        "node_overlap_share": float(node_overlap / filtered_rows) if filtered_rows else 0.0,
        "real_gatv2conv_attention_extracted_all": bool(step165.get("quality", {}).get("real_gatv2conv_attention_extracted_all", False)),
        "attempt_specific_route_path_filter_applied": bool(step165.get("quality", {}).get("attempt_specific_route_path_filter_applied", False)),
        "connection_mode": step165.get("connection_mode"),
        **NON_CLAIM_FLAGS,
    }


def maybe_write_plots(output_root: Path, attempt_summary: pd.DataFrame, success_path: pd.DataFrame) -> List[str]:
    warnings: List[str] = []
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        warnings.append(f"matplotlib_unavailable_plots_skipped: {exc}")
        return warnings

    try:
        plot_df = success_path.copy()
        if not plot_df.empty:
            plot_df["label"] = plot_df["zero_loss_success"].astype(str) + ":" + plot_df["path_segment"].astype(str) + ":" + plot_df["filter_match_type"].astype(str)
            fig = plt.figure()
            plot_df.set_index("label")["attention_mass_sum"].plot(kind="bar")
            plt.xlabel("zero_loss:path_segment:match_type")
            plt.ylabel("filtered_attention_mass")
            plt.title("Step 166 Filtered Real GATv2 Attention Mass")
            fig.tight_layout()
            fig.savefig(output_root / "zero_loss_attention_mass_by_success_path_v2.png", dpi=160)
            plt.close(fig)
    except Exception as exc:  # pragma: no cover
        warnings.append(f"attention_mass_success_path_plot_failed: {exc}")

    try:
        if {"delta_eta_existing_passenger_sec", "filtered_attention_mass_total"}.issubset(attempt_summary.columns):
            fig = plt.figure()
            plt.scatter(
                pd.to_numeric(attempt_summary["delta_eta_existing_passenger_sec"], errors="coerce"),
                pd.to_numeric(attempt_summary["filtered_attention_mass_total"], errors="coerce"),
            )
            plt.axvline(0.0, linestyle="--")
            plt.xlabel("delta_eta_existing_passenger_sec")
            plt.ylabel("filtered_attention_mass_total")
            plt.title("Step 166 Delta ETA vs Filtered Attention Mass")
            fig.tight_layout()
            fig.savefig(output_root / "zero_loss_delta_vs_attention_mass_v2.png", dpi=160)
            plt.close(fig)
    except Exception as exc:  # pragma: no cover
        warnings.append(f"delta_vs_attention_mass_plot_failed: {exc}")
    return warnings


def write_report(path: Path, summary: Dict[str, Any], by_condition: pd.DataFrame, attempt_summary: pd.DataFrame, success_path: pd.DataFrame, top_edges: pd.DataFrame, warnings: List[str]) -> None:
    lines: List[str] = []
    lines.append("# Step 166 Patent Evidence Report v2")
    lines.append("")
    lines.append("## 1. Status")
    lines.append("")
    lines.append("This report is simulation evidence only. It is not an actual operational claim and not a causal performance claim.")
    lines.append("")
    lines.append("```text")
    for key in ["simulation_evidence_only", "actual_operational_claim_allowed", "paper_level_claim_allowed", "causal_performance_claim_allowed", "train_allowed"]:
        lines.append(f"{key} = {summary.get(key)}")
    lines.append("```")
    lines.append("")
    lines.append("## 2. Zero-Loss Pickup Summary")
    lines.append("")
    lines.append("```text")
    lines.append(f"total_pickup_attempts     = {summary['total_pickup_attempts']}")
    lines.append(f"zero_loss_success_count  = {summary['zero_loss_success_count']}")
    lines.append(f"zero_loss_failure_count  = {summary['zero_loss_failure_count']}")
    lines.append(f"zero_loss_success_rate   = {summary['zero_loss_success_rate']:.6f}")
    lines.append(f"zero_loss_epsilon_sec    = {summary['zero_loss_epsilon_sec']}")
    lines.append("```")
    lines.append("")
    lines.append("## 3. Attempt-Specific Real GATv2 Attention Summary")
    lines.append("")
    lines.append("```text")
    lines.append(f"connection_mode                         = {summary['connection_mode']}")
    lines.append(f"filtered_attention_row_count            = {summary['filtered_attention_row_count']}")
    lines.append(f"attempt_count_with_filtered_attention   = {summary['attempt_count_with_filtered_attention']}")
    lines.append(f"fallback_attempt_count                  = {summary['fallback_attempt_count']}")
    lines.append(f"exact_edge_overlap_count                = {summary['exact_edge_overlap_count']}")
    lines.append(f"node_overlap_count                      = {summary['node_overlap_count']}")
    lines.append(f"exact_edge_overlap_share                = {summary['exact_edge_overlap_share']:.6f}")
    lines.append(f"node_overlap_share                      = {summary['node_overlap_share']:.6f}")
    lines.append(f"real_gatv2conv_attention_extracted_all  = {summary['real_gatv2conv_attention_extracted_all']}")
    lines.append(f"attempt_specific_route_path_filter_applied = {summary['attempt_specific_route_path_filter_applied']}")
    lines.append("```")
    lines.append("")
    lines.append("## 4. By-Condition Preview")
    lines.append("")
    lines.append(dataframe_to_markdown_simple(by_condition, max_rows=20))
    lines.append("")
    lines.append("## 5. Attempt-Level Evidence Preview")
    lines.append("")
    preview_cols = [c for c in ["attempt_id", "condition_id", "route_id", "direction_id", "decision", "delta_eta_existing_passenger_sec", "zero_loss_success", "filtered_attention_row_count", "edge_overlap_count", "node_overlap_count", "fallback_row_count", "filtered_attention_mass_total"] if c in attempt_summary.columns]
    lines.append(dataframe_to_markdown_simple(attempt_summary[preview_cols], max_rows=20))
    lines.append("")
    lines.append("## 6. Filtered Attention Mass by Success / Path / Match Type")
    lines.append("")
    lines.append(dataframe_to_markdown_simple(success_path, max_rows=30))
    lines.append("")
    lines.append("## 7. Top Filtered Attention Edges Preview")
    lines.append("")
    lines.append(dataframe_to_markdown_simple(top_edges, max_rows=30))
    lines.append("")
    lines.append("## 8. Required Interpretation")
    lines.append("")
    lines.append("A zero-loss success means `eta_with_new_pickup_sec - eta_without_new_pickup_sec <= zero_loss_epsilon_sec` inside the configured simulator/counterfactual pipeline.")
    lines.append("")
    lines.append("The attention evidence in this report is route/path-filtered GATv2 attention evidence produced by the Step 165 non-claim pipeline. It should not be described as field-observed proof of operational performance.")
    lines.append("")
    lines.append("## 9. Warnings")
    lines.append("")
    if warnings:
        for w in warnings:
            lines.append(f"- {w}")
    else:
        lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_report_v2(step160_manifest_path: Path, step165_manifest_path: Path, output_root: Path, top_k_edges: int = 10) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs(step160_manifest_path, step165_manifest_path)

    eta_delta = inputs["eta_delta"]
    filtered_attention = inputs["filtered_attention"]
    filter_report = inputs["filter_report"]
    by_condition = inputs["zero_loss_by_condition"]
    step160 = inputs["step160_manifest"]
    step165 = inputs["step165_manifest"]

    attempt_summary = enrich_attempt_attention(eta_delta, filtered_attention, filter_report)
    success_path = aggregate_success_path(filtered_attention, eta_delta)
    top_edges = select_top_filtered_edges(filtered_attention, eta_delta, top_k=top_k_edges)
    summary = build_summary(step160, step165, attempt_summary, success_path)

    attempt_summary_path = output_root / "zero_loss_attempt_attention_summary_v2.csv"
    path_mass_path = output_root / "attention_mass_by_attempt_path_v2.csv"
    success_path_path = output_root / "attention_mass_by_success_path_v2.csv"
    top_edges_path = output_root / "top_filtered_attention_edges_v2.csv"
    summary_path = output_root / "patent_evidence_summary_v2.json"
    report_path = output_root / "patent_evidence_report_v2.md"
    manifest_path = output_root / "patent_evidence_report_v2_manifest.json"

    write_csv(attempt_summary_path, attempt_summary)
    write_csv(path_mass_path, inputs["path_mass_by_attempt"])
    write_csv(success_path_path, success_path)
    write_csv(top_edges_path, top_edges)
    dump_json(summary_path, summary)

    warnings = maybe_write_plots(output_root, attempt_summary, success_path)
    write_report(report_path, summary, by_condition, attempt_summary, success_path, top_edges, warnings)

    output_files = {
        "patent_evidence_report_v2": str(report_path),
        "patent_evidence_summary_v2": str(summary_path),
        "zero_loss_attempt_attention_summary_v2": str(attempt_summary_path),
        "attention_mass_by_attempt_path_v2": str(path_mass_path),
        "attention_mass_by_success_path_v2": str(success_path_path),
        "top_filtered_attention_edges_v2": str(top_edges_path),
        "manifest": str(manifest_path),
    }
    for optional_name in ["zero_loss_attention_mass_by_success_path_v2.png", "zero_loss_delta_vs_attention_mass_v2.png"]:
        p = output_root / optional_name
        if p.exists():
            output_files[optional_name] = str(p)

    manifest: Dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "audit_status": "PASS",
        "bundle_status": BUNDLE_STATUS,
        "input_manifests": {
            "step160_zero_loss_evidence_manifest": str(step160_manifest_path),
            "step165_attempt_route_path_attention_filter_manifest": str(step165_manifest_path),
        },
        "output_root": str(output_root),
        "output_files": output_files,
        "summary": summary,
        "row_counts": {
            "attempt_summary_rows": int(len(attempt_summary)),
            "attention_mass_by_success_path_rows": int(len(success_path)),
            "top_filtered_attention_edges_rows": int(len(top_edges)),
        },
        "file_hashes": {
            "patent_evidence_report_v2": sha256_file(report_path),
            "patent_evidence_summary_v2": sha256_file(summary_path),
            "zero_loss_attempt_attention_summary_v2": sha256_file(attempt_summary_path),
            "attention_mass_by_attempt_path_v2": sha256_file(path_mass_path),
            "attention_mass_by_success_path_v2": sha256_file(success_path_path),
            "top_filtered_attention_edges_v2": sha256_file(top_edges_path),
        },
        "warnings": warnings,
        **NON_CLAIM_FLAGS,
    }
    dump_json(manifest_path, manifest)

    print("[OK] Step 166 Patent Evidence Report v2 completed")
    print("[OK] audit_status              : PASS")
    print(f"[OK] bundle_status             : {BUNDLE_STATUS}")
    print(f"[OK] total_pickup_attempts     : {summary['total_pickup_attempts']}")
    print(f"[OK] zero_loss_success_count   : {summary['zero_loss_success_count']}")
    print(f"[OK] zero_loss_success_rate    : {summary['zero_loss_success_rate']:.6f}")
    print(f"[OK] filtered_attention_row_count : {summary['filtered_attention_row_count']}")
    print(f"[OK] exact_edge_overlap_count  : {summary['exact_edge_overlap_count']}")
    print(f"[OK] node_overlap_count        : {summary['node_overlap_count']}")
    print(f"[OK] fallback_attempt_count    : {summary['fallback_attempt_count']}")
    print(f"[OK] report                    : {report_path}")
    print(f"[OK] manifest                  : {manifest_path}")
    print(f"[OK] paper_level_claim_allowed : {manifest['paper_level_claim_allowed']}")
    print(f"[OK] causal_performance_claim_allowed : {manifest['causal_performance_claim_allowed']}")
    return manifest


def run_dependency_command(cmd: List[str], label: str) -> None:
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")


def run_sample_pipeline(output_root: Path, top_k_edges: int) -> Dict[str, Any]:
    """Run Step 165 sample + Step 160 integration when scripts are available."""
    project_root = Path.cwd()
    py = sys.executable
    step165 = project_root / "05_training" / "patent_evidence" / "attempt_route_path_attention_filter_step165.py"
    step160 = project_root / "05_training" / "patent_evidence" / "zero_loss_pickup_evidence_reporter_step160.py"
    if not step165.exists() or not step160.exists():
        raise RuntimeError("sample pipeline requires Step 160 and Step 165 scripts in the project tree")

    step165_root = output_root / "step165_input"
    step160_root = output_root / "step160_input"
    run_dependency_command([
        py, str(step165),
        "--mode", "sample",
        "--output-root", str(step165_root),
        "--top-k-per-attempt", "8",
    ], "Step 165 sample generation")
    step165_manifest = step165_root / "attempt_route_path_attention_filter_manifest.json"
    if not step165_manifest.exists():
        raise RuntimeError(f"Step 165 manifest not found after sample generation: {step165_manifest}")
    s165 = load_json_any_encoding(step165_manifest)
    s165_outputs = s165.get("output_files", {})

    # Step 160 contract expects ETA values to live only in eta_counterfactual.
    # Some older Step 165 sample outputs may still include ETA columns in
    # pickup_attempt_events, which would make pandas merge add _x/_y suffixes.
    # Sanitize the attempt file before invoking Step 160 so Step 166 remains
    # compatible with both old and fixed Step 165 artifacts.
    sanitized_root = output_root / "step160_sanitized_input"
    sanitized_root.mkdir(parents=True, exist_ok=True)
    raw_attempts = read_table(resolve_path(s165_outputs["pickup_attempt_events"], step165_manifest))
    eta_cols = ["eta_without_new_pickup_sec", "eta_with_new_pickup_sec"]
    sanitized_attempts = raw_attempts.drop(columns=[c for c in eta_cols if c in raw_attempts.columns])
    sanitized_attempt_path = sanitized_root / "pickup_attempt_events.csv"
    write_csv(sanitized_attempt_path, sanitized_attempts)

    run_dependency_command([
        py, str(step160),
        "--pickup-attempt-events", str(sanitized_attempt_path),
        "--eta-counterfactual", str(s165_outputs["eta_counterfactual"]),
        "--attention-weights", str(s165_outputs["gatv2_attention"]),
        "--run-manifest", str(s165_outputs["run_manifest"]),
        "--output-root", str(step160_root),
        "--top-k-attention", "10",
    ], "Step 160 evidence report generation")
    step160_manifest = step160_root / "zero_loss_evidence_manifest.json"
    if not step160_manifest.exists():
        raise RuntimeError(f"Step 160 manifest not found after sample generation: {step160_manifest}")
    return run_report_v2(step160_manifest, step165_manifest, output_root / "report_v2", top_k_edges=top_k_edges)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 166 Patent Evidence Report v2")
    parser.add_argument("--mode", choices=["sample", "from-manifests"], default="sample")
    parser.add_argument("--step160-manifest", default="")
    parser.add_argument("--step165-manifest", default="")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--top-k-edges", type=int, default=10)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if args.mode == "sample":
        run_sample_pipeline(output_root, top_k_edges=args.top_k_edges)
        return

    if not args.step160_manifest or not args.step165_manifest:
        raise RuntimeError("from-manifests mode requires --step160-manifest and --step165-manifest")
    run_report_v2(Path(args.step160_manifest), Path(args.step165_manifest), output_root, top_k_edges=args.top_k_edges)


if __name__ == "__main__":
    main()
