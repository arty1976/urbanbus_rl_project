from __future__ import annotations

import argparse
import json
import math
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


ARTIFACT_VERSION = "zero_loss_pickup_evidence_reporter_step160_v1"
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
REQUIRED_ATTENTION_COLUMNS = [
    "attempt_id",
    "state_ts",
    "layer_id",
    "head_id",
    "src_node",
    "dst_node",
    "attention_weight",
]

NON_CLAIM_FLAGS = {
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
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    raise RuntimeError(f"unsupported table format: {path}")


def write_table_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def require_columns(df: pd.DataFrame, required: List[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"{label} missing required columns: {missing}")


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


def make_sample_inputs(sample_root: Path) -> Dict[str, Path]:
    """Create deterministic toy inputs so the reporter can be smoke-tested."""
    sample_root.mkdir(parents=True, exist_ok=True)

    attempts = pd.DataFrame(
        [
            {
                "attempt_id": "att_001",
                "state_ts": "2026-01-01T08:00:00+09:00",
                "condition_id": "A",
                "seed": 1,
                "vehicle_id": "bus_01",
                "existing_passenger_id": "p_old_001",
                "new_passenger_id": "p_new_001",
                "route_id": "route_100",
                "direction_id": "0",
                "pickup_stop_id": "stop_010",
                "dropoff_stop_id": "stop_020",
                "decision": "accepted",
                "time_band": "peak",
                "window_id": "w_001",
            },
            {
                "attempt_id": "att_002",
                "state_ts": "2026-01-01T08:05:00+09:00",
                "condition_id": "A90",
                "seed": 1,
                "vehicle_id": "bus_02",
                "existing_passenger_id": "p_old_002",
                "new_passenger_id": "p_new_002",
                "route_id": "route_100",
                "direction_id": "0",
                "pickup_stop_id": "stop_011",
                "dropoff_stop_id": "stop_021",
                "decision": "rejected",
                "time_band": "peak",
                "window_id": "w_001",
            },
            {
                "attempt_id": "att_003",
                "state_ts": "2026-01-01T13:00:00+09:00",
                "condition_id": "A",
                "seed": 2,
                "vehicle_id": "bus_03",
                "existing_passenger_id": "p_old_003",
                "new_passenger_id": "p_new_003",
                "route_id": "route_200",
                "direction_id": "1",
                "pickup_stop_id": "stop_110",
                "dropoff_stop_id": "stop_130",
                "decision": "accepted",
                "time_band": "offpeak",
                "window_id": "w_002",
            },
        ]
    )

    eta = pd.DataFrame(
        [
            {
                "attempt_id": "att_001",
                "existing_passenger_id": "p_old_001",
                "eta_without_new_pickup_sec": 600.0,
                "eta_with_new_pickup_sec": 600.0,
                "counterfactual_method": "sample_counterfactual_v1",
            },
            {
                "attempt_id": "att_002",
                "existing_passenger_id": "p_old_002",
                "eta_without_new_pickup_sec": 480.0,
                "eta_with_new_pickup_sec": 510.0,
                "counterfactual_method": "sample_counterfactual_v1",
            },
            {
                "attempt_id": "att_003",
                "existing_passenger_id": "p_old_003",
                "eta_without_new_pickup_sec": 720.0,
                "eta_with_new_pickup_sec": 719.0,
                "counterfactual_method": "sample_counterfactual_v1",
            },
        ]
    )

    attention_rows: List[Dict[str, Any]] = []
    segments = [
        "existing_passenger_path",
        "candidate_pickup_path",
        "candidate_dropoff_path",
        "unrelated",
    ]
    for attempt_id in ["att_001", "att_002", "att_003"]:
        for i, segment in enumerate(segments):
            for head in [0, 1]:
                weight = 0.45 - i * 0.09 + head * 0.03
                if attempt_id == "att_002" and segment == "unrelated":
                    weight += 0.2
                attention_rows.append(
                    {
                        "attempt_id": attempt_id,
                        "state_ts": "2026-01-01T08:00:00+09:00",
                        "layer_id": 1,
                        "head_id": head,
                        "src_node": f"stop_{10+i}",
                        "dst_node": f"stop_{11+i}",
                        "edge_id": f"edge_{attempt_id}_{i}_{head}",
                        "attention_weight": round(weight, 6),
                        "distance_m": 300.0 + 50.0 * i,
                        "time_sec": 45.0 + 10.0 * i,
                        "generalized_cost": 1.0 + 0.1 * i,
                        "path_segment": segment,
                    }
                )
    attention = pd.DataFrame(attention_rows)

    run_manifest = {
        "artifact_version": "zero_loss_pickup_evidence_input_sample_v1",
        "project": "urbanbus_rl_project",
        "git_commit": "sample_git_commit",
        "dataset_artifact": "sample_dataset",
        "checkpoint_path": "artifacts/gatv2_v1/best_mappo.pt",
        "checkpoint_sha256": "sample_checkpoint_sha256",
        "gatv2_artifact_contract": "artifacts/gatv2_v1/artifact_contract.json",
        "simulator_version": "route_aware_simulator_v2_scaffold_sample",
        "zero_loss_epsilon_sec": 0.0,
        **NON_CLAIM_FLAGS,
    }

    attempt_path = sample_root / "pickup_attempt_events.csv"
    eta_path = sample_root / "eta_counterfactual.csv"
    attention_path = sample_root / "gatv2_attention.csv"
    manifest_path = sample_root / "run_manifest.json"
    attempts.to_csv(attempt_path, index=False, encoding="utf-8-sig")
    eta.to_csv(eta_path, index=False, encoding="utf-8-sig")
    attention.to_csv(attention_path, index=False, encoding="utf-8-sig")
    dump_json(manifest_path, run_manifest)

    return {
        "attempt_events": attempt_path,
        "eta_counterfactual": eta_path,
        "attention_weights": attention_path,
        "run_manifest": manifest_path,
    }


def compute_eta_delta(attempts: pd.DataFrame, eta: pd.DataFrame, epsilon_sec: float) -> pd.DataFrame:
    require_columns(attempts, REQUIRED_ATTEMPT_COLUMNS, "pickup_attempt_events")
    require_columns(eta, REQUIRED_ETA_COLUMNS, "eta_counterfactual")

    out = attempts.merge(
        eta,
        on=["attempt_id", "existing_passenger_id"],
        how="inner",
        validate="one_to_one",
    )
    if out.empty:
        raise RuntimeError("attempt_events and eta_counterfactual have no joined rows")

    out["eta_without_new_pickup_sec"] = pd.to_numeric(out["eta_without_new_pickup_sec"], errors="raise")
    out["eta_with_new_pickup_sec"] = pd.to_numeric(out["eta_with_new_pickup_sec"], errors="raise")
    out["delta_eta_existing_passenger_sec"] = (
        out["eta_with_new_pickup_sec"] - out["eta_without_new_pickup_sec"]
    )
    out["zero_loss_epsilon_sec"] = float(epsilon_sec)
    out["zero_loss_success"] = out["delta_eta_existing_passenger_sec"] <= float(epsilon_sec)
    return out


def summarize_by_condition(eta_delta: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["condition_id"]
    optional = ["seed", "time_band", "route_id", "direction_id"]
    group_cols.extend([c for c in optional if c in eta_delta.columns])

    rows: List[Dict[str, Any]] = []
    for keys, grp in eta_delta.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        total = int(len(grp))
        success = int(grp["zero_loss_success"].sum())
        row.update(
            {
                "pickup_attempt_count": total,
                "zero_loss_success_count": success,
                "zero_loss_failure_count": total - success,
                "zero_loss_success_rate": float(success / total) if total else 0.0,
                "delta_eta_mean_sec": float(grp["delta_eta_existing_passenger_sec"].mean()),
                "delta_eta_median_sec": float(grp["delta_eta_existing_passenger_sec"].median()),
                "delta_eta_max_sec": float(grp["delta_eta_existing_passenger_sec"].max()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def compute_attention_outputs(
    attention: pd.DataFrame,
    eta_delta: pd.DataFrame,
    top_k: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    require_columns(attention, REQUIRED_ATTENTION_COLUMNS, "gatv2_attention")
    if "path_segment" not in attention.columns:
        attention = attention.copy()
        attention["path_segment"] = "unknown"

    success_attempts = set(
        eta_delta.loc[eta_delta["zero_loss_success"], "attempt_id"].astype(str).tolist()
    )
    attention = attention.copy()
    attention["attempt_id"] = attention["attempt_id"].astype(str)
    attention["attention_weight"] = pd.to_numeric(attention["attention_weight"], errors="raise")
    attention["zero_loss_success"] = attention["attempt_id"].isin(success_attempts)

    if attention.empty:
        raise RuntimeError("gatv2_attention is empty")

    segment_rows: List[Dict[str, Any]] = []
    for (success, segment), grp in attention.groupby(["zero_loss_success", "path_segment"], dropna=False):
        segment_rows.append(
            {
                "zero_loss_success": bool(success),
                "path_segment": normalize_str(segment) or "unknown",
                "edge_attention_row_count": int(len(grp)),
                "attention_mass_sum": float(grp["attention_weight"].sum()),
                "attention_weight_mean": float(grp["attention_weight"].mean()),
                "attention_weight_max": float(grp["attention_weight"].max()),
            }
        )
    mass_by_segment = pd.DataFrame(segment_rows).sort_values(
        ["zero_loss_success", "attention_mass_sum"], ascending=[False, False]
    )

    sort_cols = ["attempt_id", "attention_weight"]
    top = attention.sort_values(sort_cols, ascending=[True, False]).groupby("attempt_id", as_index=False).head(top_k)
    keep_cols = [
        "attempt_id",
        "zero_loss_success",
        "state_ts",
        "layer_id",
        "head_id",
        "src_node",
        "dst_node",
        "attention_weight",
        "path_segment",
    ]
    for c in ["edge_id", "distance_m", "time_sec", "generalized_cost", "route_id", "direction_id"]:
        if c in top.columns:
            keep_cols.append(c)
    top = top[keep_cols].reset_index(drop=True)
    return mass_by_segment, top


def maybe_write_plots(output_root: Path, eta_delta: pd.DataFrame, mass_by_segment: pd.DataFrame) -> List[str]:
    warnings: List[str] = []
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment-dependent
        warnings.append(f"matplotlib_unavailable_plots_skipped: {exc}")
        return warnings

    try:
        fig = plt.figure()
        eta_delta["delta_eta_existing_passenger_sec"].plot(kind="hist", bins=20)
        plt.axvline(0.0, linestyle="--")
        plt.xlabel("delta_eta_existing_passenger_sec")
        plt.ylabel("attempt_count")
        plt.title("Zero-Loss Pickup ETA Delta Distribution")
        fig.tight_layout()
        fig.savefig(output_root / "eta_delta_distribution.png", dpi=160)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
        warnings.append(f"eta_delta_distribution_plot_failed: {exc}")

    try:
        plot_df = mass_by_segment.copy()
        plot_df["label"] = plot_df["zero_loss_success"].astype(str) + ":" + plot_df["path_segment"].astype(str)
        fig = plt.figure()
        plot_df.set_index("label")["attention_mass_sum"].plot(kind="bar")
        plt.xlabel("zero_loss_success:path_segment")
        plt.ylabel("attention_mass_sum")
        plt.title("GATv2 Attention Mass by Path Segment")
        fig.tight_layout()
        fig.savefig(output_root / "attention_mass_by_segment.png", dpi=160)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
        warnings.append(f"attention_mass_plot_failed: {exc}")

    return warnings


def build_summary(
    eta_delta: pd.DataFrame,
    epsilon_sec: float,
    run_manifest: Dict[str, Any],
) -> Dict[str, Any]:
    total = int(len(eta_delta))
    success = int(eta_delta["zero_loss_success"].sum())
    rejected_loss = int((~eta_delta["zero_loss_success"]).sum())
    summary = {
        "artifact_version": ARTIFACT_VERSION,
        "step": "Step 160",
        "evidence_type": "simulation_zero_loss_pickup_threshold",
        "zero_loss_epsilon_sec": float(epsilon_sec),
        "total_pickup_attempts": total,
        "zero_loss_success_count": success,
        "zero_loss_failure_count": rejected_loss,
        "zero_loss_success_rate": float(success / total) if total else 0.0,
        "delta_eta_existing_passenger_sec": {
            "min": float(eta_delta["delta_eta_existing_passenger_sec"].min()),
            "mean": float(eta_delta["delta_eta_existing_passenger_sec"].mean()),
            "median": float(eta_delta["delta_eta_existing_passenger_sec"].median()),
            "max": float(eta_delta["delta_eta_existing_passenger_sec"].max()),
        },
        "input_manifest_project": run_manifest.get("project", ""),
        "input_manifest_dataset_artifact": run_manifest.get("dataset_artifact", ""),
        "input_manifest_checkpoint_sha256": run_manifest.get("checkpoint_sha256", ""),
        **NON_CLAIM_FLAGS,
    }
    return summary



def dataframe_to_markdown_simple(df: pd.DataFrame) -> str:
    """Return a GitHub-style Markdown table without pandas optional tabulate dependency."""
    if df.empty:
        return "_No rows._"

    preview = df.copy()
    columns = [str(c) for c in preview.columns]

    def fmt(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.6g}"
        text = str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    rows = []
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for record in preview.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(fmt(v) for v in record) + " |")
    return "\n".join(rows)

def write_markdown_report(
    path: Path,
    summary: Dict[str, Any],
    by_condition: pd.DataFrame,
    warnings: List[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# Step 160 Zero-Loss Pickup Evidence Report")
    lines.append("")
    lines.append("## 1. Status")
    lines.append("")
    lines.append("This report is simulation evidence only. It is not an actual operational claim.")
    lines.append("")
    lines.append("```text")
    for key in [
        "simulation_evidence_only",
        "actual_operational_claim_allowed",
        "paper_level_claim_allowed",
        "causal_performance_claim_allowed",
        "train_allowed",
    ]:
        lines.append(f"{key} = {summary.get(key)}")
    lines.append("```")
    lines.append("")
    lines.append("## 2. Zero-Loss Pickup Summary")
    lines.append("")
    lines.append("```text")
    lines.append(f"total_pickup_attempts    = {summary['total_pickup_attempts']}")
    lines.append(f"zero_loss_success_count = {summary['zero_loss_success_count']}")
    lines.append(f"zero_loss_failure_count = {summary['zero_loss_failure_count']}")
    lines.append(f"zero_loss_success_rate  = {summary['zero_loss_success_rate']:.6f}")
    lines.append(f"zero_loss_epsilon_sec   = {summary['zero_loss_epsilon_sec']}")
    lines.append("```")
    lines.append("")
    lines.append("## 3. Delta ETA Statistics")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(summary["delta_eta_existing_passenger_sec"], ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## 4. By-Condition Preview")
    lines.append("")
    lines.append(dataframe_to_markdown_simple(by_condition.head(20)))
    lines.append("")
    lines.append("## 5. Required Interpretation")
    lines.append("")
    lines.append(
        "A zero-loss success means `eta_with_new_pickup_sec - eta_without_new_pickup_sec <= zero_loss_epsilon_sec` "
        "inside the configured simulator/counterfactual pipeline. It does not mean that actual field operation observed zero arrival-time loss."
    )
    lines.append("")
    lines.append("## 6. Warnings")
    lines.append("")
    if warnings:
        for w in warnings:
            lines.append(f"- {w}")
    else:
        lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_reporter(
    attempt_events_path: Path,
    eta_counterfactual_path: Path,
    attention_weights_path: Path,
    run_manifest_path: Path,
    output_root: Path,
    epsilon_sec: Optional[float],
    top_k_attention: int,
) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)

    run_manifest = load_json_any_encoding(run_manifest_path)
    if epsilon_sec is None:
        epsilon_sec = float(run_manifest.get("zero_loss_epsilon_sec", 0.0))
    epsilon_sec = float(epsilon_sec)
    if epsilon_sec < 0:
        raise RuntimeError("zero_loss_epsilon_sec must be >= 0")

    # Preserve non-claim discipline even if input manifest is incomplete.
    for key, expected in NON_CLAIM_FLAGS.items():
        observed = run_manifest.get(key, expected)
        if bool_from_any(observed) != expected:
            raise RuntimeError(f"run_manifest guard mismatch: {key} must be {expected}, got {observed}")

    attempts = read_table(attempt_events_path)
    eta = read_table(eta_counterfactual_path)
    attention = read_table(attention_weights_path)

    eta_delta = compute_eta_delta(attempts, eta, epsilon_sec=epsilon_sec)
    by_condition = summarize_by_condition(eta_delta)
    mass_by_segment, top_edges = compute_attention_outputs(attention, eta_delta, top_k=top_k_attention)
    summary = build_summary(eta_delta, epsilon_sec, run_manifest)

    eta_delta_path = output_root / "eta_delta_by_attempt.csv"
    by_condition_path = output_root / "zero_loss_by_condition.csv"
    mass_path = output_root / "attention_mass_by_segment.csv"
    top_edges_path = output_root / "top_attention_edges.csv"
    summary_path = output_root / "zero_loss_summary.json"
    report_path = output_root / "patent_evidence_report.md"
    manifest_path = output_root / "zero_loss_evidence_manifest.json"

    write_table_csv(eta_delta_path, eta_delta)
    write_table_csv(by_condition_path, by_condition)
    write_table_csv(mass_path, mass_by_segment)
    write_table_csv(top_edges_path, top_edges)
    dump_json(summary_path, summary)

    warnings = maybe_write_plots(output_root, eta_delta, mass_by_segment)
    write_markdown_report(report_path, summary, by_condition, warnings)

    output_files = {
        "zero_loss_summary": str(summary_path),
        "zero_loss_by_condition": str(by_condition_path),
        "eta_delta_by_attempt": str(eta_delta_path),
        "attention_mass_by_segment": str(mass_path),
        "top_attention_edges": str(top_edges_path),
        "patent_evidence_report": str(report_path),
        "zero_loss_evidence_manifest": str(manifest_path),
    }
    for optional_name in ["eta_delta_distribution.png", "attention_mass_by_segment.png"]:
        p = output_root / optional_name
        if p.exists():
            output_files[optional_name] = str(p)

    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "audit_status": "PASS",
        "bundle_status": "ZERO_LOSS_PICKUP_EVIDENCE_BUNDLE_READY_NONCLAIM",
        "input_files": {
            "pickup_attempt_events": str(attempt_events_path),
            "eta_counterfactual": str(eta_counterfactual_path),
            "gatv2_attention": str(attention_weights_path),
            "run_manifest": str(run_manifest_path),
        },
        "input_sha256": {
            "pickup_attempt_events": sha256_file(attempt_events_path),
            "eta_counterfactual": sha256_file(eta_counterfactual_path),
            "gatv2_attention": sha256_file(attention_weights_path),
            "run_manifest": sha256_file(run_manifest_path),
        },
        "output_root": str(output_root),
        "output_files": output_files,
        "summary": summary,
        "warnings": warnings,
        **NON_CLAIM_FLAGS,
    }
    dump_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 160 Zero-Loss Pickup Evidence Reporter")
    parser.add_argument("--pickup-attempt-events", default="")
    parser.add_argument("--eta-counterfactual", default="")
    parser.add_argument("--attention-weights", default="")
    parser.add_argument("--run-manifest", default="")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--zero-loss-epsilon-sec", type=float, default=None)
    parser.add_argument("--top-k-attention", type=int, default=10)
    parser.add_argument("--make-sample-inputs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)

    if args.make_sample_inputs:
        sample_paths = make_sample_inputs(output_root / "sample_inputs")
        attempt_events_path = sample_paths["attempt_events"]
        eta_counterfactual_path = sample_paths["eta_counterfactual"]
        attention_weights_path = sample_paths["attention_weights"]
        run_manifest_path = sample_paths["run_manifest"]
    else:
        missing_args = [
            name
            for name, value in [
                ("--pickup-attempt-events", args.pickup_attempt_events),
                ("--eta-counterfactual", args.eta_counterfactual),
                ("--attention-weights", args.attention_weights),
                ("--run-manifest", args.run_manifest),
            ]
            if not value
        ]
        if missing_args:
            raise SystemExit(f"missing required args unless --make-sample-inputs is used: {missing_args}")
        attempt_events_path = Path(args.pickup_attempt_events)
        eta_counterfactual_path = Path(args.eta_counterfactual)
        attention_weights_path = Path(args.attention_weights)
        run_manifest_path = Path(args.run_manifest)

    manifest = run_reporter(
        attempt_events_path=attempt_events_path,
        eta_counterfactual_path=eta_counterfactual_path,
        attention_weights_path=attention_weights_path,
        run_manifest_path=run_manifest_path,
        output_root=output_root,
        epsilon_sec=args.zero_loss_epsilon_sec,
        top_k_attention=int(args.top_k_attention),
    )

    summary = manifest["summary"]
    print("[OK] Step 160 zero-loss pickup evidence reporter completed")
    print(f"[OK] audit_status              : {manifest['audit_status']}")
    print(f"[OK] bundle_status             : {manifest['bundle_status']}")
    print(f"[OK] total_pickup_attempts     : {summary['total_pickup_attempts']}")
    print(f"[OK] zero_loss_success_count   : {summary['zero_loss_success_count']}")
    print(f"[OK] zero_loss_success_rate    : {summary['zero_loss_success_rate']:.6f}")
    print(f"[OK] output_root               : {output_root}")
    print(f"[OK] manifest                  : {manifest['output_files']['zero_loss_evidence_manifest']}")
    print("[OK] paper_level_claim_allowed : False")
    print("[OK] causal_performance_claim_allowed : False")


if __name__ == "__main__":
    main()
