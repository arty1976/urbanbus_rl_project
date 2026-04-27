import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


EXPECTED_SHARED_KPIS = [
    "cv_headway",
    "avg_wait_seconds",
    "bunching_rate",
    "on_time_rate",
    "intervention_rate",
    "energy_proxy",
]

VALID_TIME_BANDS = {"peak", "offpeak", "night"}

OPTIONAL_EXTENDED_KPIS = [
    "active_bus_count",
    "base_num_agents_b0r",
    "fleet_ratio_vs_b0r",
    "fleet_reduction_ratio",
    "passengers_served",
    "passenger_demand_generated",
    "passenger_service_rate",
    "passenger_wait_p95_seconds",
    "energy_proxy_per_passenger",
    "intervention_events",
]

NON_CAUSAL_SOURCE_MODES = {
    "historical",
    "legacy_bridge",
    "stub_b1_noop_smoke",
    "stub_b2_rulebased_smoke",
    "stub_A_pure_mappo_smoke",
    "replay_b1_noop_noncausal",
    "replay_b2_rulebased_noncausal",
    "replay_A_pure_mappo_noncausal",
}

REQUIRED_OFFICIAL_COLUMNS = [
    "condition_id",
    "seed",
    "window_id",
    "state_ts",
    "service_date",
    "time_band",
    "evaluation_horizon_minutes",
    "headway_mean_seconds",
    "headway_std_seconds",
    "headway_sample_count",
    "bunching_event_count",
    "headway_event_count",
    "wait_total_passenger_seconds",
    "wait_passenger_count",
    "ontime_event_count",
    "schedulable_arrival_count",
    "intervention_count",
    "decision_step_count",
    "energy_proxy_total",
    "source_mode",
]


# Step 55: preserve policy provenance metadata from window_rollup inputs.
# These columns are not KPI inputs, but they are required to prove whether
# a canonical KPI row came from mock, MAPPO smoke, or actual MAPPO policy.
POLICY_METADATA_PRESERVE_COLUMNS = [
    "policy_metadata_version",
    "policy_source",
    "policy_action_source_version",
    "policy_action_source_mode",
    "checkpoint_path",
    "checkpoint_validation_mode",
    "checkpoint_validator_ran",
    "checkpoint_loaded",
    "trained_model",
    "performance_claim_allowed",
    "placeholder_fallback_used",
    "mock_action_used",
    "qwen_train",
    "qwen_inference",
    "reward_version",
    "energy_proxy_model_version",
    "k_dist_kwh_per_m",
    "k_acc_kwh_per_event",
    "k_idle_kwh_per_sec",
    "actual_policy_claim_ready",
    "causal_policy_claim_ready",
    "policy_action_count",
    "policy_nonzero_action_count",
    "distance_m",
    "acceleration_event_count",
    "hold_seconds",
    "passenger_served_count",
    "energy_proxy_per_passenger",
]



class AggregationError(RuntimeError):
    pass


def load_json_any_encoding(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise AggregationError(f"failed to read json: {path}")


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def safe_mean(series: pd.Series) -> Optional[float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.mean())


def safe_std(series: pd.Series) -> Optional[float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2:
        return None
    return float(s.std(ddof=1))


def compute_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    out = num / den
    out = out.where(den > 0)
    return out


def validate_contract(contract: Dict[str, Any]) -> None:
    shared = contract.get("shared_kpis", [])
    if shared != EXPECTED_SHARED_KPIS:
        raise AggregationError(
            f"shared_kpis mismatch. expected={EXPECTED_SHARED_KPIS}, got={shared}"
        )

    horizon = int(contract.get("evaluation_horizon_minutes", -1))
    if horizon != 30:
        raise AggregationError(
            f"evaluation_horizon_minutes must be 30, got {horizon}"
        )

    time_bands = set(str(x).strip().lower() for x in contract.get("time_bands", []))
    if time_bands != VALID_TIME_BANDS:
        raise AggregationError(
            f"time_bands mismatch. expected={sorted(VALID_TIME_BANDS)}, got={sorted(time_bands)}"
        )

    fairness = contract.get("fairness_constraints", {})
    for key in ("same_initial_state", "same_exogenous_events", "same_eval_window"):
        if not fairness.get(key, False):
            raise AggregationError(f"fairness_constraints.{key} must be true")

    if contract.get("condition_id") == "A":
        if bool(contract.get("qwen_train", False)):
            raise AggregationError("Experiment A contract must have qwen_train=false")
        if bool(contract.get("qwen_inference", False)):
            raise AggregationError("Experiment A contract must have qwen_inference=false")


def aggregate_by_seed(window_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for (condition_id, seed), grp in window_df.groupby(["condition_id", "seed"], dropna=False):
        row: Dict[str, Any] = {
            "condition_id": condition_id,
            "seed": int(seed),
            "window_count": int(len(grp)),
            "time_band_count": int(grp["time_band"].nunique()),
            "causal_comparison_allowed": bool(grp["causal_comparison_allowed"].all()),
        }

        for kpi in EXPECTED_SHARED_KPIS:
            s = pd.to_numeric(grp[kpi], errors="coerce")
            row[f"{kpi}_mean"] = safe_mean(s)
            row[f"{kpi}_std"] = safe_std(s)
            row[f"{kpi}_valid_window_count"] = int(s.notna().sum())
            row[f"{kpi}_null_window_count"] = int(s.isna().sum())

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["condition_id", "seed"]).reset_index(drop=True)


def aggregate_by_time_band(window_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for (condition_id, seed, time_band), grp in window_df.groupby(
        ["condition_id", "seed", "time_band"],
        dropna=False,
    ):
        row: Dict[str, Any] = {
            "condition_id": condition_id,
            "seed": int(seed),
            "time_band": str(time_band),
            "window_count": int(len(grp)),
            "causal_comparison_allowed": bool(grp["causal_comparison_allowed"].all()),
        }

        for kpi in EXPECTED_SHARED_KPIS:
            s = pd.to_numeric(grp[kpi], errors="coerce")
            row[f"{kpi}_mean"] = safe_mean(s)
            row[f"{kpi}_std"] = safe_std(s)
            row[f"{kpi}_valid_window_count"] = int(s.notna().sum())
            row[f"{kpi}_null_window_count"] = int(s.isna().sum())

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["condition_id", "seed", "time_band"]).reset_index(drop=True)


def aggregate_overall(
    window_df: pd.DataFrame,
    seed_df: pd.DataFrame,
    mode: str,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "artifact_version": "canonical_kpi_aggregator_v1_step8",
        "mode": mode,
        "condition_ids": sorted(window_df["condition_id"].astype(str).unique().tolist()),
        "n_condition_seed_pairs": int(seed_df[["condition_id", "seed"]].drop_duplicates().shape[0]),
        "n_windows_total": int(len(window_df)),
        "causal_comparison_allowed": bool(window_df["causal_comparison_allowed"].all()),
        "kpis": {},
    }

    for kpi in EXPECTED_SHARED_KPIS:
        s = pd.to_numeric(window_df[kpi], errors="coerce")
        payload["kpis"][kpi] = {
            "mean": safe_mean(s),
            "std": safe_std(s),
            "valid_window_count": int(s.notna().sum()),
            "null_window_count": int(s.isna().sum()),
        }

    return payload


def validate_b0_window_smoke(out: pd.DataFrame) -> None:
    if out.empty:
        raise AggregationError("smoke failed: B0 canonical window output is empty")

    if not bool((out["evaluation_horizon_minutes"] == 30).all()):
        raise AggregationError("smoke failed: evaluation_horizon_minutes must be 30")

    dup_mask = out.duplicated(["condition_id", "seed", "window_id"], keep=False)
    if bool(dup_mask.any()):
        dup = out.loc[dup_mask, ["condition_id", "seed", "window_id"]].head(10)
        raise AggregationError(
            f"smoke failed: duplicate condition/seed/window rows found: {dup.to_dict('records')}"
        )

    if int(out["avg_wait_seconds"].notna().sum()) == 0:
        raise AggregationError("smoke failed: avg_wait_seconds has no valid values")

    if int(out["intervention_rate"].notna().sum()) == 0:
        raise AggregationError("smoke failed: intervention_rate has no valid values")


def run_legacy_b0_passthrough(
    contract: Dict[str, Any],
    input_root: Path,
    output_root: Path,
    smoke: bool,
) -> None:
    legacy_path = first_existing([
        input_root / "kpi_by_window.parquet",
        input_root / "canonical_eval" / "kpi_by_window.parquet",
    ])

    if legacy_path is None:
        raise AggregationError(
            f"legacy B0 kpi_by_window.parquet not found under input_root: {input_root}"
        )

    df = pd.read_parquet(legacy_path).copy()

    required = ["state_ts", "avg_wait_seconds", "intervention_rate"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise AggregationError(f"legacy B0 file missing columns: {missing}")

    if "condition_id" not in df.columns:
        df["condition_id"] = "B0"

    if "seed" not in df.columns:
        df["seed"] = 0

    if "window_id" not in df.columns:
        df["window_id"] = range(1, len(df) + 1)

    if "time_band" not in df.columns:
        df["time_band"] = "offpeak"

    if "service_date" not in df.columns:
        df["service_date"] = pd.to_datetime(df["state_ts"], errors="raise").dt.date.astype(str)

    if "evaluation_horizon_minutes" not in df.columns:
        df["evaluation_horizon_minutes"] = int(contract["evaluation_horizon_minutes"])

    for kpi in EXPECTED_SHARED_KPIS:
        if kpi not in df.columns:
            df[kpi] = pd.NA

    df["strict_canonical"] = False
    df["computation_mode"] = "legacy_b0_passthrough"
    df["causal_comparison_allowed"] = False

    if "source_mode" not in df.columns:
        df["source_mode"] = "historical"

    if "qwen_trigger_rate" not in df.columns:
        df["qwen_trigger_rate"] = 0.0

    if "effective_replay_step_minutes" not in df.columns:
        df["effective_replay_step_minutes"] = 60

    df["input_source_path"] = str(legacy_path)

    optional_extended_cols = [c for c in OPTIONAL_EXTENDED_KPIS if c in df.columns]

    out_cols = [
        "condition_id",
        "seed",
        "window_id",
        "state_ts",
        "service_date",
        "time_band",
        "evaluation_horizon_minutes",
        *EXPECTED_SHARED_KPIS,
        "strict_canonical",
        "computation_mode",
        "causal_comparison_allowed",
        "source_mode",
        "qwen_trigger_rate",
        "effective_replay_step_minutes",
        "input_source_path",
    ]

    window_df = df[out_cols].copy()

    if smoke:
        validate_b0_window_smoke(window_df)

    seed_df = aggregate_by_seed(window_df)
    time_band_df = aggregate_by_time_band(window_df)
    overall_payload = aggregate_overall(window_df, seed_df, mode="legacy_b0_passthrough")

    if smoke:
        if seed_df.empty:
            raise AggregationError("smoke failed: kpi_by_seed is empty")
        if time_band_df.empty:
            raise AggregationError("smoke failed: kpi_by_time_band is empty")
        if overall_payload["n_windows_total"] != int(len(window_df)):
            raise AggregationError("smoke failed: overall window count mismatch")

    output_root.mkdir(parents=True, exist_ok=True)

    window_path = output_root / "kpi_by_window.parquet"
    seed_path = output_root / "kpi_by_seed.parquet"
    time_band_path = output_root / "kpi_by_time_band.parquet"
    overall_path = output_root / "kpi_overall.json"
    manifest_path = output_root / "aggregation_manifest.json"

    window_df.to_parquet(window_path, index=False)
    seed_df.to_parquet(seed_path, index=False)
    time_band_df.to_parquet(time_band_path, index=False)
    dump_json(overall_path, overall_payload)

    manifest = {
        "artifact_version": "canonical_kpi_aggregator_v1_step8",
        "mode": "legacy_b0_passthrough",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "input_source_path": str(legacy_path),
        "output_files": {
            "kpi_by_window": str(window_path),
            "kpi_by_seed": str(seed_path),
            "kpi_by_time_band": str(time_band_path),
            "kpi_overall": str(overall_path),
            "aggregation_manifest": str(manifest_path),
        },
        "row_counts": {
            "kpi_by_window": int(len(window_df)),
            "kpi_by_seed": int(len(seed_df)),
            "kpi_by_time_band": int(len(time_band_df)),
        },
        "causal_comparison_allowed": False,
        "strict_canonical": False,
        "warnings": [
            {
                "code": "legacy_b0_passthrough",
                "message": (
                    "B0 historical baseline was passed through into canonical schema. "
                    "This step does not recompute strict official rollout KPIs."
                ),
            }
        ],
        "smoke": {
            "enabled": bool(smoke),
            "passed": True,
        },
    }

    dump_json(manifest_path, manifest)

    print("[OK] B0 legacy passthrough completed")
    print(f"[OK] input             : {legacy_path}")
    print(f"[OK] kpi_by_window     : {window_path}")
    print(f"[OK] kpi_by_seed       : {seed_path}")
    print(f"[OK] kpi_by_time_band  : {time_band_path}")
    print(f"[OK] kpi_overall       : {overall_path}")
    print(f"[OK] manifest          : {manifest_path}")
    print(f"[OK] window_rows       : {len(window_df)}")
    print(f"[OK] seed_rows         : {len(seed_df)}")
    print(f"[OK] time_band_rows    : {len(time_band_df)}")


def discover_window_rollups(input_root: Path) -> List[Path]:
    direct = input_root / "window_rollup.parquet"
    if direct.exists():
        return [direct]
    return sorted(input_root.rglob("window_rollup.parquet"))


def validate_official_input_df(
    df: pd.DataFrame,
    contract: Dict[str, Any],
) -> Dict[str, Any]:
    missing = [c for c in REQUIRED_OFFICIAL_COLUMNS if c not in df.columns]
    if missing:
        raise AggregationError(f"official_rollup input missing columns: {missing}")

    if df.empty:
        raise AggregationError("official_rollup input is empty")

    numeric_required = [
        "seed",
        "evaluation_horizon_minutes",
        "headway_mean_seconds",
        "headway_std_seconds",
        "headway_sample_count",
        "bunching_event_count",
        "headway_event_count",
        "wait_total_passenger_seconds",
        "wait_passenger_count",
        "ontime_event_count",
        "schedulable_arrival_count",
        "intervention_count",
        "decision_step_count",
        "energy_proxy_total",
    ]

    for c in numeric_required:
        converted = pd.to_numeric(df[c], errors="coerce")
        if converted.isna().any():
            bad_count = int(converted.isna().sum())
            raise AggregationError(f"column has non-numeric or null values: {c}, bad_count={bad_count}")

    bad_time_bands = sorted(
        set(df["time_band"].astype(str).str.strip().str.lower()) - VALID_TIME_BANDS
    )
    if bad_time_bands:
        raise AggregationError(f"invalid time_band values: {bad_time_bands}")

    horizon = int(contract["evaluation_horizon_minutes"])
    bad_horizon = pd.to_numeric(df["evaluation_horizon_minutes"], errors="coerce") != horizon
    if bool(bad_horizon.any()):
        raise AggregationError("evaluation_horizon_minutes mismatch inside window_rollup input")

    dup_mask = df.duplicated(subset=["condition_id", "seed", "window_id"], keep=False)
    if bool(dup_mask.any()):
        dup = df.loc[dup_mask, ["condition_id", "seed", "window_id"]].head(10)
        raise AggregationError(
            f"duplicate (condition_id, seed, window_id) rows found: {dup.to_dict('records')}"
        )

    non_negative_cols = [
        "headway_mean_seconds",
        "headway_std_seconds",
        "headway_sample_count",
        "bunching_event_count",
        "headway_event_count",
        "wait_total_passenger_seconds",
        "wait_passenger_count",
        "ontime_event_count",
        "schedulable_arrival_count",
        "intervention_count",
        "decision_step_count",
        "energy_proxy_total",
    ]

    for c in non_negative_cols:
        values = pd.to_numeric(df[c], errors="coerce")
        if bool((values < 0).any()):
            raise AggregationError(f"negative values found in {c}")

    if bool((pd.to_numeric(df["headway_mean_seconds"], errors="coerce") <= 0).any()):
        raise AggregationError("headway_mean_seconds must be > 0")

    if bool((pd.to_numeric(df["bunching_event_count"], errors="coerce") > pd.to_numeric(df["headway_event_count"], errors="coerce")).any()):
        raise AggregationError("bunching_event_count cannot exceed headway_event_count")

    if bool((pd.to_numeric(df["ontime_event_count"], errors="coerce") > pd.to_numeric(df["schedulable_arrival_count"], errors="coerce")).any()):
        raise AggregationError("ontime_event_count cannot exceed schedulable_arrival_count")

    if bool((pd.to_numeric(df["intervention_count"], errors="coerce") > pd.to_numeric(df["decision_step_count"], errors="coerce")).any()):
        raise AggregationError("intervention_count cannot exceed decision_step_count")

    if contract.get("condition_id") == "A":
        observed_condition_ids = sorted(df["condition_id"].astype(str).str.upper().unique().tolist())
        if observed_condition_ids != ["A"]:
            raise AggregationError(f"Experiment A input must contain only condition_id=A, got={observed_condition_ids}")

        if "qwen_trigger_rate" in df.columns:
            qwen_rate = pd.to_numeric(df["qwen_trigger_rate"], errors="coerce").fillna(0.0)
            if bool((qwen_rate != 0.0).any()):
                raise AggregationError("Experiment A must have qwen_trigger_rate=0.0")

    summary = {
        "row_count": int(len(df)),
        "condition_ids": sorted(df["condition_id"].astype(str).str.upper().unique().tolist()),
        "seed_values": sorted([int(x) for x in pd.to_numeric(df["seed"], errors="raise").unique().tolist()]),
        "window_count": int(df["window_id"].nunique()),
        "time_bands": sorted(df["time_band"].astype(str).str.strip().str.lower().unique().tolist()),
        "source_modes": sorted(df["source_mode"].astype(str).unique().tolist()),
        "horizon": horizon,
    }

    return summary


def compute_official_kpi_by_window(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    df["condition_id"] = df["condition_id"].astype(str).str.upper()
    df["seed"] = pd.to_numeric(df["seed"], errors="raise").astype(int)
    df["state_ts"] = pd.to_datetime(df["state_ts"], errors="raise")
    df["service_date"] = pd.to_datetime(df["service_date"], errors="coerce").dt.date.astype(str)
    df["time_band"] = df["time_band"].astype(str).str.strip().str.lower()
    df["evaluation_horizon_minutes"] = pd.to_numeric(
        df["evaluation_horizon_minutes"], errors="raise"
    ).astype(int)

    df["cv_headway"] = compute_ratio(
        df["headway_std_seconds"],
        df["headway_mean_seconds"],
    )

    sample_count = pd.to_numeric(df["headway_sample_count"], errors="coerce")
    df.loc[sample_count < 2, "cv_headway"] = pd.NA

    df["avg_wait_seconds"] = compute_ratio(
        df["wait_total_passenger_seconds"],
        df["wait_passenger_count"],
    )

    df["bunching_rate"] = compute_ratio(
        df["bunching_event_count"],
        df["headway_event_count"],
    )

    df["on_time_rate"] = compute_ratio(
        df["ontime_event_count"],
        df["schedulable_arrival_count"],
    )

    df["intervention_rate"] = compute_ratio(
        df["intervention_count"],
        df["decision_step_count"],
    )

    df["energy_proxy"] = pd.to_numeric(df["energy_proxy_total"], errors="coerce")

    df["strict_canonical"] = True
    df["computation_mode"] = "official_rollup"

    source_lower = df["source_mode"].astype(str).str.lower()
    non_causal_pattern = "historical|legacy|stub|replay|smoke|non_causal|non-causal"
    df["causal_comparison_allowed"] = ~source_lower.str.contains(
        non_causal_pattern,
        regex=True,
        na=False,
    )

    if "qwen_trigger_rate" not in df.columns:
        df["qwen_trigger_rate"] = 0.0
    df["qwen_trigger_rate"] = pd.to_numeric(
        df["qwen_trigger_rate"],
        errors="coerce",
    ).fillna(0.0)

    if "effective_replay_step_minutes" not in df.columns:
        df["effective_replay_step_minutes"] = pd.NA
    df["effective_replay_step_minutes"] = pd.to_numeric(
        df["effective_replay_step_minutes"],
        errors="coerce",
    )

    optional_extended_cols = [c for c in OPTIONAL_EXTENDED_KPIS if c in df.columns]

    out_cols = [
        "condition_id",
        "seed",
        "window_id",
        "state_ts",
        "service_date",
        "time_band",
        "evaluation_horizon_minutes",
        *EXPECTED_SHARED_KPIS,
        "strict_canonical",
        "computation_mode",
        "causal_comparison_allowed",
        "source_mode",
        "qwen_trigger_rate",
        "effective_replay_step_minutes",
        "headway_mean_seconds",
        "headway_std_seconds",
        "headway_sample_count",
        "bunching_event_count",
        "headway_event_count",
        "wait_total_passenger_seconds",
        "wait_passenger_count",
        "ontime_event_count",
        "schedulable_arrival_count",
        "intervention_count",
        "decision_step_count",
        "energy_proxy_total",
        *optional_extended_cols,
        "input_source_path",
    ]

    preserve_cols = [
        c for c in POLICY_METADATA_PRESERVE_COLUMNS
        if c in df.columns and c not in out_cols
    ]
    out_cols = [*out_cols, *preserve_cols]

    return df[out_cols].copy()


def validate_official_window_smoke(window_df: pd.DataFrame) -> None:
    if window_df.empty:
        raise AggregationError("smoke failed: official kpi_by_window is empty")

    if not bool((window_df["evaluation_horizon_minutes"] == 30).all()):
        raise AggregationError("smoke failed: evaluation_horizon_minutes must be 30")

    observed_time_bands = set(window_df["time_band"].astype(str).unique())
    if not observed_time_bands.issubset(VALID_TIME_BANDS):
        raise AggregationError(f"smoke failed: invalid time_band values: {sorted(observed_time_bands)}")

    dup_mask = window_df.duplicated(["condition_id", "seed", "window_id"], keep=False)
    if bool(dup_mask.any()):
        dup = window_df.loc[dup_mask, ["condition_id", "seed", "window_id"]].head(10)
        raise AggregationError(
            f"smoke failed: duplicate condition/seed/window rows found: {dup.to_dict('records')}"
        )

    for kpi in EXPECTED_SHARED_KPIS:
        s = pd.to_numeric(window_df[kpi], errors="coerce")
        if int(s.notna().sum()) == 0:
            raise AggregationError(f"smoke failed: {kpi} has no valid values")

    for bounded in ["bunching_rate", "on_time_rate", "intervention_rate"]:
        s = pd.to_numeric(window_df[bounded], errors="coerce").dropna()
        if bool((s < 0).any()) or bool((s > 1).any()):
            raise AggregationError(f"smoke failed: {bounded} must be between 0 and 1")


def build_scenario_summary(scenario_index_path: Optional[Path]) -> Dict[str, Any]:
    scenario_summary: Dict[str, Any] = {
        "provided": scenario_index_path is not None,
        "path": str(scenario_index_path) if scenario_index_path else None,
    }

    if not scenario_index_path:
        return scenario_summary

    if not scenario_index_path.exists():
        raise AggregationError(f"scenario_index not found: {scenario_index_path}")

    scenario_df = pd.read_parquet(scenario_index_path)
    scenario_required = ["window_id", "state_ts", "service_date", "time_band"]
    missing = [c for c in scenario_required if c not in scenario_df.columns]
    if missing:
        raise AggregationError(f"scenario_index missing columns: {missing}")

    scenario_summary.update({
        "rows": int(len(scenario_df)),
        "window_count": int(scenario_df["window_id"].nunique()),
        "time_bands": sorted(
            scenario_df["time_band"].astype(str).str.strip().str.lower().unique().tolist()
        ),
    })

    return scenario_summary


def run_official_rollup(
    contract: Dict[str, Any],
    input_root: Path,
    output_root: Path,
    scenario_index_path: Optional[Path],
    smoke: bool,
) -> None:
    input_paths = discover_window_rollups(input_root)

    if not input_paths:
        raise AggregationError(
            f"no window_rollup.parquet files found under input_root: {input_root}"
        )

    frames = []
    per_file_rows = []

    for path in input_paths:
        df_one = pd.read_parquet(path).copy()
        df_one["input_source_path"] = str(path)
        frames.append(df_one)
        per_file_rows.append({
            "path": str(path),
            "rows": int(len(df_one)),
        })

    raw = pd.concat(frames, ignore_index=True)

    validation_summary = validate_official_input_df(raw, contract)
    scenario_summary = build_scenario_summary(scenario_index_path)

    window_df = compute_official_kpi_by_window(raw)
    seed_df = aggregate_by_seed(window_df)
    time_band_df = aggregate_by_time_band(window_df)
    overall_payload = aggregate_overall(window_df, seed_df, mode="official_rollup")

    if smoke:
        validate_official_window_smoke(window_df)
        if seed_df.empty:
            raise AggregationError("smoke failed: kpi_by_seed is empty")
        if time_band_df.empty:
            raise AggregationError("smoke failed: kpi_by_time_band is empty")
        if overall_payload["n_windows_total"] != int(len(window_df)):
            raise AggregationError("smoke failed: overall window count mismatch")

    output_root.mkdir(parents=True, exist_ok=True)

    validation_path = output_root / "official_rollup_input_validation.json"
    window_path = output_root / "kpi_by_window.parquet"
    seed_path = output_root / "kpi_by_seed.parquet"
    time_band_path = output_root / "kpi_by_time_band.parquet"
    overall_path = output_root / "kpi_overall.json"
    manifest_path = output_root / "aggregation_manifest.json"

    validation_payload = {
        "artifact_version": "canonical_kpi_aggregator_v1_step8",
        "mode": "official_rollup_input_validation",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "window_rollup_file_count": int(len(input_paths)),
        "window_rollup_files": per_file_rows,
        "validation_summary": validation_summary,
        "scenario_index_summary": scenario_summary,
        "smoke": {
            "enabled": bool(smoke),
            "passed": True,
        },
    }

    dump_json(validation_path, validation_payload)

    window_df.to_parquet(window_path, index=False)
    seed_df.to_parquet(seed_path, index=False)
    time_band_df.to_parquet(time_band_path, index=False)
    dump_json(overall_path, overall_payload)

    warnings = []
    if not bool(window_df["causal_comparison_allowed"].all()):
        warnings.append({
            "code": "non_causal_or_smoke_source",
            "message": (
                "At least one source_mode indicates historical, replay, stub, smoke, or non-causal data. "
                "These KPI outputs are valid for contract/smoke validation, not causal performance comparison."
            ),
        })

    manifest = {
        "artifact_version": "canonical_kpi_aggregator_v1_step8",
        "mode": "official_rollup",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "scenario_index_path": str(scenario_index_path) if scenario_index_path else None,
        "window_rollup_file_count": int(len(input_paths)),
        "window_rollup_files": per_file_rows,
        "output_files": {
            "official_rollup_input_validation": str(validation_path),
            "kpi_by_window": str(window_path),
            "kpi_by_seed": str(seed_path),
            "kpi_by_time_band": str(time_band_path),
            "kpi_overall": str(overall_path),
            "aggregation_manifest": str(manifest_path),
        },
        "row_counts": {
            "kpi_by_window": int(len(window_df)),
            "kpi_by_seed": int(len(seed_df)),
            "kpi_by_time_band": int(len(time_band_df)),
        },
        "validation_summary": validation_summary,
        "scenario_index_summary": scenario_summary,
        "causal_comparison_allowed": bool(window_df["causal_comparison_allowed"].all()),
        "strict_canonical": True,
        "warnings": warnings,
        "smoke": {
            "enabled": bool(smoke),
            "passed": True,
        },
    }

    dump_json(manifest_path, manifest)

    print("[OK] official_rollup completed")
    print(f"[OK] input_root        : {input_root}")
    print(f"[OK] output_root       : {output_root}")
    print(f"[OK] validation_json   : {validation_path}")
    print(f"[OK] kpi_by_window     : {window_path}")
    print(f"[OK] kpi_by_seed       : {seed_path}")
    print(f"[OK] kpi_by_time_band  : {time_band_path}")
    print(f"[OK] kpi_overall       : {overall_path}")
    print(f"[OK] manifest          : {manifest_path}")
    print(f"[OK] files_found       : {len(input_paths)}")
    print(f"[OK] window_rows       : {len(window_df)}")
    print(f"[OK] seed_rows         : {len(seed_df)}")
    print(f"[OK] time_band_rows    : {len(time_band_df)}")
    print(f"[OK] condition_ids     : {validation_summary['condition_ids']}")
    print(f"[OK] seeds             : {validation_summary['seed_values']}")
    print(f"[OK] windows           : {validation_summary['window_count']}")
    print(f"[OK] time_bands        : {validation_summary['time_bands']}")
    print(f"[OK] causal_allowed    : {bool(window_df['causal_comparison_allowed'].all())}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Canonical KPI aggregator for B0/B1/B2/A"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["legacy_b0_passthrough", "official_rollup"],
        help="Aggregation mode",
    )
    parser.add_argument(
        "--contract",
        required=True,
        help="Path to baseline or experiment contract JSON",
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Root directory for input artifacts",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory where canonical outputs will be written",
    )
    parser.add_argument(
        "--scenario-index",
        default="",
        help="Optional scenario_index.parquet path",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Enable smoke validation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    contract_path = Path(args.contract)
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    scenario_index_path = Path(args.scenario_index) if args.scenario_index else None

    if not contract_path.exists():
        raise SystemExit(f"[STOP] contract not found: {contract_path}")

    if not input_root.exists():
        raise SystemExit(f"[STOP] input_root not found: {input_root}")

    contract = load_json_any_encoding(contract_path)
    validate_contract(contract)

    if args.mode == "legacy_b0_passthrough":
        run_legacy_b0_passthrough(
            contract=contract,
            input_root=input_root,
            output_root=output_root,
            smoke=bool(args.smoke),
        )
    elif args.mode == "official_rollup":
        run_official_rollup(
            contract=contract,
            input_root=input_root,
            output_root=output_root,
            scenario_index_path=scenario_index_path,
            smoke=bool(args.smoke),
        )
    else:
        raise SystemExit(f"[STOP] unsupported mode: {args.mode}")

    print("[OK] shared_kpis       : validated")
    print("[OK] horizon           : 30")
    print("[OK] time_bands        : peak/offpeak/night")
    print("[OK] fairness          : validated")
    print("SMOKE PASS" if args.smoke else "DONE")


if __name__ == "__main__":
    main()
