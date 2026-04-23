import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

EXPECTED_SHARED_KPIS = [
    "cv_headway",
    "avg_wait_seconds",
    "bunching_rate",
    "on_time_rate",
    "intervention_rate",
    "energy_proxy",
]

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

VALID_TIME_BANDS = {"peak", "offpeak", "night"}


@dataclass
class AggregationWarning:
    code: str
    message: str


class AggregationError(RuntimeError):
    pass


def load_json_any_encoding(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except Exception:
            pass
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


class CanonicalKpiAggregator:
    def __init__(
        self,
        mode: str,
        contract_path: Path,
        input_root: Path,
        output_root: Path,
        scenario_index_path: Optional[Path] = None,
        smoke: bool = False,
    ) -> None:
        self.mode = mode
        self.contract_path = contract_path
        self.input_root = input_root
        self.output_root = output_root
        self.scenario_index_path = scenario_index_path
        self.smoke = smoke
        self.warnings: List[AggregationWarning] = []
        self.contract: Dict[str, Any] = {}

    def add_warning(self, code: str, message: str) -> None:
        self.warnings.append(AggregationWarning(code=code, message=message))

    def run(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.contract = load_json_any_encoding(self.contract_path)
        self.validate_contract(self.contract)

        if self.mode == "legacy_b0_passthrough":
            window_df, input_summary = self.run_legacy_b0_passthrough()
        elif self.mode == "official_rollup":
            window_df, input_summary = self.run_official_rollup()
        else:
            raise AggregationError(f"unsupported mode: {self.mode}")

        seed_df = self.aggregate_by_seed(window_df)
        time_band_df = self.aggregate_by_time_band(window_df)
        overall_payload = self.aggregate_overall(window_df, seed_df)
        smoke_summary = self.run_smoke_validation(window_df, seed_df, time_band_df)
        manifest = self.build_manifest(input_summary, smoke_summary, window_df, seed_df, time_band_df)

        window_path = self.output_root / "kpi_by_window.parquet"
        seed_path = self.output_root / "kpi_by_seed.parquet"
        time_band_path = self.output_root / "kpi_by_time_band.parquet"
        overall_path = self.output_root / "kpi_overall.json"
        manifest_path = self.output_root / "aggregation_manifest.json"

        window_df.to_parquet(window_path, index=False)
        seed_df.to_parquet(seed_path, index=False)
        time_band_df.to_parquet(time_band_path, index=False)
        dump_json(overall_path, overall_payload)
        dump_json(manifest_path, manifest)

        print(f"[OK] mode              : {self.mode}")
        print(f"[OK] contract          : {self.contract_path}")
        print(f"[OK] input_root        : {self.input_root}")
        if self.scenario_index_path:
            print(f"[OK] scenario_index    : {self.scenario_index_path}")
        print(f"[OK] output_root       : {self.output_root}")
        print(f"[OK] kpi_by_window     : {window_path}")
        print(f"[OK] kpi_by_seed       : {seed_path}")
        print(f"[OK] kpi_by_time_band  : {time_band_path}")
        print(f"[OK] kpi_overall       : {overall_path}")
        print(f"[OK] manifest          : {manifest_path}")
        print(f"[OK] warnings          : {len(self.warnings)}")
        print("SMOKE PASS" if self.smoke else "DONE")

    def validate_contract(self, contract: Dict[str, Any]) -> None:
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

        time_bands = set(contract.get("time_bands", []))
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

    def run_legacy_b0_passthrough(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        legacy_path = first_existing([
            self.input_root / "kpi_by_window.parquet",
            self.input_root / "canonical_eval" / "kpi_by_window.parquet",
        ])
        if legacy_path is None:
            raise AggregationError(
                f"legacy B0 file not found under input_root: {self.input_root}"
            )

        df = pd.read_parquet(legacy_path).copy()
        required = {"state_ts", "avg_wait_seconds", "intervention_rate"}
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise AggregationError(f"legacy B0 file missing columns: {missing}")

        meta_path = first_existing([
            self.input_root / "metadata.json",
            self.input_root / "canonical_eval" / "metadata.json",
        ])
        metadata = load_json_any_encoding(meta_path) if meta_path else {}

        if "condition_id" not in df.columns:
            df["condition_id"] = "B0"
        if "seed" not in df.columns:
            df["seed"] = 0
        if "window_id" not in df.columns:
            df["window_id"] = range(1, len(df) + 1)
        if "time_band" not in df.columns:
            df["time_band"] = "offpeak"
            self.add_warning(
                "legacy_missing_time_band",
                "legacy B0 file does not contain time_band; defaulted to offpeak",
            )
        if "service_date" not in df.columns:
            df["service_date"] = pd.to_datetime(df["state_ts"]).dt.date.astype(str)
        if "evaluation_horizon_minutes" not in df.columns:
            df["evaluation_horizon_minutes"] = int(self.contract["evaluation_horizon_minutes"])

        for c in EXPECTED_SHARED_KPIS:
            if c not in df.columns:
                df[c] = pd.NA

        df["source_mode"] = df.get("source_mode", "historical")
        df["strict_canonical"] = False
        df["computation_mode"] = "legacy_passthrough"
        df["causal_comparison_allowed"] = False
        df["qwen_trigger_rate"] = df.get("qwen_trigger_rate", 0.0)
        df["effective_replay_step_minutes"] = df.get("effective_replay_step_minutes", 60)
        df["headway_mean_seconds"] = df.get("headway_mean_seconds", pd.NA)
        df["headway_std_seconds"] = df.get("headway_std_seconds", pd.NA)
        df["headway_sample_count"] = df.get("headway_sample_count", pd.NA)
        df["bunching_event_count"] = df.get("bunching_event_count", pd.NA)
        df["headway_event_count"] = df.get("headway_event_count", pd.NA)
        df["wait_total_passenger_seconds"] = df.get("wait_total_passenger_seconds", pd.NA)
        df["wait_passenger_count"] = df.get("wait_passenger_count", pd.NA)
        df["ontime_event_count"] = df.get("ontime_event_count", pd.NA)
        df["schedulable_arrival_count"] = df.get("schedulable_arrival_count", pd.NA)
        df["intervention_count"] = df.get("intervention_count", pd.NA)
        df["decision_step_count"] = df.get("decision_step_count", pd.NA)
        df["energy_proxy_total"] = df.get("energy_proxy_total", df.get("energy_proxy", pd.NA))
        df["input_source_path"] = str(legacy_path)

        cols = [
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
            "input_source_path",
        ]
        df = df[cols].copy()

        self.add_warning(
            "legacy_passthrough",
            "B0 was aggregated in legacy_passthrough mode; strict canonical re-computation was not performed",
        )

        return df, {
            "input_mode": "legacy_b0_passthrough",
            "legacy_path": str(legacy_path),
            "metadata_path": str(meta_path) if meta_path else None,
            "metadata_snapshot": metadata,
        }

    def run_official_rollup(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        input_paths = self.discover_window_rollups()
        if not input_paths:
            raise AggregationError(
                f"no window_rollup.parquet files found under input_root: {self.input_root}"
            )

        frames = []
        for p in input_paths:
            df = pd.read_parquet(p).copy()
            df["input_source_path"] = str(p)
            frames.append(df)
        raw = pd.concat(frames, ignore_index=True)
        self.validate_official_input_df(raw)
        window_df = self.compute_kpi_by_window(raw)

        input_summary = {
            "input_mode": "official_rollup",
            "window_rollup_paths": [str(p) for p in input_paths],
            "window_rollup_file_count": len(input_paths),
            "scenario_index_path": str(self.scenario_index_path) if self.scenario_index_path else None,
        }

        if self.scenario_index_path and self.scenario_index_path.exists():
            scenario_df = pd.read_parquet(self.scenario_index_path)
            input_summary["scenario_index_rows"] = int(len(scenario_df))
            required = ["window_id", "state_ts", "service_date", "time_band"]
            missing = [c for c in required if c not in scenario_df.columns]
            if missing:
                self.add_warning(
                    "scenario_index_missing_columns",
                    f"scenario_index is missing columns: {missing}",
                )

        return window_df, input_summary

    def discover_window_rollups(self) -> List[Path]:
        direct = self.input_root / "window_rollup.parquet"
        if direct.exists():
            return [direct]
        return sorted(self.input_root.rglob("window_rollup.parquet"))

    def validate_official_input_df(self, df: pd.DataFrame) -> None:
        missing = [c for c in REQUIRED_OFFICIAL_COLUMNS if c not in df.columns]
        if missing:
            raise AggregationError(f"official_rollup input missing columns: {missing}")

        if df.empty:
            raise AggregationError("official_rollup input is empty")

        for c in [
            "seed",
            "evaluation_horizon_minutes",
            "headway_sample_count",
            "bunching_event_count",
            "headway_event_count",
            "wait_passenger_count",
            "ontime_event_count",
            "schedulable_arrival_count",
            "intervention_count",
            "decision_step_count",
        ]:
            if df[c].isna().any():
                raise AggregationError(f"column contains nulls: {c}")

        bad_time_bands = sorted(set(df["time_band"].astype(str).str.lower()) - VALID_TIME_BANDS)
        if bad_time_bands:
            raise AggregationError(f"invalid time_band values: {bad_time_bands}")

        bad_horizon = df["evaluation_horizon_minutes"] != int(self.contract["evaluation_horizon_minutes"])
        if bool(bad_horizon.any()):
            raise AggregationError(
                "evaluation_horizon_minutes mismatch inside window_rollup input"
            )

        dup_mask = df.duplicated(subset=["condition_id", "seed", "window_id"], keep=False)
        if bool(dup_mask.any()):
            dup = df.loc[dup_mask, ["condition_id", "seed", "window_id"]].head(10).to_dict("records")
            raise AggregationError(f"duplicate (condition_id, seed, window_id) rows found: {dup}")

        non_negative = [
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
        for c in non_negative:
            bad = pd.to_numeric(df[c], errors="coerce") < 0
            if bool(bad.fillna(False).any()):
                raise AggregationError(f"negative values found in {c}")

        if bool((df["bunching_event_count"] > df["headway_event_count"]).any()):
            raise AggregationError("bunching_event_count cannot exceed headway_event_count")
        if bool((df["ontime_event_count"] > df["schedulable_arrival_count"]).any()):
            raise AggregationError("ontime_event_count cannot exceed schedulable_arrival_count")
        if bool((df["intervention_count"] > df["decision_step_count"]).any()):
            raise AggregationError("intervention_count cannot exceed decision_step_count")

    def compute_ratio(self, numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        num = pd.to_numeric(numerator, errors="coerce")
        den = pd.to_numeric(denominator, errors="coerce")
        out = num / den
        out = out.where(den > 0)
        return out

    def compute_kpi_by_window(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()
        df["condition_id"] = df["condition_id"].astype(str)
        df["seed"] = pd.to_numeric(df["seed"], errors="raise").astype(int)
        df["window_id"] = df["window_id"]
        df["state_ts"] = pd.to_datetime(df["state_ts"], errors="raise")
        df["service_date"] = pd.to_datetime(df["service_date"], errors="coerce").dt.date.astype(str)
        df["time_band"] = df["time_band"].astype(str).str.lower()
        df["strict_canonical"] = True
        df["computation_mode"] = "official_rollup"
        df["causal_comparison_allowed"] = ~df["source_mode"].astype(str).isin(["historical", "legacy_bridge"])

        df["cv_headway"] = self.compute_ratio(df["headway_std_seconds"], df["headway_mean_seconds"])
        sample_count = pd.to_numeric(df["headway_sample_count"], errors="coerce")
        df.loc[sample_count < 2, "cv_headway"] = pd.NA

        df["avg_wait_seconds"] = self.compute_ratio(
            df["wait_total_passenger_seconds"], df["wait_passenger_count"]
        )
        df["bunching_rate"] = self.compute_ratio(
            df["bunching_event_count"], df["headway_event_count"]
        )
        df["on_time_rate"] = self.compute_ratio(
            df["ontime_event_count"], df["schedulable_arrival_count"]
        )
        df["intervention_rate"] = self.compute_ratio(
            df["intervention_count"], df["decision_step_count"]
        )
        df["energy_proxy"] = pd.to_numeric(df["energy_proxy_total"], errors="coerce")
        if "qwen_trigger_rate" in df.columns:
            df["qwen_trigger_rate"] = pd.to_numeric(df["qwen_trigger_rate"], errors="coerce").fillna(0.0)
        else:
            df["qwen_trigger_rate"] = pd.Series(0.0, index=df.index, dtype=float)
        df["effective_replay_step_minutes"] = pd.to_numeric(
            df.get("effective_replay_step_minutes", pd.NA), errors="coerce"
        )

        return df[
            [
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
                "input_source_path",
            ]
        ].copy()

    def aggregate_by_seed(self, window_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        group_cols = ["condition_id", "seed"]
        for keys, grp in window_df.groupby(group_cols, dropna=False):
            row: Dict[str, Any] = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
            row["window_count"] = int(len(grp))
            row["time_band_count"] = int(grp["time_band"].nunique())
            row["causal_comparison_allowed"] = bool(grp["causal_comparison_allowed"].all())
            for kpi in EXPECTED_SHARED_KPIS:
                s = pd.to_numeric(grp[kpi], errors="coerce")
                row[f"{kpi}_mean"] = self.safe_mean(s)
                row[f"{kpi}_std"] = self.safe_std(s)
                row[f"{kpi}_valid_window_count"] = int(s.notna().sum())
                row[f"{kpi}_null_window_count"] = int(s.isna().sum())
            rows.append(row)
        return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)

    def aggregate_by_time_band(self, window_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        group_cols = ["condition_id", "seed", "time_band"]
        for keys, grp in window_df.groupby(group_cols, dropna=False):
            row: Dict[str, Any] = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
            row["window_count"] = int(len(grp))
            row["causal_comparison_allowed"] = bool(grp["causal_comparison_allowed"].all())
            for kpi in EXPECTED_SHARED_KPIS:
                s = pd.to_numeric(grp[kpi], errors="coerce")
                row[f"{kpi}_mean"] = self.safe_mean(s)
                row[f"{kpi}_std"] = self.safe_std(s)
                row[f"{kpi}_valid_window_count"] = int(s.notna().sum())
                row[f"{kpi}_null_window_count"] = int(s.isna().sum())
            rows.append(row)
        return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)

    def aggregate_overall(self, window_df: pd.DataFrame, seed_df: pd.DataFrame) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "artifact_version": "canonical_kpi_aggregator_v1",
            "mode": self.mode,
            "condition_ids": sorted(window_df["condition_id"].astype(str).unique().tolist()),
            "n_seeds": int(seed_df[["condition_id", "seed"]].drop_duplicates().shape[0]),
            "n_windows_total": int(len(window_df)),
            "causal_comparison_allowed": bool(window_df["causal_comparison_allowed"].all()),
            "warnings": [w.__dict__ for w in self.warnings],
            "kpis": {},
        }
        for kpi in EXPECTED_SHARED_KPIS:
            s = pd.to_numeric(window_df[kpi], errors="coerce")
            payload["kpis"][kpi] = {
                "mean": self.safe_mean(s),
                "std": self.safe_std(s),
                "valid_window_count": int(s.notna().sum()),
                "null_window_count": int(s.isna().sum()),
            }
        return payload

    def run_smoke_validation(
        self,
        window_df: pd.DataFrame,
        seed_df: pd.DataFrame,
        time_band_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        checks: List[Dict[str, Any]] = []

        def add(name: str, passed: bool, detail: str) -> None:
            checks.append({"name": name, "passed": bool(passed), "detail": detail})
            if not passed:
                raise AggregationError(f"smoke failed: {name} :: {detail}")

        add(
            "non_empty_window_df",
            not window_df.empty,
            f"window_rows={len(window_df)}",
        )
        add(
            "horizon_is_30",
            bool((window_df["evaluation_horizon_minutes"] == 30).all()),
            "all rows must have evaluation_horizon_minutes=30",
        )
        add(
            "time_band_domain",
            set(window_df["time_band"].astype(str).unique()) <= VALID_TIME_BANDS,
            f"observed={sorted(window_df['time_band'].astype(str).unique().tolist())}",
        )
        add(
            "unique_condition_seed_window",
            not window_df.duplicated(["condition_id", "seed", "window_id"]).any(),
            "(condition_id, seed, window_id) must be unique",
        )
        add(
            "non_negative_intervention_rate",
            bool((pd.to_numeric(window_df["intervention_rate"], errors="coerce").dropna() >= 0).all()),
            "intervention_rate must be >= 0 when present",
        )
        add(
            "bounded_intervention_rate",
            bool((pd.to_numeric(window_df["intervention_rate"], errors="coerce").dropna() <= 1).all()),
            "intervention_rate must be <= 1 when present",
        )
        add(
            "seed_aggregation_not_empty",
            not seed_df.empty,
            f"seed_rows={len(seed_df)}",
        )
        add(
            "time_band_aggregation_not_empty",
            not time_band_df.empty,
            f"time_band_rows={len(time_band_df)}",
        )

        if self.contract.get("condition_id") == "A":
            add(
                "experiment_A_qwen_off",
                not bool(self.contract.get("qwen_train", False)) and not bool(self.contract.get("qwen_inference", False)),
                "A must have qwen_train=false and qwen_inference=false",
            )

        return {
            "smoke_enabled": self.smoke,
            "checks": checks,
            "passed": True,
        }

    def build_manifest(
        self,
        input_summary: Dict[str, Any],
        smoke_summary: Dict[str, Any],
        window_df: pd.DataFrame,
        seed_df: pd.DataFrame,
        time_band_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        return {
            "artifact_version": "canonical_kpi_aggregator_v1",
            "mode": self.mode,
            "contract_path": str(self.contract_path),
            "input_root": str(self.input_root),
            "output_root": str(self.output_root),
            "scenario_index_path": str(self.scenario_index_path) if self.scenario_index_path else None,
            "input_summary": input_summary,
            "row_counts": {
                "kpi_by_window": int(len(window_df)),
                "kpi_by_seed": int(len(seed_df)),
                "kpi_by_time_band": int(len(time_band_df)),
            },
            "causal_comparison_allowed": bool(window_df["causal_comparison_allowed"].all()),
            "warnings": [w.__dict__ for w in self.warnings],
            "smoke_summary": smoke_summary,
        }

    @staticmethod
    def safe_mean(series: pd.Series) -> Optional[float]:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            return None
        return float(s.mean())

    @staticmethod
    def safe_std(series: pd.Series) -> Optional[float]:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if len(s) < 2:
            return None
        return float(s.std(ddof=1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonical KPI aggregator for B0/B1/B2/A")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["legacy_b0_passthrough", "official_rollup"],
        help="Aggregation mode",
    )
    parser.add_argument("--contract", required=True, help="Path to baseline or experiment contract JSON")
    parser.add_argument("--input-root", required=True, help="Root directory for rollout artifacts")
    parser.add_argument("--output-root", required=True, help="Directory where canonical outputs will be written")
    parser.add_argument("--scenario-index", default="", help="Optional scenario_index.parquet path")
    parser.add_argument("--smoke", action="store_true", help="Enable smoke validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregator = CanonicalKpiAggregator(
        mode=args.mode,
        contract_path=Path(args.contract),
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        scenario_index_path=Path(args.scenario_index) if args.scenario_index else None,
        smoke=bool(args.smoke),
    )
    aggregator.run()


if __name__ == "__main__":
    main()

