from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping


BASIC_WINDOW_ROLLUP_COLUMNS = [
    "condition_id",
    "seed",
    "window_id",
    "state_ts",
    "service_date",
    "time_band",
    "evaluation_horizon_minutes",
    "source_mode",
    "qwen_trigger_rate",
    "energy_proxy_total",
]


MOCK_REQUIRED_POLICY_VALUES = {
    "policy_metadata_version": "policy_source_metadata_v1",
    "policy_source": "mock_policy",
    "policy_action_source_version": "mock_policy_action_source_v1",
    "checkpoint_validator_ran": False,
    "checkpoint_loaded": False,
    "trained_model": False,
    "performance_claim_allowed": False,
    "placeholder_fallback_used": True,
    "mock_action_used": True,
    "qwen_train": False,
    "qwen_inference": False,
    "qwen_trigger_rate": 0.0,
    "reward_version": "mappo_reward_v1",
    "energy_proxy_model_version": "daegu_energy_proxy_v1",
    "actual_policy_claim_ready": False,
    "causal_policy_claim_ready": False,
}


EXPECTED_K = {
    "k_dist_kwh_per_m": 0.0012,
    "k_acc_kwh_per_event": 0.1800,
    "k_idle_kwh_per_sec": 0.0080,
}


class WindowRollupPolicyMetadataError(RuntimeError):
    pass


def append_training_dir_to_path() -> Path:
    here = Path(__file__).resolve()
    training_dir = here.parents[1]
    if str(training_dir) not in sys.path:
        sys.path.insert(0, str(training_dir))
    return training_dir


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n", ""}:
            return False
    try:
        import pandas as pd

        if pd.isna(value):
            return False
    except Exception:
        pass
    return bool(value)


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        import pandas as pd

        if pd.isna(value):
            return float(default)
    except Exception:
        pass

    try:
        return float(value)
    except Exception:
        return float(default)


def float_equal(a: Any, b: Any, tol: float = 1.0e-12) -> bool:
    return abs(as_float(a) - as_float(b)) <= tol


def row_to_plain_dict(row: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in dict(row).items():
        try:
            import pandas as pd

            if pd.isna(value):
                out[str(key)] = None
                continue
        except Exception:
            pass

        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                pass

        out[str(key)] = value
    return out


def validate_mock_policy_row(row: Mapping[str, Any], row_index: int) -> List[str]:
    errors: List[str] = []

    for key, expected in MOCK_REQUIRED_POLICY_VALUES.items():
        actual = row.get(key)

        if isinstance(expected, bool):
            if as_bool(actual) is not expected:
                errors.append(
                    f"row {row_index}: {key} must be {expected}, got {actual}"
                )
        elif isinstance(expected, float):
            if not float_equal(actual, expected):
                errors.append(
                    f"row {row_index}: {key} must be {expected}, got {actual}"
                )
        else:
            if actual != expected:
                errors.append(
                    f"row {row_index}: {key} must be {expected!r}, got {actual!r}"
                )

    for key, expected in EXPECTED_K.items():
        if not float_equal(row.get(key), expected):
            errors.append(
                f"row {row_index}: {key} mismatch. expected={expected}, got={row.get(key)}"
            )

    source_mode = str(row.get("source_mode", ""))
    if "mock_policy" not in source_mode:
        errors.append(
            f"row {row_index}: mock_policy row source_mode should contain mock_policy, got {source_mode}"
        )

    if source_mode.startswith("causal_"):
        errors.append(
            f"row {row_index}: mock_policy row cannot have causal source_mode: {source_mode}"
        )

    return errors


def validate_mappo_policy_row(
    row: Mapping[str, Any],
    *,
    row_index: int,
    require_actual_ready: bool,
    require_causal_ready: bool,
) -> List[str]:
    from policies.policy_source_metadata import validate_rollout_policy_metadata

    result = validate_rollout_policy_metadata(
        row,
        require_actual_ready=bool(require_actual_ready),
        require_causal_ready=bool(require_causal_ready),
    )

    errors = [f"row {row_index}: {err}" for err in result.errors]

    source_mode = str(row.get("source_mode", ""))

    if row.get("policy_source") != "mappo_policy":
        errors.append(f"row {row_index}: policy_source must be mappo_policy")

    if source_mode.startswith("causal_") and as_bool(row.get("causal_policy_claim_ready")) is not True:
        errors.append(
            f"row {row_index}: causal source_mode requires causal_policy_claim_ready=true"
        )

    if source_mode.startswith("noncausal_") and as_bool(row.get("causal_policy_claim_ready")) is True:
        errors.append(
            f"row {row_index}: noncausal source_mode cannot have causal_policy_claim_ready=true"
        )

    return errors


def validate_window_rollup_policy_metadata(
    input_path: Path,
    *,
    allow_mock: bool = False,
    require_actual_ready: bool = False,
    require_causal_ready: bool = False,
) -> Dict[str, Any]:
    append_training_dir_to_path()

    import pandas as pd
    from policies.policy_source_metadata import policy_metadata_columns

    input_path = Path(input_path)
    if not input_path.exists():
        raise WindowRollupPolicyMetadataError(f"window_rollup not found: {input_path}")

    df = pd.read_parquet(input_path)

    errors: List[str] = []
    warnings: List[str] = []

    if df.empty:
        errors.append("window_rollup is empty")

    required_cols = list(dict.fromkeys(BASIC_WINDOW_ROLLUP_COLUMNS + policy_metadata_columns()))
    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        errors.append(f"missing required columns: {missing_cols}")

    policy_source_counts: Dict[str, int] = {}
    source_mode_counts: Dict[str, int] = {}
    actual_ready_count = 0
    causal_ready_count = 0
    mock_row_count = 0
    mappo_row_count = 0

    if not missing_cols:
        for idx, row_raw in enumerate(df.to_dict(orient="records")):
            row = row_to_plain_dict(row_raw)

            policy_source = str(row.get("policy_source", ""))
            source_mode = str(row.get("source_mode", ""))

            policy_source_counts[policy_source] = policy_source_counts.get(policy_source, 0) + 1
            source_mode_counts[source_mode] = source_mode_counts.get(source_mode, 0) + 1

            if as_bool(row.get("actual_policy_claim_ready", False)):
                actual_ready_count += 1

            if as_bool(row.get("causal_policy_claim_ready", False)):
                causal_ready_count += 1

            if as_float(row.get("qwen_trigger_rate", 1.0)) != 0.0:
                errors.append(
                    f"row {idx}: qwen_trigger_rate must be 0.0, got {row.get('qwen_trigger_rate')}"
                )

            if policy_source == "mock_policy":
                mock_row_count += 1
                if not allow_mock:
                    errors.append(
                        f"row {idx}: mock_policy row found but allow_mock=false"
                    )
                errors.extend(validate_mock_policy_row(row, idx))

            elif policy_source == "mappo_policy":
                mappo_row_count += 1
                errors.extend(
                    validate_mappo_policy_row(
                        row,
                        row_index=idx,
                        require_actual_ready=bool(require_actual_ready),
                        require_causal_ready=bool(require_causal_ready),
                    )
                )

            else:
                errors.append(
                    f"row {idx}: unsupported policy_source={policy_source!r}"
                )

    if require_actual_ready and actual_ready_count != int(len(df)):
        errors.append(
            f"require_actual_ready=true but actual_ready_count={actual_ready_count}, rows={len(df)}"
        )

    if require_causal_ready and causal_ready_count != int(len(df)):
        errors.append(
            f"require_causal_ready=true but causal_ready_count={causal_ready_count}, rows={len(df)}"
        )

    if mock_row_count > 0:
        warnings.append(
            "mock_policy rows are valid only for development smoke paths and never for performance claims"
        )

    if actual_ready_count == 0:
        warnings.append(
            "actual_policy_claim_ready count is zero; output is not actual performance-claim ready"
        )

    if causal_ready_count == 0:
        warnings.append(
            "causal_policy_claim_ready count is zero; output is not causal performance-claim ready"
        )

    payload = {
        "artifact_version": "window_rollup_policy_metadata_validation_v1",
        "input_path": str(input_path),
        "valid": len(errors) == 0,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "missing_columns": missing_cols,
        "allow_mock": bool(allow_mock),
        "require_actual_ready": bool(require_actual_ready),
        "require_causal_ready": bool(require_causal_ready),
        "policy_source_counts": policy_source_counts,
        "source_mode_counts": source_mode_counts,
        "mock_row_count": int(mock_row_count),
        "mappo_row_count": int(mappo_row_count),
        "actual_policy_claim_ready_count": int(actual_ready_count),
        "causal_policy_claim_ready_count": int(causal_ready_count),
        "errors": errors,
        "warnings": warnings,
    }

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate policy metadata columns in window_rollup.parquet"
    )
    parser.add_argument("--input", required=True, help="window_rollup.parquet path")
    parser.add_argument("--json-output", default="")
    parser.add_argument("--allow-mock", action="store_true")
    parser.add_argument("--require-actual-ready", action="store_true")
    parser.add_argument("--require-causal-ready", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    payload = validate_window_rollup_policy_metadata(
        Path(args.input),
        allow_mock=bool(args.allow_mock),
        require_actual_ready=bool(args.require_actual_ready),
        require_causal_ready=bool(args.require_causal_ready),
    )

    if args.json_output:
        dump_json(Path(args.json_output), payload)

    if payload["valid"]:
        print("[OK] window_rollup policy metadata validation PASS")
        print(f"[OK] input              : {payload['input_path']}")
        print(f"[OK] rows               : {payload['row_count']}")
        print(f"[OK] policy_sources     : {payload['policy_source_counts']}")
        print(f"[OK] source_modes       : {payload['source_mode_counts']}")
        print(f"[OK] actual_ready_count : {payload['actual_policy_claim_ready_count']}")
        print(f"[OK] causal_ready_count : {payload['causal_policy_claim_ready_count']}")
        for warning in payload["warnings"]:
            print(f"[WARN] {warning}")
        return 0

    print("[FAIL] window_rollup policy metadata validation FAIL")
    print(f"[FAIL] input: {payload['input_path']}")
    for err in payload["errors"]:
        print(f"[FAIL] {err}")
    for warning in payload["warnings"]:
        print(f"[WARN] {warning}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
