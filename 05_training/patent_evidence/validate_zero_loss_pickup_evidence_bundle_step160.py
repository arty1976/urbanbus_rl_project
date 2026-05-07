from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


REQUIRED_MANIFEST_KEYS = [
    "artifact_version",
    "audit_status",
    "bundle_status",
    "output_files",
    "summary",
    "simulation_evidence_only",
    "actual_operational_claim_allowed",
    "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
    "train_allowed",
]


class ValidationError(RuntimeError):
    pass


def load_json_any_encoding(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise ValidationError(f"failed to read json: {path}")


def bool_from_any(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def require_file(path_str: str, label: str) -> Path:
    path = Path(path_str)
    if not path.exists():
        raise ValidationError(f"required output missing: {label}: {path}")
    if path.stat().st_size <= 0:
        raise ValidationError(f"required output empty: {label}: {path}")
    return path


def validate_manifest(manifest_path: Path) -> Dict[str, Any]:
    manifest = load_json_any_encoding(manifest_path)

    missing = [k for k in REQUIRED_MANIFEST_KEYS if k not in manifest]
    if missing:
        raise ValidationError(f"manifest missing required keys: {missing}")

    if manifest.get("audit_status") != "PASS":
        raise ValidationError(f"audit_status must be PASS, got {manifest.get('audit_status')}")

    if bool_from_any(manifest.get("simulation_evidence_only")) is not True:
        raise ValidationError("simulation_evidence_only must be true")
    for key in [
        "actual_operational_claim_allowed",
        "paper_level_claim_allowed",
        "causal_performance_claim_allowed",
        "train_allowed",
    ]:
        if bool_from_any(manifest.get(key)) is not False:
            raise ValidationError(f"{key} must be false")

    summary = manifest["summary"]
    total = int(summary.get("total_pickup_attempts", -1))
    success = int(summary.get("zero_loss_success_count", -1))
    failure = int(summary.get("zero_loss_failure_count", -1))
    epsilon = float(summary.get("zero_loss_epsilon_sec", -1))

    if total <= 0:
        raise ValidationError("total_pickup_attempts must be > 0")
    if success < 0 or failure < 0:
        raise ValidationError("success/failure counts must be >= 0")
    if success + failure != total:
        raise ValidationError("success + failure must equal total")
    if epsilon < 0:
        raise ValidationError("zero_loss_epsilon_sec must be >= 0")

    output_files = manifest["output_files"]
    required_outputs = [
        "zero_loss_summary",
        "zero_loss_by_condition",
        "eta_delta_by_attempt",
        "attention_mass_by_segment",
        "top_attention_edges",
        "patent_evidence_report",
        "zero_loss_evidence_manifest",
    ]
    for key in required_outputs:
        if key not in output_files:
            raise ValidationError(f"output_files missing {key}")
        require_file(output_files[key], key)

    eta_delta = pd.read_csv(output_files["eta_delta_by_attempt"])
    required_eta_cols = [
        "attempt_id",
        "delta_eta_existing_passenger_sec",
        "zero_loss_epsilon_sec",
        "zero_loss_success",
    ]
    missing_eta = [c for c in required_eta_cols if c not in eta_delta.columns]
    if missing_eta:
        raise ValidationError(f"eta_delta_by_attempt missing columns: {missing_eta}")

    if int(len(eta_delta)) != total:
        raise ValidationError("eta_delta_by_attempt row count must equal total_pickup_attempts")

    eta_delta["delta_eta_existing_passenger_sec"] = pd.to_numeric(
        eta_delta["delta_eta_existing_passenger_sec"], errors="raise"
    )
    eta_delta["zero_loss_epsilon_sec"] = pd.to_numeric(
        eta_delta["zero_loss_epsilon_sec"], errors="raise"
    )
    normalized_success = eta_delta["zero_loss_success"].astype(str).str.lower().isin(["true", "1", "yes"])
    expected_success = eta_delta["delta_eta_existing_passenger_sec"] <= eta_delta["zero_loss_epsilon_sec"]
    if not bool((normalized_success == expected_success).all()):
        bad = eta_delta.loc[normalized_success != expected_success].head(5)
        raise ValidationError(f"zero_loss_success mismatch: {bad.to_dict('records')}")

    by_condition = pd.read_csv(output_files["zero_loss_by_condition"])
    if by_condition.empty:
        raise ValidationError("zero_loss_by_condition must not be empty")
    if int(pd.to_numeric(by_condition["pickup_attempt_count"], errors="raise").sum()) != total:
        raise ValidationError("zero_loss_by_condition attempt sum must equal total")

    top_edges = pd.read_csv(output_files["top_attention_edges"])
    if top_edges.empty:
        raise ValidationError("top_attention_edges must not be empty")
    if "attention_weight" not in top_edges.columns:
        raise ValidationError("top_attention_edges missing attention_weight")

    mass = pd.read_csv(output_files["attention_mass_by_segment"])
    if mass.empty:
        raise ValidationError("attention_mass_by_segment must not be empty")
    if "attention_mass_sum" not in mass.columns:
        raise ValidationError("attention_mass_by_segment missing attention_mass_sum")

    return {
        "audit_status": "PASS",
        "manifest": str(manifest_path),
        "total_pickup_attempts": total,
        "zero_loss_success_count": success,
        "zero_loss_success_rate": float(summary.get("zero_loss_success_rate", 0.0)),
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Step 160 Zero-Loss Pickup Evidence Bundle")
    parser.add_argument("--manifest", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_manifest(Path(args.manifest))
    print("[OK] Step 160 zero-loss pickup evidence bundle validation PASS")
    print(f"[OK] manifest                : {result['manifest']}")
    print(f"[OK] total_pickup_attempts   : {result['total_pickup_attempts']}")
    print(f"[OK] zero_loss_success_count : {result['zero_loss_success_count']}")
    print(f"[OK] zero_loss_success_rate  : {result['zero_loss_success_rate']:.6f}")
    print("[OK] paper_level_claim_allowed : False")
    print("[OK] causal_performance_claim_allowed : False")


if __name__ == "__main__":
    main()
