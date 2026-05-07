from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


BUNDLE_STATUS = "PATENT_EVIDENCE_REPORT_V2_READY_NONCLAIM"
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


def bool_from_any(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def resolve_path(raw_path: str, manifest_path: Path) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    for c in [Path.cwd() / p, manifest_path.parent / p]:
        if c.exists():
            return c
    return Path.cwd() / p


def validate_manifest(manifest_path: Path) -> Dict[str, Any]:
    m = load_json_any_encoding(manifest_path)
    if m.get("audit_status") != "PASS":
        raise RuntimeError(f"audit_status must be PASS, got {m.get('audit_status')}")
    if m.get("bundle_status") != BUNDLE_STATUS:
        raise RuntimeError(f"bundle_status mismatch: {m.get('bundle_status')}")

    for key, expected in NON_CLAIM_FLAGS.items():
        observed = m.get(key)
        if bool_from_any(observed) != expected:
            raise RuntimeError(f"guard flag mismatch: {key}={observed}, expected={expected}")

    summary = m.get("summary", {})
    required_summary = [
        "total_pickup_attempts",
        "zero_loss_success_count",
        "zero_loss_success_rate",
        "filtered_attention_row_count",
        "exact_edge_overlap_count",
        "node_overlap_count",
        "fallback_attempt_count",
        "real_gatv2conv_attention_extracted_all",
        "attempt_specific_route_path_filter_applied",
    ]
    for key in required_summary:
        if key not in summary:
            raise RuntimeError(f"summary missing {key}")

    if int(summary["total_pickup_attempts"]) <= 0:
        raise RuntimeError("total_pickup_attempts must be positive")
    if int(summary["filtered_attention_row_count"]) <= 0:
        raise RuntimeError("filtered_attention_row_count must be positive")
    if bool_from_any(summary["real_gatv2conv_attention_extracted_all"]) is not True:
        raise RuntimeError("real_gatv2conv_attention_extracted_all must be true")
    if bool_from_any(summary["attempt_specific_route_path_filter_applied"]) is not True:
        raise RuntimeError("attempt_specific_route_path_filter_applied must be true")

    output_files = m.get("output_files", {})
    required_outputs = [
        "patent_evidence_report_v2",
        "patent_evidence_summary_v2",
        "zero_loss_attempt_attention_summary_v2",
        "attention_mass_by_attempt_path_v2",
        "attention_mass_by_success_path_v2",
        "top_filtered_attention_edges_v2",
    ]
    for key in required_outputs:
        if key not in output_files:
            raise RuntimeError(f"output_files missing {key}")
        p = resolve_path(output_files[key], manifest_path)
        if not p.exists():
            raise RuntimeError(f"output file does not exist: {key} -> {p}")

    row_counts = m.get("row_counts", {})
    if int(row_counts.get("attempt_summary_rows", 0)) != int(summary["total_pickup_attempts"]):
        raise RuntimeError("attempt_summary_rows must match total_pickup_attempts")

    return m


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Step 166 Patent Evidence Report v2 manifest")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    m = validate_manifest(Path(args.manifest))
    s = m["summary"]
    print("[OK] Step 166 Patent Evidence Report v2 validation PASS")
    print(f"[OK] manifest                  : {args.manifest}")
    print(f"[OK] total_pickup_attempts     : {s['total_pickup_attempts']}")
    print(f"[OK] zero_loss_success_count   : {s['zero_loss_success_count']}")
    print(f"[OK] zero_loss_success_rate    : {s['zero_loss_success_rate']:.6f}")
    print(f"[OK] filtered_attention_row_count : {s['filtered_attention_row_count']}")
    print(f"[OK] exact_edge_overlap_count  : {s['exact_edge_overlap_count']}")
    print(f"[OK] node_overlap_count        : {s['node_overlap_count']}")
    print(f"[OK] paper_level_claim_allowed : {m['paper_level_claim_allowed']}")
    print(f"[OK] causal_performance_claim_allowed : {m['causal_performance_claim_allowed']}")


if __name__ == "__main__":
    main()
