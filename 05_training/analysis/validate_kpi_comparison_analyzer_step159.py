from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


EXPECTED_TOOL_STATUS = "KPI_COMPARISON_ANALYZER_READY_SCAFFOLD_STILL_LOCKED"
EXPECTED_DECISION = "SCAFFOLD_ONLY_NOT_ACTUAL_COMPARISON"
EXPECTED_NEXT_GATE = "WAITING_FOR_REAL_H200_STEP149_EXPECT_H200_RESULT"

REQUIRED_LOCK_VALUES: Dict[str, Any] = {
    "comparison_tool_status": EXPECTED_TOOL_STATUS,
    "comparison_allowed": True,
    "winner_selected": False,
    "trainable_reward_promoted": False,
    "actual_results": False,
    "actual_execution_allowed": False,
    "actual_execution_released": False,
    "train_allowed": False,
    "paper_level_claim_allowed": False,
    "causal_performance_claim_allowed": False,
    "live_mutation_allowed": False,
    "db_write_allowed": False,
    "h200_required_for_actual_claim": True,
    "next_gate": EXPECTED_NEXT_GATE,
}


def load_json(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read JSON: {path}")


def add_failure(failures: List[str], message: str) -> None:
    failures.append(message)


def check_locks(payload: Dict[str, Any], source: str, failures: List[str]) -> None:
    for key, expected in REQUIRED_LOCK_VALUES.items():
        if key not in payload:
            add_failure(failures, f"[{source}] missing required key: {key}")
            continue
        actual = payload[key]
        if actual != expected:
            add_failure(
                failures,
                f"[{source}] {key} must be {expected!r}; got {actual!r}",
            )


def file_non_empty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def count_csv_data_rows(path: Path) -> int:
    if not path.exists():
        return -1
    with path.open("r", encoding="utf-8", newline="") as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    if not lines:
        return 0
    return max(0, len(lines) - 1)


def find_baseline_in_summary(summary_csv: Path, baseline_condition: str) -> bool:
    if not summary_csv.exists():
        return False
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        first = f.readline().strip().split(",")
        if "condition_id" not in first:
            return False
        cond_idx = first.index("condition_id")
        for line in f:
            parts = line.strip().split(",")
            if len(parts) <= cond_idx:
                continue
            if parts[cond_idx] == baseline_condition:
                return True
    return False


def validate(manifest_path: Path, guard_path: Path) -> Tuple[List[str], Dict[str, Any]]:
    failures: List[str] = []

    if not manifest_path.exists():
        add_failure(failures, f"manifest not found: {manifest_path}")
        return failures, {}

    if not guard_path.exists():
        add_failure(failures, f"guard not found: {guard_path}")
        return failures, {}

    manifest = load_json(manifest_path)
    guard = load_json(guard_path)

    if manifest.get("comparison_tool_status") != EXPECTED_TOOL_STATUS:
        add_failure(
            failures,
            f"manifest.comparison_tool_status must be {EXPECTED_TOOL_STATUS}; "
            f"got {manifest.get('comparison_tool_status')}",
        )

    if manifest.get("decision") != EXPECTED_DECISION:
        add_failure(
            failures,
            f"manifest.decision must be {EXPECTED_DECISION}; got {manifest.get('decision')}",
        )

    check_locks(manifest, "manifest", failures)
    check_locks(guard, "guard", failures)

    output_files = manifest.get("output_files", {})
    required_keys = [
        "manifest",
        "guard",
        "summary_csv",
        "delta_by_condition_csv",
        "delta_by_seed_csv",
        "delta_by_time_band_csv",
        "report_md",
    ]
    for key in required_keys:
        path_str = output_files.get(key)
        if not path_str:
            add_failure(failures, f"manifest.output_files.{key} missing")
            continue
        path = Path(path_str)
        if not file_non_empty(path):
            add_failure(failures, f"output file empty or missing: {path}")

    delta_files = [
        ("delta_by_condition_csv", output_files.get("delta_by_condition_csv")),
        ("delta_by_seed_csv", output_files.get("delta_by_seed_csv")),
        ("delta_by_time_band_csv", output_files.get("delta_by_time_band_csv")),
    ]
    for name, path_str in delta_files:
        if not path_str:
            continue
        rows = count_csv_data_rows(Path(path_str))
        if rows <= 0:
            add_failure(failures, f"delta file has no data rows: {name} ({path_str})")

    summary_csv_str = output_files.get("summary_csv")
    baseline_condition = manifest.get("baseline_condition")
    if summary_csv_str and baseline_condition:
        if not find_baseline_in_summary(Path(summary_csv_str), str(baseline_condition)):
            add_failure(
                failures,
                f"baseline_condition '{baseline_condition}' not found in summary CSV",
            )

    return failures, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--guard", default=None)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    if args.guard:
        guard_path = Path(args.guard).resolve()
    else:
        guard_path = manifest_path.parent / "claim_guard_status_step159.json"

    failures, manifest = validate(manifest_path, guard_path)

    if failures:
        print(f"[FAIL] Step 159 KPI comparison analyzer validation: {len(failures)} failure(s)")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)

    print("[OK] Step 159 KPI comparison analyzer validation PASS")
    print(f"[OK] comparison_tool_status: {manifest.get('comparison_tool_status')}")
    print(f"[OK] decision              : {manifest.get('decision')}")
    print(f"[OK] baseline_condition    : {manifest.get('baseline_condition')}")
    print(f"[OK] conditions            : {manifest.get('conditions')}")
    print(f"[OK] next_gate             : {manifest.get('next_gate')}")


if __name__ == "__main__":
    main()
