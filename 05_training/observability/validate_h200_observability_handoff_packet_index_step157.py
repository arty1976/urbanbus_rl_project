from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict

EXPECTED_PACKET_STATUS = "H200_OBSERVABILITY_HANDOFF_PACKET_INDEX_READY_MONITORING_ONLY_STILL_LOCKED"
EXPECTED_DECISION = "OBSERVABILITY_PACKET_ONLY_NOT_RELEASED"
EXPECTED_DASHBOARD_MODE = "MONITORING_ONLY"
EXPECTED_CONTROL_POLICY = "NO_LIVE_MUTATION_CONFIG_BASED_NEXT_RUN_ONLY"

FALSE_GUARDS = [
    "actual_execution_allowed",
    "actual_execution_released",
    "train_allowed",
    "live_mutation_allowed",
    "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
    "operator_approval_recorded",
    "operator_approval_granted",
]


def load_json(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


def fail(message: str) -> None:
    raise SystemExit(f"[FAIL] {message}")


def validate_manifest(manifest_path: Path) -> Dict[str, Any]:
    if not manifest_path.exists():
        fail(f"manifest not found: {manifest_path}")
    m = load_json(manifest_path)

    if m.get("packet_status") != EXPECTED_PACKET_STATUS:
        fail(f"packet_status mismatch: {m.get('packet_status')}")
    if m.get("decision") != EXPECTED_DECISION:
        fail(f"decision mismatch: {m.get('decision')}")
    if m.get("dashboard_mode") != EXPECTED_DASHBOARD_MODE:
        fail(f"dashboard_mode mismatch: {m.get('dashboard_mode')}")
    if m.get("control_policy") != EXPECTED_CONTROL_POLICY:
        fail(f"control_policy mismatch: {m.get('control_policy')}")

    for key in FALSE_GUARDS:
        if bool(m.get(key)) is not False:
            fail(f"{key} must be false, got {m.get(key)}")

    output_files = m.get("output_files", {})
    required = ["summary_md", "operator_packet_json", "component_index_csv", "command_index_sh", "local_notes_md"]
    for key in required:
        p = Path(str(output_files.get(key, "")))
        if not p.exists():
            fail(f"required output file missing: {key}={p}")

    component_csv = Path(str(output_files["component_index_csv"]))
    with component_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != int(m.get("component_count", -1)):
        fail("component CSV row count mismatch")
    if len(rows) < 5:
        fail("component CSV must contain Step 152-156 rows")

    forbidden_tokens = [
        "actual_execution_allowed = true",
        "actual_execution_released = true",
        "train_allowed = true",
        "live_mutation_allowed = true",
        "actual reward ablation released",
    ]
    for key in ["summary_md", "command_index_sh", "local_notes_md"]:
        text = Path(str(output_files[key])).read_text(encoding="utf-8", errors="ignore").lower()
        for token in forbidden_tokens:
            if token in text:
                fail(f"forbidden token found in {key}: {token}")

    return m


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    m = validate_manifest(manifest_path)
    print("[OK] Step 157 H200 observability handoff packet index validation PASS")
    print(f"[OK] manifest: {manifest_path}")
    print(f"[OK] packet_status: {m.get('packet_status')}")
    print(f"[OK] decision     : {m.get('decision')}")
    print(f"[OK] components   : {m.get('component_manifest_present_count')}/{m.get('component_count')} present")


if __name__ == "__main__":
    main()
