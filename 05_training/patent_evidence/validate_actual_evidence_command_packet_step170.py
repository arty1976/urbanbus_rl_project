from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

EXPECTED_FALSE_FLAGS = [
    "actual_evidence_execution_allowed",
    "command_execution_allowed",
    "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
    "actual_operational_claim_allowed",
    "train_allowed",
    "winner_selected",
]


def load_json(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with path.open("r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


def validate_manifest(path: Path, require_pass: bool = False, require_dry_run: bool = False) -> Dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"manifest not found: {path}")
    m = load_json(path)
    if m.get("artifact_version") != "actual_evidence_command_packet_step170_v1":
        raise RuntimeError(f"artifact_version mismatch: {m.get('artifact_version')}")
    if require_pass and m.get("audit_status") != "PASS":
        raise RuntimeError(f"audit_status must be PASS, got {m.get('audit_status')}")
    if int(m.get("command_count", 0)) < 6:
        raise RuntimeError(f"command_count too small: {m.get('command_count')}")
    if require_dry_run and not bool(m.get("dry_run_only", False)):
        raise RuntimeError("dry_run_only must be true")
    for flag in EXPECTED_FALSE_FLAGS:
        if bool(m.get(flag, False)) is not False:
            raise RuntimeError(f"guard flag mismatch: {flag}={m.get(flag)}, expected=False")
    files = m.get("output_files", {})
    for key in ("command_sequence_json", "operator_script_ps1", "report_md"):
        p = Path(files.get(key, ""))
        if not p.exists():
            raise RuntimeError(f"missing output file {key}: {p}")
    seq = load_json(Path(files["command_sequence_json"]))
    commands = seq.get("commands", [])
    if len(commands) != int(m.get("command_count")):
        raise RuntimeError("command sequence count mismatch")
    required_steps = ["Step 162", "Step 161", "Step 163", "Step 165", "Step 160", "Step 166"]
    got_steps = [str(c.get("step")) for c in commands]
    for step in required_steps:
        if step not in got_steps:
            raise RuntimeError(f"missing command step: {step}")
    print("[OK] Step 170 actual evidence command packet validation PASS")
    print(f"[OK] manifest                  : {path}")
    print(f"[OK] audit_status              : {m.get('audit_status')}")
    print(f"[OK] bundle_status             : {m.get('bundle_status')}")
    print(f"[OK] command_count             : {m.get('command_count')}")
    print(f"[OK] dry_run_only              : {m.get('dry_run_only')}")
    print(f"[OK] command_execution_allowed : {m.get('command_execution_allowed')}")
    print(f"[OK] paper_level_claim_allowed : {m.get('paper_level_claim_allowed')}")
    print(f"[OK] causal_performance_claim_allowed : {m.get('causal_performance_claim_allowed')}")
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Step 170 actual evidence command packet")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--require-pass", action="store_true")
    parser.add_argument("--require-dry-run", action="store_true")
    args = parser.parse_args()
    validate_manifest(Path(args.manifest), require_pass=args.require_pass, require_dry_run=args.require_dry_run)


if __name__ == "__main__":
    main()
