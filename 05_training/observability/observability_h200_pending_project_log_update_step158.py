from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

STEP_ID = "step158"
UPDATE_STATUS = "OBSERVABILITY_H200_PENDING_PROJECT_LOG_UPDATED_STILL_LOCKED"
FINAL_LOCAL_STATUS = "LOCAL_PREPARATION_AND_OBSERVABILITY_HANDOFF_COMPLETE"
H200_PENDING_STATUS = "WAITING_FOR_H200_SERVER_AVAILABILITY"
BLOCK_REASON = "H200_SERVER_NOT_AVAILABLE_YET"
NEXT_GATE = "WAITING_FOR_REAL_H200_STEP149_EXPECT_H200_RESULT"
CONTROL_POLICY = "NO_LIVE_MUTATION_CONFIG_BASED_NEXT_RUN_ONLY"
DASHBOARD_MODE = "MONITORING_ONLY"

START_MARKER = "<!-- STEP_158_OBSERVABILITY_H200_PENDING_PROJECT_LOG_UPDATE_START -->"
END_MARKER = "<!-- STEP_158_OBSERVABILITY_H200_PENDING_PROJECT_LOG_UPDATE_END -->"

OBSERVABILITY_COMPONENTS = [
    {
        "step": "152",
        "name": "H200 monitoring dashboard contract",
        "manifest": "artifacts/observability/h200_monitoring_dashboard_contract_step152/h200_monitoring_dashboard_contract_step152_manifest.json",
        "expected_status_key": "contract_status",
        "expected_status": "OBSERVABILITY_CONTRACT_READY_MONITORING_ONLY_STILL_LOCKED",
    },
    {
        "step": "153",
        "name": "H200 TensorBoard + JSONL logger integration",
        "manifest": "artifacts/observability/h200_tensorboard_jsonl_logger_step153/h200_tensorboard_jsonl_logger_step153_manifest.json",
        "expected_status_key": "integration_status",
        "expected_status": "LOGGER_INTEGRATION_READY_MONITORING_ONLY_STILL_LOCKED",
    },
    {
        "step": "154",
        "name": "H200 nvidia-smi GPU monitor logger scaffold",
        "manifest": "artifacts/observability/h200_nvidia_smi_gpu_monitor_step154/h200_nvidia_smi_gpu_monitor_step154_manifest.json",
        "expected_status_key": "monitor_status",
        "expected_status": "NVIDIA_SMI_GPU_MONITOR_SCAFFOLD_READY_MONITORING_ONLY_STILL_LOCKED",
    },
    {
        "step": "155",
        "name": "H200 observability dashboard index / status page",
        "manifest": "artifacts/observability/h200_observability_dashboard_index_step155/h200_observability_dashboard_index_step155_manifest.json",
        "expected_status_key": "index_status",
        "expected_status": "OBSERVABILITY_DASHBOARD_INDEX_READY_MONITORING_ONLY_STILL_LOCKED",
    },
    {
        "step": "156",
        "name": "H200 observability operator runbook",
        "manifest": "artifacts/observability/h200_observability_operator_runbook_step156/h200_observability_operator_runbook_step156_manifest.json",
        "expected_status_key": "runbook_status",
        "expected_status": "H200_OBSERVABILITY_OPERATOR_RUNBOOK_READY_MONITORING_ONLY_STILL_LOCKED",
    },
    {
        "step": "157",
        "name": "H200 observability handoff packet index",
        "manifest": "artifacts/observability/h200_observability_handoff_packet_index_step157/h200_observability_handoff_packet_index_step157_manifest.json",
        "expected_status_key": "packet_status",
        "expected_status": "H200_OBSERVABILITY_HANDOFF_PACKET_INDEX_READY_MONITORING_ONLY_STILL_LOCKED",
    },
]

LOCKS = {
    "dashboard_mode": DASHBOARD_MODE,
    "control_policy": CONTROL_POLICY,
    "actual_execution_allowed": False,
    "actual_execution_released": False,
    "train_allowed": False,
    "live_mutation_allowed": False,
    "actual_results": False,
    "winner_selected": False,
    "trainable_reward_promoted": False,
    "paper_level_claim_allowed": False,
    "causal_performance_claim_allowed": False,
    "operator_approval_recorded": False,
    "operator_approval_granted": False,
}


def load_json(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def inspect_components(project_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in OBSERVABILITY_COMPONENTS:
        manifest_path = project_root / item["manifest"]
        row: Dict[str, Any] = {
            "step": item["step"],
            "name": item["name"],
            "manifest": item["manifest"],
            "manifest_exists": manifest_path.exists(),
            "expected_status_key": item["expected_status_key"],
            "expected_status": item["expected_status"],
            "observed_status": None,
            "status_match": False,
            "locked_flags_ok": False,
            "notes": [],
        }
        if manifest_path.exists():
            payload = load_json(manifest_path)
            observed = payload.get(item["expected_status_key"])
            row["observed_status"] = observed
            row["status_match"] = observed == item["expected_status"]
            locked_ok = True
            for key in ("actual_execution_allowed", "actual_execution_released", "train_allowed", "live_mutation_allowed"):
                if key in payload and bool(payload.get(key)) is not False:
                    locked_ok = False
                    row["notes"].append(f"{key} is not false")
            if payload.get("dashboard_mode", DASHBOARD_MODE) != DASHBOARD_MODE:
                locked_ok = False
                row["notes"].append("dashboard_mode is not MONITORING_ONLY")
            if payload.get("control_policy", CONTROL_POLICY) != CONTROL_POLICY:
                locked_ok = False
                row["notes"].append("control_policy mismatch")
            row["locked_flags_ok"] = locked_ok
        else:
            row["notes"].append("manifest missing on this machine; project log still records pending summary")
        rows.append(row)
    return rows


def to_csv(rows: List[Dict[str, Any]]) -> str:
    header = [
        "step",
        "name",
        "manifest",
        "manifest_exists",
        "expected_status_key",
        "expected_status",
        "observed_status",
        "status_match",
        "locked_flags_ok",
        "notes",
    ]
    lines = [",".join(header)]
    for r in rows:
        vals = []
        for h in header:
            v = r.get(h)
            if isinstance(v, list):
                v = "; ".join(str(x) for x in v)
            s = "" if v is None else str(v)
            s = '"' + s.replace('"', '""') + '"'
            vals.append(s)
        lines.append(",".join(vals))
    return "\n".join(lines) + "\n"


def build_project_log_section(component_rows: List[Dict[str, Any]], now_utc: str) -> str:
    component_lines = []
    for row in component_rows:
        exists = "present" if row["manifest_exists"] else "missing_on_this_machine"
        match = "match" if row["status_match"] else "not_verified"
        locks = "locked_ok" if row["locked_flags_ok"] else "locked_not_verified"
        component_lines.append(
            f"- Step {row['step']} — {row['name']}: manifest={exists}, status={match}, locks={locks}"
        )

    components_md = "\n".join(component_lines)
    return f"""{START_MARKER}

## 📅 2026-04-30 — Step 158 Observability + H200 Pending Project Log Update

### Summary

Step 158 closes the local observability preparation record after Step 152~157 and explicitly records that the real H200 server is not available yet.

Current final local status:

```text
{FINAL_LOCAL_STATUS}
{H200_PENDING_STATUS}
```

Blocking reason:

```text
{BLOCK_REASON}
```

### Observability scaffold status

{components_md}

### Locked guard state

```text
actual_execution_allowed = false
actual_execution_released = false
train_allowed = false
live_mutation_allowed = false
actual_results = false
winner_selected = false
trainable_reward_promoted = false
operator_approval_recorded = false
operator_approval_granted = false
paper_level_claim_allowed = false
causal_performance_claim_allowed = false
```

### Interpretation

The local notebook has completed the monitoring-only observability scaffold and handoff documentation. However, the actual H200 Step 149 `--expect-h200` preflight cannot be executed until the real H200 server is available.

No local placeholder, mock, expected checklist, or simulated GPU result may be substituted for the real H200 preflight manifest.

### Next external gate

```text
{NEXT_GATE}
```

The next real action remains execution of Step 149 on the real H200 server:

```bash
python 05_training/rewards/h200_environment_preflight_result_manifest_step149.py \
  --project-root /workspace/urbanbus_rl_project \
  --expect-h200 \
  --min-gpu-count 1 \
  --output-root artifacts/rewards/h200_environment_preflight_result_manifest_step149_h200_actual
```

Generated at UTC: `{now_utc}`

{END_MARKER}
"""


def upsert_section(text: str, section: str) -> str:
    if START_MARKER in text and END_MARKER in text:
        start = text.index(START_MARKER)
        end = text.index(END_MARKER) + len(END_MARKER)
        return text[:start].rstrip() + "\n\n" + section.rstrip() + "\n" + text[end:].lstrip("\n")
    return text.rstrip() + "\n\n" + section.rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--output-root",
        default="artifacts/observability/observability_h200_pending_project_log_update_step158",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_root = (project_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    project_log_path = project_root / "project_log.md"
    project_log_exists_before = project_log_path.exists()
    existing_log = project_log_path.read_text(encoding="utf-8-sig") if project_log_exists_before else "# 🚌 UrbanBus RL Project - 작업 일지 (Project Journal)\n"

    now_utc = datetime.now(timezone.utc).isoformat()
    component_rows = inspect_components(project_root)
    section = build_project_log_section(component_rows, now_utc)
    updated_log = upsert_section(existing_log, section)

    updated_log_path = output_root / "project_log_step158_updated_preview.md"
    updated_log_path.write_text(updated_log, encoding="utf-8")

    section_path = output_root / "project_log_step158_section.md"
    section_path.write_text(section, encoding="utf-8")

    component_csv_path = output_root / "observability_h200_pending_component_status_step158.csv"
    component_csv_path.write_text(to_csv(component_rows), encoding="utf-8")

    if not args.dry_run:
        project_log_path.write_text(updated_log, encoding="utf-8")

    component_manifest_count = sum(1 for r in component_rows if r["manifest_exists"])
    component_status_matches = sum(1 for r in component_rows if r["status_match"])
    hard_failures = 0
    warnings: List[str] = []
    if component_manifest_count < len(component_rows):
        warnings.append("some_observability_manifests_missing_on_this_machine")
    if component_status_matches < component_manifest_count:
        warnings.append("some_present_observability_manifests_have_unverified_status")

    manifest = {
        "artifact_version": "observability_h200_pending_project_log_update_step158_v1",
        "step_id": STEP_ID,
        "update_status": UPDATE_STATUS,
        "final_local_status": FINAL_LOCAL_STATUS,
        "h200_pending_status": H200_PENDING_STATUS,
        "block_reason": BLOCK_REASON,
        "next_gate": NEXT_GATE,
        **LOCKS,
        "project_root": str(project_root),
        "project_log": str(project_log_path),
        "project_log_exists_before": bool(project_log_exists_before),
        "project_log_updated": bool(not args.dry_run),
        "dry_run": bool(args.dry_run),
        "component_count": len(component_rows),
        "component_manifest_count": int(component_manifest_count),
        "component_status_matches": int(component_status_matches),
        "hard_failures": hard_failures,
        "warnings": warnings,
        "output_files": {
            "project_log_step158_section": str(section_path),
            "project_log_step158_updated_preview": str(updated_log_path),
            "observability_h200_pending_component_status": str(component_csv_path),
            "manifest": str(output_root / "observability_h200_pending_project_log_update_step158_manifest.json"),
        },
        "generated_at_utc": now_utc,
    }

    manifest_path = output_root / "observability_h200_pending_project_log_update_step158_manifest.json"
    dump_json(manifest_path, manifest)

    latest_path = project_root / "05_training" / "observability" / "observability_h200_pending_project_log_update_step158.latest.json"
    dump_json(latest_path, {"latest_manifest": str(manifest_path), "updated_at_utc": now_utc})

    print("[OK] Step 158 observability + H200 pending project log update generated")
    print(f"[OK] update_status      : {manifest['update_status']}")
    print(f"[OK] final_local_status : {manifest['final_local_status']}")
    print(f"[OK] h200_pending_status: {manifest['h200_pending_status']}")
    print(f"[OK] block_reason       : {manifest['block_reason']}")
    print(f"[OK] project_log_updated: {manifest['project_log_updated']}")
    print(f"[OK] train_allowed      : {manifest['train_allowed']}")
    print(f"[OK] manifest           : {manifest_path}")
    if warnings:
        print(f"[WARN] warnings          : {warnings}")


if __name__ == "__main__":
    main()
