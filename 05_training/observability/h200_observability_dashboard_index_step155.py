from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

INDEX_STATUS = "OBSERVABILITY_DASHBOARD_INDEX_READY_MONITORING_ONLY_STILL_LOCKED"
DASHBOARD_MODE = "MONITORING_ONLY"
CONTROL_POLICY = "NO_LIVE_MUTATION_CONFIG_BASED_NEXT_RUN_ONLY"

REQUIRED_COMPONENTS = [
    {
        "step": "152",
        "name": "h200_monitoring_dashboard_contract",
        "manifest_relpath": "artifacts/observability/h200_monitoring_dashboard_contract_step152/h200_monitoring_dashboard_contract_step152_manifest.json",
        "status_key_candidates": ["contract_status", "status"],
        "expected_status": "OBSERVABILITY_CONTRACT_READY_MONITORING_ONLY_STILL_LOCKED",
    },
    {
        "step": "153",
        "name": "h200_tensorboard_jsonl_logger",
        "manifest_relpath": "artifacts/observability/h200_tensorboard_jsonl_logger_step153/h200_tensorboard_jsonl_logger_step153_manifest.json",
        "status_key_candidates": ["integration_status", "status"],
        "expected_status": "LOGGER_INTEGRATION_READY_MONITORING_ONLY_STILL_LOCKED",
    },
    {
        "step": "154",
        "name": "h200_nvidia_smi_gpu_monitor",
        "manifest_relpath": "artifacts/observability/h200_nvidia_smi_gpu_monitor_step154/h200_nvidia_smi_gpu_monitor_step154_manifest.json",
        "status_key_candidates": ["monitor_status", "status"],
        "expected_status": "NVIDIA_SMI_GPU_MONITOR_SCAFFOLD_READY_MONITORING_ONLY_STILL_LOCKED",
    },
]

LOCK_FIELDS = [
    "actual_execution_allowed",
    "actual_execution_released",
    "train_allowed",
    "live_mutation_allowed",
    "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def find_status(payload: Optional[Dict[str, Any]], keys: List[str]) -> Optional[str]:
    if not payload:
        return None
    for key in keys:
        if key in payload:
            return str(payload.get(key))
    # Some earlier manifests put status under nested summary-like fields.
    for container_key in ("summary", "status_summary", "dashboard", "logger", "monitor"):
        nested = payload.get(container_key)
        if isinstance(nested, dict):
            for key in keys:
                if key in nested:
                    return str(nested.get(key))
    return None


def get_bool(payload: Optional[Dict[str, Any]], key: str, default: bool = False) -> bool:
    if not payload:
        return default
    if key in payload:
        return bool(payload[key])
    locks = payload.get("locks")
    if isinstance(locks, dict) and key in locks:
        return bool(locks[key])
    guards = payload.get("guards")
    if isinstance(guards, dict) and key in guards:
        return bool(guards[key])
    return default


def find_tensorboard_status(payload: Optional[Dict[str, Any]]) -> str:
    if not payload:
        return "unknown_missing_manifest"
    for key in ("tensorboard_status", "tensorboard"):
        if key in payload:
            return str(payload[key])
    tb = payload.get("tensorboard")
    if isinstance(tb, dict):
        for key in ("status", "availability", "writer_status"):
            if key in tb:
                return str(tb[key])
    return "unknown_not_reported"


def find_gpu_probe_status(payload: Optional[Dict[str, Any]]) -> str:
    if not payload:
        return "unknown_missing_manifest"
    for key in ("nvidia_smi_status", "gpu_probe_status", "probe_status"):
        if key in payload:
            return str(payload[key])
    gpu = payload.get("gpu")
    if isinstance(gpu, dict):
        for key in ("nvidia_smi_status", "probe_status", "status"):
            if key in gpu:
                return str(gpu[key])
    return "unknown_not_reported"


def build_component_summary(project_root: Path) -> List[Dict[str, Any]]:
    components: List[Dict[str, Any]] = []
    for spec in REQUIRED_COMPONENTS:
        path = project_root / spec["manifest_relpath"]
        payload = read_json(path)
        exists = payload is not None
        status_value = find_status(payload, spec["status_key_candidates"])
        status_matches_expected = bool(status_value == spec["expected_status"])
        locks = {key: get_bool(payload, key, default=False) for key in LOCK_FIELDS}
        lock_violation_fields = [key for key, value in locks.items() if bool(value)]

        component: Dict[str, Any] = {
            "step": spec["step"],
            "name": spec["name"],
            "manifest_relpath": spec["manifest_relpath"],
            "manifest_path": str(path),
            "manifest_exists": exists,
            "expected_status": spec["expected_status"],
            "observed_status": status_value,
            "status_matches_expected": status_matches_expected,
            "locks": locks,
            "lock_violation_fields": lock_violation_fields,
            "component_ready": bool(exists and status_matches_expected and not lock_violation_fields),
        }

        if spec["step"] == "153":
            component["tensorboard_status"] = find_tensorboard_status(payload)
        if spec["step"] == "154":
            component["nvidia_smi_status"] = find_gpu_probe_status(payload)

        components.append(component)
    return components


def build_status_rows(components: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for c in components:
        rows.append({
            "step": c["step"],
            "name": c["name"],
            "manifest_exists": c["manifest_exists"],
            "observed_status": c.get("observed_status"),
            "expected_status": c.get("expected_status"),
            "status_matches_expected": c.get("status_matches_expected"),
            "component_ready": c.get("component_ready"),
            "lock_violations": ",".join(c.get("lock_violation_fields", [])),
            "manifest_relpath": c.get("manifest_relpath"),
        })
    return rows


def render_status_page(index: Dict[str, Any]) -> str:
    components = index["components"]
    lines = [
        "# H200 Observability Dashboard Status Page — Step 155",
        "",
        f"- generated_at_utc: `{index['generated_at_utc']}`",
        f"- index_status: `{index['index_status']}`",
        f"- dashboard_mode: `{index['dashboard_mode']}`",
        f"- control_policy: `{index['control_policy']}`",
        f"- observability_scaffold_ready: `{index['observability_scaffold_ready']}`",
        f"- warning_count: `{len(index['warnings'])}`",
        "",
        "## Global locks",
        "",
        "| Field | Value |",
        "|---|---:|",
    ]
    for key, value in index["locks"].items():
        lines.append(f"| `{key}` | `{str(value).lower()}` |")

    lines.extend([
        "",
        "## Component summary",
        "",
        "| Step | Component | Manifest | Status match | Ready | Lock violations |",
        "|---:|---|---:|---:|---:|---|",
    ])
    for c in components:
        lock_violations = ", ".join(c.get("lock_violation_fields", [])) or "none"
        lines.append(
            f"| {c['step']} | `{c['name']}` | `{c['manifest_exists']}` | "
            f"`{c['status_matches_expected']}` | `{c['component_ready']}` | `{lock_violations}` |"
        )

    lines.extend([
        "",
        "## Operator quick interpretation",
        "",
        "- This page is monitoring-only.",
        "- It does not release actual reward ablation.",
        "- It does not allow live mutation of reward weights, learning rate, batch size, or run configuration.",
        "- Any config change must be applied to a future run configuration, not to a live run.",
        "",
        "## Warnings",
        "",
    ])
    if index["warnings"]:
        for w in index["warnings"]:
            lines.append(f"- `{w['code']}`: {w['message']}")
    else:
        lines.append("- none")

    lines.extend([
        "",
        "## H200 viewing commands",
        "",
        "```bash",
        "# TensorBoard, if installed and event files are present",
        "tensorboard --logdir artifacts/observability --host 0.0.0.0 --port 6006",
        "",
        "# One-shot GPU probe, when on H200",
        "python 05_training/observability/h200_nvidia_smi_gpu_monitor_step154.py \\",
        "  --project-root /workspace/urbanbus_rl_project \\",
        "  --output-root artifacts/observability/h200_nvidia_smi_gpu_monitor_step154_h200_probe \\",
        "  --probe-once",
        "```",
        "",
    ])
    return "\n".join(lines)


def render_operator_quick_view(index: Dict[str, Any]) -> str:
    lines = [
        "# H200 Observability Operator Quick View — Step 155",
        "",
        "```text",
        f"index_status                 : {index['index_status']}",
        f"dashboard_mode               : {index['dashboard_mode']}",
        f"control_policy               : {index['control_policy']}",
        f"observability_scaffold_ready : {index['observability_scaffold_ready']}",
        f"hard_failures                : {index['hard_failures']}",
        f"warnings                     : {len(index['warnings'])}",
        f"actual_execution_allowed     : {index['locks']['actual_execution_allowed']}",
        f"actual_execution_released    : {index['locks']['actual_execution_released']}",
        f"train_allowed                : {index['locks']['train_allowed']}",
        f"live_mutation_allowed        : {index['locks']['live_mutation_allowed']}",
        "```",
        "",
        "## Components",
        "",
    ]
    for c in index["components"]:
        lines.extend([
            f"### Step {c['step']} — {c['name']}",
            "",
            "```text",
            f"manifest_exists         : {c['manifest_exists']}",
            f"observed_status         : {c.get('observed_status')}",
            f"status_matches_expected : {c.get('status_matches_expected')}",
            f"component_ready         : {c.get('component_ready')}",
            f"lock_violations         : {c.get('lock_violation_fields')}",
            "```",
            "",
        ])
    return "\n".join(lines)


def build_index(project_root: Path, output_root: Path) -> Dict[str, Any]:
    components = build_component_summary(project_root)
    warnings: List[Dict[str, str]] = []
    hard_failures: List[Dict[str, str]] = []

    required_steps = {spec["step"] for spec in REQUIRED_COMPONENTS}
    observed_steps = {str(c["step"]) for c in components}
    missing_declared_steps = sorted(required_steps - observed_steps)
    if missing_declared_steps:
        hard_failures.append({
            "code": "component_declaration_missing",
            "message": f"Required component declarations missing: {missing_declared_steps}",
        })

    for c in components:
        if not c["manifest_exists"]:
            warnings.append({
                "code": f"step_{c['step']}_manifest_missing",
                "message": f"{c['name']} manifest was not found at {c['manifest_relpath']}; this is warning-only for the index scaffold.",
            })
        elif not c["status_matches_expected"]:
            warnings.append({
                "code": f"step_{c['step']}_status_mismatch",
                "message": f"{c['name']} status is {c.get('observed_status')}, expected {c.get('expected_status')}.",
            })
        if c.get("lock_violation_fields"):
            hard_failures.append({
                "code": f"step_{c['step']}_lock_violation",
                "message": f"{c['name']} has lock violations: {c['lock_violation_fields']}",
            })

    locks = {key: False for key in LOCK_FIELDS}
    index = {
        "artifact_version": "h200_observability_dashboard_index_step155_v1",
        "generated_at_utc": utc_now(),
        "project_root": str(project_root),
        "output_root": str(output_root),
        "index_status": INDEX_STATUS,
        "dashboard_mode": DASHBOARD_MODE,
        "control_policy": CONTROL_POLICY,
        "observability_scaffold_ready": bool(not hard_failures),
        "monitoring_only": True,
        "release_step": False,
        "locks": locks,
        "actual_execution_allowed": False,
        "actual_execution_released": False,
        "train_allowed": False,
        "live_mutation_allowed": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "components": components,
        "required_component_steps": sorted(required_steps),
        "hard_failures": hard_failures,
        "warnings": warnings,
        "next_gate": "OBSERVABILITY_SCAFFOLD_CLOSED_WAITING_FOR_REAL_H200_STEP149_EXPECT_H200_RESULT",
    }
    return index


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step", "name", "manifest_exists", "observed_status", "expected_status",
        "status_matches_expected", "component_ready", "lock_violations", "manifest_relpath",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 155 H200 observability dashboard index/status page")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-root", default="artifacts/observability/h200_observability_dashboard_index_step155")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_root = (project_root / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    index = build_index(project_root=project_root, output_root=output_root)

    manifest_path = output_root / "h200_observability_dashboard_index_step155_manifest.json"
    index_path = output_root / "observability_dashboard_index_step155.json"
    status_page_path = output_root / "h200_observability_status_page_step155.md"
    quick_view_path = output_root / "h200_observability_operator_quick_view_step155.md"
    csv_path = output_root / "observability_component_status_step155.csv"

    index["output_files"] = {
        "manifest": str(manifest_path),
        "dashboard_index": str(index_path),
        "status_page": str(status_page_path),
        "operator_quick_view": str(quick_view_path),
        "component_status_csv": str(csv_path),
    }

    write_json(manifest_path, index)
    write_json(index_path, index)
    status_page_path.write_text(render_status_page(index), encoding="utf-8")
    quick_view_path.write_text(render_operator_quick_view(index), encoding="utf-8")
    write_csv(csv_path, build_status_rows(index["components"]))

    latest_path = project_root / "05_training" / "observability" / "h200_observability_dashboard_index_step155.latest.json"
    write_json(latest_path, {
        "artifact_version": "h200_observability_dashboard_index_step155_latest_pointer_v1",
        "generated_at_utc": index["generated_at_utc"],
        "manifest": str(manifest_path),
        "index_status": index["index_status"],
        "dashboard_mode": index["dashboard_mode"],
        "control_policy": index["control_policy"],
        "train_allowed": False,
        "live_mutation_allowed": False,
    })

    print("[OK] Step 155 H200 observability dashboard index/status page generated")
    print(f"[OK] index_status  : {index['index_status']}")
    print(f"[OK] dashboard_mode: {index['dashboard_mode']}")
    print(f"[OK] control_policy: {index['control_policy']}")
    print(f"[OK] ready         : {index['observability_scaffold_ready']}")
    print(f"[OK] hard_failures : {len(index['hard_failures'])}")
    print(f"[OK] warnings      : {len(index['warnings'])}")
    print(f"[OK] train_allowed : {index['train_allowed']}")
    print(f"[OK] manifest      : {manifest_path}")


if __name__ == "__main__":
    main()
