from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


STEP_ID = 131
ARTIFACT_VERSION = "reward_ablation_execution_environment_preflight_step131_v1"

FORBIDDEN_TRUE_GUARDS = [
    "actual_results",
    "winner_selected",
    "trainable_reward_promoted",
    "train_with_this_reward_allowed",
    "actual_training_allowed",
    "final_reward_design_claim_allowed",
    "best_reward_claim_allowed",
    "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
]

STEP_REQUIREMENTS = [
    {"step": 111, "name": "final_reward_specification_draft"},
    {"step": 112, "name": "reward_candidate_protocol"},
    {"step": 113, "name": "reward_normalization_baseline_lock"},
    {"step": 114, "name": "hard_constraint_review"},
    {"step": 115, "name": "reward_ablation_matrix"},
    {"step": 116, "name": "trainable_reward_promotion_gate"},
    {"step": 117, "name": "reward_ablation_result_schema"},
    {"step": 118, "name": "reward_ablation_result_writer_guard"},
    {"step": 119, "name": "reward_ablation_runner_dry_run_plan"},
    {"step": 120, "name": "reward_ablation_execution_preflight"},
    {"step": 121, "name": "reward_ablation_execution_manifest"},
    {"step": 122, "name": "reward_ablation_sandbox_noop_runner_guard"},
    {"step": 123, "name": "reward_ablation_noop_result_ingestion_guard"},
    {"step": 124, "name": "reward_ablation_actual_result_schema_bridge"},
    {"step": 125, "name": "reward_ablation_selection_criteria_gate"},
    {"step": 126, "name": "reward_ablation_actual_result_ingestion_preflight"},
    {"step": 127, "name": "trainable_reward_promotion_decision_package"},
    {"step": 128, "name": "reward_pipeline_actual_ablation_wait_state"},
    {"step": 129, "name": "actual_reward_ablation_runbook"},
    {"step": 130, "name": "reward_pipeline_project_log_update"},
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_text_any(path: Path) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    return ""


def load_json_any(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except Exception:
            continue
    raise RuntimeError(f"failed_to_read_json: {path}")


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def discover_step_marker(project_root: Path, step: int) -> Dict[str, Any]:
    search_dirs = [
        project_root / "05_training" / "rewards",
        project_root / "artifacts" / "rewards",
    ]
    project_log = project_root / "project_log.md"
    token_a = f"step{step}"
    token_b = f"Step {step}"
    token_c = f"Step-{step}"

    matched_files: List[str] = []
    for search_dir in search_dirs:
        if search_dir.exists():
            for p in sorted(search_dir.rglob("*")):
                if p.is_file() and token_a.lower() in p.name.lower():
                    matched_files.append(str(p.relative_to(project_root)))

    log_has_step = False
    if project_log.exists():
        text = read_text_any(project_log)
        lowered = text.lower()
        log_has_step = token_b in text or token_c in text or token_a in lowered

    return {
        "step": step,
        "file_markers": matched_files,
        "project_log_marker": bool(log_has_step),
        "present": bool(matched_files or log_has_step),
    }


def check_step130_guard_file(project_root: Path) -> Dict[str, Any]:
    path = project_root / "05_training" / "rewards" / "reward_pipeline_project_log_update_step130.json"
    result: Dict[str, Any] = {
        "path": str(path.relative_to(project_root)),
        "exists": path.exists(),
        "violations": [],
        "guard_values": {},
    }

    if not path.exists():
        result["violations"].append("step130_guard_json_missing")
        return result

    try:
        payload = load_json_any(path)
    except Exception as exc:
        result["violations"].append(f"step130_guard_json_unreadable: {exc}")
        return result

    # Some older step outputs may use aliases. Missing keys default to False.
    alias_map = {
        "trainable_reward_promoted": ["promoted"],
        "train_with_this_reward_allowed": ["train_allowed"],
    }

    for key in FORBIDDEN_TRUE_GUARDS:
        raw_value = payload.get(key, False)
        if not raw_value:
            for alias in alias_map.get(key, []):
                raw_value = payload.get(alias, False)
                if raw_value:
                    break
        value = bool(raw_value)
        result["guard_values"][key] = value
        if value:
            result["violations"].append(f"forbidden_true_guard: {key}")

    return result


def check_python_imports() -> Dict[str, Any]:
    required = ["json", "pathlib", "argparse"]
    recommended = ["pandas", "numpy"]
    out: Dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "required_imports": {},
        "recommended_imports": {},
        "missing_required": [],
        "missing_recommended": [],
    }

    for mod in required:
        try:
            __import__(mod)
            out["required_imports"][mod] = True
        except Exception:
            out["required_imports"][mod] = False
            out["missing_required"].append(mod)

    for mod in recommended:
        try:
            __import__(mod)
            out["recommended_imports"][mod] = True
        except Exception:
            out["recommended_imports"][mod] = False
            out["missing_recommended"].append(mod)

    return out


def check_write_access(output_root: Path) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    probe = output_root / "_write_probe.txt"
    result = {"path": str(output_root), "write_ok": False, "error": None}
    try:
        probe.write_text("ok", encoding="utf-8")
        result["write_ok"] = probe.read_text(encoding="utf-8") == "ok"
        probe.unlink(missing_ok=True)
    except Exception as exc:
        result["error"] = str(exc)
    return result


def run_preflight(project_root: Path, output_root: Path) -> Dict[str, Any]:
    project_root = project_root.resolve()
    output_root = output_root.resolve()

    step_markers = [discover_step_marker(project_root, item["step"]) for item in STEP_REQUIREMENTS]
    missing_steps = [item for item in step_markers if not item["present"]]

    step130_guard = check_step130_guard_file(project_root)
    python_env = check_python_imports()
    write_access = check_write_access(output_root)

    blocking_reasons: List[str] = []

    if missing_steps:
        blocking_reasons.append(
            "missing_step_markers: " + ",".join(str(x["step"]) for x in missing_steps)
        )

    for violation in step130_guard["violations"]:
        blocking_reasons.append(violation)

    if python_env["missing_required"]:
        blocking_reasons.append(
            "missing_required_python_imports: " + ",".join(python_env["missing_required"])
        )

    if not write_access["write_ok"]:
        blocking_reasons.append("output_root_not_writable")

    audit_status = "PASS" if not blocking_reasons else "BLOCKED"
    actual_reward_ablation_execution_allowed = audit_status == "PASS"

    payload: Dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "step": STEP_ID,
        "created_at_utc": utc_now(),
        "project_root": str(project_root),
        "output_root": str(output_root),
        "audit_status": audit_status,
        "blocking_reasons": blocking_reasons,
        "step_markers": step_markers,
        "missing_steps": missing_steps,
        "step130_guard_check": step130_guard,
        "python_environment": python_env,
        "write_access": write_access,
        "decision": {
            "actual_reward_ablation_execution_allowed": actual_reward_ablation_execution_allowed,
            "reason": (
                "environment_preflight_passed"
                if actual_reward_ablation_execution_allowed
                else "environment_preflight_blocked"
            ),
        },
        "non_claim_guards": {
            "actual_results": False,
            "winner_selected": False,
            "trainable_reward_promoted": False,
            "train_with_this_reward_allowed": False,
            "actual_training_allowed": False,
            "final_reward_design_claim_allowed": False,
            "best_reward_claim_allowed": False,
            "paper_level_claim_allowed": False,
            "causal_performance_claim_allowed": False,
        },
        "scope_note": (
            "Step 131 only checks whether the environment is ready to execute actual reward ablation. "
            "It does not ingest actual results, select a winner, promote a reward, or allow MAPPO training."
        ),
    }

    manifest_path = output_root / "reward_ablation_execution_environment_preflight_step131_manifest.json"
    dump_json(manifest_path, payload)

    pointer_path = project_root / "05_training" / "rewards" / "reward_ablation_execution_environment_preflight_step131.latest.json"
    dump_json(pointer_path, {
        "manifest_path": str(manifest_path),
        "audit_status": audit_status,
        "actual_reward_ablation_execution_allowed": actual_reward_ablation_execution_allowed,
        "created_at_utc": payload["created_at_utc"],
    })

    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--output-root",
        default="artifacts/rewards/reward_ablation_execution_environment_preflight_step131",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root)
    output_root = project_root / args.output_root

    payload = run_preflight(project_root, output_root)

    print("[OK] Step 131 actual reward ablation execution environment preflight completed")
    print(f"[OK] audit_status   : {payload['audit_status']}")
    print(f"[OK] execution_ready: {payload['decision']['actual_reward_ablation_execution_allowed']}")
    print(f"[OK] output_root    : {output_root}")
    print("[OK] train_allowed  : False")
    print("[OK] winner_selected: False")
    print("[OK] paper_claim    : False")

    if payload["audit_status"] != "PASS":
        print("[BLOCKED] Step 131 preflight failed")
        for reason in payload["blocking_reasons"]:
            print(f"[BLOCKED] {reason}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
