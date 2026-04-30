$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\ryujo\urbanbus_rl_project"
if (-not (Test-Path $ProjectRoot)) { throw "[STOP] Project root not found: $ProjectRoot" }
Set-Location $ProjectRoot

$RewardsDir = Join-Path $ProjectRoot "05_training\rewards"
if (-not (Test-Path $RewardsDir)) { New-Item -ItemType Directory -Force -Path $RewardsDir | Out-Null }

$PyMain = Join-Path $RewardsDir "h200_receive_side_transfer_runbook_step147.py"
$PyValidate = Join-Path $RewardsDir "validate_h200_receive_side_transfer_runbook_step147.py"
$PyTest = Join-Path $RewardsDir "test_h200_receive_side_transfer_runbook_step147.py"
$MdDoc = Join-Path $RewardsDir "h200_receive_side_transfer_runbook_step147.md"

$MainCode = @'
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

STEP = 147
ARTIFACT_VERSION = "h200_receive_side_transfer_runbook_step147_v1"
STEP146_POINTER = "05_training/rewards/h200_transfer_package_integrity_verifier_step146.latest.json"
STEP146_FALLBACK = "artifacts/rewards/h200_transfer_package_integrity_verifier_step146/h200_transfer_package_integrity_verifier_step146_manifest.json"
EXPECTED_RUN_COUNT = 72
EXPECTED_TRANSFER_FILE_COUNT = 10
CONDITIONS = ["A", "A90", "A80", "A70"]
REWARD_IDS = ["R0", "R1", "R2", "R3", "R4", "R5"]
SEEDS = [1, 2, 3]
VERIFY_METADATA_FILES = [
    "artifacts/rewards/h200_transfer_package_export_manifest_step145/h200_transfer_package_export_manifest_step145.json",
    "artifacts/rewards/h200_transfer_package_export_manifest_step145/h200_transfer_package_filelist_step145.txt",
]
FORBIDDEN_TRUE_KEYS = [
    "actual_execution_allowed", "actual_execution_released", "actual_executed",
    "actual_results", "reward_result_written", "winner_selected",
    "trainable_reward_promoted", "train_with_this_reward_allowed",
    "actual_training_allowed", "final_reward_design_claim_allowed",
    "best_reward_claim_allowed", "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except Exception:
            continue
    raise RuntimeError(f"failed_to_read_json: {path}")


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def dump_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(project_root: Path, raw: str) -> Path:
    p = Path(str(raw))
    return p if p.is_absolute() else project_root / p


def git_cmd(project_root: Path, args: List[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=str(project_root), stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""


def find_step146(project_root: Path) -> Tuple[Optional[Path], List[str]]:
    warnings: List[str] = []
    pointer = project_root / STEP146_POINTER
    if pointer.exists():
        try:
            ptr = load_json(pointer)
            raw = ptr.get("manifest_path")
            if raw:
                p = resolve(project_root, str(raw))
                if p.exists():
                    return p, warnings
                warnings.append(f"step146_pointer_manifest_missing: {p}")
            else:
                warnings.append("step146_pointer_has_no_manifest_path")
        except Exception as exc:
            warnings.append(f"step146_pointer_unreadable: {exc}")
    else:
        warnings.append(f"step146_pointer_missing: {STEP146_POINTER}")

    fallback = project_root / STEP146_FALLBACK
    if fallback.exists():
        warnings.append("using_step146_fallback_manifest")
        return fallback, warnings
    return None, warnings


def recursive_forbidden_true(obj: Any, prefix: str = "") -> List[str]:
    out: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            if k == "required_source_tokens":
                continue
            if k in FORBIDDEN_TRUE_KEYS and bool(v):
                out.append(f"forbidden_true: {path}")
            if isinstance(v, (dict, list)):
                out.extend(recursive_forbidden_true(v, path))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, (dict, list)):
                out.extend(recursive_forbidden_true(v, f"{prefix}[{i}]"))
    return out


def validate_step146(path: Optional[Path]) -> Tuple[Dict[str, Any], List[str]]:
    errors: List[str] = []
    if path is None or not path.exists():
        return {}, ["step146_manifest_missing"]
    try:
        p = load_json(path)
    except Exception as exc:
        return {}, [f"step146_manifest_unreadable: {exc}"]

    checks = [
        (p.get("step") == 146, f"step146_step_mismatch: {p.get('step')}"),
        (p.get("artifact_version") == "h200_transfer_package_integrity_verifier_step146_v1", "step146_artifact_version_mismatch"),
        (p.get("audit_status") == "PASS", f"step146_audit_not_pass: {p.get('audit_status')}"),
        (p.get("integrity_status") == "H200_TRANSFER_PACKAGE_INTEGRITY_VERIFIED_ACTUAL_STILL_LOCKED", f"step146_integrity_status_invalid: {p.get('integrity_status')}"),
        (int(p.get("planned_run_count", -1)) == EXPECTED_RUN_COUNT, "step146_planned_run_count_not_72"),
        (p.get("conditions") == CONDITIONS, "step146_conditions_mismatch"),
        (p.get("reward_ids") == REWARD_IDS, "step146_reward_ids_mismatch"),
        (p.get("seeds") == SEEDS, "step146_seeds_mismatch"),
        (int(p.get("verified_file_count", -1)) == EXPECTED_TRANSFER_FILE_COUNT, "step146_verified_file_count_not_10"),
    ]
    for ok, msg in checks:
        if not ok:
            errors.append(msg)

    entries = p.get("verified_entries", [])
    if not isinstance(entries, list) or len(entries) != EXPECTED_TRANSFER_FILE_COUNT:
        errors.append("step146_verified_entries_count_mismatch")
    else:
        for e in entries:
            rel = e.get("relative_path")
            if e.get("integrity_status") != "PASS":
                errors.append(f"{rel}: step146_integrity_not_pass")
            if not bool(e.get("exists", False)):
                errors.append(f"{rel}: step146_exists_false")
            if str(e.get("expected_sha256", "")).lower() != str(e.get("actual_sha256", "")).lower():
                errors.append(f"{rel}: step146_sha_mismatch")
            if int(e.get("expected_size_bytes", -1)) != int(e.get("actual_size_bytes", -2)):
                errors.append(f"{rel}: step146_size_mismatch")

    errors.extend([f"step146_manifest.{x}" for x in recursive_forbidden_true(p)])
    return p, errors


def command_plan(h200_root: str, repo_url: str, branch: str, commit: str) -> Dict[str, List[str]]:
    root = h200_root.rstrip("/")
    return {
        "local_before_transfer": [
            "git status --short",
            "git log --oneline origin/main..HEAD",
            "git push origin main",
        ],
        "h200_git_receive": [
            f"mkdir -p {root}",
            f"cd {root}",
            f"git clone {repo_url} .  # only if directory is empty",
            f"git fetch origin {branch}",
            f"git checkout {branch}",
            f"git pull --ff-only origin {branch}",
            f"git rev-parse HEAD  # expected {commit or '<PINNED_COMMIT>'}",
        ],
        "h200_metadata_receive_note": [
            "# Copy these Step 145 files as verification metadata only, not execution source-of-truth:",
            *[f"#   {p}" for p in VERIFY_METADATA_FILES],
        ],
        "h200_integrity_verify": [
            f"cd {root}",
            "python 05_training/rewards/h200_transfer_package_integrity_verifier_step146.py --project-root . --transfer-root . --output-root artifacts/rewards/h200_transfer_package_integrity_verifier_step146_h200",
        ],
        "blocked_execution_reminder": [
            "# Do not run actual reward ablation yet.",
            "# actual_execution_allowed must remain false until explicit operator release.",
            "# Do not select winner, do not promote reward, do not make paper-level claims.",
        ],
    }


def runbook_md(payload: Dict[str, Any]) -> str:
    plan = payload["receive_command_plan"]
    sections = [
        ("Local before transfer", "powershell", plan["local_before_transfer"]),
        ("H200 git receive", "bash", plan["h200_git_receive"]),
        ("Step 145 metadata receive note", "text", plan["h200_metadata_receive_note"]),
        ("H200 integrity verification", "bash", plan["h200_integrity_verify"]),
        ("Blocked execution reminder", "text", plan["blocked_execution_reminder"]),
    ]
    lines = [
        "# Step 147 H200 receive-side transfer runbook",
        "",
        "This runbook defines how the H200 side should receive and verify the transfer package.",
        "",
        "It does not execute MAPPO and does not release actual reward ablation.",
        "",
        "## Status",
        "",
        f"- audit_status: `{payload['audit_status']}`",
        f"- receive_plan_status: `{payload['receive_plan_status']}`",
        f"- transfer_file_count: `{payload['transfer_file_count']}`",
        f"- planned_run_count: `{payload['planned_run_count']}`",
        f"- pinned_git_commit: `{payload['pinned_git_commit']}`",
        f"- actual_execution_allowed: `{payload['actual_execution_allowed']}`",
        f"- actual_execution_released: `{payload['actual_execution_released']}`",
        f"- actual_results: `{payload['actual_results']}`",
        f"- winner_selected: `{payload['winner_selected']}`",
        f"- paper_level_claim_allowed: `{payload['paper_level_claim_allowed']}`",
        "",
    ]
    for title, lang, cmds in sections:
        lines.extend([f"## {title}", "", f"```{lang}", *cmds, "```", ""])
    lines.extend([
        "## Guard",
        "",
        "- Step 147 only defines receive-side commands and verification order.",
        "- Step 146 integrity PASS only proves file identity against Step 145 hashes.",
        "- Actual execution remains locked until a later explicit operator release manifest.",
        "",
    ])
    return "\n".join(lines)


def summary_md(payload: Dict[str, Any]) -> str:
    return "\n".join([
        "# Step 147 H200 receive-side transfer runbook",
        "",
        "Step 147 creates the H200-side receive and verification command plan.",
        "",
        f"- audit_status: `{payload['audit_status']}`",
        f"- receive_plan_status: `{payload['receive_plan_status']}`",
        f"- transfer_file_count: `{payload['transfer_file_count']}`",
        f"- planned_run_count: `{payload['planned_run_count']}`",
        f"- actual_execution_allowed: `{payload['actual_execution_allowed']}`",
        f"- paper_level_claim_allowed: `{payload['paper_level_claim_allowed']}`",
        "",
    ])


def generate(project_root: Path, output_root: Path, h200_root: str, repo_url: str, branch: str) -> Dict[str, Any]:
    project_root = project_root.resolve()
    output_root = output_root.resolve()
    warnings: List[str] = []
    blocking: List[str] = []

    step146_path, warn = find_step146(project_root)
    warnings.extend(warn)
    step146, errors = validate_step146(step146_path)
    blocking.extend(errors)

    commit = git_cmd(project_root, ["git", "rev-parse", "HEAD"])
    dirty = git_cmd(project_root, ["git", "status", "--short"])
    if dirty:
        warnings.append("git_status_not_clean_at_step147_generation_time")

    entries = step146.get("verified_entries", []) if step146 else []
    metadata = []
    for rel in VERIFY_METADATA_FILES:
        p = project_root / rel
        metadata.append({
            "relative_path": rel,
            "exists": p.exists(),
            "size_bytes": int(p.stat().st_size) if p.exists() else 0,
            "role": "verification_metadata_for_h200_integrity_check_not_source_of_truth",
        })
        if not p.exists():
            warnings.append(f"verification_metadata_missing_locally: {rel}")

    plan = command_plan(h200_root, repo_url, branch, commit)
    audit = "PASS" if not blocking else "BLOCKED"
    status = "H200_RECEIVE_SIDE_TRANSFER_RUNBOOK_READY_ACTUAL_STILL_LOCKED" if audit == "PASS" else "H200_RECEIVE_SIDE_TRANSFER_RUNBOOK_BLOCKED"

    cmd_path = output_root / "h200_receive_side_command_plan_step147.json"
    runbook_path = output_root / "h200_receive_side_transfer_runbook_step147.md"
    manifest_path = output_root / "h200_receive_side_transfer_runbook_step147_manifest.json"

    payload: Dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "step": STEP,
        "created_at_utc": utc_now(),
        "project_root": str(project_root),
        "output_root": str(output_root),
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "audit_status": audit,
        "receive_plan_status": status,
        "pinned_git_commit": commit,
        "git_status_clean": not bool(dirty),
        "git_status_short": dirty,
        "repo_url": repo_url,
        "branch": branch,
        "h200_project_root": h200_root,
        "planned_run_count": EXPECTED_RUN_COUNT,
        "conditions": CONDITIONS,
        "reward_ids": REWARD_IDS,
        "seeds": SEEDS,
        "step146_manifest_path": str(step146_path) if step146_path else "",
        "transfer_file_count": len(entries),
        "transfer_paths": [str(e.get("relative_path", "")) for e in entries],
        "verification_metadata_files": metadata,
        "receive_command_plan": plan,
        "blocking_reasons": blocking,
        "warnings": warnings,
        "actual_execution_allowed": False,
        "actual_execution_released": False,
        "actual_executed": False,
        "actual_results": False,
        "reward_result_written": False,
        "winner_selected": False,
        "trainable_reward_promoted": False,
        "train_with_this_reward_allowed": False,
        "actual_training_allowed": False,
        "final_reward_design_claim_allowed": False,
        "best_reward_claim_allowed": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "scope_note": "Step 147 creates H200 receive-side runbook and command plan only. It does not copy files to H200, execute MAPPO, or unlock actual execution.",
        "next_step_recommendation": "Step 148 should create a H200 receive preflight or explicit operator release request/review gate.",
        "command_plan_path": str(cmd_path),
        "runbook_path": str(runbook_path),
        "manifest_path": str(manifest_path),
    }

    dump_json(cmd_path, plan)
    dump_text(runbook_path, runbook_md(payload))
    dump_json(manifest_path, payload)
    dump_text(project_root / "05_training/rewards/h200_receive_side_transfer_runbook_step147.md", summary_md(payload))
    dump_json(project_root / "05_training/rewards/h200_receive_side_transfer_runbook_step147.latest.json", {
        "manifest_path": str(manifest_path),
        "command_plan_path": str(cmd_path),
        "runbook_path": str(runbook_path),
        "audit_status": audit,
        "receive_plan_status": status,
        "transfer_file_count": len(entries),
        "planned_run_count": EXPECTED_RUN_COUNT,
        "actual_execution_allowed": False,
        "actual_execution_released": False,
        "actual_results": False,
        "created_at_utc": payload["created_at_utc"],
    })
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--output-root", default="artifacts/rewards/h200_receive_side_transfer_runbook_step147")
    ap.add_argument("--h200-root", default="/workspace/urbanbus_rl_project")
    ap.add_argument("--repo-url", default="https://github.com/arty1976/urbanbus_rl_project")
    ap.add_argument("--branch", default="main")
    args = ap.parse_args()

    payload = generate(Path(args.project_root), Path(args.project_root) / args.output_root, args.h200_root, args.repo_url, args.branch)

    print("[OK] Step 147 H200 receive-side transfer runbook completed")
    print(f"[OK] audit_status              : {payload['audit_status']}")
    print(f"[OK] receive_plan_status       : {payload['receive_plan_status']}")
    print(f"[OK] transfer_file_count       : {payload['transfer_file_count']}")
    print(f"[OK] planned_run_count         : {payload['planned_run_count']}")
    print(f"[OK] conditions                : {payload['conditions']}")
    print(f"[OK] reward_ids                : {payload['reward_ids']}")
    print(f"[OK] seeds                     : {payload['seeds']}")
    print(f"[OK] pinned_git_commit         : {payload['pinned_git_commit']}")
    print(f"[OK] actual_execution_allowed  : {payload['actual_execution_allowed']}")
    print(f"[OK] actual_execution_released : {payload['actual_execution_released']}")
    print(f"[OK] actual_executed           : {payload['actual_executed']}")
    print(f"[OK] actual_results            : {payload['actual_results']}")
    print(f"[OK] winner_selected           : {payload['winner_selected']}")
    print(f"[OK] training_allowed          : {payload['train_with_this_reward_allowed']}")
    print(f"[OK] paper_claim               : {payload['paper_level_claim_allowed']}")
    print(f"[OK] runbook                   : {payload['runbook_path']}")
    for w in payload["warnings"]:
        print(f"[WARN] {w}")
    if payload["blocking_reasons"]:
        print("[BLOCKED] Step 147 receive-side runbook failed")
        for r in payload["blocking_reasons"]:
            print(f"[BLOCKED] {r}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
'@

$ValidateCode = @'
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

ARTIFACT_VERSION = "h200_receive_side_transfer_runbook_step147_v1"
CONDITIONS = ["A", "A90", "A80", "A70"]
REWARD_IDS = ["R0", "R1", "R2", "R3", "R4", "R5"]
SEEDS = [1, 2, 3]
FORBIDDEN_TRUE_KEYS = [
    "actual_execution_allowed", "actual_execution_released", "actual_executed",
    "actual_results", "reward_result_written", "winner_selected",
    "trainable_reward_promoted", "train_with_this_reward_allowed",
    "actual_training_allowed", "final_reward_design_claim_allowed",
    "best_reward_claim_allowed", "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
]


def load_json(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except Exception:
            continue
    raise RuntimeError(f"failed_to_read_json: {path}")


def recursive_forbidden_true(obj: Any, prefix: str = "") -> List[str]:
    errors: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            if k in FORBIDDEN_TRUE_KEYS and bool(v):
                errors.append(f"forbidden_true: {path}")
            if isinstance(v, (dict, list)):
                errors.extend(recursive_forbidden_true(v, path))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, (dict, list)):
                errors.extend(recursive_forbidden_true(v, f"{prefix}[{i}]"))
    return errors


def validate_payload(payload: Dict[str, Any], require_pass: bool = True) -> Dict[str, Any]:
    errors: List[str] = []
    if payload.get("artifact_version") != ARTIFACT_VERSION:
        errors.append("artifact_version_mismatch")
    if payload.get("step") != 147:
        errors.append("step_must_be_147")
    if payload.get("audit_status") not in ("PASS", "BLOCKED"):
        errors.append("invalid_audit_status")
    if require_pass and payload.get("audit_status") != "PASS":
        errors.append("audit_status_not_pass")
    if payload.get("receive_plan_status") not in {
        "H200_RECEIVE_SIDE_TRANSFER_RUNBOOK_READY_ACTUAL_STILL_LOCKED",
        "H200_RECEIVE_SIDE_TRANSFER_RUNBOOK_BLOCKED",
    }:
        errors.append("invalid_receive_plan_status")
    if require_pass and payload.get("receive_plan_status") != "H200_RECEIVE_SIDE_TRANSFER_RUNBOOK_READY_ACTUAL_STILL_LOCKED":
        errors.append("receive_plan_status_not_ready_locked")
    if int(payload.get("planned_run_count", -1)) != 72:
        errors.append("planned_run_count_must_be_72")
    if payload.get("conditions") != CONDITIONS:
        errors.append("conditions_mismatch")
    if payload.get("reward_ids") != REWARD_IDS:
        errors.append("reward_ids_mismatch")
    if payload.get("seeds") != SEEDS:
        errors.append("seeds_mismatch")
    if int(payload.get("transfer_file_count", -1)) != 10:
        errors.append("transfer_file_count_must_be_10")
    if not str(payload.get("pinned_git_commit", "")):
        errors.append("pinned_git_commit_missing")
    plan = payload.get("receive_command_plan", {})
    for key in ["local_before_transfer", "h200_git_receive", "h200_metadata_receive_note", "h200_integrity_verify", "blocked_execution_reminder"]:
        if key not in plan or not isinstance(plan.get(key), list) or not plan.get(key):
            errors.append(f"command_plan_missing_or_empty: {key}")
    metadata = payload.get("verification_metadata_files", [])
    if not isinstance(metadata, list) or len(metadata) != 2:
        errors.append("verification_metadata_files_must_have_2_entries")
    errors.extend(recursive_forbidden_true(payload))
    return {"validation_status": "PASS" if not errors else "FAIL", "errors": errors}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--allow-blocked", action="store_true")
    args = ap.parse_args()
    payload = load_json(Path(args.manifest))
    result = validate_payload(payload, require_pass=not args.allow_blocked)
    print(f"[OK] validation_status: {result['validation_status']}")
    if result["errors"]:
        for e in result["errors"]:
            print(f"[FAIL] {e}")
        raise SystemExit(2)
    print("[OK] Step 147 H200 receive-side transfer runbook validation PASS")


if __name__ == "__main__":
    main()
'@

$TestCode = @'
from __future__ import annotations

from validate_h200_receive_side_transfer_runbook_step147 import validate_payload


def base_payload() -> dict:
    return {
        "artifact_version": "h200_receive_side_transfer_runbook_step147_v1",
        "step": 147,
        "audit_status": "PASS",
        "receive_plan_status": "H200_RECEIVE_SIDE_TRANSFER_RUNBOOK_READY_ACTUAL_STILL_LOCKED",
        "pinned_git_commit": "a" * 40,
        "planned_run_count": 72,
        "conditions": ["A", "A90", "A80", "A70"],
        "reward_ids": ["R0", "R1", "R2", "R3", "R4", "R5"],
        "seeds": [1, 2, 3],
        "transfer_file_count": 10,
        "receive_command_plan": {
            "local_before_transfer": ["git status --short"],
            "h200_git_receive": ["git pull --ff-only origin main"],
            "h200_metadata_receive_note": ["# copy metadata"],
            "h200_integrity_verify": ["python 05_training/rewards/h200_transfer_package_integrity_verifier_step146.py --project-root . --transfer-root ."],
            "blocked_execution_reminder": ["# do not run actual execution"],
        },
        "verification_metadata_files": [
            {"relative_path": "artifacts/rewards/h200_transfer_package_export_manifest_step145/h200_transfer_package_export_manifest_step145.json"},
            {"relative_path": "artifacts/rewards/h200_transfer_package_export_manifest_step145/h200_transfer_package_filelist_step145.txt"},
        ],
        "actual_execution_allowed": False,
        "actual_execution_released": False,
        "actual_executed": False,
        "actual_results": False,
        "reward_result_written": False,
        "winner_selected": False,
        "trainable_reward_promoted": False,
        "train_with_this_reward_allowed": False,
        "actual_training_allowed": False,
        "final_reward_design_claim_allowed": False,
        "best_reward_claim_allowed": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
    }


def test_pass_payload() -> None:
    assert validate_payload(base_payload(), True)["validation_status"] == "PASS"


def test_actual_execution_true_fails() -> None:
    p = base_payload(); p["actual_execution_allowed"] = True
    r = validate_payload(p, True)
    assert r["validation_status"] == "FAIL"
    assert any("actual_execution_allowed" in e for e in r["errors"])


def test_missing_command_plan_fails() -> None:
    p = base_payload(); p["receive_command_plan"].pop("h200_integrity_verify")
    r = validate_payload(p, True)
    assert r["validation_status"] == "FAIL"
    assert any("command_plan_missing_or_empty" in e for e in r["errors"])


def test_wrong_transfer_count_fails() -> None:
    p = base_payload(); p["transfer_file_count"] = 9
    r = validate_payload(p, True)
    assert r["validation_status"] == "FAIL"
    assert any("transfer_file_count" in e for e in r["errors"])


def test_blocked_payload_can_pass_with_allow_blocked() -> None:
    p = base_payload(); p["audit_status"] = "BLOCKED"; p["receive_plan_status"] = "H200_RECEIVE_SIDE_TRANSFER_RUNBOOK_BLOCKED"
    assert validate_payload(p, False)["validation_status"] == "PASS"


def main() -> None:
    test_pass_payload(); test_actual_execution_true_fails(); test_missing_command_plan_fails(); test_wrong_transfer_count_fails(); test_blocked_payload_can_pass_with_allow_blocked()
    print("[OK] Step 147 H200 receive-side transfer runbook self-test PASS")


if __name__ == "__main__":
    main()
'@

$DocText = @'
# Step 147 H200 receive-side transfer runbook

Step 147 creates the H200-side receive and verification command plan.

It defines:
- local pre-transfer commands
- H200 git receive commands
- Step 145 metadata receive note
- H200 Step 146 integrity verification command
- blocked execution reminder

This step does not copy files to H200.
This step does not execute H200 jobs.
This step does not release actual execution.
This step does not select a reward winner.
This step does not allow paper-level claims.
'@

Set-Content -Path $PyMain -Value $MainCode -Encoding UTF8
Set-Content -Path $PyValidate -Value $ValidateCode -Encoding UTF8
Set-Content -Path $PyTest -Value $TestCode -Encoding UTF8
Set-Content -Path $MdDoc -Value $DocText -Encoding UTF8

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

$OutputRoot = "artifacts/rewards/h200_receive_side_transfer_runbook_step147"
$Manifest = Join-Path $ProjectRoot "$OutputRoot\h200_receive_side_transfer_runbook_step147_manifest.json"

& $py $PyMain `
  --project-root "." `
  --output-root $OutputRoot `
  --h200-root "/workspace/urbanbus_rl_project" `
  --repo-url "https://github.com/arty1976/urbanbus_rl_project" `
  --branch "main"

if ($LASTEXITCODE -ne 0) { throw "[FAIL] Step 147 H200 receive-side transfer runbook failed" }

& $py $PyValidate --manifest $Manifest
if ($LASTEXITCODE -ne 0) { throw "[FAIL] Step 147 receive-side runbook validation failed" }

& $py $PyTest
if ($LASTEXITCODE -ne 0) { throw "[FAIL] Step 147 self-test failed" }

Write-Host "[DONE] Step 147 H200 receive-side transfer runbook complete."

