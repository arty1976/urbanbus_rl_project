from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


STEP_ID = 134
ARTIFACT_VERSION = "reward_ablation_command_runner_integration_step134_v1"

VALID_REWARD_IDS = ["R0", "R1", "R2", "R3", "R4", "R5"]
DEFAULT_SEEDS = [1, 2, 3]
VALID_CONDITIONS = ["A"]

FORBIDDEN_TRUE_KEYS = [
    "actual_executed",
    "actual_results",
    "reward_result_written",
    "winner_selected",
    "trainable_reward_promoted",
    "train_with_this_reward_allowed",
    "actual_training_allowed",
    "final_reward_design_claim_allowed",
    "best_reward_claim_allowed",
    "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def find_latest_step132_manifest(project_root: Path) -> Optional[Path]:
    rewards_dir = project_root / "05_training" / "rewards"
    pointer = rewards_dir / "reward_ablation_command_dry_run_executor_step132.latest.json"
    if pointer.exists():
        try:
            payload = load_json_any(pointer)
            manifest = payload.get("manifest_path") or payload.get("path")
            if manifest:
                p = Path(str(manifest))
                if not p.is_absolute():
                    p = project_root / p
                if p.exists():
                    return p
        except Exception:
            pass

    candidates = []
    artifact_root = project_root / "artifacts" / "rewards"
    if artifact_root.exists():
        for p in artifact_root.rglob("*step132*manifest*.json"):
            if p.is_file():
                candidates.append(p)

    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def extract_flag(payload: Dict[str, Any], key: str) -> bool:
    if bool(payload.get(key, False)):
        return True

    for container_key in ("decision", "non_claim_guards", "guards"):
        obj = payload.get(container_key)
        if isinstance(obj, dict) and bool(obj.get(key, False)):
            return True

    return False


def check_step132_manifest(path: Optional[Path]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "provided": bool(path),
        "path": str(path) if path else "",
        "exists": bool(path and path.exists()),
        "status": "not_found",
        "violations": [],
        "warnings": [],
        "payload": None,
    }

    if path is None:
        result["warnings"].append("step132_manifest_not_found")
        return result

    if not path.exists():
        result["warnings"].append("step132_manifest_missing")
        return result

    try:
        payload = load_json_any(path)
    except Exception as exc:
        result["status"] = "unreadable"
        result["violations"].append(f"step132_manifest_unreadable: {exc}")
        return result

    result["status"] = "read"
    result["payload"] = payload
    result["step"] = payload.get("step")
    result["artifact_version"] = payload.get("artifact_version")
    result["audit_status"] = payload.get("audit_status")

    for key in FORBIDDEN_TRUE_KEYS:
        if extract_flag(payload, key):
            result["violations"].append(f"forbidden_true_in_step132: {key}")

    return result


def parse_command_string(command: str) -> Optional[Dict[str, Any]]:
    text = str(command)

    reward_id = None
    seed = None
    condition = "A"

    m = re.search(r"--reward-id\s+([A-Za-z0-9_-]+)", text)
    if m:
        reward_id = m.group(1).upper()

    m = re.search(r"--seed\s+([0-9]+)", text)
    if m:
        seed = int(m.group(1))

    m = re.search(r"--condition\s+([A-Za-z0-9_-]+)", text)
    if m:
        condition = m.group(1).upper()

    if reward_id is None:
        # Fallback for compact tokens like A_R0_seed_001
        m = re.search(r"(R[0-9]+)", text, re.IGNORECASE)
        if m:
            reward_id = m.group(1).upper()

    if seed is None:
        m = re.search(r"seed[_-]?0*([0-9]+)", text, re.IGNORECASE)
        if m:
            seed = int(m.group(1))

    if reward_id is None or seed is None:
        return None

    return {
        "condition": condition,
        "reward_id": reward_id,
        "seed": int(seed),
        "source": "parsed_command_string",
        "raw_command": text,
    }


def normalize_planned_command(item: Any) -> Optional[Dict[str, Any]]:
    if isinstance(item, str):
        return parse_command_string(item)

    if not isinstance(item, dict):
        return None

    command_text = item.get("command") or item.get("cmd") or item.get("dry_run_command")
    if command_text:
        parsed = parse_command_string(str(command_text))
        if parsed:
            parsed["raw_item"] = item
            return parsed

    reward_id = (
        item.get("reward_id")
        or item.get("candidate_reward_id")
        or item.get("reward")
        or item.get("reward_candidate")
    )
    seed = item.get("seed") or item.get("training_seed")
    condition = item.get("condition") or item.get("condition_id") or "A"

    if reward_id is None or seed is None:
        return None

    return {
        "condition": str(condition).upper(),
        "reward_id": str(reward_id).upper(),
        "seed": int(seed),
        "source": "dict_fields",
        "raw_item": item,
    }


def collect_lists(obj: Any, key_hint: str = "command") -> List[List[Any]]:
    found: List[List[Any]] = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            lower = str(key).lower()
            if isinstance(value, list) and (
                "command" in lower
                or "planned" in lower
                or "candidate" in lower
                or "matrix" in lower
            ):
                found.append(value)
            if isinstance(value, (dict, list)):
                found.extend(collect_lists(value, key_hint=key_hint))
    elif isinstance(obj, list):
        for value in obj:
            if isinstance(value, (dict, list)):
                found.extend(collect_lists(value, key_hint=key_hint))

    return found


def extract_planned_commands_from_step132(payload: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    commands: List[Dict[str, Any]] = []

    candidate_lists = collect_lists(payload)
    seen = set()

    for lst in candidate_lists:
        for item in lst:
            normalized = normalize_planned_command(item)
            if not normalized:
                continue

            key = (
                normalized.get("condition", "A"),
                normalized.get("reward_id"),
                int(normalized.get("seed", 0)),
            )
            if key in seen:
                continue
            seen.add(key)
            commands.append(normalized)

    if commands:
        commands.sort(key=lambda x: (x["condition"], x["reward_id"], int(x["seed"])))
        return commands, warnings

    warnings.append("no_parseable_planned_commands_in_step132_manifest")
    return [], warnings


def reconstruct_expected_commands() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for reward_id in VALID_REWARD_IDS:
        for seed in DEFAULT_SEEDS:
            out.append({
                "condition": "A",
                "reward_id": reward_id,
                "seed": seed,
                "source": "reconstructed_expected_R0_to_R5_seed1_to_seed3",
            })
    return out


def validate_command_matrix(commands: List[Dict[str, Any]]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    normalized_keys = []
    for cmd in commands:
        condition = str(cmd.get("condition", "A")).upper()
        reward_id = str(cmd.get("reward_id", "")).upper()
        seed = int(cmd.get("seed", 0))
        normalized_keys.append((condition, reward_id, seed))

        if condition not in VALID_CONDITIONS:
            errors.append(f"invalid_condition: {condition}")
        if reward_id not in VALID_REWARD_IDS:
            errors.append(f"invalid_reward_id: {reward_id}")
        if seed <= 0:
            errors.append(f"invalid_seed: {seed}")

    unique_keys = sorted(set(normalized_keys))
    if len(unique_keys) != len(normalized_keys):
        errors.append("duplicate_condition_reward_seed_commands")

    expected = sorted(("A", reward_id, seed) for reward_id in VALID_REWARD_IDS for seed in DEFAULT_SEEDS)
    missing = [x for x in expected if x not in unique_keys]
    extra = [x for x in unique_keys if x not in expected]

    if missing:
        errors.append("missing_expected_commands: " + ",".join(f"{c}_{r}_seed{s}" for c, r, s in missing))

    if extra:
        warnings.append("extra_commands_present: " + ",".join(f"{c}_{r}_seed{s}" for c, r, s in extra))

    return {
        "errors": errors,
        "warnings": warnings,
        "expected_count": len(expected),
        "actual_count": len(unique_keys),
        "missing": missing,
        "extra": extra,
    }


def run_step133_guard_for_command(
    *,
    project_root: Path,
    python_executable: str,
    runner_path: Path,
    command: Dict[str, Any],
    step132_manifest: Optional[Path],
    output_base: Path,
) -> Dict[str, Any]:
    condition = str(command["condition"]).upper()
    reward_id = str(command["reward_id"]).upper()
    seed = int(command["seed"])
    candidate_key = f"{condition}_{reward_id}_seed_{seed:03d}"

    output_root = output_base / candidate_key
    args = [
        python_executable,
        str(runner_path),
        "--project-root",
        str(project_root),
        "--reward-id",
        reward_id,
        "--seed",
        str(seed),
        "--condition",
        condition,
        "--mode",
        "dry-run",
        "--output-root",
        safe_rel(output_root, project_root),
        "--candidate-label",
        "step134",
    ]

    if step132_manifest is not None and step132_manifest.exists():
        args.extend(["--step132-manifest", safe_rel(step132_manifest, project_root)])

    proc = subprocess.run(
        args,
        cwd=str(project_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    manifest_path = output_root / "actual_reward_ablation_runner_guard_step133_manifest.json"
    result: Dict[str, Any] = {
        "candidate_key": candidate_key,
        "condition": condition,
        "reward_id": reward_id,
        "seed": seed,
        "returncode": int(proc.returncode),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
        "manifest_path": str(manifest_path),
        "manifest_exists": manifest_path.exists(),
        "guard_passed": False,
        "violations": [],
    }

    if proc.returncode != 0:
        result["violations"].append("step133_runner_returned_nonzero")

    if not manifest_path.exists():
        result["violations"].append("step133_manifest_missing")
        return result

    try:
        manifest = load_json_any(manifest_path)
    except Exception as exc:
        result["violations"].append(f"step133_manifest_unreadable: {exc}")
        return result

    result["audit_status"] = manifest.get("audit_status")
    for key in FORBIDDEN_TRUE_KEYS:
        if bool(manifest.get(key, False)):
            result["violations"].append(f"forbidden_true_in_step133_manifest: {key}")

    result["guard_passed"] = (
        proc.returncode == 0
        and manifest.get("audit_status") == "PASS"
        and not result["violations"]
    )

    return result


def run_integration(project_root: Path, output_root: Path, step132_manifest: Optional[Path]) -> Dict[str, Any]:
    project_root = project_root.resolve()
    output_root = output_root.resolve()

    if step132_manifest is None:
        step132_manifest = find_latest_step132_manifest(project_root)

    upstream_check = check_step132_manifest(step132_manifest)
    blocking_reasons: List[str] = []
    warnings: List[str] = list(upstream_check.get("warnings", []))
    blocking_reasons.extend(upstream_check.get("violations", []))

    payload132 = upstream_check.get("payload") if isinstance(upstream_check.get("payload"), dict) else None
    if payload132:
        planned_commands, parse_warnings = extract_planned_commands_from_step132(payload132)
        warnings.extend(parse_warnings)
    else:
        planned_commands = []

    if not planned_commands:
        planned_commands = reconstruct_expected_commands()
        warnings.append("using_reconstructed_expected_command_matrix")

    matrix_check = validate_command_matrix(planned_commands)
    blocking_reasons.extend(matrix_check["errors"])
    warnings.extend(matrix_check["warnings"])

    runner_path = project_root / "05_training" / "rewards" / "run_actual_reward_ablation_candidate.py"
    if not runner_path.exists():
        blocking_reasons.append("step133_runner_missing: 05_training/rewards/run_actual_reward_ablation_candidate.py")

    command_results: List[Dict[str, Any]] = []
    if not blocking_reasons:
        step133_output_base = output_root / "step133_guard_outputs"
        for command in planned_commands:
            command_results.append(run_step133_guard_for_command(
                project_root=project_root,
                python_executable=sys.executable,
                runner_path=runner_path,
                command=command,
                step132_manifest=step132_manifest,
                output_base=step133_output_base,
            ))

        for result in command_results:
            if not result.get("guard_passed", False):
                blocking_reasons.append(f"step133_guard_failed: {result.get('candidate_key')}")

    audit_status = "PASS" if not blocking_reasons else "BLOCKED"

    summary = {
        "planned_command_count": len(planned_commands),
        "step133_invocation_count": len(command_results),
        "step133_guard_pass_count": sum(1 for r in command_results if r.get("guard_passed", False)),
        "step133_guard_fail_count": sum(1 for r in command_results if not r.get("guard_passed", False)),
    }

    manifest_path = output_root / "reward_ablation_command_runner_integration_step134_manifest.json"

    payload: Dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "step": STEP_ID,
        "created_at_utc": utc_now(),
        "project_root": str(project_root),
        "output_root": str(output_root),
        "step132_manifest": str(step132_manifest) if step132_manifest else "",
        "upstream_step132_check": {k: v for k, v in upstream_check.items() if k != "payload"},
        "audit_status": audit_status,
        "blocking_reasons": blocking_reasons,
        "warnings": warnings,
        "matrix_check": matrix_check,
        "summary": summary,
        "planned_commands": planned_commands,
        "command_results": command_results,
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
        "scope_note": (
            "Step 134 links Step 132 planned reward-ablation commands to the Step 133 guarded runner. "
            "All invocations are dry-run guard checks only. No actual ablation is executed."
        ),
    }

    dump_json(manifest_path, payload)
    latest_path = project_root / "05_training" / "rewards" / "reward_ablation_command_runner_integration_step134.latest.json"
    dump_json(latest_path, {
        "manifest_path": str(manifest_path),
        "audit_status": audit_status,
        "summary": summary,
        "actual_executed": False,
        "actual_results": False,
        "created_at_utc": payload["created_at_utc"],
    })

    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-root", default="artifacts/rewards/reward_ablation_command_runner_integration_step134")
    parser.add_argument("--step132-manifest", default="")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    output_root = project_root / args.output_root

    step132_manifest = Path(args.step132_manifest) if args.step132_manifest else None
    if step132_manifest is not None and not step132_manifest.is_absolute():
        step132_manifest = project_root / step132_manifest

    payload = run_integration(project_root, output_root, step132_manifest)

    print("[OK] Step 134 reward ablation command-runner integration completed")
    print(f"[OK] audit_status          : {payload['audit_status']}")
    print(f"[OK] planned_commands      : {payload['summary']['planned_command_count']}")
    print(f"[OK] step133_invocations   : {payload['summary']['step133_invocation_count']}")
    print(f"[OK] guard_pass_count      : {payload['summary']['step133_guard_pass_count']}")
    print(f"[OK] actual_executed       : {payload['actual_executed']}")
    print(f"[OK] actual_results        : {payload['actual_results']}")
    print(f"[OK] winner_selected       : {payload['winner_selected']}")
    print(f"[OK] training_allowed      : {payload['train_with_this_reward_allowed']}")
    print(f"[OK] paper_claim           : {payload['paper_level_claim_allowed']}")

    if payload["warnings"]:
        for warning in payload["warnings"]:
            print(f"[WARN] {warning}")

    if payload["audit_status"] != "PASS":
        print("[BLOCKED] Step 134 integration failed")
        for reason in payload["blocking_reasons"]:
            print(f"[BLOCKED] {reason}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
