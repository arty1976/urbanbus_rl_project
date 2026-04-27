from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class FolderContractError(RuntimeError):
    pass


PASS_STATUSES = {
    "PASS",
    "PASSED",
    "READY",
    "READY_TO_EXECUTE",
    "READY_FOR_MANUAL_EXECUTE",
    "ACTUAL_CHECKPOINT_READY",
}

EXPECTED_CONDITION_ID = "A"
EXPECTED_REWARD_VERSION = "mappo_reward_v1"
EXPECTED_ENERGY_PROXY_MODEL_VERSION = "daegu_energy_proxy_v1"
EXPECTED_K_DIST = 0.0012
EXPECTED_K_ACC = 0.1800
EXPECTED_K_IDLE = 0.0080


REQUIRED_RELATIVE_FILES = [
    "environment_report.json",
    "training_config.json",
    "training_command.txt",
    "git_commit.txt",
    "logs/train_stdout.log",
    "logs/train_stderr.log",
    "logs/train_metrics.jsonl",
    "checkpoints/best_mappo.pt",
    "checkpoints/checkpoint_manifest.json",
    "validation/validate_mappo_checkpoint_actual_report.json",
    "validation/h200_actual_checkpoint_preflight_report.json",
    "README_run_summary.md",
]


REQUIRED_RELATIVE_DIRS = [
    "logs",
    "checkpoints",
    "validation",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise FolderContractError(f"failed to read JSON: {path}")


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def norm(value: Any) -> str:
    return str(value or "").strip().upper()


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return default


def as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def collect_values_by_key(obj: Any, target_key: str) -> List[Any]:
    out: List[Any] = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == target_key:
                out.append(value)
            out.extend(collect_values_by_key(value, target_key))
    elif isinstance(obj, list):
        for item in obj:
            out.extend(collect_values_by_key(item, target_key))

    return out


def first_present(payloads: List[Dict[str, Any]], key: str, default: Any = None) -> Any:
    for payload in payloads:
        if key in payload:
            return payload.get(key)
        nested = collect_values_by_key(payload, key)
        if nested:
            return nested[0]
    return default


def first_status(payload: Dict[str, Any]) -> str:
    for key in (
        "report_status",
        "preflight_status",
        "checkpoint_status",
        "validation_status",
        "status",
        "runner_status",
    ):
        if key in payload:
            return norm(payload.get(key))

    for key in (
        "report_status",
        "preflight_status",
        "checkpoint_status",
        "validation_status",
        "status",
        "runner_status",
    ):
        nested = collect_values_by_key(payload, key)
        if nested:
            return norm(nested[0])

    return ""


def is_pass_like(payload: Dict[str, Any]) -> bool:
    status = first_status(payload)
    return status in PASS_STATUSES


def normalize_path_string(path_value: Any) -> str:
    return str(path_value or "").replace("\\", "/").strip()


def path_mentions_best_mappo(path_value: Any) -> bool:
    return normalize_path_string(path_value).endswith("checkpoints/best_mappo.pt") or normalize_path_string(path_value).endswith("best_mappo.pt")


def validate_required_paths(run_root: Path) -> Dict[str, Any]:
    missing_files = []
    empty_files = []
    missing_dirs = []

    for rel in REQUIRED_RELATIVE_DIRS:
        path = run_root / rel
        if not path.exists() or not path.is_dir():
            missing_dirs.append(rel)

    for rel in REQUIRED_RELATIVE_FILES:
        path = run_root / rel
        if not path.exists() or not path.is_file():
            missing_files.append(rel)
        elif path.stat().st_size <= 0:
            empty_files.append(rel)

    if missing_dirs or missing_files or empty_files:
        raise FolderContractError(
            f"folder contract path check failed. "
            f"missing_dirs={missing_dirs}, missing_files={missing_files}, empty_files={empty_files}"
        )

    return {
        "required_dirs": REQUIRED_RELATIVE_DIRS,
        "required_files": REQUIRED_RELATIVE_FILES,
        "missing_dirs": missing_dirs,
        "missing_files": missing_files,
        "empty_files": empty_files,
        "passed": True,
    }


def validate_training_metadata(
    environment_report: Dict[str, Any],
    training_config: Dict[str, Any],
    checkpoint_manifest: Dict[str, Any],
    actual_validation_report: Dict[str, Any],
    preflight_report: Dict[str, Any],
) -> Dict[str, Any]:
    payloads = [
        training_config,
        checkpoint_manifest,
        actual_validation_report,
        preflight_report,
        environment_report,
    ]

    condition_id = str(first_present(payloads, "condition_id", "")).strip().upper()
    qwen_train = as_bool(first_present(payloads, "qwen_train", None), default=False)
    qwen_inference = as_bool(first_present(payloads, "qwen_inference", None), default=False)
    qwen_trigger_rate = as_float(first_present(payloads, "qwen_trigger_rate", 0.0), default=0.0)
    reward_version = str(first_present(payloads, "reward_version", "")).strip()
    energy_proxy_model_version = str(first_present(payloads, "energy_proxy_model_version", "")).strip()
    k_dist = as_float(first_present(payloads, "k_dist_kwh_per_m", None), default=-1.0)
    k_acc = as_float(first_present(payloads, "k_acc_kwh_per_event", None), default=-1.0)
    k_idle = as_float(first_present(payloads, "k_idle_kwh_per_sec", None), default=-1.0)
    trained_model = as_bool(first_present(payloads, "trained_model", None), default=False)
    performance_claim_allowed = as_bool(
        first_present(payloads, "performance_claim_allowed", None),
        default=False,
    )

    errors = []

    if condition_id != EXPECTED_CONDITION_ID:
        errors.append(f"condition_id must be {EXPECTED_CONDITION_ID}, got={condition_id or '<missing>'}")

    if qwen_train:
        errors.append("qwen_train must be false")

    if qwen_inference:
        errors.append("qwen_inference must be false")

    if qwen_trigger_rate != 0.0:
        errors.append(f"qwen_trigger_rate must be 0.0, got={qwen_trigger_rate}")

    if reward_version != EXPECTED_REWARD_VERSION:
        errors.append(f"reward_version must be {EXPECTED_REWARD_VERSION}, got={reward_version or '<missing>'}")

    if energy_proxy_model_version != EXPECTED_ENERGY_PROXY_MODEL_VERSION:
        errors.append(
            f"energy_proxy_model_version must be {EXPECTED_ENERGY_PROXY_MODEL_VERSION}, "
            f"got={energy_proxy_model_version or '<missing>'}"
        )

    if abs(k_dist - EXPECTED_K_DIST) > 1e-12:
        errors.append(f"k_dist_kwh_per_m mismatch, got={k_dist}")

    if abs(k_acc - EXPECTED_K_ACC) > 1e-12:
        errors.append(f"k_acc_kwh_per_event mismatch, got={k_acc}")

    if abs(k_idle - EXPECTED_K_IDLE) > 1e-12:
        errors.append(f"k_idle_kwh_per_sec mismatch, got={k_idle}")

    if not trained_model:
        errors.append("trained_model must be true")

    if not performance_claim_allowed:
        errors.append("performance_claim_allowed must be true")

    if errors:
        raise FolderContractError("metadata contract failed: " + "; ".join(errors))

    return {
        "condition_id": condition_id,
        "qwen_train": qwen_train,
        "qwen_inference": qwen_inference,
        "qwen_trigger_rate": qwen_trigger_rate,
        "reward_version": reward_version,
        "energy_proxy_model_version": energy_proxy_model_version,
        "k_dist_kwh_per_m": k_dist,
        "k_acc_kwh_per_event": k_acc,
        "k_idle_kwh_per_sec": k_idle,
        "trained_model": trained_model,
        "performance_claim_allowed": performance_claim_allowed,
        "passed": True,
    }


def validate_validation_reports(
    run_root: Path,
    actual_validation_report: Dict[str, Any],
    preflight_report: Dict[str, Any],
) -> Dict[str, Any]:
    errors = []

    if not is_pass_like(actual_validation_report):
        errors.append(
            "validate_mappo_checkpoint_actual_report.json must be PASS-like, "
            f"got={first_status(actual_validation_report) or '<missing>'}"
        )

    if not is_pass_like(preflight_report):
        errors.append(
            "h200_actual_checkpoint_preflight_report.json must be PASS-like, "
            f"got={first_status(preflight_report) or '<missing>'}"
        )

    expected_best = run_root / "checkpoints" / "best_mappo.pt"
    expected_best_norm = normalize_path_string(expected_best.resolve())

    checkpoint_paths = []
    checkpoint_paths.extend(collect_values_by_key(actual_validation_report, "checkpoint_path"))
    checkpoint_paths.extend(collect_values_by_key(preflight_report, "checkpoint_path"))

    checked_paths = []
    for value in checkpoint_paths:
        value_norm = normalize_path_string(value)
        if not value_norm:
            continue
        checked_paths.append(value_norm)

        # Allow absolute path or relative path, but it must point to best_mappo.pt.
        if not path_mentions_best_mappo(value_norm):
            errors.append(f"validation report checkpoint_path does not reference best_mappo.pt: {value_norm}")

    if errors:
        raise FolderContractError("validation report contract failed: " + "; ".join(errors))

    return {
        "actual_validation_status": first_status(actual_validation_report),
        "preflight_status": first_status(preflight_report),
        "expected_best_checkpoint": str(expected_best),
        "expected_best_checkpoint_resolved": expected_best_norm,
        "observed_checkpoint_paths": checked_paths,
        "passed": True,
    }


def validate_text_files(run_root: Path) -> Dict[str, Any]:
    training_command = (run_root / "training_command.txt").read_text(encoding="utf-8-sig")
    git_commit = (run_root / "git_commit.txt").read_text(encoding="utf-8-sig")
    summary = (run_root / "README_run_summary.md").read_text(encoding="utf-8-sig")

    required_command_phrases = [
        "--condition-id A",
        "--qwen-train false",
        "--qwen-inference false",
        "--qwen-trigger-rate 0.0",
        "--reward-version mappo_reward_v1",
        "--energy-proxy-model-version daegu_energy_proxy_v1",
    ]

    missing_command = [p for p in required_command_phrases if p not in training_command]
    if missing_command:
        raise FolderContractError(f"training_command.txt missing required phrases: {missing_command}")

    if len(git_commit.strip()) < 7:
        raise FolderContractError("git_commit.txt must contain a commit hash")

    required_summary_phrases = [
        "best_mappo.pt",
        "trained_model",
        "performance_claim_allowed",
        "causal",
    ]

    missing_summary = [p for p in required_summary_phrases if p not in summary]
    if missing_summary:
        raise FolderContractError(f"README_run_summary.md missing required phrases: {missing_summary}")

    return {
        "training_command_required_phrases": required_command_phrases,
        "git_commit_length": len(git_commit.strip()),
        "summary_required_phrases": required_summary_phrases,
        "passed": True,
    }


def build_report(run_root: Path) -> Dict[str, Any]:
    path_summary = validate_required_paths(run_root)

    environment_report = load_json(run_root / "environment_report.json")
    training_config = load_json(run_root / "training_config.json")
    checkpoint_manifest = load_json(run_root / "checkpoints" / "checkpoint_manifest.json")
    actual_validation_report = load_json(run_root / "validation" / "validate_mappo_checkpoint_actual_report.json")
    preflight_report = load_json(run_root / "validation" / "h200_actual_checkpoint_preflight_report.json")

    metadata_summary = validate_training_metadata(
        environment_report=environment_report,
        training_config=training_config,
        checkpoint_manifest=checkpoint_manifest,
        actual_validation_report=actual_validation_report,
        preflight_report=preflight_report,
    )

    validation_summary = validate_validation_reports(
        run_root=run_root,
        actual_validation_report=actual_validation_report,
        preflight_report=preflight_report,
    )

    text_summary = validate_text_files(run_root)

    return {
        "artifact_version": "h200_actual_checkpoint_folder_contract_v1_step68",
        "created_at_utc": utc_now(),
        "folder_contract_status": "PASS",
        "run_root": str(run_root),
        "path_summary": path_summary,
        "metadata_summary": metadata_summary,
        "validation_summary": validation_summary,
        "text_summary": text_summary,
        "claim_boundary": {
            "ready_for_step62_arrival_workflow": True,
            "causal_performance_claim_allowed": False,
            "note": "Folder contract PASS does not prove causal performance superiority.",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate H200 actual MAPPO checkpoint folder contract."
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument(
        "--report-path",
        default="",
        help="Output report path. Default: <run-root>/validation/h200_actual_checkpoint_folder_contract_report.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root)
    report_path = Path(args.report_path) if args.report_path else (
        run_root / "validation" / "h200_actual_checkpoint_folder_contract_report.json"
    )

    try:
        if not run_root.exists():
            raise FolderContractError(f"run root not found: {run_root}")

        report = build_report(run_root)
        dump_json(report_path, report)

        print("[OK] Step 68 H200 actual checkpoint folder contract PASS")
        print(f"[OK] run_root    : {run_root}")
        print(f"[OK] report_path : {report_path}")
        print("[OK] ready_for_step62_arrival_workflow: true")
        return 0

    except FolderContractError as exc:
        failure = {
            "artifact_version": "h200_actual_checkpoint_folder_contract_v1_step68",
            "created_at_utc": utc_now(),
            "folder_contract_status": "BLOCKED",
            "run_root": str(run_root),
            "error": str(exc),
        }

        try:
            dump_json(report_path, failure)
        except Exception:
            pass

        print(f"[BLOCKED] {exc}", file=sys.stderr)
        print(f"[BLOCKED] report_path: {report_path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
