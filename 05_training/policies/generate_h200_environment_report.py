from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class EnvironmentReportError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run_command(
    command: List[str],
    cwd: Optional[Path] = None,
    timeout_sec: int = 30,
) -> Dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
        )
        return {
            "command": command,
            "returncode": int(completed.returncode),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "available": True,
        }
    except FileNotFoundError as exc:
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "available": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": f"timeout after {timeout_sec}s",
            "available": True,
            "timeout": True,
        }


def load_torch_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "torch_importable": False,
        "torch_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "cudnn_version": None,
        "gpu_count": 0,
        "gpu_names": [],
        "gpu_capabilities": [],
        "torch_error": None,
    }

    try:
        import torch  # type: ignore
    except Exception as exc:
        info["torch_error"] = repr(exc)
        return info

    info["torch_importable"] = True
    info["torch_version"] = getattr(torch, "__version__", None)
    info["cuda_available"] = bool(torch.cuda.is_available())
    info["cuda_version"] = getattr(torch.version, "cuda", None)

    try:
        info["cudnn_version"] = torch.backends.cudnn.version()
    except Exception:
        info["cudnn_version"] = None

    try:
        count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:
        count = 0

    info["gpu_count"] = count

    names = []
    capabilities = []

    for idx in range(count):
        try:
            names.append(torch.cuda.get_device_name(idx))
        except Exception as exc:
            names.append(f"<error: {exc}>")

        try:
            major, minor = torch.cuda.get_device_capability(idx)
            capabilities.append({"index": idx, "capability": f"{major}.{minor}"})
        except Exception as exc:
            capabilities.append({"index": idx, "capability": None, "error": repr(exc)})

    info["gpu_names"] = names
    info["gpu_capabilities"] = capabilities

    return info


def git_info(project_root: Path) -> Dict[str, Any]:
    rev = run_command(["git", "rev-parse", "HEAD"], cwd=project_root)
    short = run_command(["git", "rev-parse", "--short", "HEAD"], cwd=project_root)
    branch = run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=project_root)
    status = run_command(["git", "status", "--short"], cwd=project_root)

    status_short = status.get("stdout", "")
    dirty = bool(status_short.strip())

    return {
        "git_available": bool(rev.get("available")) and rev.get("returncode") == 0,
        "git_commit": rev.get("stdout", "").strip() if rev.get("returncode") == 0 else None,
        "git_commit_short": short.get("stdout", "").strip() if short.get("returncode") == 0 else None,
        "git_branch": branch.get("stdout", "").strip() if branch.get("returncode") == 0 else None,
        "git_status_short": status_short,
        "repo_dirty": dirty,
        "commands": {
            "rev_parse_head": rev,
            "rev_parse_short": short,
            "branch": branch,
            "status_short": status,
        },
    }


def nvidia_smi_info() -> Dict[str, Any]:
    query_cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free",
        "--format=csv,noheader",
    ]
    query = run_command(query_cmd, timeout_sec=30)
    raw = run_command(["nvidia-smi"], timeout_sec=30)

    gpus = []
    if query.get("available") and query.get("returncode") == 0:
        for line in query.get("stdout", "").splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                gpus.append({
                    "index": parts[0],
                    "name": parts[1],
                    "driver_version": parts[2],
                    "memory_total": parts[3],
                    "memory_used": parts[4],
                    "memory_free": parts[5],
                })

    return {
        "nvidia_smi_available": bool(query.get("available")),
        "nvidia_smi_query": query,
        "nvidia_smi_raw": raw,
        "gpu_rows": gpus,
    }


def selected_environment_variables() -> Dict[str, Optional[str]]:
    keys = [
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "CONDA_DEFAULT_ENV",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ]
    return {key: os.environ.get(key) for key in keys}


def build_report(project_root: Path) -> Dict[str, Any]:
    torch_info = load_torch_info()
    git = git_info(project_root)
    smi = nvidia_smi_info()

    report: Dict[str, Any] = {
        "artifact_version": "h200_environment_report_v1_step67",
        "created_at_utc": utc_now(),
        "project_root": str(project_root.resolve()),
        "hostname": socket.gethostname(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
        },
        "python": {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "python_implementation": platform.python_implementation(),
        },
        "torch": torch_info,
        "nvidia_smi": smi,
        "git": git,
        "environment_variables": selected_environment_variables(),
        "training_relevance": {
            "intended_for_h200_training": True,
            "cuda_required_for_actual_h200_training": True,
            "qwen_train_expected": False,
            "qwen_inference_expected": False,
            "qwen_trigger_rate_expected": 0.0,
            "reward_version_expected": "mappo_reward_v1",
            "energy_proxy_model_version_expected": "daegu_energy_proxy_v1",
            "k_dist_kwh_per_m_expected": 0.0012,
            "k_acc_kwh_per_event_expected": 0.1800,
            "k_idle_kwh_per_sec_expected": 0.0080,
        },
    }

    return report


def validate_report(
    report: Dict[str, Any],
    require_cuda: bool,
    require_clean_git: bool,
    require_h200_name: bool,
) -> None:
    if require_cuda:
        if not bool(report["torch"]["torch_importable"]):
            raise EnvironmentReportError("torch is not importable")
        if not bool(report["torch"]["cuda_available"]):
            raise EnvironmentReportError("CUDA is required but torch.cuda.is_available() is false")
        if int(report["torch"]["gpu_count"]) <= 0:
            raise EnvironmentReportError("CUDA is required but gpu_count <= 0")

    if require_h200_name:
        names = [str(x).lower() for x in report["torch"].get("gpu_names", [])]
        if not any("h200" in name for name in names):
            raise EnvironmentReportError(f"H200 GPU name required but not found: {report['torch'].get('gpu_names')}")

    if require_clean_git:
        if bool(report["git"].get("repo_dirty", True)):
            raise EnvironmentReportError("clean git tree required but repo_dirty=true")


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate H200 MAPPO training environment report."
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root for git metadata. Default: current directory.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path to write environment_report.json.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail if CUDA is not available. Use on actual H200 server.",
    )
    parser.add_argument(
        "--require-h200-name",
        action="store_true",
        help="Fail if no GPU name contains H200. Use on actual H200 server.",
    )
    parser.add_argument(
        "--require-clean-git",
        action="store_true",
        help="Fail if git status --short is non-empty.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    project_root = Path(args.project_root).resolve()
    output_path = Path(args.output_path)

    try:
        if not project_root.exists():
            raise EnvironmentReportError(f"project root not found: {project_root}")

        report = build_report(project_root=project_root)
        validate_report(
            report=report,
            require_cuda=bool(args.require_cuda),
            require_clean_git=bool(args.require_clean_git),
            require_h200_name=bool(args.require_h200_name),
        )
        report["report_status"] = "PASS"

        dump_json(output_path, report)

        print("[OK] Step 67 H200 environment report generated")
        print(f"[OK] output_path    : {output_path}")
        print(f"[OK] hostname       : {report['hostname']}")
        print(f"[OK] git_commit     : {report['git'].get('git_commit_short')}")
        print(f"[OK] repo_dirty     : {report['git'].get('repo_dirty')}")
        print(f"[OK] torch_version  : {report['torch'].get('torch_version')}")
        print(f"[OK] cuda_available : {report['torch'].get('cuda_available')}")
        print(f"[OK] gpu_count      : {report['torch'].get('gpu_count')}")
        return 0

    except EnvironmentReportError as exc:
        failure = {
            "artifact_version": "h200_environment_report_v1_step67",
            "created_at_utc": utc_now(),
            "project_root": str(project_root),
            "report_status": "BLOCKED",
            "error": str(exc),
        }
        try:
            dump_json(output_path, failure)
        except Exception:
            pass

        print(f"[BLOCKED] {exc}", file=sys.stderr)
        print(f"[BLOCKED] output_path: {output_path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
