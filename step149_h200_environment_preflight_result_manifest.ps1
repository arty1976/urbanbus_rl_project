$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\ryujo\urbanbus_rl_project"
if (-not (Test-Path $ProjectRoot)) {
    throw "[STOP] Project root not found: $ProjectRoot"
}
Set-Location $ProjectRoot

$RewardDir = ".\05_training\rewards"
New-Item -ItemType Directory -Force -Path $RewardDir | Out-Null

# ---------------------------------------------------------------------
# Step 149 main python
# ---------------------------------------------------------------------
$MainPy = @'
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


STEP_ID = 149
ARTIFACT_VERSION = "h200_environment_preflight_result_manifest_step149_v1"

STEP148_MANIFEST = "artifacts/rewards/h200_receive_side_preflight_gate_step148/h200_receive_side_preflight_gate_step148_manifest.json"

REQUIRED_FALSE_LOCKS = [
    "actual_execution_allowed",
    "actual_execution_released",
    "train_allowed",
    "actual_results",
    "winner_selected",
    "trainable_reward_promoted",
    "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
]

REQUIRED_ENV_CHECKS = [
    "python_version_recorded",
    "platform_recorded",
    "project_root_exists",
    "step148_gate_ready",
    "execution_locks_false",
    "torch_import_checked",
    "cuda_availability_checked",
    "gpu_count_checked",
    "disk_usage_recorded",
]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def dump_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_cmd(args: List[str], cwd: Optional[Path] = None, timeout: int = 20) -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "cmd": args,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "available": True,
        }
    except FileNotFoundError:
        return {
            "cmd": args,
            "returncode": None,
            "stdout": "",
            "stderr": "command_not_found",
            "available": False,
        }
    except subprocess.TimeoutExpired:
        return {
            "cmd": args,
            "returncode": None,
            "stdout": "",
            "stderr": "timeout",
            "available": True,
        }
    except Exception as exc:
        return {
            "cmd": args,
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "available": False,
        }


def collect_git(project_root: Path) -> Dict[str, Any]:
    return {
        "branch": run_cmd(["git", "branch", "--show-current"], cwd=project_root),
        "commit": run_cmd(["git", "rev-parse", "HEAD"], cwd=project_root),
        "status_short": run_cmd(["git", "status", "--short"], cwd=project_root),
        "remote_origin": run_cmd(["git", "remote", "get-url", "origin"], cwd=project_root),
    }


def collect_torch() -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "import_ok": False,
        "torch_version": "",
        "cuda_available": False,
        "cuda_version": "",
        "gpu_count": 0,
        "gpu_names": [],
        "error": "",
    }

    try:
        import torch  # type: ignore

        payload["import_ok"] = True
        payload["torch_version"] = str(getattr(torch, "__version__", ""))
        payload["cuda_available"] = bool(torch.cuda.is_available())
        payload["cuda_version"] = str(getattr(torch.version, "cuda", "") or "")
        payload["gpu_count"] = int(torch.cuda.device_count()) if payload["cuda_available"] else 0
        if payload["cuda_available"]:
            payload["gpu_names"] = [
                str(torch.cuda.get_device_name(i))
                for i in range(int(torch.cuda.device_count()))
            ]
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"

    return payload


def collect_disk(path: Path) -> Dict[str, Any]:
    try:
        usage = shutil.disk_usage(path)
        return {
            "path": str(path),
            "exists": path.exists(),
            "total_bytes": int(usage.total),
            "used_bytes": int(usage.used),
            "free_bytes": int(usage.free),
        }
    except Exception as exc:
        return {
            "path": str(path),
            "exists": path.exists(),
            "error": f"{type(exc).__name__}: {exc}",
        }


def validate_step148(project_root: Path) -> Tuple[Dict[str, Any], List[str], List[str]]:
    warnings: List[str] = []
    hard_failures: List[str] = []

    path = project_root / STEP148_MANIFEST
    payload: Dict[str, Any] = {}

    if not path.exists():
        hard_failures.append(f"step148_manifest_missing: {path}")
        return payload, warnings, hard_failures

    try:
        payload = load_json(path)
    except Exception as exc:
        hard_failures.append(f"step148_manifest_unreadable: {exc}")
        return payload, warnings, hard_failures

    if payload.get("artifact_version") != "h200_receive_side_preflight_gate_step148_v1":
        hard_failures.append("step148_artifact_version_mismatch")

    if int(payload.get("step_id", -1)) != 148:
        hard_failures.append("step148_step_id_mismatch")

    if payload.get("gate_status") != "READY_FOR_H200_RECEIVE_SIDE_PREFLIGHT_REVIEW":
        hard_failures.append(f"step148_gate_status_not_ready: {payload.get('gate_status')}")

    locks = payload.get("execution_locks", {})
    for key in REQUIRED_FALSE_LOCKS:
        if locks.get(key) is not False:
            hard_failures.append(f"step148_lock_not_false: {key}")

    if payload.get("hard_failures"):
        hard_failures.append(f"step148_has_hard_failures: {payload.get('hard_failures')}")

    return payload, warnings, hard_failures


def build_markdown(payload: Dict[str, Any]) -> str:
    torch_info = payload.get("torch", {})
    lines = [
        "# Step 149 H200 environment preflight result manifest",
        "",
        "This step records environment facts before any H200 actual execution.",
        "",
        "It does not run training and does not release actual reward ablation.",
        "",
        "## Status",
        "",
        f"- audit_status: `{payload.get('audit_status')}`",
        f"- environment_status: `{payload.get('environment_status')}`",
        f"- expect_h200: `{payload.get('expect_h200')}`",
        f"- project_root: `{payload.get('project_root')}`",
        f"- actual_execution_allowed: `{payload.get('execution_locks', {}).get('actual_execution_allowed')}`",
        f"- train_allowed: `{payload.get('execution_locks', {}).get('train_allowed')}`",
        "",
        "## Python / platform",
        "",
        f"- python_executable: `{payload.get('python', {}).get('executable')}`",
        f"- python_version: `{payload.get('python', {}).get('version')}`",
        f"- platform: `{payload.get('platform', {}).get('platform')}`",
        "",
        "## PyTorch / CUDA",
        "",
        f"- torch_import_ok: `{torch_info.get('import_ok')}`",
        f"- torch_version: `{torch_info.get('torch_version')}`",
        f"- cuda_available: `{torch_info.get('cuda_available')}`",
        f"- cuda_version: `{torch_info.get('cuda_version')}`",
        f"- gpu_count: `{torch_info.get('gpu_count')}`",
        f"- gpu_names: `{torch_info.get('gpu_names')}`",
        "",
        "## Guard",
        "",
        "- PASS in local mode only means the environment facts were recorded.",
        "- Use `--expect-h200` on H200 to require CUDA/GPU availability.",
        "- Actual execution remains locked until a later explicit release manifest.",
        "",
    ]
    return "\n".join(lines)


def build_manifest(
    project_root: Path,
    output_root: Path,
    expect_h200: bool,
    min_gpu_count: int,
) -> Dict[str, Any]:
    project_root = project_root.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    warnings: List[str] = []
    hard_failures: List[str] = []

    if not project_root.exists():
        hard_failures.append(f"project_root_not_found: {project_root}")

    step148_payload, step148_warnings, step148_failures = validate_step148(project_root)
    warnings.extend(step148_warnings)
    hard_failures.extend(step148_failures)

    torch_info = collect_torch()
    nvidia_smi = run_cmd(["nvidia-smi"], cwd=project_root, timeout=20)
    git_info = collect_git(project_root)
    disk_info = {
        "project_root": collect_disk(project_root),
        "artifacts_root": collect_disk(project_root / "artifacts"),
        "rewards_root": collect_disk(project_root / "05_training" / "rewards"),
    }

    if expect_h200:
        if not torch_info.get("import_ok"):
            hard_failures.append("expect_h200_requires_torch_import_ok")
        if not torch_info.get("cuda_available"):
            hard_failures.append("expect_h200_requires_cuda_available")
        if int(torch_info.get("gpu_count", 0)) < int(min_gpu_count):
            hard_failures.append(
                f"expect_h200_requires_gpu_count_at_least_{min_gpu_count}"
            )
        if not nvidia_smi.get("available") or nvidia_smi.get("returncode") != 0:
            hard_failures.append("expect_h200_requires_nvidia_smi_available")
    else:
        if not torch_info.get("import_ok"):
            warnings.append("torch_import_failed_local_record_only")
        if not torch_info.get("cuda_available"):
            warnings.append("cuda_not_available_local_record_only")
        if int(torch_info.get("gpu_count", 0)) == 0:
            warnings.append("gpu_count_zero_local_record_only")

    status_short = git_info.get("status_short", {}).get("stdout", "")
    if status_short:
        warnings.append("git_worktree_not_clean_at_step149_generation_time")

    execution_locks = {
        "actual_execution_allowed": False,
        "actual_execution_released": False,
        "train_allowed": False,
        "actual_results": False,
        "winner_selected": False,
        "trainable_reward_promoted": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
    }

    audit_status = "PASS" if not hard_failures else "BLOCKED"
    if audit_status == "PASS" and expect_h200:
        environment_status = "H200_ENVIRONMENT_PREFLIGHT_RECORDED_READY_ACTUAL_STILL_LOCKED"
    elif audit_status == "PASS":
        environment_status = "LOCAL_ENVIRONMENT_PREFLIGHT_RECORDED_H200_NOT_ASSERTED_ACTUAL_STILL_LOCKED"
    else:
        environment_status = "H200_ENVIRONMENT_PREFLIGHT_BLOCKED"

    manifest_path = output_root / "h200_environment_preflight_result_manifest_step149.json"
    md_path = output_root / "h200_environment_preflight_result_manifest_step149.md"

    payload: Dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "step_id": STEP_ID,
        "created_at_utc": now_utc(),
        "project_root": str(project_root),
        "output_root": str(output_root),
        "expect_h200": bool(expect_h200),
        "min_gpu_count": int(min_gpu_count),
        "audit_status": audit_status,
        "environment_status": environment_status,
        "hard_failures": hard_failures,
        "warnings": warnings,
        "required_env_checks": REQUIRED_ENV_CHECKS,
        "step148_manifest_path": str(project_root / STEP148_MANIFEST),
        "step148_summary": {
            "gate_status": step148_payload.get("gate_status"),
            "h200_project_root": step148_payload.get("h200_project_root"),
            "required_source_file_count": step148_payload.get("required_source_file_count"),
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version.replace("\n", " "),
            "version_info": list(sys.version_info[:5]),
        },
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "system": platform.system(),
            "release": platform.release(),
            "python_implementation": platform.python_implementation(),
        },
        "environment_variables_subset": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "PATH_length": len(os.environ.get("PATH", "")),
        },
        "torch": torch_info,
        "nvidia_smi": nvidia_smi,
        "git": git_info,
        "disk": disk_info,
        "execution_locks": execution_locks,
        "operator_interpretation": {
            "local_pass_meaning": (
                "Local PASS records environment facts only. It does not prove H200 GPU readiness."
            ),
            "h200_pass_meaning": (
                "H200 PASS with --expect-h200 means required GPU/CUDA visibility was observed, "
                "but actual training remains locked."
            ),
            "not_allowed": [
                "actual_reward_ablation_execution",
                "winner_selection",
                "trainable_reward_promotion",
                "paper_level_claim",
                "causal_performance_claim",
            ],
        },
        "next_step_recommendation": {
            "step": 150,
            "title": "actual reward ablation operator release checklist",
            "meaning": (
                "Prepare the final human-readable checklist before creating any explicit release request."
            ),
        },
        "manifest_path": str(manifest_path),
        "markdown_path": str(md_path),
    }

    dump_json(manifest_path, payload)
    dump_text(md_path, build_markdown(payload))
    dump_text(
        project_root / "05_training" / "rewards" / "h200_environment_preflight_result_manifest_step149.md",
        build_markdown(payload),
    )

    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--output-root",
        default="artifacts/rewards/h200_environment_preflight_result_manifest_step149",
    )
    parser.add_argument("--expect-h200", action="store_true")
    parser.add_argument("--min-gpu-count", type=int, default=1)
    args = parser.parse_args()

    payload = build_manifest(
        project_root=Path(args.project_root),
        output_root=Path(args.output_root),
        expect_h200=bool(args.expect_h200),
        min_gpu_count=int(args.min_gpu_count),
    )

    print("[OK] Step 149 H200 environment preflight result manifest completed")
    print(f"[OK] audit_status        : {payload['audit_status']}")
    print(f"[OK] environment_status  : {payload['environment_status']}")
    print(f"[OK] expect_h200         : {payload['expect_h200']}")
    print(f"[OK] torch_import_ok     : {payload['torch']['import_ok']}")
    print(f"[OK] cuda_available      : {payload['torch']['cuda_available']}")
    print(f"[OK] gpu_count           : {payload['torch']['gpu_count']}")
    print(f"[OK] hard_failures       : {len(payload['hard_failures'])}")
    print(f"[OK] warnings            : {len(payload['warnings'])}")
    print(f"[OK] actual_execution_allowed : {payload['execution_locks']['actual_execution_allowed']}")
    print(f"[OK] train_allowed            : {payload['execution_locks']['train_allowed']}")
    print(f"[OK] manifest            : {payload['manifest_path']}")

    if payload["warnings"]:
        for warning in payload["warnings"]:
            print(f"[WARN] {warning}")

    if payload["hard_failures"]:
        for failure in payload["hard_failures"]:
            print(f"[BLOCKED] {failure}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
'@

$MainPy | Set-Content -Path "$RewardDir\h200_environment_preflight_result_manifest_step149.py" -Encoding UTF8

# ---------------------------------------------------------------------
# Step 149 validator
# ---------------------------------------------------------------------
$ValidatePy = @'
from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_FALSE_LOCKS = [
    "actual_execution_allowed",
    "actual_execution_released",
    "train_allowed",
    "actual_results",
    "winner_selected",
    "trainable_reward_promoted",
    "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def validate_manifest(path: Path, allow_local: bool) -> None:
    payload = load_json(path)

    if payload.get("artifact_version") != "h200_environment_preflight_result_manifest_step149_v1":
        raise RuntimeError("artifact_version mismatch")

    if int(payload.get("step_id", -1)) != 149:
        raise RuntimeError("step_id must be 149")

    if payload.get("audit_status") != "PASS":
        raise RuntimeError(f"audit_status must be PASS: {payload.get('audit_status')}")

    status = payload.get("environment_status")
    valid_status = {
        "H200_ENVIRONMENT_PREFLIGHT_RECORDED_READY_ACTUAL_STILL_LOCKED",
        "LOCAL_ENVIRONMENT_PREFLIGHT_RECORDED_H200_NOT_ASSERTED_ACTUAL_STILL_LOCKED",
    }
    if status not in valid_status:
        raise RuntimeError(f"invalid environment_status: {status}")

    if not allow_local and status != "H200_ENVIRONMENT_PREFLIGHT_RECORDED_READY_ACTUAL_STILL_LOCKED":
        raise RuntimeError("H200-ready status required but local status was recorded")

    if payload.get("hard_failures"):
        raise RuntimeError(f"hard_failures found: {payload.get('hard_failures')}")

    locks = payload.get("execution_locks", {})
    for key in REQUIRED_FALSE_LOCKS:
        if locks.get(key) is not False:
            raise RuntimeError(f"execution lock must remain false: {key}")

    if not payload.get("python", {}).get("version"):
        raise RuntimeError("python version missing")

    if "torch" not in payload:
        raise RuntimeError("torch section missing")

    if "disk" not in payload:
        raise RuntimeError("disk section missing")

    print("[OK] Step 149 manifest validation PASS")
    print(f"[OK] manifest: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--allow-local", action="store_true")
    args = parser.parse_args()
    validate_manifest(Path(args.manifest), allow_local=bool(args.allow_local))


if __name__ == "__main__":
    main()
'@

$ValidatePy | Set-Content -Path "$RewardDir\validate_h200_environment_preflight_result_manifest_step149.py" -Encoding UTF8

# ---------------------------------------------------------------------
# Step 149 tests
# ---------------------------------------------------------------------
$TestPy = @'
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import h200_environment_preflight_result_manifest_step149 as step149


def make_step148_manifest(root: Path) -> None:
    path = root / step149.STEP148_MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_version": "h200_receive_side_preflight_gate_step148_v1",
        "step_id": 148,
        "gate_status": "READY_FOR_H200_RECEIVE_SIDE_PREFLIGHT_REVIEW",
        "hard_failures": [],
        "h200_project_root": "/workspace/urbanbus_rl_project",
        "required_source_file_count": 23,
        "execution_locks": {
            "actual_execution_allowed": False,
            "actual_execution_released": False,
            "train_allowed": False,
            "actual_results": False,
            "winner_selected": False,
            "trainable_reward_promoted": False,
            "paper_level_claim_allowed": False,
            "causal_performance_claim_allowed": False,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_local_manifest_passes_without_h200_assertion() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        make_step148_manifest(root)
        payload = step149.build_manifest(
            project_root=root,
            output_root=root / "artifacts" / "step149",
            expect_h200=False,
            min_gpu_count=1,
        )

        assert payload["audit_status"] == "PASS"
        assert payload["environment_status"] == "LOCAL_ENVIRONMENT_PREFLIGHT_RECORDED_H200_NOT_ASSERTED_ACTUAL_STILL_LOCKED"
        assert payload["execution_locks"]["actual_execution_allowed"] is False
        assert Path(payload["manifest_path"]).exists()


def test_missing_step148_blocks() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        payload = step149.build_manifest(
            project_root=root,
            output_root=root / "artifacts" / "step149",
            expect_h200=False,
            min_gpu_count=1,
        )

        assert payload["audit_status"] == "BLOCKED"
        assert any("step148_manifest_missing" in x for x in payload["hard_failures"])


def test_bad_step148_lock_blocks() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        make_step148_manifest(root)
        path = root / step149.STEP148_MANIFEST
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["execution_locks"]["train_allowed"] = True
        path.write_text(json.dumps(payload), encoding="utf-8")

        out = step149.build_manifest(
            project_root=root,
            output_root=root / "artifacts" / "step149",
            expect_h200=False,
            min_gpu_count=1,
        )

        assert out["audit_status"] == "BLOCKED"
        assert any("step148_lock_not_false" in x for x in out["hard_failures"])


def main() -> None:
    test_local_manifest_passes_without_h200_assertion()
    test_missing_step148_blocks()
    test_bad_step148_lock_blocks()
    print("[OK] Step 149 H200 environment preflight result manifest self-test PASS")


if __name__ == "__main__":
    main()
'@

$TestPy | Set-Content -Path "$RewardDir\test_h200_environment_preflight_result_manifest_step149.py" -Encoding UTF8

# ---------------------------------------------------------------------
# Step 149 markdown
# ---------------------------------------------------------------------
$Md = @'
# Step 149 — H200 environment preflight result manifest

## Purpose

Step 149 records environment facts before any H200 actual reward ablation execution.

This step does not run MAPPO training.  
This step does not release actual execution.  
This step only records Python, PyTorch, CUDA, GPU, disk, Git, and Step 148 gate facts.

## Modes

### Local/default mode

Local mode records the current environment and allows CUDA/GPU absence.

Expected status:

```text
LOCAL_ENVIRONMENT_PREFLIGHT_RECORDED_H200_NOT_ASSERTED_ACTUAL_STILL_LOCKED
```

### H200 assertion mode

On the H200 server, run with:

```bash
python 05_training/rewards/h200_environment_preflight_result_manifest_step149.py \
  --project-root . \
  --output-root artifacts/rewards/h200_environment_preflight_result_manifest_step149 \
  --expect-h200 \
  --min-gpu-count 1
```

Expected status:

```text
H200_ENVIRONMENT_PREFLIGHT_RECORDED_READY_ACTUAL_STILL_LOCKED
```

## Execution locks

The following must remain false:

- actual_execution_allowed
- actual_execution_released
- train_allowed
- actual_results
- winner_selected
- trainable_reward_promoted
- paper_level_claim_allowed
- causal_performance_claim_allowed

## Next step

Recommended next step:

```text
Step 150 — actual reward ablation operator release checklist
```
'@

$Md | Set-Content -Path "$RewardDir\h200_environment_preflight_result_manifest_step149.md" -Encoding UTF8

# ---------------------------------------------------------------------
# Run Step 149 in local/default mode
# ---------------------------------------------------------------------
if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

$OutputRoot = ".\artifacts\rewards\h200_environment_preflight_result_manifest_step149"
$Manifest = Join-Path $OutputRoot "h200_environment_preflight_result_manifest_step149.json"

& $py "$RewardDir\test_h200_environment_preflight_result_manifest_step149.py"
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 149 self-test failed"
}

& $py "$RewardDir\h200_environment_preflight_result_manifest_step149.py" `
  --project-root "." `
  --output-root $OutputRoot

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 149 environment preflight failed"
}

& $py "$RewardDir\validate_h200_environment_preflight_result_manifest_step149.py" `
  --manifest $Manifest `
  --allow-local

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 149 manifest validation failed"
}

Write-Host "[DONE] Step 149 H200 environment preflight result manifest complete."
