#!/usr/bin/env python3
"""R18-R18B external-MPS recovery binding and fresh factorized one-shot issuer.

This stage never trains.  It proves that the exact project interpreter can use
native MPS only from the approved external execution context, binds that fact
to three fresh-process probes, then emits one new and distinct authorization.
The previously attempted R18-R18 authorization is evidence only and is never
reused.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
sys.path.insert(0, str(ROOT))

import run_h4m_ae_ls3_bt8_r18_r17_factorized_actor_authorization as R17  # noqa: E402


STAGE = "H4M-AE-R9.8-LS3-BT8-R18-R18B"
AUTHORIZATION = "R18R18B_FACTORIZED_ACTOR_EXTERNAL_MPS_BOUNDED_TRAINING_ONE_SHOT_ONLY"
REVISION = "R18-R18B-FACTORIZED-ACTOR-EXTERNAL-MPS-ONE-SHOT-V1"
PASS_GATE = (
    "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_R18B_"
    "EXTERNAL_MPS_RECOVERY_AND_FRESH_FACTORIZED_ONE_SHOT_AUTHORIZATION_COMPLETE"
)
CLASSIFICATION = "A_EXTERNAL_MPS_CONTEXT_BOUND_AND_FRESH_FACTORIZED_BOUNDED_EXECUTION_AUTHORIZED"
BLOCK_BINDING = "BLOCKED_R18R18B_SOURCE_OR_EVIDENCE_BINDING_FAILURE"
BLOCK_MPS = "BLOCKED_R18R18B_EXTERNAL_MPS_PREFLIGHT_FAILURE"
BLOCK_DRY_RUN = "BLOCKED_R18R18B_AUTHORIZATION_DRY_RUN_FAILURE"

R17_ROOT = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r17_factorized_actor_authorization_20260828_201708+09:00"
BLOCKED_R18_ROOT = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r18_factorized_actor_bounded_training_execution_20260828_201708+09:00"
R17_AUTH_PATH = R17_ROOT / "r18_factorized_one_shot_authorization_manifest.json"
EXECUTOR = ROOT / "run_h4m_ae_ls3_bt8_r18_e1_bounded_training.py"
INTERPRETER = PROJECT / ".venv" / "bin" / "python"


class R18R18BError(RuntimeError):
    def __init__(self, gate: str, detail: str) -> None:
        super().__init__(f"{gate}: {detail}")
        self.gate = gate
        self.detail = detail


def require(condition: bool, detail: str, gate: str = BLOCK_BINDING) -> None:
    if not condition:
        raise R18R18BError(gate, detail)


def dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def load(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing={path}")
    return json.loads(path.read_text(encoding="utf-8"))


def git(args: list[str]) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT, text=True, capture_output=True, check=True).stdout.strip()


def artifact_root() -> Path:
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M%S%z")[:-2] + ":00"
    return ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r18b_external_mps_one_shot_authorization_{stamp}"


def future_output_root(stamp: str) -> Path:
    return ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r18b_factorized_actor_bounded_training_execution_{stamp}"


def _probe_command() -> list[str]:
    probe = """
import json
import platform
import sys
import torch

result = {
    'interpreter': sys.executable,
    'platform_machine': platform.machine(),
    'torch_version': torch.__version__,
    'mps_built': bool(torch.backends.mps.is_built()),
    'mps_available': bool(torch.backends.mps.is_available()),
    'cuda_available': bool(torch.cuda.is_available()),
}
try:
    x = torch.tensor([[1.0, 2.0]], device='mps:0')
    product = x @ torch.tensor([[2.0], [3.0]], device='mps:0')
    linear = torch.nn.Linear(2, 1, bias=True, device='mps:0')
    y = linear(x)
    torch.mps.synchronize()
    result.update({
        'mps_tensor_device': str(x.device),
        'matmul_result': float(product.detach().cpu().item()),
        'linear_forward_shape': list(y.shape),
        'linear_forward_finite': bool(torch.isfinite(y).all().detach().cpu()),
        'passed': True,
    })
except Exception as exc:
    result.update({'passed': False, 'error_type': type(exc).__name__, 'error': str(exc)})
print(json.dumps(result, sort_keys=True))
"""
    return [str(INTERPRETER), "-c", probe]


def external_mps_preflight() -> dict[str, Any]:
    require(INTERPRETER.is_file(), f"interpreter_missing={INTERPRETER}", BLOCK_MPS)
    records = []
    for index in range(3):
        completed = subprocess.run(_probe_command(), cwd=PROJECT, text=True, capture_output=True)
        parsed: dict[str, Any] | None = None
        if completed.returncode == 0:
            try:
                parsed = json.loads(completed.stdout)
            except json.JSONDecodeError:
                parsed = None
        records.append({
            "fresh_process_index": index + 1,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "result": parsed,
        })
    passed = all(
        row["returncode"] == 0
        and isinstance(row["result"], dict)
        and row["result"].get("passed") is True
        and row["result"].get("interpreter") == str(INTERPRETER)
        and row["result"].get("platform_machine") == "arm64"
        and row["result"].get("mps_built") is True
        and row["result"].get("mps_available") is True
        and row["result"].get("mps_tensor_device") == "mps:0"
        and row["result"].get("matmul_result") == 8.0
        and row["result"].get("linear_forward_shape") == [1, 1]
        and row["result"].get("linear_forward_finite") is True
        and row["result"].get("cuda_available") is False
        for row in records
    )
    require(passed, "fresh_process_mps_probe", BLOCK_MPS)
    return {
        "passed": True,
        "execution_context": "EXTERNAL_MPS_APPROVED_CONTEXT_ONLY",
        "cpu_fallback": 0,
        "fresh_process_count": 3,
        "records": records,
    }


def bind_evidence() -> tuple[dict[str, Any], dict[str, Any]]:
    r17_gate = load(R17_ROOT / "gate_decision.json")
    require(r17_gate.get("gate") == R17.PASS_GATE, f"r17_gate={r17_gate.get('gate')}")
    base = load(R17_AUTH_PATH)
    base_hash = R17.canonical_sha256({key: value for key, value in base.items() if key != "authorization_sha256"})
    require(base.get("authorization") == R17.R18R17_AUTHORIZATION and base.get("authorization_sha256") == base_hash,
            "r17_authorization_hash")
    previous = load(BLOCKED_R18_ROOT / "gate_decision.json")
    require(previous.get("gate") == "BLOCKED_MPS_EXECUTION_ENVIRONMENT_UNAVAILABLE", f"previous_r18_gate={previous.get('gate')}")
    prior_results = load(BLOCKED_R18_ROOT / "test_results.json")
    prior_counters = dict(prior_results.get("execution_counters", {}))
    require(all(int(value) == 0 for value in prior_counters.values()), "previous_r18_attempt_counters_not_zero")
    return base, {
        "passed": True,
        "r18_r17_authorization": {
            "root": str(R17_ROOT.resolve()),
            "manifest": str(R17_AUTH_PATH.resolve()),
            "manifest_sha256": R17.sha256(R17_AUTH_PATH),
            "authorization_sha256": base_hash,
            "source_commit": base.get("source_commit"),
        },
        "previous_r18_attempt": {
            "root": str(BLOCKED_R18_ROOT.resolve()),
            "gate": previous.get("gate"),
            "gate_sha256": R17.sha256(BLOCKED_R18_ROOT / "gate_decision.json"),
            "counters_all_zero": True,
        },
    }


def source_binding() -> dict[str, Any]:
    require(git(["status", "--porcelain=v1"]) == "", "dirty_tree_before_authorization")
    base = load(R17_AUTH_PATH)
    base_expected = dict(dict(base["module_freeze_contract"])["frozen_source_hashes"]["expected"])
    executor_rel = "05_training/run_h4m_ae_ls3_bt8_r18_e1_bounded_training.py"
    require(executor_rel in base_expected, "base_executor_missing")
    unchanged = {
        rel: {"expected": digest, "actual": R17.sha256(PROJECT / rel)}
        for rel, digest in base_expected.items() if rel != executor_rel
    }
    mismatches = {rel: payload for rel, payload in unchanged.items() if payload["expected"] != payload["actual"]}
    require(not mismatches, f"frozen_source_mismatches={mismatches}")
    expected = {rel: payload["actual"] for rel, payload in unchanged.items()}
    expected[executor_rel] = R17.sha256(EXECUTOR)
    expected["05_training/run_h4m_ae_ls3_bt8_r18_r18b_external_mps_one_shot_authorization.py"] = R17.sha256(Path(__file__))
    actual = {rel: R17.sha256(PROJECT / rel) for rel in expected}
    require(expected == actual, "current_source_hash_map")
    changed = [item for item in git(["diff", "--name-only", f"{base['source_commit']}..HEAD"]).splitlines() if item]
    require(set(changed) == {
        executor_rel,
        "05_training/run_h4m_ae_ls3_bt8_r18_r18b_external_mps_one_shot_authorization.py",
    }, f"changed_files_since_r17={changed}")
    return {
        "passed": True,
        "base_r18_r17_source_commit": base["source_commit"],
        "current_source_commit": git(["rev-parse", "HEAD"]),
        "changed_files_since_r18_r17": changed,
        "allowed_change_scope": [executor_rel, "05_training/run_h4m_ae_ls3_bt8_r18_r18b_external_mps_one_shot_authorization.py"],
        "frozen_files_unchanged": True,
        "expected": expected,
        "actual": actual,
    }


def build_authorization(base: Mapping[str, Any], evidence: Mapping[str, Any], source: Mapping[str, Any],
                        mps: Mapping[str, Any], output_root: Path, mps_path: Path) -> dict[str, Any]:
    payload = copy.deepcopy(dict(base))
    payload.update({
        "stage": STAGE,
        "authorization": AUTHORIZATION,
        "authorization_revision": REVISION,
        "authorized": True,
        "one_shot": True,
        "consumed_on_attempt": True,
        "source_commit": source["current_source_commit"],
        "authorization_replaces": {
            "authorization": base["authorization"],
            "reason": "previous R18-R18 attempt was blocked before MPS preflight completion; never reuse a consumed manifest",
        },
        "execution_context": "EXTERNAL_MPS_APPROVED_CONTEXT_ONLY",
        "mps_recovery_audit": {"path": str(mps_path.resolve()), "sha256": R17.sha256(mps_path), **dict(mps)},
        "future_output_root": str(output_root.resolve()),
        "training_authorized_in_R18_R18B": False,
        "bounded_execution_authorized_for_future_one_shot": True,
        "authorization_sha256": "PENDING",
    })
    payload["upstream"] = {
        **dict(base["upstream"]),
        "r18_r17_authorization": dict(evidence["r18_r17_authorization"]),
        "previous_r18_attempt": dict(evidence["previous_r18_attempt"]),
        "r18_executor": {"path": str(EXECUTOR.resolve()), "sha256": R17.sha256(EXECUTOR)},
    }
    module = dict(payload["module_freeze_contract"])
    module["frozen_source_hashes"] = {
        "expected": dict(source["expected"]),
        "actual": dict(source["actual"]),
        "all_unchanged": True,
        "actor_architecture": "FACTORIZED_ASSIGN_THEN_CANDIDATE",
        "factorized_ppo_loss_binding": "DIRECT_RECONSTRUCTED_ACTION_LOG_PROBS_NO_SECOND_SHARED_SOFTMAX",
    }
    payload["module_freeze_contract"] = module
    guard = dict(payload["execution_guard_contract"])
    guard.update({
        "execute_flag_required": "--execute-exact-r18-envelope",
        "external_mps_execution_context_required": True,
        "external_mps_fresh_process_preflight_required": 3,
        "CPU_fallback_allowed": False,
        "previous_authorization_reuse_allowed": False,
    })
    payload["execution_guard_contract"] = guard
    checkpoint = dict(payload["checkpoint_contract"])
    checkpoint["R18_output_root"] = str(output_root.resolve())
    checkpoint["output_root_must_not_exist_before_execution"] = True
    checkpoint["overwrite_existing_checkpoint"] = False
    payload["checkpoint_contract"] = checkpoint
    payload["authorization_sha256"] = R17.canonical_sha256(
        {key: value for key, value in payload.items() if key != "authorization_sha256"}
    )
    return payload


def dry_run(auth_path: Path, authorization_sha256: str) -> dict[str, Any]:
    command = [str(INTERPRETER), str(EXECUTOR), "--authorization-manifest", str(auth_path.resolve()),
               "--authorization-sha256", authorization_sha256, "--dry-run"]
    result = subprocess.run(command, cwd=PROJECT, text=True, capture_output=True)
    parsed = json.loads(result.stdout) if result.returncode == 0 else None
    passed = bool(
        result.returncode == 0 and isinstance(parsed, dict)
        and parsed.get("authorization_valid") is True
        and parsed.get("authorization") == AUTHORIZATION
        and parsed.get("actor_architecture") == "FACTORIZED_ASSIGN_THEN_CANDIDATE"
        and parsed.get("training") == parsed.get("rollout") == parsed.get("optimizer_step") == parsed.get("checkpoint_write") == 0
    )
    require(passed, f"dry_run={result.stdout}|{result.stderr}", BLOCK_DRY_RUN)
    return {"passed": True, "command": command, "stdout": result.stdout, "stderr": result.stderr}


def write_manifest(root: Path, gate: str, source_commit: str, classification: str) -> None:
    files = {item.relative_to(root).as_posix(): R17.sha256(item)
             for item in root.rglob("*") if item.is_file() and item.name != "manifest.json"}
    dump(root / "manifest.json", {"stage": STAGE, "gate": gate, "classification": classification,
                                   "source_commit": source_commit, "file_sha256": files, "github_push_performed": False})


def main() -> None:
    root = artifact_root()
    root.mkdir(parents=True)
    counters = {"training": 0, "rollout": 0, "simulator": 0, "optimizer_creation": 0, "optimizer_step": 0,
                "backward": 0, "checkpoint_write": 0, "policy_mutation": 0, "github_push": 0}
    source_commit = None
    try:
        base, evidence = bind_evidence()
        source = source_binding()
        source_commit = str(source["current_source_commit"])
        mps = external_mps_preflight()
        mps_path = root / "external_mps_preflight.json"
        dump(mps_path, mps)
        stamp = root.name.split("r18_r18b_external_mps_one_shot_authorization_", 1)[1]
        output_root = future_output_root(stamp)
        require(not output_root.exists(), f"output_root_exists={output_root}")
        authorization = build_authorization(base, evidence, source, mps, output_root, mps_path)
        auth_path = root / "r18r18b_factorized_one_shot_authorization_manifest.json"
        dump(auth_path, authorization)
        dry = dry_run(auth_path, str(authorization["authorization_sha256"]))
        output = {
            "evidence_binding_audit.json": {"passed": True, "evidence": evidence, "execution_counters": counters},
            "source_binding_guard.json": source,
            "external_mps_preflight.json": mps,
            "r18r18b_factorized_one_shot_authorization_manifest.json": authorization,
            "r18r18b_dry_run_validation.json": dry,
            "execution_guard_contract.json": authorization["execution_guard_contract"],
            "test_results.json": {"passed": True, "execution_counters": counters, "hard_failures": [], "warnings": []},
            "gate_decision.json": {"stage": STAGE, "gate": PASS_GATE, "classification": CLASSIFICATION,
                                   "source_commit": source_commit, "authorization_sha256": authorization["authorization_sha256"],
                                   "future_output_root": str(output_root.resolve()), "next_step": "execute exactly once with the sealed command"},
        }
        for name, value in output.items():
            if name != "external_mps_preflight.json":
                dump(root / name, value)
        command = [str(INTERPRETER), str(EXECUTOR), "--authorization-manifest", str(auth_path.resolve()),
                   "--authorization-sha256", str(authorization["authorization_sha256"]), "--execute-exact-r18-envelope"]
        (root / "exact_execution_command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
        (root / "final_report.md").write_text(
            "# R18-R18B external MPS recovery and fresh authorization\n\n"
            f"- gate: `{PASS_GATE}`\n- source: `{source_commit}`\n"
            "- external MPS fresh-process preflight: `3/3 PASS`\n"
            "- training / rollout / optimizer / backward / checkpoint write: `0`\n"
            "- previous R18-R18 manifest is consumed evidence only; this authorization is new and one-shot.\n",
            encoding="utf-8",
        )
        write_manifest(root, PASS_GATE, source_commit, CLASSIFICATION)
        (root / "_SUCCESS.lock").write_text(PASS_GATE + "\n", encoding="utf-8")
        print(f"[PASS] {PASS_GATE}")
        print(root)
    except R18R18BError as exc:
        dump(root / "test_results.json", {"passed": False, "execution_counters": counters, "hard_failures": [exc.detail], "warnings": []})
        dump(root / "gate_decision.json", {"stage": STAGE, "gate": exc.gate, "classification": "BLOCKED",
                                             "source_commit": source_commit, "hard_failures": [exc.detail], "next_step": "STOP"})
        (root / "final_report.md").write_text(f"# R18-R18B blocked\n\n- gate: `{exc.gate}`\n- detail: `{exc.detail}`\n", encoding="utf-8")
        write_manifest(root, exc.gate, source_commit or "UNKNOWN", "BLOCKED")
        print(f"[BLOCKED] {exc.gate}")
        print(root)


if __name__ == "__main__":
    main()
