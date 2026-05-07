from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

EXPECTED_BUNDLE_STATUS = "ACTUAL_ROUTE_AWARE_ROLLOUT_EVIDENCE_RUNBOOK_READY_NONCLAIM"
EXPECTED_GUARDS = {
    "actual_operational_claim_allowed": False,
    "paper_level_claim_allowed": False,
    "causal_performance_claim_allowed": False,
    "train_allowed": False,
    "actual_results_claimed": False,
}
EXPECTED_STAGES = [
    "A_step162_adapt_rollout",
    "B_step161_pickup_attempt_writer",
    "C_step163_real_attention_extractor",
    "D_step165_attempt_path_filter",
    "E_step160_zero_loss_reporter",
    "F_step166_patent_report_v2",
]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_manifest(path: Path, require_actual_ready: bool = False) -> Dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"manifest not found: {path}")
    m = load_json(path)

    if m.get("bundle_status") != EXPECTED_BUNDLE_STATUS:
        raise RuntimeError(f"bundle_status mismatch: {m.get('bundle_status')}")
    if m.get("audit_status") not in {"PASS", "BLOCKED"}:
        raise RuntimeError(f"invalid audit_status: {m.get('audit_status')}")
    for key, expected in EXPECTED_GUARDS.items():
        if m.get(key) is not expected:
            raise RuntimeError(f"guard flag mismatch: {key}={m.get(key)}, expected={expected}")

    stages = list(m.get("execution_chain", []))
    if stages != EXPECTED_STAGES:
        raise RuntimeError(f"execution_chain mismatch: {stages}")

    cmd_md = Path(m.get("command_plan_md", ""))
    cmd_json = Path(m.get("command_plan_json", ""))
    if not cmd_md.exists():
        raise RuntimeError(f"command plan md missing: {cmd_md}")
    if not cmd_json.exists():
        raise RuntimeError(f"command plan json missing: {cmd_json}")

    if require_actual_ready and not bool(m.get("ready_for_actual_like_execution", False)):
        raise RuntimeError("actual-like execution requested but manifest is not ready")
    if require_actual_ready and m.get("template_only", True):
        raise RuntimeError("actual-like execution requested but manifest is template_only")

    print("[OK] Step 167 actual route-aware rollout evidence runbook validation PASS")
    print(f"[OK] manifest                  : {path}")
    print(f"[OK] audit_status              : {m.get('audit_status')}")
    print(f"[OK] template_only             : {m.get('template_only')}")
    print(f"[OK] ready_for_actual_like_execution : {m.get('ready_for_actual_like_execution')}")
    print(f"[OK] paper_level_claim_allowed : {m.get('paper_level_claim_allowed')}")
    print(f"[OK] causal_performance_claim_allowed : {m.get('causal_performance_claim_allowed')}")
    return m


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--require-actual-ready", action="store_true")
    args = p.parse_args()
    validate_manifest(Path(args.manifest), require_actual_ready=bool(args.require_actual_ready))


if __name__ == "__main__":
    main()
