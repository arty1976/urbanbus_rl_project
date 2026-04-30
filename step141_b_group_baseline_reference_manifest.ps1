$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\ryujo\urbanbus_rl_project"
if (-not (Test-Path $ProjectRoot)) {
    throw "[STOP] Project root not found: $ProjectRoot"
}
Set-Location $ProjectRoot

$RewardsDir = Join-Path $ProjectRoot "05_training\rewards"
if (-not (Test-Path $RewardsDir)) {
    New-Item -ItemType Directory -Force -Path $RewardsDir | Out-Null
}

$PyMain = Join-Path $RewardsDir "baseline_reference_manifest_generator_step141.py"
$PyValidate = Join-Path $RewardsDir "validate_baseline_reference_manifest_step141.py"
$PyTest = Join-Path $RewardsDir "test_baseline_reference_manifest_step141.py"
$MdDoc = Join-Path $RewardsDir "baseline_reference_manifest_step141.md"

$MainCode = @'
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


STEP_ID = 141
ARTIFACT_VERSION = "baseline_reference_manifest_step141_v1"

REQUIRED_CANONICAL_FILES = [
    "kpi_by_window.parquet",
    "kpi_by_seed.parquet",
    "kpi_by_time_band.parquet",
    "kpi_overall.json",
    "aggregation_manifest.json",
]

BASELINE_SPECS = [
    {
        "baseline_id": "B0R",
        "legacy_condition_id": "B0",
        "display_name": "historical replay baseline",
        "reference_role": "historical_replay_reference",
        "root_candidates": [
            "artifacts/baseline_v1/B0_historical/canonical_eval",
            "artifacts/baseline_v1/B0_historical",
        ],
    },
    {
        "baseline_id": "B1",
        "legacy_condition_id": "B1",
        "display_name": "no-op baseline",
        "reference_role": "no_op_reference",
        "root_candidates": [
            "artifacts/baseline_v1/B1_noop/canonical_eval",
            "artifacts/baseline_v1/B1_noop",
        ],
    },
    {
        "baseline_id": "B2",
        "legacy_condition_id": "B2",
        "display_name": "rule-based baseline",
        "reference_role": "rule_based_reference",
        "root_candidates": [
            "artifacts/baseline_v1/B2_rulebased/canonical_eval",
            "artifacts/baseline_v1/B2_rulebased",
        ],
    },
]

FORBIDDEN_TRUE_KEYS = [
    "causal_comparison_allowed",
    "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
    "best_reward_claim_allowed",
    "train_with_this_reward_allowed",
    "actual_training_allowed",
    "winner_selected",
    "trainable_reward_promoted",
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


def dump_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def find_root(project_root: Path, candidates: List[str]) -> Optional[Path]:
    for raw in candidates:
        p = project_root / raw
        if p.exists():
            return p
    return None


def recursive_forbidden_true(obj: Any, prefix: str = "") -> List[str]:
    violations: List[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in FORBIDDEN_TRUE_KEYS and bool(value):
                violations.append(f"forbidden_true: {path}")
            if isinstance(value, (dict, list)):
                violations.extend(recursive_forbidden_true(value, path))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            path = f"{prefix}[{idx}]"
            if isinstance(value, (dict, list)):
                violations.extend(recursive_forbidden_true(value, path))
    return violations


def read_json_if_exists(path: Path, warnings: List[str], label: str) -> Dict[str, Any]:
    if not path.exists():
        warnings.append(f"{label}_missing")
        return {}
    try:
        return load_json_any(path)
    except Exception as exc:
        warnings.append(f"{label}_unreadable: {exc}")
        return {}


def inspect_baseline(project_root: Path, spec: Dict[str, Any]) -> Dict[str, Any]:
    root = find_root(project_root, spec["root_candidates"])
    warnings: List[str] = []
    missing: List[str] = []

    result: Dict[str, Any] = {
        "baseline_id": spec["baseline_id"],
        "legacy_condition_id": spec["legacy_condition_id"],
        "display_name": spec["display_name"],
        "reference_role": spec["reference_role"],
        "root": str(root) if root else "",
        "root_exists": bool(root),
        "canonical_files": {},
        "missing_canonical_files": missing,
        "kpi_overall_summary": {},
        "aggregation_manifest_summary": {},
        "reference_status": "MISSING",
        "causal_comparison_allowed": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "best_reward_claim_allowed": False,
        "train_with_this_reward_allowed": False,
        "actual_training_allowed": False,
        "winner_selected": False,
        "trainable_reward_promoted": False,
        "warnings": warnings,
    }

    if root is None:
        missing.extend(REQUIRED_CANONICAL_FILES)
        warnings.append("baseline_root_missing")
        return result

    for fname in REQUIRED_CANONICAL_FILES:
        p = root / fname
        exists = p.exists()
        result["canonical_files"][fname] = {
            "path": str(p),
            "exists": exists,
            "size_bytes": int(p.stat().st_size) if exists else 0,
        }
        if not exists:
            missing.append(fname)

    overall = read_json_if_exists(root / "kpi_overall.json", warnings, "kpi_overall")
    if overall:
        result["kpi_overall_summary"] = {
            "mode": overall.get("mode"),
            "condition_ids": overall.get("condition_ids", []),
            "n_windows_total": overall.get("n_windows_total"),
            "n_condition_seed_pairs": overall.get("n_condition_seed_pairs"),
            "causal_comparison_allowed": bool(overall.get("causal_comparison_allowed", False)),
            "kpi_names": sorted(list(overall.get("kpis", {}).keys())) if isinstance(overall.get("kpis"), dict) else [],
        }
        if bool(overall.get("causal_comparison_allowed", False)):
            warnings.append("kpi_overall_has_causal_comparison_allowed_true")

    agg = read_json_if_exists(root / "aggregation_manifest.json", warnings, "aggregation_manifest")
    if agg:
        result["aggregation_manifest_summary"] = {
            "mode": agg.get("mode"),
            "row_counts": agg.get("row_counts", {}),
            "strict_canonical": agg.get("strict_canonical"),
            "causal_comparison_allowed": bool(agg.get("causal_comparison_allowed", False)),
            "warnings": agg.get("warnings", []),
        }
        if bool(agg.get("causal_comparison_allowed", False)):
            warnings.append("aggregation_manifest_has_causal_comparison_allowed_true")

    if not missing:
        result["reference_status"] = "READY_AS_NONCAUSAL_BASELINE_REFERENCE"
    elif (root / "kpi_by_window.parquet").exists():
        result["reference_status"] = "PARTIAL_REFERENCE_HAS_WINDOW_KPI"
    else:
        result["reference_status"] = "INCOMPLETE_REFERENCE_MISSING_CANONICAL_OUTPUTS"

    return result


def build_markdown(payload: Dict[str, Any]) -> str:
    lines = [
        "# Step 141 B-group baseline reference manifest",
        "",
        "This step records B0R/B1/B2 as local baseline references for later comparison with A-family MAPPO outputs.",
        "",
        "This step does not execute MAPPO and does not permit causal or paper-level claims.",
        "",
        "## Status",
        "",
        f"- audit_status: `{payload['audit_status']}`",
        f"- baseline_reference_status: `{payload['baseline_reference_status']}`",
        f"- ready_baseline_count: `{payload['ready_baseline_count']}`",
        f"- required_baseline_count: `{payload['required_baseline_count']}`",
        f"- causal_comparison_allowed: `{payload['causal_comparison_allowed']}`",
        f"- paper_level_claim_allowed: `{payload['paper_level_claim_allowed']}`",
        "",
        "## Baseline references",
        "",
        "| baseline | role | status | root |",
        "|---|---|---|---|",
    ]
    for ref in payload["baseline_references"]:
        lines.append(
            f"| {ref['baseline_id']} | {ref['reference_role']} | {ref['reference_status']} | `{ref['root']}` |"
        )
    lines.extend([
        "",
        "## Guard",
        "",
        "- B0R/B1/B2 are reference baselines, not reward candidates.",
        "- A-family MAPPO reward ablation remains separate.",
        "- Historical/replay/stub outputs remain non-causal references unless rerun through a validated causal simulator.",
        "",
    ])
    return "\n".join(lines)


def run_generator(project_root: Path, output_root: Path) -> Dict[str, Any]:
    project_root = project_root.resolve()
    output_root = output_root.resolve()

    refs = [inspect_baseline(project_root, spec) for spec in BASELINE_SPECS]
    ready_count = sum(
        1 for ref in refs
        if ref["reference_status"] == "READY_AS_NONCAUSAL_BASELINE_REFERENCE"
    )
    required_count = len(BASELINE_SPECS)

    blocking_reasons: List[str] = []
    warnings: List[str] = []

    for ref in refs:
        baseline_id = ref["baseline_id"]
        for w in ref.get("warnings", []):
            warnings.append(f"{baseline_id}: {w}")
        if ref["reference_status"] != "READY_AS_NONCAUSAL_BASELINE_REFERENCE":
            blocking_reasons.append(f"{baseline_id}_not_ready: {ref['reference_status']}")
        for v in recursive_forbidden_true(ref):
            blocking_reasons.append(f"{baseline_id}: {v}")

    audit_status = "PASS" if not blocking_reasons else "PASS_WITH_INCOMPLETE_BASELINES"
    baseline_reference_status = (
        "B_GROUP_BASELINE_REFERENCE_READY"
        if ready_count == required_count
        else "B_GROUP_BASELINE_REFERENCE_INCOMPLETE"
    )

    manifest_path = output_root / "baseline_reference_manifest_step141.json"
    md_path = output_root / "baseline_reference_manifest_step141.md"

    payload: Dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "step": STEP_ID,
        "created_at_utc": utc_now(),
        "project_root": str(project_root),
        "output_root": str(output_root),
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "audit_status": audit_status,
        "baseline_reference_status": baseline_reference_status,
        "ready_baseline_count": ready_count,
        "required_baseline_count": required_count,
        "blocking_reasons": blocking_reasons,
        "warnings": warnings,
        "baseline_references": refs,
        "comparison_role": {
            "B0R": "historical replay reference",
            "B1": "no-op reference",
            "B2": "rule-based reference",
            "A_family": "future A/A90/A80/A70 MAPPO reward ablation outputs",
        },
        "causal_comparison_allowed": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "best_reward_claim_allowed": False,
        "train_with_this_reward_allowed": False,
        "actual_training_allowed": False,
        "winner_selected": False,
        "trainable_reward_promoted": False,
        "scope_note": (
            "Step 141 creates a B-group baseline reference manifest only. "
            "It does not execute MAPPO, select a reward winner, or permit causal claims."
        ),
        "manifest_path": str(manifest_path),
        "markdown_path": str(md_path),
    }

    dump_json(manifest_path, payload)
    dump_text(md_path, build_markdown(payload))
    dump_text(project_root / "05_training" / "rewards" / "baseline_reference_manifest_step141.md", build_markdown(payload))
    dump_json(project_root / "05_training" / "rewards" / "baseline_reference_manifest_step141.latest.json", {
        "manifest_path": str(manifest_path),
        "audit_status": audit_status,
        "baseline_reference_status": baseline_reference_status,
        "ready_baseline_count": ready_count,
        "required_baseline_count": required_count,
        "causal_comparison_allowed": False,
        "paper_level_claim_allowed": False,
        "created_at_utc": payload["created_at_utc"],
    })

    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-root", default="artifacts/rewards/baseline_reference_manifest_step141")
    args = parser.parse_args()

    payload = run_generator(Path(args.project_root), Path(args.project_root) / args.output_root)

    print("[OK] Step 141 B-group baseline reference manifest generator completed")
    print(f"[OK] audit_status                 : {payload['audit_status']}")
    print(f"[OK] baseline_reference_status    : {payload['baseline_reference_status']}")
    print(f"[OK] ready_baseline_count         : {payload['ready_baseline_count']}")
    print(f"[OK] required_baseline_count      : {payload['required_baseline_count']}")
    print(f"[OK] causal_comparison_allowed    : {payload['causal_comparison_allowed']}")
    print(f"[OK] paper_level_claim_allowed    : {payload['paper_level_claim_allowed']}")
    print(f"[OK] manifest                     : {payload['manifest_path']}")

    for ref in payload["baseline_references"]:
        print(f"[OK] {ref['baseline_id']} status: {ref['reference_status']}")

    if payload["warnings"]:
        for warning in payload["warnings"]:
            print(f"[WARN] {warning}")

    if payload["blocking_reasons"]:
        for reason in payload["blocking_reasons"]:
            print(f"[INFO] incomplete_baseline_reason: {reason}")


if __name__ == "__main__":
    main()
'@

$ValidateCode = @'
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


ARTIFACT_VERSION = "baseline_reference_manifest_step141_v1"
REQUIRED_BASELINES = {"B0R", "B1", "B2"}

FORBIDDEN_TRUE_KEYS = [
    "causal_comparison_allowed",
    "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
    "best_reward_claim_allowed",
    "train_with_this_reward_allowed",
    "actual_training_allowed",
    "winner_selected",
    "trainable_reward_promoted",
]


def load_json(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except Exception:
            continue
    raise RuntimeError(f"failed_to_read_json: {path}")


def recursive_forbidden_true(obj: Any, prefix: str = "") -> List[str]:
    violations: List[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in FORBIDDEN_TRUE_KEYS and bool(value):
                violations.append(f"forbidden_true: {path}")
            if isinstance(value, (dict, list)):
                violations.extend(recursive_forbidden_true(value, path))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            path = f"{prefix}[{idx}]"
            if isinstance(value, (dict, list)):
                violations.extend(recursive_forbidden_true(value, path))
    return violations


def validate_payload(payload: Dict[str, Any], require_all_ready: bool = False) -> Dict[str, Any]:
    errors: List[str] = []

    if payload.get("artifact_version") != ARTIFACT_VERSION:
        errors.append("artifact_version_mismatch")

    if payload.get("step") != 141:
        errors.append("step_must_be_141")

    status = payload.get("audit_status")
    if status not in ("PASS", "PASS_WITH_INCOMPLETE_BASELINES"):
        errors.append("invalid_audit_status")

    refs = payload.get("baseline_references", [])
    baseline_ids = {str(r.get("baseline_id")) for r in refs}
    if baseline_ids != REQUIRED_BASELINES:
        errors.append(f"baseline_ids_mismatch: {sorted(baseline_ids)}")

    if int(payload.get("required_baseline_count", -1)) != 3:
        errors.append("required_baseline_count_must_be_3")

    ready_count = int(payload.get("ready_baseline_count", -1))
    if ready_count < 0 or ready_count > 3:
        errors.append("ready_baseline_count_out_of_range")

    if require_all_ready and ready_count != 3:
        errors.append("not_all_baselines_ready")

    errors.extend(recursive_forbidden_true(payload))

    for ref in refs:
        if ref.get("baseline_id") not in REQUIRED_BASELINES:
            errors.append(f"unexpected_baseline_id: {ref.get('baseline_id')}")
        if bool(ref.get("causal_comparison_allowed", False)):
            errors.append(f"{ref.get('baseline_id')}: causal_comparison_allowed_true")
        if bool(ref.get("paper_level_claim_allowed", False)):
            errors.append(f"{ref.get('baseline_id')}: paper_level_claim_allowed_true")
        if ref.get("reference_status") not in {
            "READY_AS_NONCAUSAL_BASELINE_REFERENCE",
            "PARTIAL_REFERENCE_HAS_WINDOW_KPI",
            "INCOMPLETE_REFERENCE_MISSING_CANONICAL_OUTPUTS",
            "MISSING",
        }:
            errors.append(f"{ref.get('baseline_id')}: invalid_reference_status")

    return {
        "validation_status": "PASS" if not errors else "FAIL",
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--require-all-ready", action="store_true")
    args = parser.parse_args()

    payload = load_json(Path(args.manifest))
    result = validate_payload(payload, require_all_ready=args.require_all_ready)

    print(f"[OK] validation_status: {result['validation_status']}")
    if result["errors"]:
        for error in result["errors"]:
            print(f"[FAIL] {error}")
        raise SystemExit(2)

    print("[OK] Step 141 B-group baseline reference manifest validation PASS")


if __name__ == "__main__":
    main()
'@

$TestCode = @'
from __future__ import annotations

from validate_baseline_reference_manifest_step141 import validate_payload


def base_payload() -> dict:
    refs = []
    for baseline_id in ["B0R", "B1", "B2"]:
        refs.append({
            "baseline_id": baseline_id,
            "reference_status": "READY_AS_NONCAUSAL_BASELINE_REFERENCE",
            "causal_comparison_allowed": False,
            "paper_level_claim_allowed": False,
            "causal_performance_claim_allowed": False,
            "best_reward_claim_allowed": False,
            "train_with_this_reward_allowed": False,
            "actual_training_allowed": False,
            "winner_selected": False,
            "trainable_reward_promoted": False,
        })
    return {
        "artifact_version": "baseline_reference_manifest_step141_v1",
        "step": 141,
        "audit_status": "PASS",
        "baseline_reference_status": "B_GROUP_BASELINE_REFERENCE_READY",
        "ready_baseline_count": 3,
        "required_baseline_count": 3,
        "baseline_references": refs,
        "causal_comparison_allowed": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "best_reward_claim_allowed": False,
        "train_with_this_reward_allowed": False,
        "actual_training_allowed": False,
        "winner_selected": False,
        "trainable_reward_promoted": False,
    }


def test_ready_payload_passes() -> None:
    result = validate_payload(base_payload(), require_all_ready=True)
    assert result["validation_status"] == "PASS", result


def test_incomplete_payload_can_pass_without_require_all_ready() -> None:
    payload = base_payload()
    payload["audit_status"] = "PASS_WITH_INCOMPLETE_BASELINES"
    payload["baseline_reference_status"] = "B_GROUP_BASELINE_REFERENCE_INCOMPLETE"
    payload["ready_baseline_count"] = 2
    payload["baseline_references"][2]["reference_status"] = "PARTIAL_REFERENCE_HAS_WINDOW_KPI"
    result = validate_payload(payload, require_all_ready=False)
    assert result["validation_status"] == "PASS", result


def test_incomplete_payload_fails_when_require_all_ready() -> None:
    payload = base_payload()
    payload["audit_status"] = "PASS_WITH_INCOMPLETE_BASELINES"
    payload["baseline_reference_status"] = "B_GROUP_BASELINE_REFERENCE_INCOMPLETE"
    payload["ready_baseline_count"] = 2
    payload["baseline_references"][2]["reference_status"] = "PARTIAL_REFERENCE_HAS_WINDOW_KPI"
    result = validate_payload(payload, require_all_ready=True)
    assert result["validation_status"] == "FAIL"
    assert any("not_all_baselines_ready" in e for e in result["errors"])


def test_causal_true_fails() -> None:
    payload = base_payload()
    payload["baseline_references"][0]["causal_comparison_allowed"] = True
    result = validate_payload(payload, require_all_ready=False)
    assert result["validation_status"] == "FAIL"
    assert any("causal_comparison_allowed" in e for e in result["errors"])


def test_wrong_baseline_set_fails() -> None:
    payload = base_payload()
    payload["baseline_references"][0]["baseline_id"] = "B0"
    result = validate_payload(payload, require_all_ready=False)
    assert result["validation_status"] == "FAIL"
    assert any("baseline_ids_mismatch" in e for e in result["errors"])


def main() -> None:
    test_ready_payload_passes()
    test_incomplete_payload_can_pass_without_require_all_ready()
    test_incomplete_payload_fails_when_require_all_ready()
    test_causal_true_fails()
    test_wrong_baseline_set_fails()
    print("[OK] Step 141 B-group baseline reference manifest self-test PASS")


if __name__ == "__main__":
    main()
'@

$DocText = @'
# Step 141 B-group baseline reference manifest

Step 141 records the local B-group baseline reference status.

Baseline set:
- B0R = historical replay baseline
- B1 = no-op baseline
- B2 = rule-based baseline

This step does not execute MAPPO.
This step does not select a reward winner.
This step does not allow paper-level claims.
This step does not allow causal performance claims.

The purpose is to freeze the reference baseline set that will later be compared against A-family MAPPO reward ablation outputs.
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

$OutputRoot = "artifacts/rewards/baseline_reference_manifest_step141"
$Manifest = Join-Path $ProjectRoot "$OutputRoot\baseline_reference_manifest_step141.json"

& $py $PyMain `
  --project-root "." `
  --output-root $OutputRoot

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 141 B-group baseline reference manifest generator failed"
}

& $py $PyValidate --manifest $Manifest
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 141 baseline reference manifest validation failed"
}

& $py $PyTest
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 141 self-test failed"
}

Write-Host "[DONE] Step 141 B-group baseline reference manifest complete."
