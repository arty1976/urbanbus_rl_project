from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


TRAINING_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_DIR.parent

PHASE2_12_KPIS = [
    "cv_headway",
    "avg_wait_seconds",
    "bunching_rate",
    "on_time_rate",
    "intervention_rate",
    "energy_proxy",
    "passenger_demand_generated",
    "passenger_served_count",
    "passenger_service_rate",
    "passenger_wait_p95_seconds",
    "energy_proxy_per_passenger",
    "fleet_reduction_ratio",
]

VALID_CONDITIONS = {"A", "A90", "A80", "A70"}
REWARD_VERSION = "mappo_reward_v1"
REWARD_CLAIM_BOUNDARY = "toy_causal_training_reward_contract_not_paper_performance_claim"


def safe_rmtree(path: Path) -> Path:
    if not path.exists():
        return path

    def onexc(func, target, exc_info):
        try:
            import os
            os.chmod(target, 0o700)
            func(target)
        except Exception:
            pass

    for attempt in range(5):
        try:
            try:
                shutil.rmtree(path, onexc=onexc)
            except TypeError:
                shutil.rmtree(path, onerror=onexc)
            return path
        except PermissionError:
            time.sleep(0.5 + 0.5 * attempt)

    fallback = path.with_name(f"{path.name}_retry_{int(time.time())}")
    print(f"[WARN] could not remove locked output directory: {path}")
    print(f"[WARN] using fallback output directory instead: {fallback}")
    return fallback


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_csv(value: str) -> List[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def run_cmd(cmd: List[str], cwd: Path) -> str:
    completed = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    if completed.returncode != 0:
        print("[STDOUT]")
        print(completed.stdout)
        print("[STDERR]")
        print(completed.stderr)
        raise RuntimeError(f"command failed: {' '.join(cmd)}")
    print(completed.stdout)
    if completed.stderr:
        print(completed.stderr)
    return completed.stdout


def validate_conditions(conditions: List[str]) -> List[str]:
    normalized = [c.upper() for c in conditions]
    bad = [c for c in normalized if c not in VALID_CONDITIONS]
    if bad:
        raise SystemExit(f"invalid conditions: {bad}; allowed={sorted(VALID_CONDITIONS)}")
    if len(set(normalized)) != len(normalized):
        raise SystemExit(f"duplicate conditions are not allowed: {normalized}")
    if "A" not in normalized:
        raise SystemExit("matrix evaluation must include A baseline")
    return normalized


def ensure_checkpoint(args: argparse.Namespace, output_root: Path) -> Path:
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise SystemExit(f"--checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    training_root = output_root / "matrix_training"
    run_cmd(
        [
            sys.executable,
            str(TRAINING_DIR / "train_toy_causal_mappo_smoke.py"),
            "--output-root",
            str(training_root),
            "--condition-id",
            "A",
            "--seed",
            str(args.seed),
            "--num-agents",
            str(args.num_agents),
            "--steps",
            str(args.steps),
            "--epochs",
            "1",
            "--device",
            str(args.device),
            "--clean",
        ],
        cwd=PROJECT_ROOT,
    )

    checkpoint_path = training_root / "checkpoints" / "toy_mappo_smoke_checkpoint.pt"
    if not checkpoint_path.exists():
        raise RuntimeError(f"training did not produce checkpoint: {checkpoint_path}")
    return checkpoint_path


def run_condition_eval(
    checkpoint_path: Path,
    condition_id: str,
    args: argparse.Namespace,
    output_root: Path,
) -> Dict[str, Any]:
    condition_root = output_root / "conditions" / condition_id
    run_cmd(
        [
            sys.executable,
            str(TRAINING_DIR / "evaluation" / "evaluate_toy_mappo_smoke_checkpoint.py"),
            "--checkpoint",
            str(checkpoint_path),
            "--output-root",
            str(condition_root),
            "--condition-id",
            condition_id,
            "--seed",
            str(args.seed),
            "--num-agents",
            str(args.num_agents),
            "--steps",
            str(args.steps),
            "--time-bands",
            str(args.time_bands),
            "--windows-per-time-band",
            str(args.windows_per_time_band),
            "--action-mode",
            str(args.action_mode),
            "--device",
            str(args.device),
            "--clean",
            "--skip-inspector",
        ],
        cwd=PROJECT_ROOT,
    )

    manifest_path = condition_root / "evaluation_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"condition evaluation manifest missing: {manifest_path}")

    manifest = load_json(manifest_path)
    if manifest.get("condition_id") != condition_id:
        raise RuntimeError(f"condition manifest mismatch: expected {condition_id}, got {manifest.get('condition_id')}")
    if manifest.get("performance_claim_allowed") is not False:
        raise RuntimeError(f"{condition_id}: performance_claim_allowed must be false")
    if manifest.get("smoke_evaluation_only") is not True:
        raise RuntimeError(f"{condition_id}: smoke_evaluation_only must be true")
    if manifest.get("reward_version") != REWARD_VERSION:
        raise RuntimeError(f"{condition_id}: reward_version mismatch")

    return manifest


def combine_canonical_outputs(
    output_root: Path,
    condition_manifests: Dict[str, Dict[str, Any]],
) -> Path:
    combined_root = output_root / "matrix_canonical_eval"
    combined_root.mkdir(parents=True, exist_ok=True)

    frames_by_name: Dict[str, List[pd.DataFrame]] = {
        "kpi_by_window.parquet": [],
        "kpi_by_seed.parquet": [],
        "kpi_by_time_band.parquet": [],
    }

    for condition_id, manifest in condition_manifests.items():
        canonical_root = Path(manifest["canonical_root"])
        for filename in frames_by_name:
            path = canonical_root / filename
            if not path.exists():
                raise RuntimeError(f"{condition_id}: missing canonical file: {path}")
            df = pd.read_parquet(path)
            frames_by_name[filename].append(df)

    combined_outputs: Dict[str, str] = {}
    for filename, frames in frames_by_name.items():
        combined = pd.concat(frames, ignore_index=True)
        out_path = combined_root / filename
        combined.to_parquet(out_path, index=False)
        combined_outputs[filename] = str(out_path)

    kpi_window = pd.read_parquet(combined_root / "kpi_by_window.parquet")
    kpi_payload: Dict[str, Any] = {}
    for kpi in PHASE2_12_KPIS:
        values = pd.to_numeric(kpi_window[kpi], errors="coerce")
        if int(values.notna().sum()) == 0:
            raise RuntimeError(f"combined kpi_by_window has no numeric values for {kpi}")
        kpi_payload[kpi] = {
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
            "valid_count": int(values.notna().sum()),
        }

    overall = {
        "artifact_version": "toy_mappo_smoke_matrix_combined_overall_v1_step93",
        "condition_ids": sorted(condition_manifests.keys()),
        "kpis": kpi_payload,
        "performance_claim_allowed": False,
        "smoke_evaluation_only": True,
        "claim_boundary": "toy MAPPO smoke matrix evaluation only; not a paper-level performance claim",
        "created_at_utc": utc_now(),
    }
    dump_json(combined_root / "kpi_overall.json", overall)

    return combined_root


def run_matrix_inspector(output_root: Path, combined_canonical_root: Path, conditions: List[str]) -> Path:
    inspection_root = output_root / "matrix_inspection"
    run_cmd(
        [
            sys.executable,
            str(TRAINING_DIR / "evaluation" / "inspect_toy_causal_a_family_kpis.py"),
            "--canonical-root",
            str(combined_canonical_root),
            "--output-root",
            str(inspection_root),
            "--conditions",
            ",".join(conditions),
        ],
        cwd=PROJECT_ROOT,
    )
    return inspection_root


def build_matrix_summary(
    output_root: Path,
    condition_manifests: Dict[str, Dict[str, Any]],
    combined_canonical_root: Path,
    inspection_root: Path,
) -> Path:
    rows: List[Dict[str, Any]] = []
    for condition_id, manifest in condition_manifests.items():
        rows.append(
            {
                "condition_id": condition_id,
                "reward_total_mean": manifest["reward_summary"]["reward_total_mean"],
                "reward_total_min": manifest["reward_summary"]["reward_total_min"],
                "reward_total_max": manifest["reward_summary"]["reward_total_max"],
                "raw_event_rows": manifest["raw_event_rows"],
                "window_rollup_rows": manifest["window_rollup_rows"],
                "reward_trace_rows": manifest["reward_trace_rows"],
                "performance_claim_allowed": manifest["performance_claim_allowed"],
                "smoke_evaluation_only": manifest["smoke_evaluation_only"],
                "causal_comparison_allowed": manifest["causal_comparison_allowed"],
                "canonical_root": manifest["canonical_root"],
                "inspection_root": manifest["inspection_root"],
            }
        )

    summary_path = output_root / "matrix_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    return summary_path


def run_matrix(args: argparse.Namespace) -> Dict[str, Any]:
    conditions = validate_conditions(parse_csv(args.conditions))

    output_root = Path(args.output_root)
    if args.clean:
        output_root = safe_rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    checkpoint_path = ensure_checkpoint(args, output_root)

    condition_manifests: Dict[str, Dict[str, Any]] = {}
    for condition_id in conditions:
        condition_manifests[condition_id] = run_condition_eval(
            checkpoint_path=checkpoint_path,
            condition_id=condition_id,
            args=args,
            output_root=output_root,
        )

    combined_canonical_root = combine_canonical_outputs(
        output_root=output_root,
        condition_manifests=condition_manifests,
    )
    inspection_root = run_matrix_inspector(
        output_root=output_root,
        combined_canonical_root=combined_canonical_root,
        conditions=conditions,
    )
    summary_path = build_matrix_summary(
        output_root=output_root,
        condition_manifests=condition_manifests,
        combined_canonical_root=combined_canonical_root,
        inspection_root=inspection_root,
    )

    kpi_by_window = pd.read_parquet(combined_canonical_root / "kpi_by_window.parquet")
    matrix_manifest = {
        "artifact_version": "toy_mappo_smoke_matrix_evaluation_v1_step93",
        "output_root": str(output_root),
        "checkpoint_path": str(checkpoint_path),
        "conditions": conditions,
        "seed": int(args.seed),
        "num_agents": int(args.num_agents),
        "steps": int(args.steps),
        "time_bands": parse_csv(args.time_bands),
        "windows_per_time_band": int(args.windows_per_time_band),
        "action_mode": str(args.action_mode),
        "reward_version": REWARD_VERSION,
        "reward_claim_boundary": REWARD_CLAIM_BOUNDARY,
        "trained_model": False,
        "performance_claim_allowed": False,
        "smoke_evaluation_only": True,
        "matrix_smoke_evaluation_only": True,
        "causal_comparison_allowed": True,
        "reward_metric_keys": PHASE2_12_KPIS,
        "condition_manifests": {
            condition_id: str(output_root / "conditions" / condition_id / "evaluation_manifest.json")
            for condition_id in conditions
        },
        "combined_canonical_root": str(combined_canonical_root),
        "matrix_inspection_root": str(inspection_root),
        "matrix_summary": str(summary_path),
        "kpi_by_window_rows": int(len(kpi_by_window)),
        "condition_count": int(len(conditions)),
        "claim_boundary": "toy MAPPO smoke matrix evaluation only; not a paper-level performance claim",
        "note": "Matrix evaluation validates smoke wiring across A/A90/A80/A70 only; do not use as paper-level performance.",
        "created_at_utc": utc_now(),
    }

    manifest_path = output_root / "matrix_evaluation_manifest.json"
    dump_json(manifest_path, matrix_manifest)

    print("[OK] Step 93 toy MAPPO smoke matrix evaluation completed")
    print(f"[OK] output_root                  : {output_root}")
    print(f"[OK] matrix_manifest              : {manifest_path}")
    print(f"[OK] matrix_summary               : {summary_path}")
    print(f"[OK] combined_canonical_root       : {combined_canonical_root}")
    print(f"[OK] matrix_inspection_root        : {inspection_root}")
    print(f"[OK] conditions                   : {conditions}")
    print(f"[OK] kpi_by_window_rows           : {matrix_manifest['kpi_by_window_rows']}")
    print(f"[OK] reward_version               : {REWARD_VERSION}")
    print("[OK] performance_claim_allowed    : False")
    print("[OK] smoke_evaluation_only        : True")

    return matrix_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="artifacts/phase2_toy_mappo_smoke_matrix_eval")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--conditions", default="A,A90,A80,A70")
    parser.add_argument("--seed", type=int, default=93)
    parser.add_argument("--num-agents", type=int, default=8)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--time-bands", default="peak,offpeak,night")
    parser.add_argument("--windows-per-time-band", type=int, default=1)
    parser.add_argument("--action-mode", default="argmax", choices=["argmax", "sample"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_matrix(args)


if __name__ == "__main__":
    main()
