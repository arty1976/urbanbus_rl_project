from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Optional, List

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from h200_tensorboard_jsonl_logger_step153 import (  # noqa: E402
    H200ExperimentMetricLogger,
    ObservabilityRunContext,
    emit_dry_run_sample,
    get_git_commit,
)
from validate_h200_tensorboard_jsonl_logger_step153 import (  # noqa: E402
    ValidationError,
    validate_manifest,
)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def expect_failure(label: str, fn) -> None:
    try:
        fn()
    except Exception:
        print(f"[OK] expected failure: {label}")
        return
    raise RuntimeError(f"expected failure did not occur: {label}")


def run_selftest(project_root: Path) -> None:
    output_root = project_root / "artifacts" / "observability" / "h200_tensorboard_jsonl_logger_step153_selftest"
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    context = ObservabilityRunContext(
        run_id="step153_selftest_A_R0_seed001",
        condition_id="A",
        reward_id="R0",
        seed=1,
        project_root=str(project_root),
        output_root=str(output_root),
        git_commit=get_git_commit(project_root),
    )
    logger = H200ExperimentMetricLogger(context=context, enable_tensorboard=False)
    emit_dry_run_sample(logger)

    expect_failure(
        "live mutation attempt is blocked",
        lambda: logger.record_control_mutation_attempt("learning_rate", 0.001, "self-test"),
    )
    manifest = logger.finalize()
    manifest_path = output_root / "h200_tensorboard_jsonl_logger_step153_manifest.json"
    validate_manifest(manifest_path)

    # Mutated manifest must fail if train_allowed becomes true.
    bad_manifest_path = output_root / "bad_train_allowed_manifest.json"
    bad = load_json(manifest_path)
    bad["train_allowed"] = True
    dump_json(bad_manifest_path, bad)
    expect_failure("train_allowed=true rejected", lambda: validate_manifest(bad_manifest_path))

    # Mutated JSONL must fail if a required metric group is missing.
    bad_group_root = output_root / "bad_missing_metric_group"
    shutil.copytree(output_root, bad_group_root, ignore=shutil.ignore_patterns("bad_*", "bad_missing_metric_group"))
    bad_group_manifest_path = bad_group_root / "h200_tensorboard_jsonl_logger_step153_manifest.json"
    bad_group_manifest = load_json(bad_group_manifest_path)
    metrics_path = Path(bad_group_manifest["output_files"]["metrics_events_jsonl"])
    if not metrics_path.is_absolute():
        metrics_path = bad_group_manifest_path.parent / metrics_path
    rows = []
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event_type") == "scalar" and event.get("metric_group") == "kpi":
                continue
            rows.append(event)
    with metrics_path.open("w", encoding="utf-8") as f:
        for event in rows:
            f.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    expect_failure("missing kpi metric group rejected", lambda: validate_manifest(bad_group_manifest_path))

    print("[OK] Step 153 H200 TensorBoard + JSONL logger self-test PASS")
    print(f"[OK] manifest: {manifest_path}")
    print(f"[OK] status  : {manifest['integration_status']}")
    print(f"[OK] control : {manifest['control_policy']}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Self-test Step 153 H200 TensorBoard + JSONL logger integration")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args(argv)
    run_selftest(Path(args.project_root).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
