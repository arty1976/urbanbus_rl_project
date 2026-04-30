from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], expect_ok: bool = True) -> subprocess.CompletedProcess[str]:
    cp = subprocess.run(cmd, text=True, capture_output=True)
    if expect_ok and cp.returncode != 0:
        print(cp.stdout)
        print(cp.stderr, file=sys.stderr)
        raise RuntimeError(f"command failed: {' '.join(cmd)}")
    if not expect_ok and cp.returncode == 0:
        print(cp.stdout)
        print(cp.stderr, file=sys.stderr)
        raise RuntimeError(f"command unexpectedly passed: {' '.join(cmd)}")
    return cp


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-test Step 155 H200 observability dashboard index/status page")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    generator = project_root / "05_training" / "observability" / "h200_observability_dashboard_index_step155.py"
    validator = project_root / "05_training" / "observability" / "validate_h200_observability_dashboard_index_step155.py"
    out_root = project_root / "artifacts" / "observability" / "h200_observability_dashboard_index_step155_selftest"
    manifest = out_root / "h200_observability_dashboard_index_step155_manifest.json"

    run([
        sys.executable,
        str(generator),
        "--project-root",
        str(project_root),
        "--output-root",
        str(out_root),
    ])
    run([sys.executable, str(validator), "--manifest", str(manifest)])

    payload = load_json(manifest)

    bad_train = copy.deepcopy(payload)
    bad_train["train_allowed"] = True
    bad_train["locks"]["train_allowed"] = True
    bad_train_path = out_root / "negative_train_allowed_true.json"
    write_json(bad_train_path, bad_train)
    run([sys.executable, str(validator), "--manifest", str(bad_train_path)], expect_ok=False)
    print("[OK] expected failure: train_allowed=true rejected")

    bad_live = copy.deepcopy(payload)
    bad_live["live_mutation_allowed"] = True
    bad_live["locks"]["live_mutation_allowed"] = True
    bad_live_path = out_root / "negative_live_mutation_allowed_true.json"
    write_json(bad_live_path, bad_live)
    run([sys.executable, str(validator), "--manifest", str(bad_live_path)], expect_ok=False)
    print("[OK] expected failure: live_mutation_allowed=true rejected")

    bad_components = copy.deepcopy(payload)
    bad_components["components"] = [c for c in bad_components["components"] if str(c.get("step")) != "154"]
    bad_components_path = out_root / "negative_missing_step154_component.json"
    write_json(bad_components_path, bad_components)
    run([sys.executable, str(validator), "--manifest", str(bad_components_path)], expect_ok=False)
    print("[OK] expected failure: missing Step 154 component rejected")

    bad_hard = copy.deepcopy(payload)
    bad_hard["hard_failures"] = [{"code": "fake", "message": "must fail"}]
    bad_hard_path = out_root / "negative_hard_failure_present.json"
    write_json(bad_hard_path, bad_hard)
    run([sys.executable, str(validator), "--manifest", str(bad_hard_path)], expect_ok=False)
    print("[OK] expected failure: hard failure present rejected")

    print("[OK] Step 155 H200 observability dashboard index/status page self-test PASS")
    print(f"[OK] manifest: {manifest}")
    print(f"[OK] status  : {payload.get('index_status')}")
    print(f"[OK] control : {payload.get('control_policy')}")


if __name__ == "__main__":
    main()
