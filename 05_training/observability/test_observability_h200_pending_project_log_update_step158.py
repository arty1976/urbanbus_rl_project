from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd, cwd: Path, expect_ok: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    if expect_ok and proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"[FAIL] command failed unexpectedly: {' '.join(cmd)}")
    if not expect_ok and proc.returncode == 0:
        print(proc.stdout)
        raise SystemExit(f"[FAIL] command succeeded unexpectedly: {' '.join(cmd)}")
    return proc


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def make_min_project(src_root: Path, tmp_root: Path) -> Path:
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    project = tmp_root / "project"
    obs = project / "05_training" / "observability"
    obs.mkdir(parents=True, exist_ok=True)
    (project / "project_log.md").write_text("# Project Log\n", encoding="utf-8")
    for name in [
        "observability_h200_pending_project_log_update_step158.py",
        "validate_observability_h200_pending_project_log_update_step158.py",
    ]:
        shutil.copy2(src_root / "05_training" / "observability" / name, obs / name)
    return project


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()

    src_root = Path(args.project_root).resolve()
    tmp_root = src_root / "artifacts" / "observability" / "step158_selftest_tmp"
    project = make_min_project(src_root, tmp_root)

    gen = project / "05_training" / "observability" / "observability_h200_pending_project_log_update_step158.py"
    val = project / "05_training" / "observability" / "validate_observability_h200_pending_project_log_update_step158.py"
    manifest = project / "artifacts" / "observability" / "observability_h200_pending_project_log_update_step158" / "observability_h200_pending_project_log_update_step158_manifest.json"

    run([sys.executable, str(gen), "--project-root", str(project)], cwd=project)
    run([sys.executable, str(val), "--project-root", str(project), "--manifest", str(manifest)], cwd=project)

    payload = load_json(manifest)
    if payload.get("train_allowed") is not False:
        raise SystemExit("[FAIL] train_allowed should be false")
    if payload.get("block_reason") != "H200_SERVER_NOT_AVAILABLE_YET":
        raise SystemExit("[FAIL] block_reason mismatch")

    # Expected failure: tamper train_allowed true.
    tampered = dict(payload)
    tampered["train_allowed"] = True
    tampered_path = manifest.parent / "tampered_train_allowed_true.json"
    tampered_path.write_text(json.dumps(tampered, ensure_ascii=False, indent=2), encoding="utf-8")
    run([sys.executable, str(val), "--project-root", str(project), "--manifest", str(tampered_path)], cwd=project, expect_ok=False)
    print("[OK] expected failure: train_allowed=true rejected")

    # Expected failure: tamper block reason.
    tampered = dict(payload)
    tampered["block_reason"] = "LOCAL_FAKE_H200_RESULT_ALLOWED"
    tampered_path = manifest.parent / "tampered_block_reason.json"
    tampered_path.write_text(json.dumps(tampered, ensure_ascii=False, indent=2), encoding="utf-8")
    run([sys.executable, str(val), "--project-root", str(project), "--manifest", str(tampered_path)], cwd=project, expect_ok=False)
    print("[OK] expected failure: wrong block_reason rejected")

    # Expected failure: tamper section by removing no-substitution phrase.
    tampered = dict(payload)
    section_path = Path(tampered["output_files"]["project_log_step158_section"])
    bad_section = section_path.with_name("bad_section.md")
    text = section_path.read_text(encoding="utf-8")
    text = text.replace("No local placeholder, mock, expected checklist, or simulated GPU result may be substituted", "placeholder substitution allowed")
    bad_section.write_text(text, encoding="utf-8")
    tampered["output_files"] = dict(tampered["output_files"])
    tampered["output_files"]["project_log_step158_section"] = str(bad_section)
    tampered_path = manifest.parent / "tampered_bad_section.json"
    tampered_path.write_text(json.dumps(tampered, ensure_ascii=False, indent=2), encoding="utf-8")
    run([sys.executable, str(val), "--project-root", str(project), "--manifest", str(tampered_path)], cwd=project, expect_ok=False)
    print("[OK] expected failure: substitution-permissive section rejected")

    print("[OK] Step 158 observability + H200 pending project log update self-test PASS")
    print(f"[OK] manifest: {manifest}")
    print(f"[OK] status  : {payload.get('update_status')}")
    print(f"[OK] pending : {payload.get('h200_pending_status')}")


if __name__ == "__main__":
    main()
