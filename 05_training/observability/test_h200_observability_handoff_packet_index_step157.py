from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
GENERATOR = SCRIPT_DIR / "h200_observability_handoff_packet_index_step157.py"
VALIDATOR = SCRIPT_DIR / "validate_h200_observability_handoff_packet_index_step157.py"


def run(cmd, expect_ok=True):
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if expect_ok and proc.returncode != 0:
        print(proc.stdout)
        raise RuntimeError(f"command failed: {cmd}")
    if (not expect_ok) and proc.returncode == 0:
        print(proc.stdout)
        raise RuntimeError(f"command unexpectedly succeeded: {cmd}")
    return proc


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    out = PROJECT_ROOT / "artifacts" / "observability" / "h200_observability_handoff_packet_index_step157_selftest"
    if out.exists():
        shutil.rmtree(out)

    run([sys.executable, str(GENERATOR), "--project-root", str(PROJECT_ROOT), "--output-root", str(out)])
    manifest = out / "h200_observability_handoff_packet_index_step157_manifest.json"
    run([sys.executable, str(VALIDATOR), "--manifest", str(manifest)])

    bad = load(manifest)
    bad["train_allowed"] = True
    bad_manifest = out / "bad_train_allowed_manifest.json"
    dump(bad_manifest, bad)
    run([sys.executable, str(VALIDATOR), "--manifest", str(bad_manifest)], expect_ok=False)
    print("[OK] expected failure: train_allowed=true rejected")

    bad = load(manifest)
    bad["live_mutation_allowed"] = True
    bad_manifest = out / "bad_live_mutation_manifest.json"
    dump(bad_manifest, bad)
    run([sys.executable, str(VALIDATOR), "--manifest", str(bad_manifest)], expect_ok=False)
    print("[OK] expected failure: live_mutation_allowed=true rejected")

    bad = load(manifest)
    bad["packet_status"] = "RELEASED"
    bad_manifest = out / "bad_packet_status_manifest.json"
    dump(bad_manifest, bad)
    run([sys.executable, str(VALIDATOR), "--manifest", str(bad_manifest)], expect_ok=False)
    print("[OK] expected failure: wrong packet status rejected")

    bad = load(manifest)
    bad["output_files"]["component_index_csv"] = str(out / "missing_component_index.csv")
    bad_manifest = out / "bad_missing_component_index_manifest.json"
    dump(bad_manifest, bad)
    run([sys.executable, str(VALIDATOR), "--manifest", str(bad_manifest)], expect_ok=False)
    print("[OK] expected failure: missing component index rejected")

    print("[OK] Step 157 H200 observability handoff packet index self-test PASS")
    print(f"[OK] manifest: {manifest}")


if __name__ == "__main__":
    main()
