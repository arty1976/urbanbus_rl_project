from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def main() -> None:
    repo_root = Path.cwd()
    generator = repo_root / "05_training" / "observability" / "h200_nvidia_smi_gpu_monitor_step154.py"
    validator = repo_root / "05_training" / "observability" / "validate_h200_nvidia_smi_gpu_monitor_step154.py"

    if not generator.exists():
        raise SystemExit(f"generator not found: {generator}")
    if not validator.exists():
        raise SystemExit(f"validator not found: {validator}")

    with tempfile.TemporaryDirectory(prefix="step154_gpu_monitor_selftest_") as tmp:
        tmp_path = Path(tmp)
        out = tmp_path / "artifacts" / "observability" / "h200_nvidia_smi_gpu_monitor_step154_selftest"

        p = run([
            sys.executable,
            str(generator),
            "--project-root",
            str(repo_root),
            "--output-root",
            str(out),
        ], cwd=repo_root)
        if p.returncode != 0:
            print(p.stdout)
            print(p.stderr)
            raise SystemExit("[FAIL] generator self-test run failed")

        manifest = out / "h200_nvidia_smi_gpu_monitor_step154_manifest.json"
        if not manifest.exists():
            raise SystemExit("[FAIL] manifest not created in self-test")

        v = run([
            sys.executable,
            str(validator),
            "--manifest",
            str(manifest),
        ], cwd=repo_root)
        if v.returncode != 0:
            print(v.stdout)
            print(v.stderr)
            raise SystemExit("[FAIL] validator rejected valid self-test manifest")

        # Expected failure 1: train_allowed=true must be rejected.
        bad_dir = tmp_path / "bad_train_allowed"
        shutil.copytree(out, bad_dir)
        bad_manifest = bad_dir / "h200_nvidia_smi_gpu_monitor_step154_manifest.json"
        payload = json.loads(bad_manifest.read_text(encoding="utf-8"))
        payload["train_allowed"] = True
        bad_manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        bad = run([
            sys.executable,
            str(validator),
            "--manifest",
            str(bad_manifest),
        ], cwd=repo_root)
        if bad.returncode == 0:
            raise SystemExit("[FAIL] validator accepted train_allowed=true")
        print("[OK] expected failure: train_allowed=true rejected")

        # Expected failure 2: required GPU metric column missing must be rejected.
        bad_dir2 = tmp_path / "bad_missing_column"
        shutil.copytree(out, bad_dir2)
        bad_manifest2 = bad_dir2 / "h200_nvidia_smi_gpu_monitor_step154_manifest.json"
        payload2 = json.loads(bad_manifest2.read_text(encoding="utf-8"))
        csv_path = Path(payload2["output_files"]["gpu_metrics_csv"])
        relocated_csv = bad_dir2 / "gpu_metrics_step154_sample.csv"
        if csv_path != relocated_csv:
            # Manifest generated with absolute temp path. Copytree preserves internal files,
            # so use the copied file path and update manifest references.
            payload2["output_files"]["gpu_metrics_csv"] = str(relocated_csv)
            payload2["output_files"]["gpu_metrics_jsonl"] = str(bad_dir2 / "gpu_metrics_step154_sample.jsonl")
            payload2["output_files"]["gpu_monitor_status"] = str(bad_dir2 / "gpu_monitor_status_step154.json")
            payload2["output_files"]["nvidia_smi_query_contract"] = str(bad_dir2 / "nvidia_smi_query_contract_step154.json")
            payload2["output_files"]["manifest"] = str(bad_manifest2)

        text = relocated_csv.read_text(encoding="utf-8").splitlines()
        header = text[0].split(",")
        if "nvidia_smi_status" not in header:
            raise SystemExit("[FAIL] self-test setup expected nvidia_smi_status column")
        idx = header.index("nvidia_smi_status")
        header.pop(idx)
        new_lines = [",".join(header)]
        for line in text[1:]:
            parts = line.split(",")
            if len(parts) > idx:
                parts.pop(idx)
            new_lines.append(",".join(parts))
        relocated_csv.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        bad_manifest2.write_text(json.dumps(payload2, ensure_ascii=False, indent=2), encoding="utf-8")

        bad2 = run([
            sys.executable,
            str(validator),
            "--manifest",
            str(bad_manifest2),
        ], cwd=repo_root)
        if bad2.returncode == 0:
            raise SystemExit("[FAIL] validator accepted missing nvidia_smi_status column")
        print("[OK] expected failure: missing nvidia_smi_status column rejected")

        # Expected failure 3: live_mutation_allowed=true in row must be rejected.
        bad_dir3 = tmp_path / "bad_live_mutation"
        shutil.copytree(out, bad_dir3)
        bad_manifest3 = bad_dir3 / "h200_nvidia_smi_gpu_monitor_step154_manifest.json"
        payload3 = json.loads(bad_manifest3.read_text(encoding="utf-8"))
        csv3 = bad_dir3 / "gpu_metrics_step154_sample.csv"
        jsonl3 = bad_dir3 / "gpu_metrics_step154_sample.jsonl"
        payload3["output_files"]["gpu_metrics_csv"] = str(csv3)
        payload3["output_files"]["gpu_metrics_jsonl"] = str(jsonl3)
        payload3["output_files"]["gpu_monitor_status"] = str(bad_dir3 / "gpu_monitor_status_step154.json")
        payload3["output_files"]["nvidia_smi_query_contract"] = str(bad_dir3 / "nvidia_smi_query_contract_step154.json")
        payload3["output_files"]["manifest"] = str(bad_manifest3)
        rows = csv3.read_text(encoding="utf-8").splitlines()
        header = rows[0].split(",")
        live_idx = header.index("live_mutation_allowed")
        parts = rows[1].split(",")
        parts[live_idx] = "True"
        rows[1] = ",".join(parts)
        csv3.write_text("\n".join(rows) + "\n", encoding="utf-8")

        jlines = []
        for line in jsonl3.read_text(encoding="utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)
                obj["live_mutation_allowed"] = True
                jlines.append(json.dumps(obj, ensure_ascii=False, sort_keys=True))
        jsonl3.write_text("\n".join(jlines) + "\n", encoding="utf-8")
        bad_manifest3.write_text(json.dumps(payload3, ensure_ascii=False, indent=2), encoding="utf-8")

        bad3 = run([
            sys.executable,
            str(validator),
            "--manifest",
            str(bad_manifest3),
        ], cwd=repo_root)
        if bad3.returncode == 0:
            raise SystemExit("[FAIL] validator accepted live_mutation_allowed=true row")
        print("[OK] expected failure: live_mutation_allowed=true row rejected")

    print("[OK] Step 154 H200 nvidia-smi GPU monitor self-test PASS")


if __name__ == "__main__":
    main()
