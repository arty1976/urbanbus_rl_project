from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


SECTION_MARKER = "<!-- STEP130_REWARD_PIPELINE_STATUS_START -->"
SECTION_END = "<!-- STEP130_REWARD_PIPELINE_STATUS_END -->"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json_any_encoding(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


def build_section(status: Dict[str, Any]) -> str:
    covered_lines = "\n".join(f"- {item}" for item in status["covered_steps"])
    guard_lines = "\n".join(
        f"- `{key}` = `{str(value).lower()}`"
        for key, value in status["guard_flags"].items()
    )

    return f"""{SECTION_MARKER}
## Step 130 ??Reward pipeline status update

?묒꽦 ?쒓컖(UTC): {utc_now()}

### 紐⑹쟻

Step 111遺??Step 129源뚯? ?댁뼱吏?reward ?ㅺ퀎쨌寃利씲룹떎?됱?鍮?pipeline??project log??怨좎젙?쒕떎. ??湲곕줉? reward ?꾨낫援곗씠 ?뺤쓽?섏뿀?붾씪???꾩쭅 ?ㅼ젣 reward ablation 寃곌낵媛 ?녾퀬, ?곕씪??winner ?좏깮쨌reward ?밴꺽쨌MAPPO ?숈뒿 ?덉슜??紐⑤몢 湲덉? ?곹깭?꾩쓣 紐낇솗???④린湲??꾪븳 寃껋씠??

### ?ы븿???④퀎

{covered_lines}

### ?꾩옱 reward pipeline ?곹깭

- reward spec: `{status["reward_pipeline_current_state"]["reward_spec_status"]}`
- candidate matrix: `{status["reward_pipeline_current_state"]["candidate_matrix"]}`
- normalization reference: `{status["reward_pipeline_current_state"]["normalization_reference"]}`
- hard constraints: `{status["reward_pipeline_current_state"]["hard_constraints"]}`
- dry-run plan rows: `{status["reward_pipeline_current_state"]["dry_run_plan_rows"]}`
- execution manifest rows: `{status["reward_pipeline_current_state"]["execution_manifest_rows"]}`
- sandbox runner: `{status["reward_pipeline_current_state"]["sandbox_runner"]}`
- actual result ingestion: `{status["reward_pipeline_current_state"]["actual_result_ingestion"]}`
- promotion decision: `{status["reward_pipeline_current_state"]["promotion_decision"]}`
- runbook: `{status["reward_pipeline_current_state"]["runbook"]}`

### Reward ablation matrix

- candidates: `{", ".join(status["candidate_ids"])}`
- conditions: `{", ".join(status["condition_ids"])}`
- seeds: `{", ".join(str(x) for x in status["seeds"])}`
- expected actual ablation runs: `{status["expected_actual_ablation_runs"]}`

### ?좎??섎뒗 guard flags

{guard_lines}

### ?댁꽍

?꾩옱源뚯????묒뾽? reward ?꾨낫 ?ㅺ퀎, ?꾨낫援?鍮꾧탳 怨꾪쉷, 寃곌낵 schema, dry-run plan, execution manifest, no-op guard, actual-result ingestion preflight, promotion decision package, actual ablation runbook??以鍮꾪븳 寃껋씠?? ?섏?留??ㅼ젣 reward ablation 寃곌낵???꾩쭅 議댁옱?섏? ?딅뒗??

?곕씪???ㅼ쓬 二쇱옣? 紐⑤몢 湲덉??쒕떎.

- R0~R5 以??뱀젙 reward媛 理쒓퀬?쇰뒗 二쇱옣
- trainable reward媛 ?밴꺽?섏뿀?ㅻ뒗 二쇱옣
- ?꾩옱 reward ?꾨낫濡?MAPPO ?숈뒿???쒖옉?대룄 ?쒕떎??二쇱옣
- ?쇰Ц ?섏? ?깅뒫 媛쒖꽑 二쇱옣
- causal performance claim

### ?ㅼ쓬 ?곹깭

`{status["next_recommended_status"]}`

{SECTION_END}
"""


def update_project_log(project_log: Path, status: Dict[str, Any]) -> Dict[str, Any]:
    project_log.parent.mkdir(parents=True, exist_ok=True)
    if project_log.exists():
        text = project_log.read_text(encoding="utf-8-sig")
    else:
        text = "# urbanbus_rl_project project log\n\n"

    new_section = build_section(status)

    if SECTION_MARKER in text and SECTION_END in text:
        before = text.split(SECTION_MARKER)[0].rstrip()
        after = text.split(SECTION_END, 1)[1].lstrip()
        updated = before + "\n\n" + new_section + "\n\n" + after
        action = "replaced_existing_step130_section"
    else:
        updated = text.rstrip() + "\n\n" + new_section + "\n"
        action = "appended_step130_section"

    project_log.write_text(updated, encoding="utf-8")
    return {
        "project_log_path": str(project_log),
        "action": action,
        "section_marker": SECTION_MARKER,
        "section_end": SECTION_END,
        "updated": True,
    }


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def resolve_path(project_root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return project_root / p


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", default="05_training/rewards/reward_pipeline_project_log_update_step130.json")
    parser.add_argument("--project-log", default="project_log.md")
    parser.add_argument("--output-root", default="artifacts/rewards/reward_pipeline_project_log_update_step130")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    status_path = resolve_path(project_root, args.status)
    project_log = resolve_path(project_root, args.project_log)
    output_root = resolve_path(project_root, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    status = load_json_any_encoding(status_path)
    update_result = update_project_log(project_log, status)

    manifest = {
        "artifact_version": "reward_pipeline_project_log_update_step130_manifest_v1",
        "created_at_utc": utc_now(),
        "log_update_status": "PROJECT_LOG_UPDATED",
        "project_log_path": str(project_log),
        "actual_results": False,
        "winner_selected": False,
        "trainable_reward_promoted": False,
        "train_with_this_reward_allowed": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "update_result": update_result,
        "source_status_snapshot": status,
    }

    manifest_path = output_root / "reward_pipeline_project_log_update_step130_manifest.json"
    dump_json(manifest_path, manifest)

    print("[OK] Step 130 reward pipeline project log update completed")
    print(f"[OK] log_update_status: {manifest['log_update_status']}")
    print(f"[OK] project_log     : {project_log}")
    print(f"[OK] action          : {update_result['action']}")
    print(f"[OK] actual_results  : {manifest['actual_results']}")
    print(f"[OK] winner_selected : {manifest['winner_selected']}")
    print(f"[OK] promoted        : {manifest['trainable_reward_promoted']}")
    print(f"[OK] train_allowed   : {manifest['train_with_this_reward_allowed']}")
    print(f"[OK] manifest        : {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
