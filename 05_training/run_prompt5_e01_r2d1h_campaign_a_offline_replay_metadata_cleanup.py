from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

import run_prompt5_e01_r2d1h_campaign_a_live_observation_actual as base


PROJECT_ROOT_DEFAULT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
SOURCE_REPAIR_DEFAULT = (
    PROJECT_ROOT_DEFAULT
    / "05_training/artifacts/prompt5_e01_r2d1h_campaign_a_offline_replay_repair_20260725_164738"
)
AUDIT_MD_DEFAULT = Path("/Users/arty/Downloads/r2d1h_offline_repair_independent_audit_20260725.md")
OUTPUT_PREFIX = "prompt5_e01_r2d1h_campaign_a_offline_replay_metadata_cleanup"

EXTRA_REQUIRED = [
    "metadata_cleanup_audit.json",
    "source_repair_artifact_reference.json",
]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    base.dump_json(path, payload)


def rewrite_manifest(output_root: Path) -> None:
    previous_manifest = load_json(output_root / "prompt5_e01_r2d1h_manifest.json")
    names = {row["path"] for row in previous_manifest.get("files", []) if row.get("path")}
    names.update(EXTRA_REQUIRED)
    names.add("prompt5_e01_r2d1h_manifest.json")
    files = []
    for name in sorted(names):
        path = output_root / name
        if name == "prompt5_e01_r2d1h_manifest.json":
            files.append(
                {
                    "path": name,
                    "exists": True,
                    "sha256": None,
                    "self_hash_exempt": True,
                    "self_hash_exemption_reason": "Stable self-hashing is not possible after final serialization.",
                }
            )
        else:
            files.append(
                {
                    "path": name,
                    "exists": path.exists(),
                    "sha256": base.sha256_file(path) if path.exists() else None,
                    "self_hash_exempt": False,
                }
            )
    missing = [row["path"] for row in files if not row["exists"]]
    write_json(
        output_root / "prompt5_e01_r2d1h_manifest.json",
        {
            "artifact_name": OUTPUT_PREFIX,
            "created_at": base.iso(),
            "files": files,
            "missing_required_file_count": len(missing),
            "missing_required_files": missing,
            "manifest_self_entry_exists": True,
            "manifest_self_hash_exempt": True,
            "source_manifest": str(previous_manifest.get("artifact_name")),
        },
    )


def update_episode_statuses(output_root: Path) -> List[Dict[str, Any]]:
    new_status = "RIGHT_CENSORED_CAMPAIGN_HARD_CAP_STOP"
    root_path = output_root / "campaign_a_terminal_recovery_episodes.parquet"
    episodes = pd.read_parquet(root_path)
    mask = (episodes["right_censored"] == True) & (episodes["episode_status"].astype(str) == "RIGHT_CENSORED_VEHICLE_RESPONSE_LOSS")
    changes = []
    for _, row in episodes[mask].iterrows():
        changes.append(
            {
                "episode_id": row["episode_id"],
                "route_id": row["route_id"],
                "vehicle_id": row["vehicle_id"],
                "old_episode_status": row["episode_status"],
                "new_episode_status": new_status,
                "reason": "No exact-ID low-sequence reentry before campaign stop at effective campaign hard-cap threshold.",
            }
        )
    episodes.loc[mask, "episode_status"] = new_status
    episodes.to_parquet(root_path, index=False)
    for route in base.TARGET_ROUTES:
        route_path = output_root / "terminal_recovery_evidence" / route / "terminal_recovery_episodes.parquet"
        if route_path.exists():
            route_episodes = episodes[episodes["route_id"].astype(str) == route]
            route_episodes.to_parquet(route_path, index=False)
    return changes


def update_counter(output_root: Path, reclass: Dict[str, Any]) -> Dict[str, Any]:
    counter_path = output_root / "terminal_counter_audit_v9.json"
    counter = load_json(counter_path)
    episodes = pd.read_parquet(output_root / "campaign_a_terminal_recovery_episodes.parquet")
    confirmation_by_route = {
        route: int(episodes[episodes["route_id"].astype(str) == route]["observed_post_terminal_confirmation_sample_count"].fillna(0).astype(int).sum())
        for route in base.TARGET_ROUTES
    }
    disappeared_by_route = {route: 0 for route in base.TARGET_ROUTES}
    for row in reclass.get("reclassifications", []):
        route = str(row.get("route_id"))
        disappeared_by_route[route] = disappeared_by_route.get(route, 0) + int(row.get("missing_route_responses_between_terminal_and_reentry") or 0)

    total_confirmation = sum(confirmation_by_route.values())
    total_disappeared = sum(disappeared_by_route.values())
    counter["total"]["post_terminal_confirmation_observation_count"] = total_confirmation
    counter["total"]["observed_post_terminal_confirmation_sample_count"] = total_confirmation
    counter["total"]["disappeared_waiting_reentry_observation_count"] = total_disappeared
    counter["total"]["post_terminal_wait_observation_count_definition"] = "Vehicle-present POST_TERMINAL_WAIT samples only; route responses where the exact post-terminal vehicle is absent before reentry are counted in disappeared_waiting_reentry_observation_count."
    for route in base.TARGET_ROUTES:
        counter["by_route"][route]["post_terminal_confirmation_observation_count"] = confirmation_by_route[route]
        counter["by_route"][route]["observed_post_terminal_confirmation_sample_count"] = confirmation_by_route[route]
        counter["by_route"][route]["disappeared_waiting_reentry_observation_count"] = disappeared_by_route.get(route, 0)
        counter["by_route"][route]["post_terminal_wait_observation_count_definition"] = counter["total"]["post_terminal_wait_observation_count_definition"]
    write_json(counter_path, counter)
    return {
        "confirmation_observation_count_total": total_confirmation,
        "confirmation_observation_count_by_route": confirmation_by_route,
        "disappeared_waiting_reentry_observation_count_total": total_disappeared,
        "disappeared_waiting_reentry_observation_count_by_route": disappeared_by_route,
    }


def update_transition_and_reclass(output_root: Path, status_changes: List[Dict[str, Any]]) -> None:
    by_episode = {row["episode_id"]: row["new_episode_status"] for row in status_changes}
    transition_path = output_root / "campaign_state_transition_audit.json"
    transitions = load_json(transition_path)
    for row in transitions.get("transitions", []):
        if row.get("episode_id") in by_episode:
            row["final_status"] = by_episode[row["episode_id"]]
    write_json(transition_path, transitions)

    reclass_path = output_root / "episode_reclassification_audit.json"
    reclass = load_json(reclass_path)
    for row in reclass.get("reclassifications", []):
        if row.get("episode_id") in by_episode:
            row["repaired_episode_status"] = by_episode[row["episode_id"]]
            row["metadata_cleanup_status_reason_updated"] = True
    write_json(reclass_path, reclass)


def update_gate_and_report(output_root: Path, cleanup: Dict[str, Any]) -> None:
    gate_path = output_root / "prompt5_e01_r2d1h_gate.json"
    gate = load_json(gate_path)
    gate["metadata_cleanup_applied"] = True
    gate["metadata_cleanup_api_calls_made"] = 0
    gate["right_censor_reason_cleanup_applied"] = True
    gate["confirmation_observation_count_total"] = cleanup["confirmation_observation_count_total"]
    gate["disappeared_waiting_reentry_observation_count_total"] = cleanup["disappeared_waiting_reentry_observation_count_total"]
    write_json(gate_path, gate)

    report_path = output_root / "prompt5_e01_r2d1h_final_report.md"
    report = report_path.read_text(encoding="utf-8")
    insert = "\n".join(
        [
            "",
            "## Metadata Cleanup",
            "",
            "- metadata_cleanup_applied: true",
            "- api_calls_made_by_cleanup: 0",
            f"- confirmation_observation_count_total: {cleanup['confirmation_observation_count_total']}",
            f"- disappeared_waiting_reentry_observation_count_total: {cleanup['disappeared_waiting_reentry_observation_count_total']}",
            "- right_censor_reason: RIGHT_CENSORED_CAMPAIGN_HARD_CAP_STOP for unrecovered terminal episodes without exact-ID low-sequence reentry before campaign stop",
        ]
    )
    if "## Metadata Cleanup" not in report:
        report = report.rstrip() + insert + "\n"
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="API-free metadata cleanup for R2D-1H offline replay repair.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT_DEFAULT)
    parser.add_argument("--source-repair-artifact", type=Path, default=SOURCE_REPAIR_DEFAULT)
    parser.add_argument("--audit-md", type=Path, default=AUDIT_MD_DEFAULT)
    parser.add_argument("--timestamp", default=base.now_stamp())
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    source = args.source_repair_artifact.resolve()
    output_root = project_root / base.ARTIFACTS_REL / f"{OUTPUT_PREFIX}_{args.timestamp}"
    shutil.copytree(source, output_root)

    reclass = load_json(output_root / "episode_reclassification_audit.json")
    status_changes = update_episode_statuses(output_root)
    counter_cleanup = update_counter(output_root, reclass)
    update_transition_and_reclass(output_root, status_changes)
    cleanup = {
        "metadata_cleanup_completed": True,
        "api_calls_made_by_cleanup": 0,
        "source_repair_artifact": str(source),
        "independent_audit_path": str(args.audit_md),
        "independent_audit_sha256": base.sha256_file(args.audit_md) if args.audit_md.exists() else None,
        "status_changes": status_changes,
        **counter_cleanup,
        "scientific_episode_reconstruction_changed": False,
        "route_level_counts_changed": False,
        "gate_changed": False,
        "gate_status": load_json(output_root / "prompt5_e01_r2d1h_gate.json").get("gate_status"),
    }
    write_json(output_root / "metadata_cleanup_audit.json", cleanup)
    write_json(
        output_root / "source_repair_artifact_reference.json",
        {
            "source_repair_artifact": str(source),
            "source_repair_manifest_sha256": base.sha256_file(source / "prompt5_e01_r2d1h_manifest.json"),
            "source_repair_scientific_reconstruction_valid": True,
            "source_repair_authoritative_metadata_superseded_by_cleanup": True,
        },
    )
    update_gate_and_report(output_root, cleanup)
    rewrite_manifest(output_root)

    note = source / f"SUPERSEDED_BY_METADATA_CLEANUP_{args.timestamp}.md"
    note.write_text(
        "\n".join(
            [
                "# Superseded By Metadata Cleanup",
                "",
                f"- superseded_at: {base.iso()}",
                f"- superseded_by: {output_root}",
                "- scope: authoritative metadata only",
                "- scientific_episode_reconstruction_status: valid and unchanged",
                "- reason: independent audit requested API-free cleanup of confirmation counters, temporary-disappearance counters, and unrecovered right-censor reasons.",
                "- api_calls_made_for_cleanup: 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("R2D-1H OFFLINE REPLAY METADATA CLEANUP COMPLETE")
    print(output_root)
    print(cleanup["gate_status"])


if __name__ == "__main__":
    main()
