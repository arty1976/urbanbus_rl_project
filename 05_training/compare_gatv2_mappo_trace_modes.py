from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional

# --- H4M-AE-R9.8 fail-closed simulator authorization -------------------------------
import sys as _authz_sys
from pathlib import Path as _AuthzPath

for _authz_dir in (_AuthzPath(__file__).resolve().parent, _AuthzPath(__file__).resolve().parent.parent):
    if (_authz_dir / "simulator_authorization.py").exists():
        if str(_authz_dir) not in _authz_sys.path:
            _authz_sys.path.insert(0, str(_authz_dir))
        break
import simulator_authorization as _authz  # noqa: E402
# -----------------------------------------------------------------------------------


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get(manifest: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    current: Any = manifest
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def abs_diff(a: Any, b: Any) -> Optional[float]:
    if a is None or b is None:
        return None
    return abs(float(a) - float(b))


def pct_change(new: float, old: float) -> Optional[float]:
    if old == 0:
        return None
    return (new - old) / old * 100.0


def improvement_percent(before: float, after: float) -> Optional[float]:
    if before == 0:
        return None
    return (before - after) / before * 100.0


def run_summary(manifest: Dict[str, Any]) -> Dict[str, Any]:
    rollout_total = float(get(manifest, "timing.rollout_collection_seconds", 0.0))
    decisions = int(get(manifest, "mappo.expected_agent_decisions", 0))
    return {
        "status": manifest.get("status"),
        "trace_mode": manifest.get("trace_mode"),
        "elapsed_sec": manifest.get("elapsed_sec"),
        "rollout_total_seconds": rollout_total,
        "rollout_online_seconds": get(manifest, "timing.rollout_online_seconds"),
        "rollout_postprocess_seconds": get(manifest, "timing.rollout_postprocess_seconds"),
        "trace_file_write_seconds": get(manifest, "timing.trace_file_write_seconds"),
        "decisions_per_second": decisions / rollout_total if rollout_total else None,
        "peak_mps_current_mb": get(manifest, "memory_summary.peak_mps_current_allocated_mb"),
        "peak_mps_driver_mb": get(manifest, "memory_summary.peak_mps_driver_allocated_mb"),
        "peak_process_rss_mb": get(manifest, "memory_summary.peak_process_rss_mb"),
        "swap_used_mb_max": get(manifest, "memory_summary.swap_used_mb_max"),
        "trace_recording_seconds": get(manifest, "mappo.rollout_profile.rollout_trace_recording_seconds"),
        "trace_file_size_bytes": get(manifest, "mappo.rollout_profile.trace_file_size_bytes"),
        "trace_output_file_bytes": get(manifest, "mappo.rollout_profile.trace_output_file_bytes"),
        "trace_buffer_estimated_bytes": get(manifest, "mappo.rollout_profile.trace_buffer_estimated_bytes"),
    }


def choose_representative_benchmark(benchmarks: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(benchmarks) == 1:
        return benchmarks[0]
    rollout_values = [float(get(item, "timing.rollout_collection_seconds", 0.0)) for item in benchmarks]
    target = median(rollout_values)
    return min(benchmarks, key=lambda item: abs(float(get(item, "timing.rollout_collection_seconds", 0.0)) - target))


def compare(audit: Dict[str, Any], benchmark: Dict[str, Any], all_benchmarks: List[Dict[str, Any]], previous: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    audit_summary = run_summary(audit)
    benchmark_summary = run_summary(benchmark)
    audit_rollout = float(audit_summary["rollout_total_seconds"])
    benchmark_rollout = float(benchmark_summary["rollout_total_seconds"])
    audit_decisions = int(get(audit, "mappo.expected_agent_decisions", 0))
    benchmark_decisions = int(get(benchmark, "mappo.expected_agent_decisions", 0))

    audit_hashes = get(audit, "mappo.rollout_tensor_hashes", {})
    benchmark_hashes = get(benchmark, "mappo.rollout_tensor_hashes", {})
    kpi_audit = get(audit, "mappo.kpi_summary", {})
    kpi_benchmark = get(benchmark, "mappo.kpi_summary", {})
    kpi_abs_diff = {
        key: abs(float(kpi_audit[key]) - float(kpi_benchmark[key]))
        for key in sorted(kpi_audit.keys() & kpi_benchmark.keys())
    }

    performance = {
        "audit_vs_benchmark_rollout_ratio": audit_rollout / benchmark_rollout if benchmark_rollout else None,
        "benchmark_speedup_over_audit": audit_rollout / benchmark_rollout if benchmark_rollout else None,
        "trace_overhead_seconds": audit_rollout - benchmark_rollout,
        "trace_overhead_percent": improvement_percent(audit_rollout, benchmark_rollout),
        "audit_trace_share_percent": (
            float(get(audit, "mappo.rollout_profile.rollout_trace_recording_seconds", 0.0)) / audit_rollout * 100.0
            if audit_rollout
            else None
        ),
        "benchmark_trace_share_percent": (
            float(get(benchmark, "mappo.rollout_profile.rollout_trace_recording_seconds", 0.0)) / benchmark_rollout * 100.0
            if benchmark_rollout
            else None
        ),
        "audit_decisions_per_second": audit_decisions / audit_rollout if audit_rollout else None,
        "benchmark_decisions_per_second": benchmark_decisions / benchmark_rollout if benchmark_rollout else None,
        "throughput_improvement_percent": pct_change(benchmark_decisions / benchmark_rollout, audit_decisions / audit_rollout)
        if audit_rollout and benchmark_rollout
        else None,
        "audit_total_elapsed": audit.get("elapsed_sec"),
        "benchmark_total_elapsed": benchmark.get("elapsed_sec"),
        "total_elapsed_improvement_percent": improvement_percent(float(audit.get("elapsed_sec", 0.0)), float(benchmark.get("elapsed_sec", 0.0))),
        "benchmark_runs": [
            {
                "path": item.get("_manifest_path"),
                "elapsed_sec": item.get("elapsed_sec"),
                "rollout_total_seconds": get(item, "timing.rollout_collection_seconds"),
                "trace_recording_seconds": get(item, "mappo.rollout_profile.rollout_trace_recording_seconds"),
            }
            for item in all_benchmarks
        ],
    }
    if previous is not None:
        previous_rollout = float(get(previous, "timing.rollout_collection_seconds", 0.0))
        performance["audit_vs_previous_rollout_change_percent"] = pct_change(audit_rollout, previous_rollout)
        performance["benchmark_vs_previous_rollout_change_percent"] = pct_change(benchmark_rollout, previous_rollout)

    equivalence = {
        "snapshot_ordering_hash_match": get(audit, "dataset.snapshot_ordering_hash") == get(benchmark, "dataset.snapshot_ordering_hash"),
        "initial_gatv2_parameter_hash_match": get(audit, "gatv2.initial_parameter_hash") == get(benchmark, "gatv2.initial_parameter_hash"),
        "initial_actor_parameter_hash_match": get(audit, "mappo.initial_actor_state_hash") == get(benchmark, "mappo.initial_actor_state_hash"),
        "initial_critic_parameter_hash_match": get(audit, "mappo.initial_critic_state_hash") == get(benchmark, "mappo.initial_critic_state_hash"),
        "embedding_hash_match": get(audit, "gatv2.embedding_tensor_hash") == get(benchmark, "gatv2.embedding_tensor_hash"),
        "action_hash_match": audit_hashes.get("actions_tensor_hash") == benchmark_hashes.get("actions_tensor_hash"),
        "old_logprob_hash_match": audit_hashes.get("old_logprobs_tensor_hash") == benchmark_hashes.get("old_logprobs_tensor_hash"),
        "reward_hash_match": audit_hashes.get("rewards_tensor_hash") == benchmark_hashes.get("rewards_tensor_hash"),
        "done_hash_match": audit_hashes.get("done_tensor_hash") == benchmark_hashes.get("done_tensor_hash"),
        "actor_logits_hash_match": audit_hashes.get("actor_logits_tensor_hash") == benchmark_hashes.get("actor_logits_tensor_hash"),
        "team_transition_hash_match": audit_hashes.get("team_transition_tensor_hash") == benchmark_hashes.get("team_transition_tensor_hash"),
        "actor_loss_abs_diff": abs_diff(get(audit, "mappo.policy_loss_mean"), get(benchmark, "mappo.policy_loss_mean")),
        "critic_loss_abs_diff": abs_diff(get(audit, "mappo.value_loss_mean"), get(benchmark, "mappo.value_loss_mean")),
        "entropy_abs_diff": abs_diff(get(audit, "mappo.entropy_mean"), get(benchmark, "mappo.entropy_mean")),
        "approx_kl_abs_diff": abs_diff(get(audit, "mappo.approx_kl_mean"), get(benchmark, "mappo.approx_kl_mean")),
        "clip_fraction_abs_diff": abs_diff(get(audit, "mappo.clip_fraction_mean"), get(benchmark, "mappo.clip_fraction_mean")),
        "final_actor_state_match": get(audit, "mappo.final_actor_state_hash") == get(benchmark, "mappo.final_actor_state_hash"),
        "final_critic_state_match": get(audit, "mappo.final_critic_state_hash") == get(benchmark, "mappo.final_critic_state_hash"),
        "checkpoint_reload_match": bool(get(audit, "mappo.checkpoint_reload_ok")) and bool(get(benchmark, "mappo.checkpoint_reload_ok")),
        "canonical_kpi_match": kpi_audit == kpi_benchmark,
        "canonical_kpi_abs_diff": kpi_abs_diff,
    }

    common_config = {
        "snapshots": get(audit, "dataset.snapshots"),
        "validation_snapshots": get(audit, "dataset.val_snapshots"),
        "agents": get(audit, "mappo.num_agents"),
        "rollout_horizon": get(audit, "mappo.rollout_horizon"),
        "ppo_epochs": get(audit, "mappo.ppo_update_epochs"),
        "minibatch_size": get(audit, "mappo.minibatch_size"),
        "seed": get(audit, "mappo.seed"),
        "condition": get(audit, "mappo.condition_id"),
        "device": audit.get("actual_device"),
        "strict_mps": audit.get("strict_mps"),
    }

    return {
        "common_config": common_config,
        "audit": audit_summary,
        "benchmark": benchmark_summary,
        "comparison": {
            "rollout_speedup": performance["benchmark_speedup_over_audit"],
            "throughput_improvement_percent": performance["throughput_improvement_percent"],
            "trace_overhead_seconds": performance["trace_overhead_seconds"],
            "trace_overhead_percent": performance["trace_overhead_percent"],
            "process_rss_change_mb": float(benchmark_summary["peak_process_rss_mb"]) - float(audit_summary["peak_process_rss_mb"]),
            "mps_current_change_mb": float(benchmark_summary["peak_mps_current_mb"]) - float(audit_summary["peak_mps_current_mb"]),
            "mps_driver_change_mb": float(benchmark_summary["peak_mps_driver_mb"]) - float(audit_summary["peak_mps_driver_mb"]),
        },
        "performance": performance,
        "equivalence": equivalence,
    }


def write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    comparison = payload["comparison"]
    equivalence = payload["equivalence"]
    lines = [
        "# 64-Agent Trace Mode Comparison",
        "",
        "## Verdict",
        "",
        f"- rollout speedup: `{comparison['rollout_speedup']:.6f}x`",
        f"- throughput improvement: `{comparison['throughput_improvement_percent']:.3f}%`",
        f"- trace overhead seconds: `{comparison['trace_overhead_seconds']:.6f}`",
        f"- action hash match: `{equivalence['action_hash_match']}`",
        f"- old logprob hash match: `{equivalence['old_logprob_hash_match']}`",
        f"- reward hash match: `{equivalence['reward_hash_match']}`",
        f"- done hash match: `{equivalence['done_hash_match']}`",
        f"- final actor state match: `{equivalence['final_actor_state_match']}`",
        f"- final critic state match: `{equivalence['final_critic_state_match']}`",
        f"- canonical KPI match: `{equivalence['canonical_kpi_match']}`",
        "",
        "This comparison validates trace-mode separation only. It does not establish convergence or policy performance.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    _authz.require_capability("performance_comparison", site="compare_gatv2_mappo_trace_modes.py::main")
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-manifest", required=True)
    parser.add_argument("--benchmark-manifest", action="append", required=True)
    parser.add_argument("--previous-manifest")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    audit = load_json(Path(args.audit_manifest))
    audit["_manifest_path"] = str(Path(args.audit_manifest))
    benchmarks = []
    for path_text in args.benchmark_manifest:
        manifest = load_json(Path(path_text))
        manifest["_manifest_path"] = str(Path(path_text))
        benchmarks.append(manifest)
    previous = load_json(Path(args.previous_manifest)) if args.previous_manifest else None
    representative = choose_representative_benchmark(benchmarks)
    payload = compare(audit, representative, benchmarks, previous)

    output_root = Path(args.output_root)
    dump_json(output_root / "mode_comparison.json", payload)
    dump_json(output_root / "determinism_comparison.json", payload["equivalence"])
    dump_json(output_root / "performance_comparison.json", payload["performance"])
    write_markdown(output_root / "mode_comparison.md", payload)
    manifest = {
        "status": "PASS",
        "artifacts": {
            name: {
                "relative_path": name,
                "absolute_path": str(output_root / name),
                "file_size_bytes": (output_root / name).stat().st_size,
                "sha256": sha256_file(output_root / name),
            }
            for name in [
                "mode_comparison.json",
                "determinism_comparison.json",
                "performance_comparison.json",
                "mode_comparison.md",
            ]
        },
    }
    dump_json(output_root / "comparison_manifest.json", manifest)
    print(json.dumps({"output_root": str(output_root), **payload}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
