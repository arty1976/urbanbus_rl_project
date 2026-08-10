from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional


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


def get(payload: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    current: Any = payload
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def pct(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old in (None, 0):
        return None
    return (float(new) - float(old)) / float(old) * 100.0


def mult(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old in (None, 0):
        return None
    return float(new) / float(old)


def classify(value: Optional[float]) -> str:
    if value is None:
        return "NOT_AVAILABLE"
    if value < 1.8:
        return "SUBLINEAR"
    if value <= 2.2:
        return "LINEAR"
    return "SUPERLINEAR"


def section_shares(profile: Dict[str, Any]) -> Dict[str, Optional[float]]:
    total = float(profile.get("rollout_total_seconds") or 0.0)
    sections = {
        "observation_build_share_percent": "rollout_observation_build_seconds",
        "actor_inference_share_percent": "rollout_actor_inference_seconds",
        "action_sampling_share_percent": "rollout_action_sampling_seconds",
        "simulator_step_share_percent": "rollout_simulator_step_seconds",
        "reward_aggregation_share_percent": "rollout_reward_aggregation_seconds",
        "buffer_append_share_percent": "rollout_buffer_append_seconds",
        "postprocess_share_percent": "rollout_postprocess_seconds",
        "other_overhead_share_percent": "rollout_other_overhead_seconds",
    }
    return {
        name: (float(profile.get(key) or 0.0) / total * 100.0 if total else None)
        for name, key in sections.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-comparison", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--target-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    source_comparison = load_json(Path(args.source_comparison))
    source_manifest = load_json(Path(args.source_manifest))
    target = load_json(Path(args.target_manifest))

    source = source_comparison.get("benchmark") or source_comparison.get("target") or {}
    target_profile = get(target, "mappo.rollout_profile", {})
    source_profile = get(source_manifest, "mappo.rollout_profile", {})
    source_mem = source.get("memory_summary", source)
    target_mem = target.get("memory_summary", {})

    source_agents = int(get(source_manifest, "mappo.num_agents", 0))
    target_agents = int(get(target, "mappo.num_agents", 0))
    source_decisions = int(get(source_manifest, "mappo.expected_agent_decisions", 0))
    target_decisions = int(get(target, "mappo.expected_agent_decisions", 0))
    target_horizon = int(get(target, "mappo.rollout_horizon", 0))
    target_update_epochs = int(get(target, "mappo.ppo_update_epochs", 0))
    expected_target_decisions = target_agents * target_horizon
    expected_actor_elements = expected_target_decisions * target_update_epochs

    performance = {
        "agent_count_multiplier": mult(target_agents, source_agents),
        "agent_decision_multiplier": mult(target_decisions, source_decisions),
        "elapsed_change_seconds": float(target.get("elapsed_sec", 0.0)) - float(source_manifest.get("elapsed_sec", 0.0)),
        "elapsed_change_percent": pct(target.get("elapsed_sec"), source_manifest.get("elapsed_sec")),
        "elapsed_multiplier": mult(target.get("elapsed_sec"), source_manifest.get("elapsed_sec")),
        "rollout_total_change_seconds": float(get(target, "timing.rollout_collection_seconds", 0.0)) - float(source.get("rollout_total_seconds", 0.0)),
        "rollout_total_change_percent": pct(get(target, "timing.rollout_collection_seconds"), source.get("rollout_total_seconds")),
        "rollout_total_multiplier": mult(get(target, "timing.rollout_collection_seconds"), source.get("rollout_total_seconds")),
        "rollout_online_change_percent": pct(get(target, "timing.rollout_online_seconds"), source.get("rollout_online_seconds")),
        "rollout_online_multiplier": mult(get(target, "timing.rollout_online_seconds"), source.get("rollout_online_seconds")),
        "rollout_postprocess_change_percent": pct(get(target, "timing.rollout_postprocess_seconds"), source.get("rollout_postprocess_seconds")),
        "ppo_update_change_seconds": float(get(target, "timing.ppo_update_seconds", 0.0)) - float(get(source_manifest, "timing.ppo_update_seconds", 0.0)),
        "ppo_update_change_percent": pct(get(target, "timing.ppo_update_seconds"), get(source_manifest, "timing.ppo_update_seconds")),
        "ppo_update_multiplier": mult(get(target, "timing.ppo_update_seconds"), get(source_manifest, "timing.ppo_update_seconds")),
        "decisions_per_second_change_percent": pct(get(target_profile, "agent_decisions_per_second_total"), source.get("decisions_per_second")),
        "online_decisions_per_second_change_percent": pct(get(target_profile, "agent_decisions_per_second_online"), source.get("decisions_per_second")),
        "actor_inference_change_percent": pct(get(target_profile, "rollout_actor_inference_seconds"), get(source_profile, "rollout_actor_inference_seconds")),
        "actor_inference_multiplier": mult(get(target_profile, "rollout_actor_inference_seconds"), get(source_profile, "rollout_actor_inference_seconds")),
        "simulator_step_change_percent": pct(get(target_profile, "rollout_simulator_step_seconds"), get(source_profile, "rollout_simulator_step_seconds")),
        "simulator_step_multiplier": mult(get(target_profile, "rollout_simulator_step_seconds"), get(source_profile, "rollout_simulator_step_seconds")),
        "action_sampling_change_percent": pct(get(target_profile, "rollout_action_sampling_seconds"), get(source_profile, "rollout_action_sampling_seconds")),
        "action_sampling_multiplier": mult(get(target_profile, "rollout_action_sampling_seconds"), get(source_profile, "rollout_action_sampling_seconds")),
        "peak_mps_current_change_mb": float(target_mem.get("peak_mps_current_allocated_mb", 0.0)) - float(source_mem.get("peak_mps_current_allocated_mb", source_mem.get("peak_mps_current_mb", 0.0))),
        "peak_mps_driver_change_mb": float(target_mem.get("peak_mps_driver_allocated_mb", 0.0)) - float(source_mem.get("peak_mps_driver_allocated_mb", source_mem.get("peak_mps_driver_mb", 0.0))),
        "peak_process_rss_change_mb": float(target_mem.get("peak_process_rss_mb", 0.0)) - float(source_mem.get("peak_process_rss_mb", 0.0)),
        "swap_change_mb": float(target_mem.get("swap_used_mb_max", 0.0)) - float(source_mem.get("swap_used_mb_max", 0.0)),
    }
    classifications = {
        "rollout": classify(performance["rollout_total_multiplier"]),
        "actor_inference": classify(performance["actor_inference_multiplier"]),
        "action_sampling": classify(mult(get(target_profile, "rollout_action_sampling_seconds"), get(source_profile, "rollout_action_sampling_seconds"))),
        "simulator_step": classify(performance["simulator_step_multiplier"]),
        "ppo_update": classify(performance["ppo_update_multiplier"]),
    }

    actor_audit = get(target, "mappo.actor_loss_audit", {})
    blocking_reasons = []
    if target.get("status") != "PASS":
        blocking_reasons.append("target_status_not_pass")
    if target.get("actual_device") != "mps" or target.get("cpu_fallback_used"):
        blocking_reasons.append("strict_mps_failed")
    if target.get("nan_detected") or target.get("inf_detected"):
        blocking_reasons.append("nan_or_inf_detected")
    if target_mem.get("swap_used_mb_max") != 0.0:
        blocking_reasons.append("swap_used")
    if get(target, "mappo.observed_agent_decisions") != expected_target_decisions:
        blocking_reasons.append("agent_decision_count_mismatch")
    if actor_audit.get("actor_elements_processed_across_all_ppo_epochs") != expected_actor_elements:
        blocking_reasons.append("ppo_actor_element_count_mismatch")
    if not get(target, "mappo.checkpoint_reload_ok"):
        blocking_reasons.append("checkpoint_reload_failed")
    if target.get("canonical_kpi_count") != 12:
        blocking_reasons.append("canonical_kpi_missing")
    severe_superlinear = any(
        performance.get(key) is not None and performance[key] > 2.5
        for key in ["rollout_total_multiplier", "ppo_update_multiplier", "actor_inference_multiplier", "simulator_step_multiplier"]
    )
    if severe_superlinear:
        blocking_reasons.append("severe_superlinear_scaling")

    next_agents = target_agents * 2 if target_agents else None
    approved = len(blocking_reasons) == 0
    next_scale = {
        "approval_reason": "No strict-MPS, actor-contract, checkpoint, KPI, swap, or severe scaling blocker detected." if approved else "Blocked by one or more gate conditions.",
        "blocking_reasons": blocking_reasons,
    }
    if next_agents:
        next_scale[f"approved_for_{next_agents}_agents"] = approved
        next_scale["next_agents"] = next_agents
    if target_agents == 128:
        next_scale["approved_for_256_agents"] = approved
    if target_agents == 256:
        next_scale["approved_for_512_agents"] = approved

    scaling_gate = {
        "source_scale": {
            "snapshots": get(source_manifest, "dataset.snapshots"),
            "agents": source_agents,
            "horizon": get(source_manifest, "mappo.rollout_horizon"),
            "trace_mode": source_manifest.get("trace_mode"),
        },
        "target_scale": {
            "snapshots": get(target, "dataset.snapshots"),
            "agents": target_agents,
            "horizon": get(target, "mappo.rollout_horizon"),
            "trace_mode": target.get("trace_mode"),
        },
        "execution": {
            "status": target.get("status"),
            "actual_device": target.get("actual_device"),
            "strict_mps": target.get("strict_mps"),
            "cpu_fallback_used": target.get("cpu_fallback_used"),
            "gpu_blocked": get(target, "device.gpu_blocked"),
            "nan_detected": target.get("nan_detected"),
            "inf_detected": target.get("inf_detected"),
            "swap_used_mb_max": target_mem.get("swap_used_mb_max"),
        },
        "actor_contract": {
            "observed_agents": get(target, "mappo.observed_unique_agent_count"),
            "observed_decisions": get(target, "mappo.observed_agent_decisions"),
            "unique_actor_elements": actor_audit.get("unique_rollout_actor_elements"),
            "processed_actor_elements": actor_audit.get("actor_elements_processed_across_all_ppo_epochs"),
            "advantage_before": actor_audit.get("team_advantage_count_before_broadcast"),
            "advantage_after": actor_audit.get("actor_advantage_count_after_broadcast"),
            "critic_targets": actor_audit.get("critic_value_target_count"),
        },
        "performance": {
            "elapsed_multiplier": performance["elapsed_multiplier"],
            "rollout_total_multiplier": performance["rollout_total_multiplier"],
            "rollout_online_multiplier": performance["rollout_online_multiplier"],
            "actor_inference_multiplier": performance["actor_inference_multiplier"],
            "action_sampling_multiplier": performance["action_sampling_multiplier"],
            "simulator_step_multiplier": performance["simulator_step_multiplier"],
            "ppo_update_multiplier": performance["ppo_update_multiplier"],
            "throughput_change_percent": performance["decisions_per_second_change_percent"],
            "classifications": classifications,
        },
        "memory": {
            "mps_current_change_mb": performance["peak_mps_current_change_mb"],
            "mps_driver_change_mb": performance["peak_mps_driver_change_mb"],
            "process_rss_change_mb": performance["peak_process_rss_change_mb"],
            "swap_change_mb": performance["swap_change_mb"],
        },
        "stability": {
            "checkpoint_reload_ok": get(target, "mappo.checkpoint_reload_ok"),
            "reload_logits_diff": get(target, "mappo.checkpoint_reload_logits_max_abs_diff"),
            "reload_value_diff": get(target, "mappo.checkpoint_reload_value_max_abs_diff"),
            "canonical_kpi_count": target.get("canonical_kpi_count"),
        },
        "next_scale": next_scale,
    }

    comparison = {
        "source_baseline": source,
        "target": {
            "status": target.get("status"),
            "elapsed_sec": target.get("elapsed_sec"),
            "rollout_total_seconds": get(target, "timing.rollout_collection_seconds"),
            "rollout_online_seconds": get(target, "timing.rollout_online_seconds"),
            "rollout_postprocess_seconds": get(target, "timing.rollout_postprocess_seconds"),
            "trace_file_write_seconds": get(target, "timing.trace_file_write_seconds"),
            "ppo_update_seconds": get(target, "timing.ppo_update_seconds"),
            "decisions_per_second": get(target_profile, "agent_decisions_per_second_total"),
            "memory_summary": target_mem,
        },
        "performance": performance,
        "classifications": classifications,
        "bottleneck_shares": section_shares(target_profile),
        "actor_inference": {
            "mode": target_profile.get("actor_inference_mode"),
            "calls": target_profile.get("actor_inference_calls"),
            "agents_per_call": target_profile.get("agents_per_actor_call"),
            "elements_per_second": target_profile.get("actor_elements_per_second"),
            "ms_per_timestep": target_profile.get("actor_inference_ms_per_timestep"),
            "ms_per_agent": target_profile.get("actor_inference_ms_per_agent"),
        },
        "simulator": {
            "mode": target_profile.get("simulator_step_mode"),
            "calls": target_profile.get("simulator_step_calls"),
        },
    }

    memory = {
        "source": {
            "peak_mps_current_mb": source_mem.get("peak_mps_current_allocated_mb", source_mem.get("peak_mps_current_mb")),
            "peak_mps_driver_mb": source_mem.get("peak_mps_driver_allocated_mb", source_mem.get("peak_mps_driver_mb")),
            "peak_process_rss_mb": source_mem.get("peak_process_rss_mb"),
            "swap_used_mb_max": source_mem.get("swap_used_mb_max"),
        },
        "target": target_mem,
        "delta": scaling_gate["memory"],
        "target_rollout_tensor_bytes": {
            "rollout_buffer_tensor_bytes": target_profile.get("rollout_buffer_tensor_bytes"),
            "actor_rollout_tensor_bytes": target_profile.get("actor_rollout_tensor_bytes"),
            "critic_rollout_tensor_bytes": target_profile.get("critic_rollout_tensor_bytes"),
            "trace_buffer_estimated_bytes": target_profile.get("trace_buffer_estimated_bytes"),
            "postprocess_tensor_bytes": target_profile.get("postprocess_tensor_bytes"),
        },
    }

    output_root = Path(args.output_root)
    dump_json(output_root / "scale_comparison.json", comparison)
    dump_json(output_root / "performance_comparison.json", performance)
    dump_json(output_root / "memory_comparison.json", memory)
    dump_json(output_root / "scaling_gate.json", scaling_gate)
    dump_json(
        output_root / "actor_inference_scaling.json",
        {
            "source_agents": source_agents,
            "target_agents": target_agents,
            "actor_inference_multiplier": performance["actor_inference_multiplier"],
            "classification": classifications["actor_inference"],
            "target_actor_inference": {
                "mode": target_profile.get("actor_inference_mode"),
                "calls": target_profile.get("actor_inference_calls"),
                "agents_per_call": target_profile.get("agents_per_actor_call"),
                "seconds": get(target_profile, "rollout_actor_inference_seconds"),
                "share_percent": section_shares(target_profile).get("actor_inference_share_percent"),
                "elements_per_second": target_profile.get("actor_elements_per_second"),
                "ms_per_timestep": target_profile.get("actor_inference_ms_per_timestep"),
                "ms_per_agent": target_profile.get("actor_inference_ms_per_agent"),
            },
        },
    )
    md = [
        f"# {source_agents} vs {target_agents} Fixed-Embedding MAPPO Benchmark Scaling",
        "",
        f"- target status: `{target.get('status')}`",
        f"- rollout multiplier: `{performance['rollout_total_multiplier']}`",
        f"- PPO update multiplier: `{performance['ppo_update_multiplier']}`",
        f"- throughput change percent: `{performance['decisions_per_second_change_percent']}`",
        f"- approved for {next_agents} agents: `{approved}`",
        f"- blocking reasons: `{scaling_gate['next_scale']['blocking_reasons']}`",
        "",
        "This is a fixed-embedding MAPPO scaling validation. It does not claim fresh GATv2 cross-process reproducibility or policy convergence.",
        "",
    ]
    (output_root / "scale_comparison.md").write_text("\n".join(md), encoding="utf-8")
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
                "scale_comparison.json",
                "scale_comparison.md",
                "performance_comparison.json",
                "memory_comparison.json",
                "actor_inference_scaling.json",
                "scaling_gate.json",
            ]
        },
    }
    dump_json(output_root / "comparison_manifest.json", manifest)
    print(json.dumps({"output_root": str(output_root), "scaling_gate": scaling_gate, "performance": performance}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
