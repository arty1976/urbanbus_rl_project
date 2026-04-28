
from __future__ import annotations

import argparse
import json
from pathlib import Path
from adapters.causal_simulator_v2_adapter import CausalSimulatorV2Adapter


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", default=".")
    p.add_argument("--num-agents", type=int, default=8)
    p.add_argument("--episode-steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--output-json", default="artifacts/causal_simulator_v2_adapter_smoke/status.json")
    args = p.parse_args()

    ad = CausalSimulatorV2Adapter.from_step100_artifacts(args.project_root, args.num_agents, args.episode_steps, args.seed)
    ad.reset(seed=args.seed)
    traj = []
    for _ in range(args.episode_steps):
        sr = ad.step({i: i % ad.action_dim for i in range(ad.num_agents)})
        traj.append(sr)
        if sr.terminated:
            break
    gs = ad.get_graph_skeleton()
    payload = {
        "artifact_version": "causal_simulator_v2_adapter_smoke_status_step101",
        "adapter_version": "causal_simulator_v2_scaffold_step101",
        "num_agents": ad.num_agents,
        "trajectory_steps": len(traj),
        "graph": {"num_nodes": gs.num_nodes, "num_edges": gs.num_edges, "edge_features_dim": gs.edge_features_dim},
        "observation_space": ad.observation_space,
        "action_space": ad.action_space,
        "kpis": ad.compute_kpis(traj),
        "claim_guardrails": {
            "performance_claim_allowed": False,
            "causal_performance_claim_allowed": False,
            "dynamic_signal_phase_claim_allowed": False,
            "red_light_delay_claim_allowed": False,
            "green_time_claim_allowed": False,
            "cycle_length_claim_allowed": False,
        },
        "source_mode": "static_signal_scaffold_nonperformance_v1",
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    ad.close()
    print("[OK] Step 101 actual artifact adapter smoke complete")
    print(f"[OK] output_json: {out}")
    print(f"[OK] graph nodes : {payload['graph']['num_nodes']}")
    print(f"[OK] graph edges : {payload['graph']['num_edges']}")


if __name__ == "__main__":
    main()
