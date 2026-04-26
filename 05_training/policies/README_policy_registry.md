# Policy Registry Contract v1

## Purpose

This document defines the policy registry boundary for UrbanBus MAPPO rollouts.

MAPPO means Multi-Agent Proximal Policy Optimization.

The goal is to prevent placeholder policy outputs from being mixed with actual MAPPO inference outputs.

## Policy Kinds

- noop
- rulebased
- placeholder
- mappo

## Main Rule

Actual MAPPO policy requires a checkpoint path.

If checkpoint_path is missing, the registry must fail loudly.

It must never silently fall back to a placeholder policy.

## A-family Conditions

A-family conditions are:

- A
- A90
- A80
- A70

In the current phase, all A-family conditions must have qwen_enabled equal to false.

Qwen intervention is not allowed in A_pure_mappo.

## Source Mode Examples

Placeholder or smoke-only examples:

    stub_A_placeholder_smoke
    stub_A90_placeholder_smoke

Replay or non-causal examples:

    replay_B1_noop_noncausal
    replay_B2_rulebased_noncausal

Actual causal MAPPO examples:

    causal_A_mappo_policy_v1
    causal_A90_mappo_policy_v1
    causal_A80_mappo_policy_v1
    causal_A70_mappo_policy_v1

## Paper-level Claim Rule

Only causal actual MAPPO rows should be used for paper-level performance claims.

Placeholder, smoke, replay, and non-causal rows are valid only for contract validation and pipeline testing.
