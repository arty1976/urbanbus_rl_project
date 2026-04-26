# Policy Inference Boundary v1

## Purpose

This document defines the hard boundary between placeholder policy outputs and actual MAPPO inference outputs.

MAPPO means Multi-Agent Proximal Policy Optimization.

Step 24 does not implement neural network inference yet.
It implements the safety gate that prevents accidental fallback from actual MAPPO to placeholder policy.

## Core Rule

If A, A90, A80, or A70 uses actual MAPPO policy, checkpoint_path is mandatory.

If checkpoint_path is missing, execution must fail.

The system must never silently fall back to placeholder policy.

## A-family Conditions

A-family conditions are:

- A
- A90
- A80
- A70

In the current phase, Qwen is disabled for all A-family conditions.

Therefore, qwen_trigger_rate must be 0.0.

## Required Rollout Provenance Fields

Every actual rollout row should record:

- policy_source
- policy_checkpoint_path
- source_mode
- qwen_trigger_rate

## Valid Actual MAPPO Source Mode Examples

- causal_A_mappo_policy_v1
- causal_A90_mappo_policy_v1
- causal_A80_mappo_policy_v1
- causal_A70_mappo_policy_v1

## Placeholder Source Mode Examples

- stub_A_placeholder_smoke
- stub_A90_placeholder_smoke
- stub_A80_placeholder_smoke
- stub_A70_placeholder_smoke

## Paper-level Claim Rule

Only actual causal MAPPO source modes can be used for paper-level performance claims.

Placeholder, smoke, replay, and non-causal outputs are for contract validation only.
