# Step 125 ??Reward Ablation Selection Criteria Gate

Step 125 defines how a future reward ablation winner may be selected.

This step does not select a winner, does not promote a reward, and does not allow MAPPO training.

Current guards:

- `actual_results = false`
- `winner_selected = false`
- `winner_selection_allowed_now = false`
- `train_with_this_reward_allowed = false`
- `paper_level_claim_allowed = false`
- `causal_performance_claim_allowed = false`

The selection rule is service-quality first. Energy and fleet reduction can only be used after hard constraints and service quality are protected.

A future winner may only be selected from actual results covering R0/R1/R2/R3/R4/R5 횞 A/A90/A80/A70 횞 seeds 1,2,3 = 72 rows.

Ranking tiers:

1. Eligibility filter
2. Service quality rank
3. Efficiency secondary rank
4. Stability and seed robustness

A candidate is disqualified if it violates hard constraints, has missing rows, uses no-op/template/dry-run outputs, or improves energy by harming passenger service.
