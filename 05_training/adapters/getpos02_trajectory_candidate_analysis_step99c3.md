# Step 99-C3 - getPos02 Trajectory Candidate Analyzer

## Purpose

This step analyzes repeated getPos02 live-position snapshots and identifies vehicle trajectory candidates.

It does not claim actual vehicle trajectory, actual headway, actual dwell time, or actual arrival/departure events. It only checks whether the same vehicle identifier candidate appears across repeated samples with changing stop order, stop id, or x/y position.

## Input

Expected input run directory:

artifacts/daegu_bis_api_audit/getpos02_repeated_sampling/20260429_090603

Required files:

- step99c2_repeated_sampling_manifest.csv
- sample_001/getpos02_sample_response_route_*.raw
- sample_002/getpos02_sample_response_route_*.raw
- ...
- sample_006/getpos02_sample_response_route_*.raw

## Output

The analyzer writes:

- getpos02_trajectory_candidate_timeseries.csv
- getpos02_trajectory_candidate_summary.csv
- getpos02_trajectory_candidate_report.json
- getpos02_trajectory_candidate_report.md

## Candidate rule

A trajectory candidate is counted when:

1. The same route_id, direction_id, and bus_id_candidate appears in at least two samples.
2. One or more of the following changes across time:
   - current_stop_order
   - current_stop_id
   - x_pos / y_pos

## Guardrails

- DB write is forbidden.
- Tensor DB overwrite is forbidden.
- paper_level_claim_allowed=false.
- causal_performance_claim_allowed=false.
- vehicle_trajectory_claim_allowed=false.
- actual_headway, actual_dwell, and actual_arrival_departure_time must not be upgraded to observed.

## Classification update

Allowed:

- vehicle_trajectory = trajectory_candidate_from_repeated_getPos02_sampling

Not allowed:

- actual_vehicle_trajectory = observed
- actual_headway = observed
- actual_dwell = observed
- actual_arrival_departure_time = observed
