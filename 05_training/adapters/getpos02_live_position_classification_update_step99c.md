# Step 99-C-Final — /getPos02 Live Bus Position Classification Update

Status: PASS.

Final evidence:
- api_called = True
- response_format_counts = {'json': 3}
- provider_status_counts = {'ok_or_unclassified': 3}
- total_candidate_rows = 18
- total_normalized_rows = 18
- live_position_possible_any = True
- audit_status = PASS

Detected schema:
- routeId -> route_id
- moveDir -> direction_id
- arTime -> live_event_time_raw
- seq -> current_route_sequence
- bsId -> current_stop_id
- xPos/yPos -> live position coordinates
- vhcNo2 -> bus_id_or_vehicle_no candidate
- routeNo -> route_no
- busTCd2/busTCd3 -> vehicle/status/type candidates; interpretation deferred

Step 97 classification update:
- bus_id_or_vehicle_no: missing -> sample_observed_candidate_from_getPos02_vhcNo2
- live_position_xy: missing -> sample_observed_candidate_from_getPos02_xPos_yPos
- current_route_sequence: missing -> sample_observed_candidate_from_getPos02_seq
- current_stop_id: missing -> sample_observed_candidate_from_getPos02_bsId
- direction_id: observed from /getBs02 and confirmed by /getPos02 moveDir
- live_event_time_raw: missing -> sample_observed_candidate_from_getPos02_arTime

Remaining limitations:
- A single /getPos02 call is a live position snapshot, not a vehicle trajectory.
- vehicle_trajectory, actual_headway, actual_arrival_departure_time, and actual_dwell require repeated sampling.

Guardrails:
- DB write not performed.
- Tensor DB overwrite not performed.
- Paper-level claim forbidden.
- Causal performance claim forbidden.

Next step: Step 99-C2 repeated /getPos02 sampling.
