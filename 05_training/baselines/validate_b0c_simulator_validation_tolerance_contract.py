from __future__ import annotations

import json
from pathlib import Path

EXPECTED_KPIS = [
    'cv_headway',
    'avg_wait_seconds',
    'bunching_rate',
    'on_time_rate',
    'intervention_rate',
    'energy_proxy',
    'passenger_demand_generated',
    'passenger_served_count',
    'passenger_service_rate',
    'passenger_wait_p95_seconds',
    'energy_proxy_per_passenger',
    'fleet_reduction_ratio',
]

def load_json(path: Path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        return json.load(f)

def main() -> None:
    path = Path('05_training/baselines/b0c_simulator_validation_tolerance_contract.json')
    if not path.exists():
        raise SystemExit(f'[FAIL] missing contract: {path}')

    payload = load_json(path)
    failures = []

    if payload.get('artifact_version') != 'b0c_simulator_validation_tolerance_contract_v1':
        failures.append('artifact_version mismatch')
    if payload.get('condition_id') != 'B0C':
        failures.append('condition_id must be B0C')
    if payload.get('baseline_id') != 'B0C_causal_shadow_v1':
        failures.append('baseline_id must be B0C_causal_shadow_v1')
    if payload.get('next_stage') != 'B0C-6_simulator_validation_report':
        failures.append('next_stage must be B0C-6_simulator_validation_report')

    required = payload.get('required_structure', {})
    if required.get('expected_seeds') != [1, 2, 3]:
        failures.append('expected_seeds must be [1, 2, 3]')
    if int(required.get('expected_window_rows', -1)) != 19710:
        failures.append('expected_window_rows must be 19710')
    if int(required.get('expected_windows_per_seed', -1)) != 6570:
        failures.append('expected_windows_per_seed must be 6570')
    if required.get('kpi_schema') != EXPECTED_KPIS:
        failures.append('12-KPI schema mismatch')

    guards = payload.get('claim_guards', {})
    for key in [
        'causal_comparison_allowed',
        'paper_level_claim_allowed',
        'actual_results',
        'winner_selected',
        'trainable_reward_promoted',
        'auto_promotion_allowed',
    ]:
        if guards.get(key) is not False:
            failures.append(f'{key} must be false')

    promotion = payload.get('promotion_policy', {})
    if promotion.get('auto_enable_causal_comparison_allowed') is not False:
        failures.append('auto_enable_causal_comparison_allowed must be false')
    if promotion.get('requires_later_explicit_release_gate') is not True:
        failures.append('requires_later_explicit_release_gate must be true')
    if promotion.get('requires_operator_approval') is not True:
        failures.append('requires_operator_approval must be true')

    rules = payload.get('tolerance_rules', {})
    for key in [
        'avg_wait_seconds',
        'cv_headway',
        'bunching_rate',
        'on_time_rate',
        'intervention_rate',
        'passenger_service_rate',
        'passenger_wait_p95_seconds',
        'energy_proxy_per_passenger',
        'fleet_reduction_ratio',
        'qwen_trigger_rate',
    ]:
        if key not in rules:
            failures.append(f'missing tolerance rule: {key}')

    obs = payload.get('observability_restrictions', {})
    for key in ['actual_headway', 'actual_arrival_departure_time', 'actual_dwell']:
        if obs.get(key) != 'not_observed_do_not_promote':
            failures.append(f'{key} must remain not_observed_do_not_promote')

    if failures:
        print('[FAIL] B0C simulator validation tolerance contract failed')
        for item in failures:
            print('[FAIL]', item)
        raise SystemExit(1)

    print('[OK] B0C simulator validation tolerance contract PASS')
    print('[OK] contract:', path)
    print('[OK] baseline_id: B0C_causal_shadow_v1')
    print('[OK] kpi_schema : 12 KPI')
    print('[OK] claim_guard: false')
    print('[OK] next_stage : B0C-6_simulator_validation_report')

if __name__ == '__main__':
    main()
