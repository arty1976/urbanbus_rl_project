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
    path = Path('05_training/baselines/b0c_causal_shadow_baseline_contract.json')
    if not path.exists():
        raise SystemExit(f'[FAIL] missing contract: {path}')

    payload = load_json(path)
    failures = []

    if payload.get('condition_id') != 'B0C':
        failures.append('condition_id must be B0C')
    if payload.get('baseline_id') != 'B0C_causal_shadow_v1':
        failures.append('baseline_id must be B0C_causal_shadow_v1')

    policy = payload.get('policy', {})
    if policy.get('qwen_train') is not False:
        failures.append('qwen_train must be false')
    if policy.get('qwen_inference') is not False:
        failures.append('qwen_inference must be false')
    if float(policy.get('qwen_trigger_rate', -1)) != 0.0:
        failures.append('qwen_trigger_rate must be 0.0')
    if float(policy.get('active_bus_ratio', -1)) != 1.0:
        failures.append('active_bus_ratio must be 1.0')
    if float(policy.get('fleet_reduction_ratio', -1)) != 0.0:
        failures.append('fleet_reduction_ratio must be 0.0')
    if policy.get('learned_policy') is not False:
        failures.append('learned_policy must be false')
    if policy.get('rule_based_policy') is not False:
        failures.append('rule_based_policy must be false')

    kpis = payload.get('evaluation', {}).get('kpi_schema', [])
    if kpis != EXPECTED_KPIS:
        failures.append(f'12-KPI schema mismatch: {kpis}')

    guards = payload.get('claim_guards', {})
    if guards.get('causal_simulator_target') is not True:
        failures.append('causal_simulator_target must be true')
    if guards.get('causal_comparison_allowed') is not False:
        failures.append('causal_comparison_allowed must remain false at contract stage')
    if guards.get('paper_level_claim_allowed') is not False:
        failures.append('paper_level_claim_allowed must remain false')
    if guards.get('actual_results') is not False:
        failures.append('actual_results must remain false')

    if failures:
        print('[FAIL] B0C contract validation failed')
        for item in failures:
            print('[FAIL]', item)
        raise SystemExit(1)

    print('[OK] B0C causal shadow baseline contract validation PASS')
    print('[OK] contract:', path)
    print('[OK] condition_id: B0C')
    print('[OK] baseline_id : B0C_causal_shadow_v1')
    print('[OK] kpi_schema  : 12 KPI')
    print('[OK] claim_guard : false')

if __name__ == '__main__':
    main()
