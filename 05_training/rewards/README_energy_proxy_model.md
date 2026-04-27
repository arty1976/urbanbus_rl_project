# Daegu Energy Proxy Model v1

## Purpose

`daegu_energy_proxy_v1` estimates an energy-equivalent proxy for Daegu bus control experiments.

This model is not a direct engineering-grade battery or fuel model. It is a KPI-side diagnostic model intended to answer this research question:

Can a MAPPO policy reduce waiting time and bus bunching without simply spending much more movement, acceleration, or idling energy?

MAPPO (Multi-Agent Proximal Policy Optimization=다중 에이전트 근접 정책 최적화) should not immediately optimize this model as reward until the values are validated across B0/B1/B2/A rollout outputs.

## Formula

energy_kwh_equiv =
  0.0012 * distance_m
  + 0.1800 * acceleration_event_count
  + 0.0080 * hold_seconds

energy_proxy_per_passenger =
  energy_kwh_equiv / max(passenger_served_count, epsilon)

## Constants

| Constant | Value | Meaning |
|---|---:|---|
| K_DIST | 0.0012 kWh/m | distance-based movement energy |
| K_ACC | 0.1800 kWh/event | acceleration/deceleration penalty |
| K_IDLE | 0.0080 kWh/sec | idling/holding/cooling-load proxy |

## Intended workflow

1. Pin constants in checkpoint contract.
2. Compute energy proxy as KPI.
3. Compare B0/B1/B2/A under the same windows and demand.
4. Check whether values are plausible and stable.
5. Only after validation, consider promotion into reward shaping.

## Safety rule

Do not use this model for paper-level causal claims by itself.

It should be reported as a proxy indicator and interpreted together with waiting time, bunching rate, intervention rate, and service quality KPIs.
