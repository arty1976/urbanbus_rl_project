from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import pandas as pd


ARTIFACT_VERSION = "terminal_dwell_reconstruction_v1"
DWELL_EVENT_COLUMNS = [
    "route_id",
    "direction_id",
    "vehicle_id",
    "terminal_stop_id",
    "arrival_time_lower",
    "arrival_time_upper",
    "departure_time_lower",
    "departure_time_upper",
    "dwell_seconds_lower",
    "dwell_seconds_upper",
    "dwell_seconds_midpoint",
    "left_censored",
    "right_censored",
    "tracking_gap_seconds",
    "next_direction_id",
    "next_sequence",
    "source_sample_count",
    "classification",
    "eligible_for_route_level_recovery",
]


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def reconstruct_events(samples: pd.DataFrame, terminals: pd.DataFrame, maximum_tracking_gap_seconds: float) -> List[Dict[str, Any]]:
    if samples.empty or terminals.empty:
        return []
    samples = samples.copy()
    samples["provider_event_time_parsed"] = pd.to_datetime(samples["provider_event_time"], errors="coerce")
    terminal_by_key = {
        (str(row["route_id"]), str(row["direction_id"])): str(row["terminal_stop_id"])
        for row in terminals.to_dict("records")
    }
    events: List[Dict[str, Any]] = []
    group_cols = ["route_id", "direction_id", "vehicle_id"]
    for (route_id, direction_id, vehicle_id), group in samples.dropna(subset=["vehicle_id"]).groupby(group_cols, dropna=False):
        terminal_stop_id = terminal_by_key.get((str(route_id), str(direction_id)))
        if terminal_stop_id is None:
            continue
        ordered = group.sort_values("provider_event_time_parsed")
        in_terminal = ordered["current_stop_id"].astype(str) == terminal_stop_id
        if not in_terminal.any():
            continue
        indices = list(ordered.index)
        terminal_indices = [idx for idx in indices if bool(in_terminal.loc[idx])]
        first = terminal_indices[0]
        last = terminal_indices[-1]
        first_pos = indices.index(first)
        last_pos = indices.index(last)
        prev_idx = indices[first_pos - 1] if first_pos > 0 else None
        next_idx = indices[last_pos + 1] if last_pos + 1 < len(indices) else None
        arrival_lower = None if prev_idx is None else ordered.loc[prev_idx, "provider_event_time_parsed"]
        arrival_upper = ordered.loc[first, "provider_event_time_parsed"]
        departure_lower = ordered.loc[last, "provider_event_time_parsed"]
        departure_upper = None if next_idx is None else ordered.loc[next_idx, "provider_event_time_parsed"]
        left_censored = prev_idx is None
        right_censored = next_idx is None
        lower = 0.0 if pd.isna(departure_lower) or pd.isna(arrival_upper) else max(0.0, (departure_lower - arrival_upper).total_seconds())
        upper = None
        if departure_upper is not None and arrival_lower is not None and not pd.isna(departure_upper) and not pd.isna(arrival_lower):
            upper = max(lower, (departure_upper - arrival_lower).total_seconds())
        gap = None
        if departure_upper is not None and departure_lower is not None and not pd.isna(departure_upper) and not pd.isna(departure_lower):
            gap = (departure_upper - departure_lower).total_seconds()
        events.append(
            {
                "route_id": str(route_id),
                "direction_id": str(direction_id),
                "vehicle_id": str(vehicle_id),
                "terminal_stop_id": terminal_stop_id,
                "arrival_time_lower": None if arrival_lower is None or pd.isna(arrival_lower) else arrival_lower.isoformat(),
                "arrival_time_upper": None if pd.isna(arrival_upper) else arrival_upper.isoformat(),
                "departure_time_lower": None if pd.isna(departure_lower) else departure_lower.isoformat(),
                "departure_time_upper": None if departure_upper is None or pd.isna(departure_upper) else departure_upper.isoformat(),
                "dwell_seconds_lower": lower,
                "dwell_seconds_upper": upper,
                "dwell_seconds_midpoint": None if upper is None else (lower + upper) / 2.0,
                "left_censored": left_censored,
                "right_censored": right_censored,
                "tracking_gap_seconds": gap,
                "next_direction_id": None if next_idx is None else ordered.loc[next_idx, "direction_id"],
                "next_sequence": None if next_idx is None else ordered.loc[next_idx, "current_sequence"],
                "source_sample_count": int(len(ordered)),
                "classification": "OBSERVED_INTERVAL_CENSORED_DWELL_CANDIDATE",
                "eligible_for_route_level_recovery": bool((not left_censored) and (not right_censored) and gap is not None and gap <= maximum_tracking_gap_seconds),
            }
        )
    return events


def audit_events(events: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    invariant_failures = 0
    for row in events:
        if row.get("vehicle_id") in {None, ""} or row.get("route_id") in {None, ""} or row.get("terminal_stop_id") in {None, ""}:
            invariant_failures += 1
        lower = row.get("dwell_seconds_lower")
        upper = row.get("dwell_seconds_upper")
        if lower is not None and float(lower) < 0:
            invariant_failures += 1
        if upper is not None and lower is not None and float(upper) < float(lower):
            invariant_failures += 1
    return {
        "artifact_version": ARTIFACT_VERSION,
        "event_count": len(events),
        "eligible_event_count": sum(1 for row in events if bool(row.get("eligible_for_route_level_recovery"))),
        "invariant_failure_count": invariant_failures,
        "event_reconstruction_invariants_pass": invariant_failures == 0,
        "exact_dwell_claim_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct interval-censored terminal dwell events from position samples.")
    parser.add_argument("--samples", required=True)
    parser.add_argument("--terminals", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--maximum-tracking-gap-seconds", type=float, required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = pd.read_parquet(args.samples)
    terminals = pd.read_parquet(args.terminals) if str(args.terminals).endswith(".parquet") else pd.read_csv(args.terminals)
    events = reconstruct_events(samples, terminals, args.maximum_tracking_gap_seconds)
    pd.DataFrame(events, columns=DWELL_EVENT_COLUMNS).to_parquet(output_dir / "terminal_dwell_events.parquet", index=False)
    audit = audit_events(events)
    audit["maximum_tracking_gap_seconds"] = args.maximum_tracking_gap_seconds
    dump_json(output_dir / "terminal_dwell_audit.json", audit)
    print(json.dumps({"event_count": len(events), "invariant_failure_count": audit["invariant_failure_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
