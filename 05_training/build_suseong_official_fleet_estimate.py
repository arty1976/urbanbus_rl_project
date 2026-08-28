from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


OFFICIAL_SOURCE_URL = "https://www.data.go.kr/data/15060936/fileData.do?recommendDataYn=Y"
OFFICIAL_DOWNLOAD_URL = (
    "https://www.data.go.kr/cmm/cmm/fileDownload.do?"
    "atchFileId=FILE_000000003527838&fileDetailSn=1&insertDataPrcus=N"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_minutes(value: Any) -> Optional[int]:
    text = str(value).strip()
    if not text or text.lower() == "nan" or ":" not in text:
        return None
    hour, minute = text.split(":", 1)
    try:
        return int(hour) * 60 + int(minute)
    except ValueError:
        return None


def span_minutes(values: pd.Series) -> Optional[float]:
    minutes = [v for v in (parse_minutes(x) for x in values) if v is not None]
    if len(minutes) < 2:
        return None
    adjusted: List[int] = []
    for minute in minutes:
        adjusted.append(minute + 24 * 60 if minute < 3 * 60 else minute)
    return float(max(adjusted) - min(adjusted))


def median_numeric(values: pd.Series) -> Optional[float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.median())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build official SUSEONG fleet estimate from Daegu timetable/headway file.")
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--official-csv", default="05_training/artifacts/official_public_sources/daegu_stop_route_avg_headway_20251114.csv")
    parser.add_argument("--service-graph-dir", default="05_training/artifacts/suseong_service_graph_v1")
    parser.add_argument("--time-table-type", default="평일")
    parser.add_argument("--central-layover-minutes-per-direction", type=float, default=10.0)
    parser.add_argument("--upper-layover-minutes-per-direction", type=float, default=20.0)
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    official_csv = (project_root / args.official_csv).resolve()
    service_graph_dir = (project_root / args.service_graph_dir).resolve()
    output_dir = service_graph_dir

    official = pd.read_csv(official_csv, dtype=str)
    routes = pd.read_csv(service_graph_dir / "service_routes.csv", dtype=str)
    service_route_nos = sorted(routes["route_no"].astype(str).unique())
    eligible_direction_count_by_route = (
        routes.assign(eligible_direction_count=pd.to_numeric(routes["eligible_direction_count"], errors="coerce").fillna(0).astype(int))
        .set_index("route_no")["eligible_direction_count"]
        .to_dict()
    )

    official["노선"] = official["노선"].astype(str)
    selected_groups: List[pd.DataFrame] = []
    selected_time_table_types: Dict[str, List[str]] = {}
    for route_no in service_route_nos:
        route_rows = official[official["노선"] == route_no]
        preferred = route_rows[route_rows["시간표유형"].astype(str) == args.time_table_type]
        if preferred.empty:
            preferred = route_rows[route_rows["시간표유형"].astype(str) == "공통"]
        if preferred.empty:
            preferred = route_rows
        if not preferred.empty:
            selected_groups.append(preferred)
            selected_time_table_types[route_no] = sorted(preferred["시간표유형"].astype(str).unique())
    official_service = pd.concat(selected_groups, ignore_index=True) if selected_groups else official.iloc[0:0].copy()

    rows: List[Dict[str, Any]] = []
    blockers: List[str] = []
    for (route_no, direction), group in official_service.groupby(["노선", "진행방향"], sort=True):
        first_span = span_minutes(group["첫차"])
        last_span = span_minutes(group["막차"])
        runtime_candidates = [v for v in [first_span, last_span] if v is not None and v > 0]
        runtime_min = float(max(runtime_candidates)) if runtime_candidates else None
        avg_headway = median_numeric(group["평균배차간격"])
        run_count = median_numeric(group["운행횟수"])
        if runtime_min is None or avg_headway is None or avg_headway <= 0:
            blockers.append(f"OFFICIAL_TIMETABLE_RUNTIME_OR_HEADWAY_MISSING:{route_no}:{direction}")
            continue
        rows.append(
            {
                "route_no": route_no,
                "official_direction": str(direction),
                "official_stop_rows": int(len(group)),
                "official_distinct_stop_count": int(group["정류소ID"].nunique()),
                "first_car_span_minutes": first_span,
                "last_car_span_minutes": last_span,
                "one_way_runtime_minutes": runtime_min,
                "weekday_run_count_median": run_count,
                "avg_headway_minutes_median": avg_headway,
                "source": "official_stop_route_avg_headway_20251114",
            }
        )

    direction_df = pd.DataFrame(rows)
    if direction_df.empty:
        blockers.append("NO_OFFICIAL_SERVICE_ROUTE_MATCH")

    route_rows: List[Dict[str, Any]] = []
    matched_route_nos = set(direction_df["route_no"].astype(str)) if not direction_df.empty else set()
    missing_route_nos = sorted(set(service_route_nos) - matched_route_nos)
    if missing_route_nos:
        blockers.append("OFFICIAL_ROUTE_MATCH_INCOMPLETE")

    if not direction_df.empty:
        for route_no, group in direction_df.groupby("route_no", sort=True):
            eligible_dirs = int(eligible_direction_count_by_route.get(route_no, len(group)))
            group_sorted = group.sort_values("one_way_runtime_minutes", ascending=False).head(eligible_dirs)
            runtime_total = float(group_sorted["one_way_runtime_minutes"].sum())
            headway = float(group_sorted["avg_headway_minutes_median"].median())
            lower = int(math.ceil(runtime_total / headway))
            central = int(math.ceil((runtime_total + args.central_layover_minutes_per_direction * len(group_sorted)) / headway))
            upper = int(math.ceil((runtime_total + args.upper_layover_minutes_per_direction * len(group_sorted)) / headway))
            route_rows.append(
                {
                    "route_no": str(route_no),
                    "eligible_direction_count": eligible_dirs,
                    "official_direction_count_used": int(len(group_sorted)),
                    "runtime_total_minutes": runtime_total,
                    "avg_headway_minutes_median": headway,
                    "fleet_lower_route": lower,
                    "fleet_central_route": central,
                    "fleet_upper_route": upper,
                    "calculation": "ceil((sum_official_direction_runtime + layover_per_direction) / official_avg_headway)",
                }
            )

    route_df = pd.DataFrame(route_rows)
    if not route_df.empty:
        direction_df.to_csv(output_dir / "official_frequency_route_direction_runtime_audit.csv", index=False)
        route_df.to_csv(output_dir / "official_fleet_route_estimate.csv", index=False)

    fleet_lower = int(route_df["fleet_lower_route"].sum()) if not route_df.empty else None
    fleet_central = int(route_df["fleet_central_route"].sum()) if not route_df.empty else None
    fleet_upper = int(route_df["fleet_upper_route"].sum()) if not route_df.empty else None
    official_frequency_pass = not any(b.startswith("OFFICIAL_TIMETABLE_RUNTIME_OR_HEADWAY_MISSING") for b in blockers)
    official_match_pass = not missing_route_nos
    status = "PASS" if official_frequency_pass and fleet_central is not None and fleet_central > 0 else "BLOCKED"
    if not official_match_pass:
        status = "PASS_WITH_ROUTE_MATCH_WARNING" if status == "PASS" else status

    manifest = {
        "created_at_utc": utc_now(),
        "artifact_version": "suseong_official_fleet_estimate_v1",
        "status": status,
        "claim_scope": "official_frequency_and_timetable_derived_fleet_estimate_for_prompt5_A_condition",
        "official_source": {
            "name": "대구광역시_시내버스 정류소별_노선별_평균배차간격_20251114",
            "url": OFFICIAL_SOURCE_URL,
            "download_url": OFFICIAL_DOWNLOAD_URL,
            "local_path": str(official_csv),
            "sha256": sha256_file(official_csv),
            "row_count": int(len(official)),
            "columns": list(official.columns),
            "dataset_date": "2025-11-14",
            "registered_or_modified_date": "2025-11-17",
            "provider": "대구광역시",
            "department": "교통정보운영과",
            "contact": "053-803-6861",
        },
        "method": {
            "time_table_type": args.time_table_type,
            "time_table_type_fallback": "공통 is used only when the requested type has no rows for a route.",
            "selected_time_table_types_by_route": selected_time_table_types,
            "route_scope": "SUSEONG_SERVICE_ELIGIBLE_SUBSET route_no list",
            "runtime_source": "official first-car/last-car stop-time span by route and direction",
            "frequency_source": "official average headway and operation count",
            "lower_formula": "ceil(sum(one_way_runtime_minutes_by_used_direction) / median_avg_headway)",
            "central_formula": "ceil((runtime_total + 10min * used_direction_count) / median_avg_headway)",
            "upper_formula": "ceil((runtime_total + 20min * used_direction_count) / median_avg_headway)",
        },
        "route_match": {
            "service_route_count": len(service_route_nos),
            "official_matched_route_count": len(matched_route_nos),
            "missing_route_nos": missing_route_nos,
            "official_route_match_pass": official_match_pass,
        },
        "fleet_lower_estimate": fleet_lower,
        "fleet_central_estimate": fleet_central,
        "fleet_upper_estimate": fleet_upper,
        "fleet_value_status": "OFFICIAL_FREQUENCY_TIMETABLE_DERIVED",
        "fleet_estimate_ready": status in {"PASS", "PASS_WITH_ROUTE_MATCH_WARNING"},
        "central_or_upper_fleet_estimate_allowed": status in {"PASS", "PASS_WITH_ROUTE_MATCH_WARNING"},
        "official_frequency_blocker_resolved": status in {"PASS", "PASS_WITH_ROUTE_MATCH_WARNING"},
        "blockers": blockers,
        "warnings": [
            "FLEET_ESTIMATE_IS_TIMETABLE_DERIVED_NOT_DIRECT_VEHICLE_ROSTER",
            "ROUTE_DIRECTION_MAPPING_USES_ROUTE_NO_AND_OFFICIAL_DIRECTION_SPANS",
        ]
        + (["OFFICIAL_ROUTE_MATCH_INCOMPLETE"] if missing_route_nos else []),
        "outputs": {
            "official_frequency_route_direction_runtime_audit": str(output_dir / "official_frequency_route_direction_runtime_audit.csv"),
            "official_fleet_route_estimate": str(output_dir / "official_fleet_route_estimate.csv"),
        },
    }
    dump_json(output_dir / "suseong_official_fleet_estimate.json", manifest)

    md = [
        "# SUSEONG Official Fleet Estimate",
        "",
        f"Status: {status}",
        f"Official source: 대구광역시_시내버스 정류소별_노선별_평균배차간격_20251114",
        f"Fleet lower: {fleet_lower}",
        f"Fleet central: {fleet_central}",
        f"Fleet upper: {fleet_upper}",
        "",
        "Claim guard: timetable-derived official frequency estimate; not a direct official vehicle roster.",
    ]
    (output_dir / "suseong_official_fleet_estimate.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    if manifest["fleet_estimate_ready"]:
        service_graph_manifest_path = output_dir / "service_graph_manifest.json"
        service_graph_manifest = json.loads(service_graph_manifest_path.read_text(encoding="utf-8"))
        service_graph_manifest["fleet_lower_estimate"] = fleet_lower
        service_graph_manifest["fleet_central_estimate"] = fleet_central
        service_graph_manifest["fleet_upper_estimate"] = fleet_upper
        service_graph_manifest["fleet_value_status"] = manifest["fleet_value_status"]
        service_graph_manifest["fleet_estimate_ready"] = True
        service_graph_manifest["official_fleet_estimate_source"] = str(output_dir / "suseong_official_fleet_estimate.json")
        service_graph_manifest["blockers"] = [
            b for b in service_graph_manifest.get("blockers", []) if b != "FLEET_FREQUENCY_NOT_OFFICIAL"
        ]
        service_graph_manifest["warnings"] = [
            w for w in service_graph_manifest.get("warnings", []) if w != "FLEET_FREQUENCY_NOT_OFFICIAL_REMAINS_FOR_PROMPT4_TO_6"
        ] + manifest["warnings"]
        dump_json(service_graph_manifest_path, service_graph_manifest)

        fleet_gate = {
            "created_at_utc": utc_now(),
            "status": "PASS",
            "fleet_lower_estimate": fleet_lower,
            "fleet_central_estimate": fleet_central,
            "fleet_upper_estimate": fleet_upper,
            "fleet_value_status": manifest["fleet_value_status"],
            "official_source": manifest["official_source"],
            "method": manifest["method"],
            "claim_scope": (
                "Prompt 5 A-condition construction is allowed using the official timetable-derived central fleet estimate. "
                "This is not a direct official vehicle roster and does not by itself allow operational performance claims."
            ),
            "blockers": [],
            "warnings": manifest["warnings"],
            "central_or_upper_fleet_estimate_allowed": True,
            "performance_claim_allowed": False,
            "fleet_scientific_claim_allowed": True,
            "operational_performance_claim_allowed": False,
        }
        dump_json(output_dir / "prompt1_scientific_fleet_gate.json", fleet_gate)

        public_manifest_path = output_dir / "official_public_sources_manifest.json"
        public_manifest = json.loads(public_manifest_path.read_text(encoding="utf-8"))
        existing_names = {entry.get("name") for entry in public_manifest.get("sources", [])}
        if "daegu_stop_route_avg_headway_20251114" not in existing_names:
            public_manifest.setdefault("sources", []).append(
                {
                    "name": "daegu_stop_route_avg_headway_20251114",
                    "url": OFFICIAL_SOURCE_URL,
                    "download_url": OFFICIAL_DOWNLOAD_URL,
                    "local_path": str(official_csv),
                    "sha256": sha256_file(official_csv),
                    "row_count": int(len(official)),
                    "columns": list(official.columns),
                    "note": "Official timetable/frequency source used for SUSEONG fleet central estimate.",
                }
            )
            dump_json(public_manifest_path, public_manifest)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
