#!/usr/bin/env python3
"""Daegu citywide observed historical demand ledger (H4M-AE-R8, Layer A).

Layer A carries only what the historical source observes: stop-hour boarding and
alighting counts with district provenance.  No passenger identity, no OD, no
sub-hour timestamp is invented here.

The ledger is built by server-side aggregation and written as a month-partitioned
Parquet dataset so a 24GB Mac mini never has to hold the 21.6M-row source in
memory, and so the same code serves a larger machine unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

LEDGER_ID = "DAEGU_CITYWIDE_HISTORICAL_DEMAND_LEDGER"
LEDGER_VERSION = "V1"
PG_BIN = "/opt/homebrew/opt/postgresql@18/bin"
DB = "urbanbus"
SESSION_GUARDS = (
    "SET default_transaction_read_only = on;\n"
    "SET statement_timeout = '120s';\n"
    "SET lock_timeout = '5s';\n"
)

# Semantic role -> observed source column.  Every field is OBSERVED at Layer A.
LEDGER_SCHEMA: Dict[str, Dict[str, str]] = {
    "service_date": {"unit": "calendar date", "provenance": "OBSERVED", "source": "fact_stop_usage_hourly.service_date"},
    "service_hour": {"unit": "hour of day 0-23", "provenance": "OBSERVED", "source": "fact_stop_usage_hourly.service_hour"},
    "stop_id": {"unit": "stop identifier", "provenance": "OBSERVED", "source": "fact_stop_usage_hourly.stop_id"},
    "boardings": {"unit": "boarding event count", "provenance": "OBSERVED", "source": "fact_stop_usage_hourly.boardings"},
    "alightings": {"unit": "alighting event count", "provenance": "OBSERVED", "source": "fact_stop_usage_hourly.alightings"},
    "admin_area": {"unit": "administrative area string", "provenance": "OBSERVED", "source": "stg_daegu_stop_usage_2023.admin_area_raw"},
    "district": {"unit": "gu/gun name", "provenance": "DERIVED", "source": "token 2 of admin_area"},
}
DISTRICT_TOKEN_INDEX = 1  # "대구광역시 <gu/gun> <dong>"


class LedgerError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceQuery:
    label: str
    sql: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.sql.encode("utf-8")).hexdigest()


def _psql(sql: str) -> str:
    env = dict(os.environ, PATH=f"{PG_BIN}:{os.environ['PATH']}")
    proc = subprocess.run([f"{PG_BIN}/psql", "-d", DB, "-v", "ON_ERROR_STOP=1", "-c", sql],
                          capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise LedgerError(f"read-only query failed: {proc.stderr.strip()[:400]}")
    return proc.stdout


def read_only_copy(sql_body: str) -> pd.DataFrame:
    """Run one read-only COPY and return it as a frame.  No write is ever issued."""
    sql = SESSION_GUARDS + f"COPY ({sql_body}) TO STDOUT WITH CSV HEADER"
    out = _psql(sql)
    lines = [ln for ln in out.splitlines() if ln and ln != "SET"]
    return pd.read_csv(StringIO("\n".join(lines)), dtype={"stop_id": str, "admin_area": str})


def district_of(admin_area: Optional[str]) -> Optional[str]:
    if not isinstance(admin_area, str):
        return None
    parts = admin_area.split()
    return parts[DISTRICT_TOKEN_INDEX] if len(parts) > DISTRICT_TOKEN_INDEX else None


def stop_district_map() -> pd.DataFrame:
    q = SourceQuery("stop_district", """
        SELECT DISTINCT stop_id_raw AS stop_id, admin_area_raw AS admin_area
        FROM stg_daegu_stop_usage_2023
        WHERE stop_id_raw IS NOT NULL
    """)
    frame = read_only_copy(q.sql)
    frame = frame.drop_duplicates("stop_id")
    frame["district"] = frame["admin_area"].map(district_of)
    return frame


def month_partition_query(year: int, month: int) -> SourceQuery:
    start = f"{year:04d}-{month:02d}-01"
    end = f"{year + (month // 12):04d}-{(month % 12) + 1:02d}-01"
    return SourceQuery(f"ledger_{year}{month:02d}", f"""
        SELECT service_date, service_hour, stop_id, boardings, alightings
        FROM public.fact_stop_usage_hourly
        WHERE service_date >= '{start}' AND service_date < '{end}'
        ORDER BY service_date, service_hour, stop_id
    """)


def source_extent() -> Dict[str, Any]:
    q = SourceQuery("extent", """
        SELECT min(service_date) AS min_date, max(service_date) AS max_date,
               count(DISTINCT service_date) AS date_count,
               min(service_hour) AS min_hour, max(service_hour) AS max_hour,
               count(DISTINCT stop_id) AS stop_count, count(*) AS row_count,
               sum(boardings) AS total_boardings, sum(alightings) AS total_alightings
        FROM public.fact_stop_usage_hourly
    """)
    row = read_only_copy(q.sql).iloc[0].to_dict()
    row["query_sha256"] = q.sha256
    return row


def build_ledger(output_root: Path, *, months: Iterable[tuple]) -> Dict[str, Any]:
    """Write a month-partitioned observed ledger.  Memory holds one month at a time."""
    output_root.mkdir(parents=True, exist_ok=True)
    districts = stop_district_map()
    partitions: List[Dict[str, Any]] = []
    totals = {"rows": 0, "boardings": 0, "alightings": 0}
    quality = {"duplicate_keys": 0, "negative_boardings": 0, "negative_alightings": 0,
               "null_counts": 0, "stops_without_district": set(), "impossible_hours": 0}
    stop_totals: Dict[str, Dict[str, int]] = {}
    hour_totals: Dict[int, Dict[str, int]] = {}
    district_totals: Dict[str, Dict[str, int]] = {}
    date_totals: Dict[str, Dict[str, int]] = {}

    for year, month in months:
        q = month_partition_query(year, month)
        frame = read_only_copy(q.sql)
        if frame.empty:
            continue
        frame = frame.merge(districts, on="stop_id", how="left")
        quality["duplicate_keys"] += int(frame.duplicated(["service_date", "service_hour", "stop_id"]).sum())
        quality["negative_boardings"] += int((frame["boardings"] < 0).sum())
        quality["negative_alightings"] += int((frame["alightings"] < 0).sum())
        quality["null_counts"] += int(frame[["boardings", "alightings"]].isna().sum().sum())
        quality["impossible_hours"] += int((~frame["service_hour"].between(0, 23)).sum())
        quality["stops_without_district"] |= set(frame.loc[frame["district"].isna(), "stop_id"].unique())

        frame = frame.sort_values(["service_date", "service_hour", "stop_id"], kind="mergesort").reset_index(drop=True)
        part = output_root / f"month={year:04d}-{month:02d}" / "part-0.parquet"
        part.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(part, index=False)

        totals["rows"] += len(frame)
        totals["boardings"] += int(frame["boardings"].sum())
        totals["alightings"] += int(frame["alightings"].sum())
        for stop, grp in frame.groupby("stop_id")[["boardings", "alightings"]].sum().iterrows():
            acc = stop_totals.setdefault(stop, {"boardings": 0, "alightings": 0})
            acc["boardings"] += int(grp["boardings"]); acc["alightings"] += int(grp["alightings"])
        for hour, grp in frame.groupby("service_hour")[["boardings", "alightings"]].sum().iterrows():
            acc = hour_totals.setdefault(int(hour), {"boardings": 0, "alightings": 0})
            acc["boardings"] += int(grp["boardings"]); acc["alightings"] += int(grp["alightings"])
        for dist, grp in frame.groupby(frame["district"].fillna("UNMAPPED"))[["boardings", "alightings"]].sum().iterrows():
            acc = district_totals.setdefault(str(dist), {"boardings": 0, "alightings": 0})
            acc["boardings"] += int(grp["boardings"]); acc["alightings"] += int(grp["alightings"])
        for date, grp in frame.groupby("service_date")[["boardings", "alightings"]].sum().iterrows():
            date_totals[str(date)] = {"boardings": int(grp["boardings"]), "alightings": int(grp["alightings"])}

        partitions.append({
            "partition": part.parent.name, "path": str(part.relative_to(output_root)),
            "rows": int(len(frame)), "boardings": int(frame["boardings"].sum()),
            "alightings": int(frame["alightings"].sum()),
            "sha256": hashlib.sha256(part.read_bytes()).hexdigest(),
            "query_sha256": q.sha256,
        })
        del frame

    quality["stops_without_district"] = sorted(quality["stops_without_district"])
    manifest = {
        "ledger_id": LEDGER_ID, "version": LEDGER_VERSION,
        "layer": "A_OBSERVED_HISTORICAL_AUTHORITY",
        "scope": "DAEGU_CITYWIDE",
        "suseong_filter_applied_during_construction": False,
        "schema": LEDGER_SCHEMA,
        "grain": "service_date x service_hour x stop_id",
        "partitioning": "month",
        "deterministic_ordering": "service_date, service_hour, stop_id",
        "partitions": partitions,
        "totals": totals,
        "distinct_stops": len(stop_totals),
        "distinct_dates": len(date_totals),
        "quality": quality,
        "dataset_sha256": hashlib.sha256(
            json.dumps([p["sha256"] for p in partitions], sort_keys=True).encode()).hexdigest(),
    }
    (output_root / "_ledger_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"manifest": manifest, "stop_totals": stop_totals, "hour_totals": hour_totals,
            "district_totals": district_totals, "date_totals": date_totals}


def load_scope_subset(output_root: Path, stop_ids: Iterable[str]) -> pd.DataFrame:
    """Deterministic scope filter over the citywide parent; never a new generator."""
    wanted = {str(s) for s in stop_ids}
    frames = []
    for part in sorted(output_root.glob("month=*/part-0.parquet")):
        frame = pd.read_parquet(part)
        hit = frame[frame["stop_id"].astype(str).isin(wanted)]
        if len(hit):
            frames.append(hit)
    if not frames:
        return pd.DataFrame(columns=list(LEDGER_SCHEMA))
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["service_date", "service_hour", "stop_id"], kind="mergesort").reset_index(drop=True)
