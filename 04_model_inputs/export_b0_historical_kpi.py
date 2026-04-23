import argparse
import json
import os
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
from sqlalchemy import create_engine, text


def build_db_url(cli_url: str) -> str:
    if cli_url and cli_url.strip():
        return cli_url

    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5432")
    user = os.getenv("PGUSER", "")
    password = os.getenv("PGPASSWORD", "")
    database = os.getenv("PGDATABASE", "")

    if not user or not database:
        raise SystemExit(
            "Database connection info missing. "
            "Provide --db-url or set PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE."
        )

    return (
        f"postgresql+psycopg2://{quote_plus(user)}:{quote_plus(password)}"
        f"@{host}:{port}/{quote_plus(database)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-url", default="")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    db_url = build_db_url(args.db_url)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = create_engine(db_url)

    kpi_sql = """
    SELECT
        window_id,
        state_ts,
        service_date,
        time_band,
        cv_headway,
        avg_wait_seconds,
        bunching_rate,
        on_time_rate,
        intervention_rate,
        energy_proxy,
        n_nodes,
        n_wait_obs,
        waiting_sum,
        next_boardings_sum,
        next_alightings_sum,
        avg_delta_t_sec
    FROM public.baseline_b0_historical_kpi_by_window
    ORDER BY state_ts
    """

    meta_sql = """
    SELECT metadata::text AS metadata_text
    FROM public.baseline_b0_historical_kpi_metadata
    LIMIT 1
    """

    df = pd.read_sql(text(kpi_sql), engine)

    if df.empty:
        raise SystemExit("baseline_b0_historical_kpi_by_window is empty.")

    parquet_path = output_dir / "kpi_by_window.parquet"
    metadata_path = output_dir / "metadata.json"

    df.to_parquet(parquet_path, index=False)

    metadata = {
        "baseline_id": "B0_historical",
        "artifact_version": "baseline_v1",
        "export_status": "fallback_metadata_used"
    }

    with engine.connect() as conn:
        meta_row = conn.execute(text(meta_sql)).fetchone()

    if meta_row and meta_row[0]:
        metadata = json.loads(meta_row[0])
        metadata["export_status"] = "metadata_table_used"

    metadata["output_dir"] = str(output_dir)
    metadata["parquet_file"] = str(parquet_path)
    metadata["row_count"] = int(len(df))
    metadata["columns"] = list(df.columns)
    metadata["min_state_ts"] = str(df["state_ts"].min())
    metadata["max_state_ts"] = str(df["state_ts"].max())

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("[OK] Export complete")
    print(f"[OK] parquet : {parquet_path}")
    print(f"[OK] metadata: {metadata_path}")
    print(f"[OK] row_count: {len(df)}")


if __name__ == "__main__":
    main()
