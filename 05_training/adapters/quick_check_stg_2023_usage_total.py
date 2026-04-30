from __future__ import annotations

import argparse
import psycopg2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-url", required=True)
    args = ap.parse_args()

    conn = psycopg2.connect(args.db_url)
    try:
        cur = conn.cursor()

        print("[1] staging raw summary")
        cur.execute("""
            SELECT
              COUNT(*) AS row_count,
              COUNT(DISTINCT service_date_raw) AS date_count,
              MIN(service_date_raw) AS min_date_raw,
              MAX(service_date_raw) AS max_date_raw,
              COUNT(DISTINCT stop_id_raw) AS stop_id_count,
              COUNT(DISTINCT mobile_id_raw) AS mobile_id_count,
              COUNT(DISTINCT usage_type_raw) AS usage_type_count,
              SUM(NULLIF(row_sum_raw::text, '')::numeric) AS row_sum_total
            FROM public.stg_daegu_stop_usage_2023;
        """)
        print(cur.fetchone())

        print("")
        print("[2] staging by usage_type")
        cur.execute("""
            SELECT
              usage_type_raw,
              COUNT(*) AS row_count,
              SUM(NULLIF(row_sum_raw::text, '')::numeric) AS row_sum_total
            FROM public.stg_daegu_stop_usage_2023
            GROUP BY usage_type_raw
            ORDER BY usage_type_raw;
        """)
        for row in cur.fetchall():
            print(row)

        print("")
        print("[EXPECTED CSV TOTALS]")
        print("boarding total  ≈ 181,556,972")
        print("alighting total ≈ 70,567,680")
        print("overall total   ≈ 252,124,652")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
