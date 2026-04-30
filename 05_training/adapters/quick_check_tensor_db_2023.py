from __future__ import annotations

import argparse
import psycopg2

CHECKS = [
    ("public.graph_state_timeslice", "state_ts"),
    ("public.rl_state_training_base", "state_ts"),
    ("public.rl_stop_transition_features", "state_ts"),
    ("public.gatv2_snapshot_stop_features_train", "state_ts"),
    ("public.gatv2_snapshot_summary", "min_state_ts"),
]

def split_rel(rel: str):
    return rel.split(".", 1) if "." in rel else ("public", rel)

def exists_table(cur, schema, table):
    cur.execute("""
        SELECT EXISTS (
          SELECT 1
          FROM information_schema.tables
          WHERE table_schema=%s AND table_name=%s
        )
    """, (schema, table))
    return bool(cur.fetchone()[0])

def exists_column(cur, schema, table, col):
    cur.execute("""
        SELECT EXISTS (
          SELECT 1
          FROM information_schema.columns
          WHERE table_schema=%s AND table_name=%s AND column_name=%s
        )
    """, (schema, table, col))
    return bool(cur.fetchone()[0])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-url", required=True)
    args = ap.parse_args()

    conn = psycopg2.connect(args.db_url)
    try:
        cur = conn.cursor()

        print("[INFO] Quick 2023 check: core tensor-source relations only")
        print("")

        all_ok_2023 = True
        checked = 0

        for rel, col in CHECKS:
            schema, table = split_rel(rel)

            if not exists_table(cur, schema, table):
                print(f"[SKIP] {rel}: table/view not found")
                continue

            if not exists_column(cur, schema, table, col):
                print(f"[SKIP] {rel}: column not found: {col}")
                continue

            checked += 1

            q = f"""
            SELECT
              MIN({col}::timestamptz),
              MAX({col}::timestamptz)
            FROM {schema}.{table}
            WHERE {col} IS NOT NULL
            """
            cur.execute(q)
            min_ts, max_ts = cur.fetchone()

            min_year = min_ts.year if min_ts else None
            max_year = max_ts.year if max_ts else None

            if min_year == 2023 and max_year == 2023:
                verdict = "2023_ONLY_BY_MIN_MAX"
            else:
                verdict = "NOT_2023_ONLY_BY_MIN_MAX"
                all_ok_2023 = False

            print(f"[CHECK] {rel}.{col}")
            print(f"        min_ts  = {min_ts}")
            print(f"        max_ts  = {max_ts}")
            print(f"        verdict = {verdict}")
            print("")

        print("[SUMMARY]")
        print(f"checked_core_relations = {checked}")
        print(f"core_relations_look_2023_only = {all_ok_2023}")

        if checked == 0:
            print("[RESULT] 판단 불가: 핵심 relation/date column을 찾지 못함")
        elif all_ok_2023:
            print("[RESULT] 기존 tensor DB 원천은 2023년 기준으로 보는 것이 타당함")
            print("[NEXT] 2023 승하차 CSV만 사용해서 alignment audit 진행")
        else:
            print("[RESULT] 2023-only가 아닐 가능성 있음")
            print("[NEXT] 2023-only view를 먼저 만들고 진행")

    finally:
        conn.close()

if __name__ == "__main__":
    main()
