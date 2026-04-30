from __future__ import annotations

import argparse
import json
import psycopg2


KEYWORDS = [
    "승차",
    "하차",
    "boarding",
    "alighting",
    "ridership",
    "passenger",
    "usage",
    "stop",
    "demand",
    "count",
    "wait",
    "load",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-url", required=True)
    args = parser.parse_args()

    conn = psycopg2.connect(args.db_url)

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  table_schema,
                  table_name,
                  column_name,
                  data_type,
                  ordinal_position
                FROM information_schema.columns
                WHERE table_schema = 'public'
                ORDER BY table_schema, table_name, ordinal_position
                """
            )

            rows = cur.fetchall()

        table_map = {}

        for schema, table, col, dtype, pos in rows:
            rel = f"{schema}.{table}"
            table_map.setdefault(rel, [])
            table_map[rel].append(
                {
                    "column_name": str(col),
                    "data_type": str(dtype),
                    "ordinal_position": int(pos),
                }
            )

        candidates = []

        for rel, cols in table_map.items():
            rel_lower = rel.lower()
            col_names = [c["column_name"] for c in cols]
            col_lower = " ".join(c.lower() for c in col_names)

            score = 0
            matched = []

            for kw in KEYWORDS:
                kw_lower = kw.lower()
                if kw_lower in rel_lower or kw_lower in col_lower:
                    score += 1
                    matched.append(kw)

            # 정류장/수요 후보로 보이는 relation만 출력
            if score > 0:
                candidates.append(
                    {
                        "relation": rel,
                        "score": score,
                        "matched_keywords": matched,
                        "columns": col_names,
                    }
                )

        candidates.sort(key=lambda x: (-x["score"], x["relation"]))

        print("[OK] Ridership / stop / demand candidate tables")
        print(f"[OK] candidate_count = {len(candidates)}")
        print("")

        for item in candidates:
            print("=" * 100)
            print(f"[RELATION] {item['relation']}")
            print(f"[SCORE]    {item['score']}")
            print(f"[MATCHED]  {', '.join(item['matched_keywords'])}")
            print("[COLUMNS]")
            for col in item["columns"]:
                print(f"  - {col}")
            print("")

        out_path = "artifacts/pre_tensor_db_year_coverage/ridership_candidate_tables.json"

        import pathlib
        pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "candidate_count": len(candidates),
                    "candidates": candidates,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"[OK] saved: {out_path}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
