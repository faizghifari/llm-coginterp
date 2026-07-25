#!/usr/bin/env python3
"""Print held-out imputation results from results/database.db, sorted by R²."""

import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = REPO_ROOT / "results" / "database.db"


def main():
    if not DB_PATH.exists():
        print(f"No database at {DB_PATH} — run imputation first.", file=sys.stderr)
        sys.exit(1)

    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    cur = con.execute("""
        SELECT dataset, method, rmse, r2, desc
        FROM imputation
        ORDER BY r2 DESC
    """)
    rows = cur.fetchall()
    con.close()

    if not rows:
        print("No rows in imputation table.")
        return

    print(f"{'dataset':24s} {'method':14s} {'rmse':>8s} {'r2':>8s}  desc")
    print("-" * 90)
    for r in rows:
        print(f"{r['dataset']:24s} {r['method']:14s} {r['rmse']:8.4f} {r['r2']:8.3f}  {r['desc']}")


if __name__ == "__main__":
    main()