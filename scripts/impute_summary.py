#!/usr/bin/env python3
"""Print held-out imputation results from a run's database.db (default results/text_only, --results-root for the multimodal run), sorted by R²."""

import argparse
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO_ROOT / "results" / "text_only"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--density", required=True, metavar="FRAC",
                    help="density slice, e.g. 0.1 (results root gains a <density> segment)")
    ap.add_argument("--results-root", default=str(DEFAULT_RESULTS),
                    help="results tree holding database.db (e.g. results/text_only)")
    return ap.parse_args()


def main(db_path):
    if not db_path.exists():
        print(f"No database at {db_path} — run imputation first.", file=sys.stderr)
        sys.exit(1)

    con = sqlite3.connect(str(db_path))
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

    print("| dataset | method | rmse | r2 | desc |")
    print("|---|---|---:|---:|---|")
    for r in rows:
        print(f"| {r['dataset']} | {r['method']} | {r['rmse']:.4f} | {r['r2']:.3f} | {r['desc']} |")


if __name__ == "__main__":
    _a = parse_args()
    main(Path(_a.results_root) / _a.density / "database.db")
