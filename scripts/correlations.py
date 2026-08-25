#!/usr/bin/env python3
"""Correlate omega_h (factoring) with R² (imputation) from a run's database.db (default results/, --results-root to override)."""

import argparse
import sqlite3
import sys
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO_ROOT / "results"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", default=str(DEFAULT_RESULTS),
                    help="results tree holding database.db (e.g. results/text_only)")
    return ap.parse_args()


def main(db_path):
    if not db_path.exists():
        print(f"No database at {db_path} — run the pipeline first.", file=sys.stderr)
        sys.exit(1)

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row

    cur = con.execute("""
        SELECT f.run, f.omega_h, i.r2, f.dataset, f.method
        FROM factoring f
        JOIN imputation i ON f.dataset = i.dataset AND f.method = i.method
        WHERE f.omega_h IS NOT NULL AND i.r2 IS NOT NULL
        ORDER BY f.run, i.r2 DESC
    """)
    rows = cur.fetchall()
    con.close()

    if not rows:
        print("No paired (omega_h, r2) rows — run both imputation and factoring.")
        return

    by_run: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        by_run.setdefault(r["run"], []).append((r["omega_h"], r["r2"]))

    def pearson(xs, ys):
        n = len(xs)
        if n < 3:
            return float("nan")
        mx = sum(xs) / n
        my = sum(ys) / n
        sx = math.sqrt(sum((x - mx) ** 2 for x in xs) / (n - 1))
        sy = math.sqrt(sum((y - my) ** 2 for y in ys) / (n - 1))
        cov = sum((x - mx) * (y - my) for y, x in zip(ys, xs)) / (n - 1)
        return cov / (sx * sy)

    print(f"{'run':>10s} {'n':>5s}  {'r':>8s}  interpretation")
    print("-" * 55)

    all_x, all_y = [], []
    all_pairs = []
    for run in sorted(by_run):
        pairs = by_run[run]
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        r_val = pearson(xs, ys)
        all_x.extend(xs)
        all_y.extend(ys)
        all_pairs.extend((run, p[0], p[1]) for p in pairs)

        print(f"{run:>10s} {len(pairs):5d}  {r_val:+.3f}")

    if len(by_run) > 1:
        r_all = pearson(all_x, all_y)
        print(f"{'overall':>10s} {len(all_x):5d}  {r_all:+.3f}")

    print()
    print(f"{'run':>10s} {'dataset':24s} {'method':14s} {'omega_h':>8s} {'r2':>8s}")
    print("-" * 70)
    for r in rows:
        print(
            f"{r['run']:>10s} {r['dataset']:24s} {r['method']:14s} "
            f"{r['omega_h']:8.3f} {r['r2']:8.3f}"
        )


if __name__ == "__main__":
    main(Path(parse_args().results_root) / "database.db")
