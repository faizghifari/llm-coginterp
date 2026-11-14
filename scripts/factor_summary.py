#!/usr/bin/env python3
"""Print bifactor factoring results from a run's database.db (default results/text_only, --results-root for the multimodal run), sorted by omega_h."""

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


def main(db_path, density):
    if not db_path.exists():
        print(f"No database at {db_path} — run factoring first.", file=sys.stderr)
        sys.exit(1)

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    cur = con.execute("""
        SELECT f.dataset, f.method, f.run, f.nf, f.var_explained, f.var_factors, f.var_avg,
               f.omega_t, f.omega_h, f.omega_hs, f.phi_avg, f.phi,
               i.r2, i.wr2
        FROM factoring f
        LEFT JOIN imputation i ON i.dataset = f.dataset AND i.method = f.method
        ORDER BY f.omega_h DESC
    """)
    rows = [r for r in cur.fetchall()
            if "_y20" not in (r["run"] or "") and str(r["run"]) == "pa"]
    con.close()

    if not rows:
        print("No rows in factoring table.")
        return

    use_wr2 = density == "0"
    col = "wr2" if use_wr2 else "r2"
    cols = ["dataset", "method", "run", "nf", "var%", "var% avg",
            "ωt", "ωh", "φ_avg", col]
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        r2 = r["r2"]
        wr2 = r["wr2"]
        picked = wr2 if use_wr2 else r2
        if picked is None:
            picked = r2 if use_wr2 else wr2
        vals = [
            r["dataset"], r["method"], r["run"],
            str(r["nf"]),
            f"{(r['var_explained'] or 0) * 100:.1f}%",
            f"{(r['var_avg'] or 0) * 100:.1f}%",
            f"{r['omega_t'] or 0:.3f}",
            f"{r['omega_h'] or 0:.3f}",
            f"{r['phi_avg'] or 0:.3f}",
            f"{picked or 0:.3f}",
        ]
        print("| " + " | ".join(str(v) for v in vals) + " |")


if __name__ == "__main__":
    _a = parse_args()
    main(Path(_a.results_root) / _a.density / "database.db", _a.density)
