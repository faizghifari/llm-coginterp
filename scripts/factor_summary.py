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
    ap.add_argument("--results-root", default=str(DEFAULT_RESULTS),
                    help="results tree holding database.db (e.g. results/text_only)")
    return ap.parse_args()


def main(db_path):
    if not db_path.exists():
        print(f"No database at {db_path} — run factoring first.", file=sys.stderr)
        sys.exit(1)

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    cur = con.execute("""
        SELECT dataset, method, run, nf, var_explained, var_factors, var_avg,
               omega_t, omega_h, omega_hs, phi_avg, phi
        FROM factoring
        ORDER BY omega_h DESC
    """)
    rows = [r for r in cur.fetchall()
            if "_y20" not in (r["run"] or "") and str(r["run"]) == "pa"]
    con.close()

    if not rows:
        print("No rows in factoring table.")
        return

    cols = ["dataset", "method", "run", "nf", "var%", "var% avg",
            "ωt", "ωh", "φ_avg"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        vals = [
            r["dataset"], r["method"], r["run"],
            str(r["nf"]),
            f"{(r['var_explained'] or 0) * 100:.1f}%",
            f"{(r['var_avg'] or 0) * 100:.1f}%",
            f"{r['omega_t'] or 0:.3f}",
            f"{r['omega_h'] or 0:.3f}",
            f"{r['phi_avg'] or 0:.3f}",
        ]
        print("| " + " | ".join(str(v) for v in vals) + " |")


if __name__ == "__main__":
    main(Path(parse_args().results_root) / "database.db")
