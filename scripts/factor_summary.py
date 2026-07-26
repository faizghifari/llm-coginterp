#!/usr/bin/env python3
"""Print bifactor factoring results from results/database.db, sorted by omega_h."""

import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = REPO_ROOT / "results" / "database.db"


def main():
    if not DB_PATH.exists():
        print(f"No database at {DB_PATH} — run factoring first.", file=sys.stderr)
        sys.exit(1)

    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    cur = con.execute("""
        SELECT dataset, method, run, nf, var_explained, var_factors, var_avg,
               omega_t, omega_h, omega_hs, phi_avg, phi
        FROM factoring
        ORDER BY phi_avg DESC
    """)
    rows = cur.fetchall()
    con.close()

    if not rows:
        print("No rows in factoring table.")
        return

    hdr = (
        f"{'dataset':24s} {'method':14s} {'run':>8s} {'nf':>3s} "
        f"{'var%':>8s} {'var% avg':>8s} {'ωt':>8s} {'ωh':>8s}  "
        f"{'φ_avg':>6s}"
    )
    print(hdr)
    print("-" * 120)
    for r in rows:
        print(
            f"{r['dataset']:24s} {r['method']:14s} {r['run']:>8s} "
            f"{r['nf']:3d} "
            f"{r['var_explained'] or 0:8.3f} "
            f"{r['var_avg'] or 0:8.3f} "
            f"{r['omega_t'] or 0:8.3f} {r['omega_h'] or 0:8.3f}  "
            f"{r['phi_avg'] or 0:6.3f}  "
        )


if __name__ == "__main__":
    main()
