"""Top-10 benchmarks by g loading per pipeline cell.

For each hardcoded (dz, st, method) combo, reads the `pa` bifactor loadings
CSV (one row per benchmark, general-factor loading in column `g`) and prints
a table: top 10 benchmarks by |g|, with benchmark names and loadings.

Usage: python scripts/top_g.py [--results-root results/text_only]
"""

import argparse
import csv
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (dz, st, method) — dz "raw" means the undensified source table, no densifier.
COMBOS = [
    ("R", "all_aggressive", "default"),
    ("R", "all_aggressive", "zeros"),
    ("C", "all_aggressive", "softimpute"),
    ("C", "all_aggressive", "default"),
    # ("C", "all_aggressive", "missforest"),
    # ("C", "all_aggressive", "onesidedmc"),
    # ("R", "all_aggressive", "default"),
    # ("S", "all_aggressive", "missforest"),
]

TOP_N = 10


def read_g(path):
    """Return {benchmark: g} from a loadings CSV."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[row["benchmark"]] = float(row["g"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default=os.path.join(REPO, "results", "text_only"))
    args = ap.parse_args()

    cols = []  # (col_name, [(bench, g), ...])
    for dz, st, method in COMBOS:
        path = os.path.join(
            args.results_root, method, f"{method}_{dz}_{st}_bifactor_pa_loadings.csv"
        )
        g = read_g(path)
        top = sorted(g.items(), key=lambda kv: abs(kv[1]), reverse=True)[:TOP_N]
        cols.append((f"{dz}_{st}_{method}", top))

    # Header hack for hcat: each combo occupies two columns (name, loading),
    # with the combo name in the first header cell and an empty second.
    print("| " + " | ".join(f"{name} |" for name, _ in cols) + " |")
    print("|" + "---|" * (2 * len(cols)))
    for i in range(TOP_N):
        row = []
        for _, top in cols:
            bench, v = top[i]
            row.extend([bench, f"{v:.3f}"])
        print("| " + " | ".join(row) + " | ")


if __name__ == "__main__":
    main()
