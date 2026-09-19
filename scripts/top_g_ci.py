"""Average normalized rank of benchmarks by |g| loading across ALL factor cells.

Globs every `*_bifactor_pa_loadings.csv` under the results root, ranks
benchmarks within each cell by |g| (rank 1 = highest), normalizes each rank
to (r - 1) / (n - 1) in [0, 1] so cells with different benchmark counts are
comparable, averages across all cells a benchmark appears in, and prints the
top 20 by lowest average normalized rank in a markdown table with sd and a
t-based 95% CI on the mean.

Usage: python scripts/top_g_ci.py [--results-root results/text_only]
"""

import argparse
import csv
import glob
import math
import os
import re
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TOP_N = 20
# t critical values (two-sided, 95%) by df, fallback z = 1.96 for df > 30
T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
    14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
    20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def read_g(path):
    """Return {benchmark: g} from a loadings CSV."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[row["benchmark"]] = float(row["g"])
    return out


def parse_cell(filename):
    """Extract (dz, st, method) from `<method>_<dz>_<st>_bifactor_pa_loadings.csv`."""
    m = re.match(r"(.+?)_(.+?)_(.+?)_bifactor_pa_loadings\.csv", os.path.basename(filename))
    return m.groups() if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default=os.path.join(REPO, "results", "text_only"))
    args = ap.parse_args()

    # benchmark -> [normalized rank (0 = top of cell) per cell]
    norm_ranks = defaultdict(list)
    cells = 0
    for path in sorted(glob.glob(
        os.path.join(args.results_root, "*", "*_bifactor_pa_loadings.csv")
    )):
        cell = parse_cell(path)
        if not cell:
            continue
        g = read_g(path)
        cells += 1
        n = len(g)
        for rank, bench in enumerate(
            sorted(g, key=lambda b: abs(g[b]), reverse=True), 1
        ):
            norm_ranks[bench].append((rank - 1) / (n - 1) if n > 1 else 0.0)

    avg = {
        bench: sum(vals) / len(vals)
        for bench, vals in norm_ranks.items()
        if len(vals) >= 2  # need more than one appearance to average meaningfully
    }
    top = sorted(avg.items(), key=lambda kv: kv[1])[:TOP_N]

    def ci95(vals):
        n = len(vals)
        m = sum(vals) / n
        sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
        t = T95.get(n - 1, 1.96)
        half = t * sd / math.sqrt(n)
        return sd, m - half, m + half

    print(f"({cells} cells, {len(norm_ranks)} benchmarks)\n")
    print(
        "| no | benchmark | avg norm rank | sd | 95% ci | n cells | best | worst |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for i, (bench, m) in enumerate(top, 1):
        vals = norm_ranks[bench]
        sd, lo, hi = ci95(vals)
        print(
            f"| {i} | {bench} | {m:.3f} | {sd:.3f} | [{lo:.3f}, {hi:.3f}] "
            f"| {len(vals)} | {min(vals):.3f} | {max(vals):.3f} |"
        )


if __name__ == "__main__":
    main()
