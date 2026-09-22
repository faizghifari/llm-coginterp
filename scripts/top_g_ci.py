"""Average normalized rank of benchmarks by |g| loading across ALL factor cells.

Globs every `*_bifactor_pa_loadings.csv` under the results root that is
registered as a `pa` run in the factoring table of that root's database.db
(orphan files from retired methods are skipped), ranks benchmarks within each
cell by |g| (rank 1 = highest), normalizes each rank to (r - 1) / (n - 1) in
[0, 1] so cells with different benchmark counts are comparable, averages
across all cells a benchmark appears in, and prints the top 20 by lowest
average normalized rank in a markdown table with sd and a t-based 95% CI on
the mean.

Usage: python scripts/top_g_ci.py [--results-root results/text_only]
"""

import argparse
import csv
import glob
import math
import os
import re
import sqlite3
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TOP_N = 20
CELL_RE = re.compile(
    r"^(?P<method>.+?)_(?P<dz>C|R|S|raw)_(?P<st>all_standard|all_aggressive)"
    r"_bifactor_pa_loadings\.csv$"
)


def registered_pa_runs(db_path):
    """{dataset} with a `pa` run in the factoring table, or None if no db."""
    if not os.path.exists(db_path):
        return None
    con = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT DISTINCT dataset FROM factoring WHERE run = 'pa'"
    ).fetchall()
    con.close()
    return {r[0] for r in rows}


def read_g(path):
    """Return {benchmark: g} from a loadings CSV."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[row["benchmark"]] = float(row["g"])
    return out


def parse_cell(filename):
    """Extract (method, dataset) from `<method>_<dz>_<st>_bifactor_pa_loadings.csv`."""
    m = CELL_RE.match(os.path.basename(filename))
    if not m:
        return None
    return m.group("method"), f"{m.group('dz')}_{m.group('st')}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default=os.path.join(REPO, "results", "text_only"))
    args = ap.parse_args()

    allowed = registered_pa_runs(os.path.join(args.results_root, "database.db"))
    if allowed is None:
        print(
            "no database.db under results root -- accepting every loadings file; "
            "run factoring to register runs"
        )

    # benchmark -> [normalized rank (0 = top of cell) per cell]
    norm_ranks = defaultdict(list)
    cells = 0
    for path in sorted(glob.glob(
        os.path.join(args.results_root, "*", "*_bifactor_pa_loadings.csv")
    )):
        cell = parse_cell(path)
        if not cell:
            continue
        if allowed is not None and cell[1] not in allowed:
            print(f"skipping {os.path.basename(path)}: {cell[1]} not in database.db")
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
        from scipy.stats import t as tdist

        n = len(vals)
        m = sum(vals) / n
        sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
        t = float(tdist.ppf(0.975, n - 1))  # exact for any df, incl. df < 1
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
