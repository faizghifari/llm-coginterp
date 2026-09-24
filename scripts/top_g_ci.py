"""Average per-cell freq-residualized normalized rank of benchmarks by |g| loading.

Globs every `*_bifactor_pa_loadings.csv` under the results root that is
registered as a `pa` run in the factoring table of that root's database.db
(orphan files from retired methods are skipped), ranks benchmarks within each
cell by |g| (rank 1 = highest), normalizes each rank to (r - 1) / (n - 1) in
[0, 1] so cells with different benchmark counts are comparable, residualizes
each cell's normalized ranks against that cell's own benchmark frequency
(non-missing proportion in the dataset's sparse table, omega_loco's
definition), averages the residuals across all cells a benchmark appears in,
and prints the top 20 sorted by lowest average residual in a markdown table
that also keeps the raw average normalized rank with a t-based 95% CI on the
mean. Because benchmark frequency is known to correlate with g loadings
within cells, the residual ranking is the primary column; the raw ranking is
kept for comparison.

Diagnostics printed before the table:
- per-cell Pearson r between frequency and normalized rank (mean r, mean |r|,
  sign counts), showing the within-cell effect;
- pooled r(avg freq, avg rank) before and after the correction, stratified by
  the number of cells a benchmark appears in (n_cells), showing that the
  within-cell effect does not survive pooling with a consistent sign.

Fallbacks when a cell cannot be residualized: a benchmark without a frequency
in that cell (or a cell with < 3 freq/rank pairs or zero freq variance) gets
its centered raw rank (rank - mean_rank) instead of an OLS residual.

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
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = Path(REPO) / "data" / "text_only"

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


def sparse_csv_path(data_root: Path, dz: str, st: str) -> Path:
    """Sparse/unimputed table for a dataset, same convention as omega_loco."""
    sub = "combinations" if dz == "raw" else f"combinations_{dz}"
    return data_root / sub / st / "model_benchmark_table.csv"


def bench_frequencies(csv_path: Path, min_obs: int = 2) -> dict[str, float]:
    """{benchmark: non-missing proportion} from a sparse CSV (omega_loco filter)."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    n = len(rows)
    if not rows:
        return {}
    freqs = {}
    for col in [c for c in fieldnames if c != "collapse_key"]:
        values = []
        for row in rows:
            v = row.get(col, "").strip()
            if v == "" or v.upper() == "NA":
                continue
            try:
                values.append(float(v))
            except ValueError:
                pass
        n_obs = len(values)
        if n_obs < min_obs:
            continue
        if n_obs > 1:
            mean = sum(values) / n_obs
            var = sum((x - mean) ** 2 for x in values) / (n_obs - 1)
        else:
            var = 0.0
        if var == 0.0:
            continue
        freqs[col] = n_obs / n
    return freqs


def main():
    from scipy.stats import linregress, pearsonr

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
    # benchmark -> [per-cell OLS residual of normalized rank on freq]
    resid_ranks = defaultdict(list)
    # dataset -> {benchmark: non-missing proportion} (cached; sparse CSV read once)
    freq_by_dataset = {}
    cell_records = []  # (method, dataset, n common, r(freq, norm rank))
    no_freq_benches = 0
    unresidualized_cells = 0
    cells = 0
    for path in sorted(glob.glob(
        os.path.join(args.results_root, "*", "*_bifactor_pa_loadings.csv")
    )):
        cell = parse_cell(path)
        if not cell:
            continue
        method, ds = cell
        if allowed is not None and ds not in allowed:
            print(f"skipping {os.path.basename(path)}: {ds} not in database.db")
            continue
        if ds not in freq_by_dataset:
            dz, st = ds.split("_", 1)
            csv_path = sparse_csv_path(DATA_ROOT, dz, st)
            freq_by_dataset[ds] = (
                bench_frequencies(csv_path) if csv_path.exists() else {}
            )
            if not freq_by_dataset[ds]:
                print(f"no sparse frequencies for {ds} -- cell cannot be residualized")
        fmap = freq_by_dataset[ds]

        g = read_g(path)
        cells += 1
        n = len(g)
        ranks = {
            bench: (rank - 1) / (n - 1) if n > 1 else 0.0
            for rank, bench in enumerate(
                sorted(g, key=lambda b: abs(g[b]), reverse=True), 1
            )
        }
        for bench, a in ranks.items():
            norm_ranks[bench].append(a)

        # per-cell OLS residual of normalized rank on this cell's freq
        common = [b for b in ranks if b in fmap]
        f_vals = [fmap[b] for b in common]
        a_vals = [ranks[b] for b in common]
        mean_a = sum(a_vals) / len(a_vals) if a_vals else 0.0
        fit = None
        if len(common) >= 3 and len(set(f_vals)) > 1:
            fit = linregress(f_vals, a_vals)
            for b, f in zip(common, f_vals):
                resid_ranks[b].append(
                    (ranks[b] - mean_a) - fit.slope * (f - sum(f_vals) / len(f_vals))
                )
            cell_records.append((method, ds, len(common), float(fit.rvalue)))
        else:
            unresidualized_cells += 1
            for b in ranks:
                resid_ranks[b].append(ranks[b] - mean_a)
        for b in ranks:
            if b not in fmap:
                no_freq_benches += 1

    avg = {
        bench: sum(vals) / len(vals)
        for bench, vals in norm_ranks.items()
        if len(vals) >= 2  # need more than one appearance to average meaningfully
    }
    avg_resid = {
        bench: sum(vals) / len(vals)
        for bench, vals in resid_ranks.items()
        if len(vals) >= 2
    }

    # benchmark frequency = mean non-missing proportion over the datasets in
    # which the benchmark appears (omega_loco's definition of frequency)
    freq = defaultdict(list)
    for bench in avg:
        for ds, fmap in freq_by_dataset.items():
            f = fmap.get(bench)
            if f is not None:
                freq[bench].append(f)
    freq = {
        bench: sum(vals) / len(vals)
        for bench, vals in freq.items()
        if vals
    }
    if len(freq) < len(avg):
        print(
            f"warning: {len(avg) - len(freq)} benchmarks have no frequency; "
            "their residual ranking uses centered ranks only"
        )

    def ci95(vals):
        from scipy.stats import t as tdist

        n = len(vals)
        m = sum(vals) / n
        sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
        t = float(tdist.ppf(0.975, n - 1))  # exact for any df, incl. df < 1
        half = t * sd / math.sqrt(n)
        return sd, m - half, m + half

    # ---- diagnostics -----------------------------------------------------
    print(f"({cells} cells, {len(norm_ranks)} benchmarks)\n")

    print("### per-cell r(benchmark freq, normalized |g| rank)\n")
    print("(r < 0 = more frequent benchmarks load higher on g)\n")
    print("| method | dataset | n | r |")
    print("|---|---|---|---|")
    for method, ds, n_common, r_val in cell_records:
        print(f"| {method} | {ds} | {n_common} | {r_val:+.3f} |")
    if cell_records:
        rs = [r for _, _, _, r in cell_records]
        n_neg = sum(r < 0 for r in rs)
        print(
            f"\nmean r = {sum(rs) / len(rs):+.3f}, mean |r| = "
            f"{sum(abs(r) for r in rs) / len(rs):.3f}, "
            f"{n_neg} negative / {len(rs) - n_neg} positive cells"
        )
    if unresidualized_cells:
        print(
            f"\n{unresidualized_cells} cells could not be residualized "
            "(< 3 freq/rank pairs or zero freq variance); centered ranks used"
        )
    if no_freq_benches:
        print(f"\n{no_freq_benches} benchmark-cell pairs had no frequency (centered ranks)")
    print()

    print("### pooled r(avg freq, avg rank), before -> after correction\n")
    print("| n cells | n bench | r raw | r resid |")
    print("|---|---|---|---|")
    common = [b for b in avg if b in freq]

    def strat_r(sub):
        if len(sub) < 3:
            return float("nan"), float("nan")
        f = [freq[b] for b in sub]
        r_raw = pearsonr(f, [avg[b] for b in sub]).statistic
        r_res = pearsonr(f, [avg_resid[b] for b in sub]).statistic
        return float(r_raw), float(r_res)

    by_ncells = defaultdict(list)
    for b in common:
        by_ncells[len(norm_ranks[b])].append(b)
    for nc in sorted(by_ncells):
        sub = by_ncells[nc]
        if len(sub) < 5:
            continue
        r_raw, r_res = strat_r(sub)
        print(f"| {nc} | {len(sub)} | {r_raw:+.3f} | {r_res:+.3f} |")
    r_raw, r_res = strat_r(common)
    print(f"| pooled | {len(common)} | {r_raw:+.3f} | {r_res:+.3f} |")
    print()

    # ---- main table ------------------------------------------------------
    top = sorted(avg_resid.items(), key=lambda kv: kv[1])[:TOP_N]

    print(
        "| no | benchmark | resid rank | 95% ci | sd | avg norm rank | 95% ci "
        "| freq | n cells | best | worst |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for i, (bench, m) in enumerate(top, 1):
        raw_vals = norm_ranks[bench]
        res_vals = resid_ranks[bench]
        sd, lo, hi = ci95(res_vals)
        raw_sd, raw_lo, raw_hi = ci95(raw_vals)
        print(
            f"| {i} | {bench} | {m:+.3f} | [{lo:.3f}, {hi:.3f}] | {sd:.3f} "
            f"| {avg[bench]:.3f} | [{raw_lo:.3f}, {raw_hi:.3f}] "
            f"| {freq.get(bench, float('nan')):.2f} "
            f"| {len(raw_vals)} | {min(raw_vals):.3f} | {max(raw_vals):.3f} |"
        )


if __name__ == "__main__":
    main()
