#!/usr/bin/env python3
"""LOCO delta omega_h vs benchmark frequency: pooled correlation table + plots.

For each (method, dataset) group in the loco table, pool delta_omega_h
across runs (elementwise mean, NaN-safe), then compute the Pearson
correlation between each benchmark's non-missing frequency in the original
sparse/unimputed dataset and its pooled delta_omega_h. Prints one row per
(method, dataset) group.

Also produces a 2x2 figure (one panel per imputation method), each panel
showing a LOWESS-smoothed line per dataset combo, so you can eyeball
whether the freq/delta_omega_h association looks linear or not.
"""

import argparse
import csv
import json
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "text_only"
RESULTS_ROOT = REPO_ROOT / "results" / "text_only"


def parse_dataset(dataset: str) -> tuple[str, str]:
    """'C_all_standard' -> ('C', 'all_standard'), 'raw_all_aggressive' -> ('raw', 'all_aggressive')."""
    idx = dataset.index("_")
    return dataset[:idx], dataset[idx + 1 :]


def sparse_csv_path(data_root: Path, dz: str, st: str) -> Path:
    sub = "combinations" if dz == "raw" else f"combinations_{dz}"
    return data_root / sub / st / "model_benchmark_table.csv"


def loadings_csv_path(results_root: Path, method: str, dataset: str, model: str) -> Path:
    return results_root / method / f"{method}_{dataset}_{model}_loadings.csv"


def bench_g_loadings(csv_path: Path) -> dict[str, float]:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        result = {}
        for row in reader:
            name = row["benchmark"].strip()
            g_val = float(row["g"])
            result[name] = g_val
    return result


def bench_frequencies(csv_path: Path, min_obs: int = 2) -> tuple[list[str], list[float]]:
    """Read sparse CSV, drop collapse_key, filter like prep_matrix (min obs + non-zero
    variance), return (names, non_missing_proportion) for kept columns."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    bench_cols = [c for c in fieldnames if c != "collapse_key"]
    if not rows:
        return [], []

    n = len(rows)
    kept_names = []
    kept_freqs = []

    for col in bench_cols:
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
        kept_names.append(col)
        kept_freqs.append(n_obs / n)

    return kept_names, kept_freqs


def pearson(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Pearson r and t-statistic. Returns (nan, nan) if n < 3 or zero variance."""
    n = len(xs)
    if n < 3:
        return float("nan"), float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs) / (n - 1))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys) / (n - 1))
    if sx == 0.0 or sy == 0.0:
        return float("nan"), float("nan")
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n - 1)
    r = cov / (sx * sy)
    t = r * math.sqrt((n - 2) / (1.0 - r * r)) if abs(r) < 1.0 else float("inf")
    return r, t


def pool_deltas(delta_lists: list[list]) -> list[float]:
    """Elementwise mean across runs, ignoring None/NaN at each position.
    Raises ValueError on length mismatch."""
    lengths = {len(d) for d in delta_lists}
    if len(lengths) != 1:
        raise ValueError(f"delta array length mismatch across runs: {sorted(lengths)}")
    n_cols = lengths.pop()
    pooled = []
    for i in range(n_cols):
        vals = [
            d[i]
            for d in delta_lists
            if d[i] is not None and not (isinstance(d[i], float) and math.isnan(d[i]))
        ]
        pooled.append(sum(vals) / len(vals) if vals else float("nan"))
    return pooled


def make_plots(
    plot_data: dict[str, dict[str, tuple[list[float], list[float]]]],
    output_path: Path,
    dpi: int,
    lowess_frac: float,
    ylabel: str = "Pooled delta omega_h (LOCO)",
    suptitle: str = "LOCO delta omega_h vs. benchmark frequency (LOWESS smoothed)",
) -> None:
    """Adaptive grid (2 cols), one panel per method, LOWESS-smoothed line per dataset."""
    import matplotlib.pyplot as plt
    from statsmodels.nonparametric.smoothers_lowess import lowess

    methods = sorted(plot_data.keys())
    ncols = 2
    nrows = math.ceil(len(methods) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows), sharex=True, sharey=True)
    if nrows == 1 and ncols == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten()
    cmap = plt.get_cmap("tab10")

    for ax, method in zip(axes_flat, methods):
        datasets = sorted(plot_data[method].keys())
        for i, dataset in enumerate(datasets):
            f_vals, d_vals = plot_data[method][dataset]
            if len(f_vals) < 3:
                continue
            color = cmap(i % 10)
            ax.scatter(f_vals, d_vals, s=12, alpha=0.3, color=color, edgecolors="none")
            smoothed = lowess(d_vals, f_vals, frac=lowess_frac, return_sorted=True)
            ax.plot(smoothed[:, 0], smoothed[:, 1], color=color, linewidth=2, label=dataset)
        ax.set_title(method)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.legend(fontsize=7, loc="best")

    for ax in axes_flat[len(methods):]:
        ax.axis("off")

    fig.supxlabel("Benchmark non-missing frequency")
    fig.supylabel(ylabel)
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="LOCO sensitivity: freq vs delta_omega_h and freq vs g-loading correlations")
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved plot")
    parser.add_argument("--lowess-frac", type=float, default=0.4, help="LOWESS smoothing span (0-1)")
    parser.add_argument("--no-plot", action="store_true", help="skip plot generation, table only")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    results_root = Path(args.results_root)
    db_path = results_root / "database.db"

    if not db_path.exists():
        print(f"No database at {db_path}", file=sys.stderr)
        sys.exit(1)

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    cur = con.execute(
        "SELECT dataset, method, nf, run, delta_omega_h FROM loco ORDER BY dataset, method, nf, run"
    )
    rows = cur.fetchall()
    con.close()

    if not rows:
        print("No rows in loco table.")
        return

    groups: dict[tuple[str, str, int], list[list]] = defaultdict(list)
    for r in rows:
        key = (r["dataset"], r["method"], r["nf"])
        groups[key].append(json.loads(r["delta_omega_h"]))

    freq_cache: dict[str, tuple[list[str], list[float]]] = {}

    results_delta = []
    plot_data_delta: dict[str, dict[str, tuple[list[float], list[float]]]] = defaultdict(dict)

    for (dataset, method, nf), delta_lists in groups.items():
        if any(len(d) == 0 for d in delta_lists):
            results_delta.append((method, dataset, nf, None, None, None, None, "empty delta array"))
            continue

        try:
            pooled = pool_deltas(delta_lists)
        except ValueError as e:
            results_delta.append((method, dataset, nf, None, None, None, None, str(e)))
            continue

        dz, st = parse_dataset(dataset)
        csv_path = sparse_csv_path(data_root, dz, st)
        if not csv_path.exists():
            results_delta.append((method, dataset, nf, None, None, None, None, "sparse CSV not found"))
            continue

        if dataset not in freq_cache:
            freq_cache[dataset] = bench_frequencies(csv_path)
        bench_names, freqs = freq_cache[dataset]
        if len(freqs) != len(pooled):
            results_delta.append((method, dataset, nf, None, None, None, None, "column-filter mismatch"))
            continue

        valid_pairs = [(f, d) for f, d in zip(freqs, pooled) if not math.isnan(d)]
        if len(valid_pairs) < 3:
            results_delta.append((method, dataset, nf, None, None, None, len(valid_pairs), "not enough pairs"))
            continue

        f_vals = [p[0] for p in valid_pairs]
        d_vals = [p[1] for p in valid_pairs]
        plot_data_delta[method][f"{dataset} (nf={nf})"] = (f_vals, d_vals)

        r_val, t_val = pearson(f_vals, d_vals)
        df = len(valid_pairs) - 2
        results_delta.append((method, dataset, nf, r_val, t_val, df, len(valid_pairs), ""))

    results_delta.sort(key=lambda x: (x[0], x[1], x[2]))

    print("=== freq vs delta_omega_h ===")
    header = f"{'method':15s} {'dataset':20s} {'nf':>4s} {'r':>8s} {'t':>8s} {'df':>5s} {'n':>5s}"
    print(header)
    print("-" * len(header))
    for method, dataset, nf_val, r_val, t_val, df, n, _ in results_delta:
        r_str = f"{r_val:+.4f}" if r_val is not None else "   -"
        t_str = f"{t_val:+.3f}" if t_val is not None else "   -"
        df_str = f"{df}" if df is not None else "-"
        n_str = f"{n}" if n is not None else "-"
        print(f"{method:15s} {dataset:20s} {nf_val:>4d} {r_str:>8s} {t_str:>8s} {df_str:>5s} {n_str:>5s}")

    valid_r_delta = [r for _, _, _, r, _, _, _, _ in results_delta if r is not None and not math.isnan(r)]
    print("-" * len(header))
    if valid_r_delta:
        avg_r = sum(valid_r_delta) / len(valid_r_delta)
        print(f"{'average':15s} {'':20s} {'':>4s} {avg_r:+8.4f} {'':>8s} {'':>5s} {len(valid_r_delta):>5d}  (mean r over {len(valid_r_delta)} groups)")
    else:
        print("average: no valid r values")

    if not args.no_plot and plot_data_delta:
        make_plots(plot_data_delta, Path("results/loco_freq_vs_delta.png"), dpi=args.dpi, lowess_frac=args.lowess_frac)
        print("Saved plot to results/loco_freq_vs_delta.png")

    nf_by_group: dict[tuple[str, str], set[int]] = defaultdict(set)
    for dataset, method, nf in groups:
        nf_by_group[(dataset, method)].add(nf)

    results_g = []
    plot_data_g: dict[str, dict[str, tuple[list[float], list[float]]]] = defaultdict(dict)

    for (dataset, method), nfs in sorted(nf_by_group.items()):
        dz, st = parse_dataset(dataset)
        csv_path = sparse_csv_path(data_root, dz, st)
        if not csv_path.exists():
            for nf in sorted(nfs):
                results_g.append((method, dataset, nf, None, None, None, None, "sparse CSV not found"))
            continue

        if dataset not in freq_cache:
            freq_cache[dataset] = bench_frequencies(csv_path)
        bench_names, freqs = freq_cache[dataset]
        freq_map = dict(zip(bench_names, freqs))

        for nf in sorted(nfs):
            if nf == 2:
                load_path = loadings_csv_path(results_root, method, dataset, "bifactor_pa")
                if not load_path.exists():
                    load_path = loadings_csv_path(results_root, method, dataset, "bifactor_2f")
            else:
                load_path = loadings_csv_path(results_root, method, dataset, "bifactor_pa")

            if not load_path.exists():
                results_g.append((method, dataset, nf, None, None, None, None, "loadings CSV not found"))
                continue

            try:
                g_map = bench_g_loadings(load_path)
            except Exception as e:
                results_g.append((method, dataset, nf, None, None, None, None, str(e)))
                continue

            common = sorted(set(freq_map) & set(g_map))
            if len(common) < 3:
                results_g.append((method, dataset, nf, None, None, None, len(common), "not enough pairs"))
                continue

            f_vals = [freq_map[b] for b in common]
            g_vals = [g_map[b] for b in common]
            plot_data_g[method][f"{dataset} (nf={nf})"] = (f_vals, g_vals)

            r_val, t_val = pearson(f_vals, g_vals)
            df = len(common) - 2
            results_g.append((method, dataset, nf, r_val, t_val, df, len(common), ""))

    results_g.sort(key=lambda x: (x[0], x[1], x[2]))

    print("\n=== freq vs g-loading (omega_h) ===")
    header_g = f"{'method':15s} {'dataset':20s} {'nf':>4s} {'r':>8s} {'t':>8s} {'df':>5s} {'n':>5s}"
    print(header_g)
    print("-" * len(header_g))
    for method, dataset, nf_val, r_val, t_val, df, n, _ in results_g:
        r_str = f"{r_val:+.4f}" if r_val is not None else "   -"
        t_str = f"{t_val:+.3f}" if t_val is not None else "   -"
        df_str = f"{df}" if df is not None else "-"
        n_str = f"{n}" if n is not None else "-"
        print(f"{method:15s} {dataset:20s} {nf_val:>4d} {r_str:>8s} {t_str:>8s} {df_str:>5s} {n_str:>5s}")

    valid_r_g = [r for _, _, _, r, _, _, _, _ in results_g if r is not None and not math.isnan(r)]
    print("-" * len(header_g))
    if valid_r_g:
        avg_r = sum(valid_r_g) / len(valid_r_g)
        print(f"{'average':15s} {'':20s} {'':>4s} {avg_r:+8.4f} {'':>8s} {'':>5s} {len(valid_r_g):>5d}  (mean r over {len(valid_r_g)} groups)")
    else:
        print("average: no valid r values")

    if not args.no_plot and plot_data_g:
        make_plots(plot_data_g, Path("results/loco_freq_vs_omegah.png"), dpi=args.dpi, lowess_frac=args.lowess_frac,
                   ylabel="General factor loading (g)", suptitle="g-loading vs. benchmark frequency (LOWESS smoothed)")
        print("Saved plot to results/loco_freq_vs_omegah.png")


if __name__ == "__main__":
    main()
