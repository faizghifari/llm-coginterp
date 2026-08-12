"""Plot KDE density (+ histogram) of pairwise-Pearson-correlation computability,
at two units of analysis, across the raw table and each densifier, both strategies:

  Per benchmark:
    1. How many other benchmarks it has a computable pairwise Pearson r with
       (pairwise-complete n >= MIN_N and non-degenerate variance).
    2. The average pairwise n (# models) underlying those computable correlations.
  Per benchmark pair (upper triangle, each pair counted once):
    3. n (shared non-missing observations) for each computable pair.

Terminal output mirrors this: an aggregate min/mean/max table at the
benchmark level, and a #pairs/#observed/%observed + n-per-pair table at the
pair level.

Pairwise Pearson correlation + pairwise n are computed for the whole
benchmark x benchmark matrix in closed form via matrix products (no python
loops over benchmark pairs):
    n_pairs = M^T M
    sum_x   = X0^T M          (mean/var/cov terms restricted to shared rows)
    sum_xy  = X0^T X0
where M is the observed-mask and X0 is the data with missing set to 0.

Reads:
  data/combinations/<strategy>/model_benchmark_table.csv
  data/combinations_<C|R|S>/<strategy>/model_benchmark_table.csv
Writes:
  results/density_corr_count.png   (# computable correlations per benchmark)
  results/density_corr_n.png       (avg n per computable correlation, per benchmark)
  results/density_pair_n.png       (n per computable benchmark pair)
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import gaussian_kde

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
STRATEGIES = ["all_standard", "all_aggressive"]
KEY = "collapse_key"
SOURCES = [
    ("raw", "combinations"),
    ("C", "combinations_C"),
    ("R", "combinations_R"),
    ("S", "combinations_S"),
]
COLORS = {"all_standard": "#2196F3", "all_aggressive": "#F44336"}
MIN_N = 4  # minimum paired (non-missing) observations for a correlation to count as "computable"

from scipy.stats import pearsonr
import os
import multiprocessing as mp

_mat = _mask = None

def _init_worker(mat, mask):
    global _mat, _mask
    _mat, _mask = mat, mask

def _corr_pair(args):
    i, j, n, min_n = args
    if not (n >= min_n):
        return i, j, np.nan
    both = _mask[:, i] & _mask[:, j]
    xi, xj = _mat[both, i], _mat[both, j]
    if np.std(xi) == 0 or np.std(xj) == 0:
        return i, j, np.nan
    r, _ = pearsonr(xi, xj)
    return i, j, r


def compute_matrices(csv_path: Path, min_n: int = MIN_N):
    df = pl.read_csv(csv_path)
    value_cols = [c for c in df.columns if c != KEY]
    mat = df.select(value_cols).to_numpy().astype(float)
    p = mat.shape[1]

    mask = ~np.isnan(mat)
    maskf = mask.astype(float)
    n_pairs = maskf.T @ maskf
    np.fill_diagonal(n_pairs, np.nan)

    iu = np.triu_indices(p, k=1)
    tasks = [(i, j, n_pairs[i, j], min_n) for i, j in zip(*iu)]

    n_procs = max(1, (os.cpu_count() or 2) - 2)
    corr = np.full((p, p), np.nan)
    with mp.Pool(n_procs, initializer=_init_worker, initargs=(mat, mask)) as pool:
        for i, j, r in pool.imap_unordered(_corr_pair, tasks, chunksize=200):
            corr[i, j] = corr[j, i] = r

    computable = np.isfinite(corr) & (n_pairs >= min_n)
    return value_cols, corr, n_pairs, computable

# def compute_matrices(csv_path: Path, min_n: int = MIN_N):
#     """Full benchmark x benchmark pairwise-complete Pearson r, pairwise n,
#     and a computable mask (finite r, |r|<=1, and n_pairs >= min_n). Diagonal is NaN."""
#     df = pl.read_csv(csv_path)
#     value_cols = [c for c in df.columns if c != KEY]
#     mat = df.select(value_cols).to_numpy().astype(float)

#     # center each column first — r is shift-invariant, and this kills most of
#     # the catastrophic cancellation in the E[XY]-E[X]E[Y] formula below
#     col_mean = np.nanmean(mat, axis=0)
#     mat = mat - col_mean

#     mask = ~np.isnan(mat)
#     maskf = mask.astype(float)
#     x0 = np.where(mask, mat, 0.0)

#     n_pairs = maskf.T @ maskf
#     sum_x = x0.T @ maskf
#     sum_x2 = (x0 ** 2).T @ maskf
#     sum_xy = x0.T @ x0
#     sum_y, sum_y2 = sum_x.T, sum_x2.T

#     with np.errstate(divide="ignore", invalid="ignore"):
#         mean_x, mean_y = sum_x / n_pairs, sum_y / n_pairs
#         cov = sum_xy / n_pairs - mean_x * mean_y
#         var_x = sum_x2 / n_pairs - mean_x ** 2
#         var_y = sum_y2 / n_pairs - mean_y ** 2
#         corr = cov / np.sqrt(var_x * var_y)

#     np.fill_diagonal(corr, np.nan)
#     np.fill_diagonal(n_pairs, np.nan)
#     computable = np.isfinite(corr) & (np.abs(corr) <= 1 + 1e-6) & (n_pairs >= min_n)
#     return value_cols, corr, n_pairs, computable

def benchmark_level_stats(n_pairs, computable):
    """Per benchmark: (n_computable, avg_n) across its computable correlations."""
    n_computable = computable.sum(axis=1).astype(float)
    sum_n = np.where(computable, n_pairs, 0.0).sum(axis=1)
    avg_n = np.divide(sum_n, n_computable, out=np.zeros_like(sum_n), where=n_computable > 0)
    return n_computable, avg_n


def pair_level_stats(corr, n_pairs, computable):
    """Unit = benchmark pair (upper triangle, each pair counted once).
    Returns (total_pairs, n_observed_pairs, pairwise_n[computable], |r|[computable])."""
    p = n_pairs.shape[0]
    iu = np.triu_indices(p, k=1)
    computable_upper = computable[iu]
    n_pairs_upper = n_pairs[iu]
    corr_upper = corr[iu]
    total_pairs = len(iu[0])
    n_observed = int(computable_upper.sum())
    pair_n = n_pairs_upper[computable_upper]
    pair_r_abs = np.abs(corr_upper[computable_upper])
    return total_pairs, n_observed, pair_n, pair_r_abs


def plot_grid(values_by_cell, title, xlabel, out_path):
    """values_by_cell[(row_idx, col_idx)] -> 1D np.array of per-benchmark values."""
    fig, axes = plt.subplots(4, 2, figsize=(12, 14), sharex=False)
    for row_idx, (label, _src_dir) in enumerate(SOURCES):
        for col_idx, strat in enumerate(STRATEGIES):
            ax = axes[row_idx, col_idx]
            values = values_by_cell[(row_idx, col_idx)]
            color = COLORS[strat]
            if len(values) == 0:
                ax.text(0.5, 0.5, "no computable pairs", transform=ax.transAxes, ha="center", va="center", fontsize=9, color="gray")
                ax.set_xticks([])
                ax.set_yticks([])
                if row_idx == 0:
                    ax.set_title(strat.replace("all_", "").title(), fontsize=11, fontweight="bold")
                continue
            x_max = max(values.max() * 1.05, 1.0)
            xs = np.linspace(0, x_max, 500)

            # Histogram on its own y-axis, drawn behind the KDE.
            ax_hist = ax.twinx()
            n_bins = min(25, max(5, int(np.sqrt(len(values)) * 2)))
            bin_edges = np.linspace(0, x_max, n_bins + 1)
            ax_hist.hist(values, bins=bin_edges, color=color, alpha=0.25, edgecolor="white", linewidth=0.5)
            ax_hist.set_ylim(bottom=0)

            ax.set_zorder(ax_hist.get_zorder() + 1)
            ax.patch.set_visible(False)

            if np.std(values) > 0:
                kde = gaussian_kde(values)
                ax.plot(xs, kde(xs), color=color, linewidth=2)
                ax.fill_between(xs, kde(xs), color=color, alpha=0.15)
            ax.set_xlim(0, x_max)
            ax.set_ylim(bottom=0)
            mean_val = values.mean()
            ax.axvline(mean_val, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.text(
                0.98, 0.92,
                f"n={len(values)}  μ={mean_val:.1f}",
                transform=ax.transAxes,
                fontsize=8,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
            )
            if row_idx == 0:
                ax.set_title(strat.replace("all_", "").title(), fontsize=11, fontweight="bold")
            if row_idx == 3:
                ax.set_xlabel(xlabel)
            if col_idx == 1:
                ax_hist.set_ylabel("Frequency", fontsize=9)
            else:
                ax_hist.set_yticklabels([])
        axes[row_idx, 0].set_ylabel(f"{label}\n\nDensity", fontsize=10)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def print_summary_header():
    print(
        f"{'source/strategy':22s} {'#bench':>7s} {'#uncomp':>8s} |"
        f" {'n_corr min':>10s} {'n_corr mean':>11s} {'n_corr max':>10s} |"
        f" {'avg_n min':>9s} {'avg_n mean':>10s} {'avg_n max':>9s}"
    )


def print_summary(label, n_computable, avg_n):
    n_uncomputable = int((n_computable == 0).sum())
    print(
        f"{label:22s} {len(n_computable):7d} {n_uncomputable:8d} |"
        f" {n_computable.min():10.0f} {n_computable.mean():11.1f} {n_computable.max():10.0f} |"
        f" {avg_n.min():9.1f} {avg_n.mean():10.1f} {avg_n.max():9.1f}"
    )


def print_pair_summary_header():
    print(
        f"\n{'source/strategy':22s} {'#pairs':>8s} {'#observed':>10s} {'%observed':>10s} |"
        f" {'n/pair min':>10s} {'n/pair mean':>11s} {'n/pair max':>10s} |"
        f" {'|r| min':>8s} {'|r| mean':>9s} {'|r| sd':>8s} {'|r| max':>8s}"
    )


def print_pair_summary(label, total_pairs, n_observed, pair_n, pair_r_abs):
    pct_observed = 100.0 * n_observed / total_pairs if total_pairs else 0.0
    if len(pair_n) == 0:
        print(
            f"{label:22s} {total_pairs:8d} {n_observed:10d} {pct_observed:9.1f}% |"
            f" {'--':>10s} {'--':>11s} {'--':>10s} |"
            f" {'--':>8s} {'--':>9s} {'--':>8s} {'--':>8s}"
        )
        return
    print(
        f"{label:22s} {total_pairs:8d} {n_observed:10d} {pct_observed:9.1f}% |"
        f" {pair_n.min():10.0f} {pair_n.mean():11.1f} {pair_n.max():10.0f} |"
        f" {pair_r_abs.min():8.3f} {pair_r_abs.mean():9.3f} {pair_r_abs.std():8.3f} {pair_r_abs.max():8.3f}"
    )


def main():
    n_computable_cells = {}
    avg_n_cells = {}
    pair_n_cells = {}
    pair_summaries = []  # (label, total_pairs, n_observed, pair_n, pair_r_abs)

    print_summary_header()
    for row_idx, (label, src_dir) in enumerate(SOURCES):
        for col_idx, strat in enumerate(STRATEGIES):
            csv_path = DATA / src_dir / strat / "model_benchmark_table.csv"
            _cols, corr, n_pairs, computable = compute_matrices(csv_path)
            cell_label = f"{label}/{strat.replace('all_', '')}"

            n_computable, avg_n = benchmark_level_stats(n_pairs, computable)
            n_computable_cells[(row_idx, col_idx)] = n_computable
            avg_n_cells[(row_idx, col_idx)] = avg_n
            print_summary(cell_label, n_computable, avg_n)

            total_pairs, n_observed, pair_n, pair_r_abs = pair_level_stats(corr, n_pairs, computable)
            pair_n_cells[(row_idx, col_idx)] = pair_n
            pair_summaries.append((cell_label, total_pairs, n_observed, pair_n, pair_r_abs))

    print_pair_summary_header()
    for cell_label, total_pairs, n_observed, pair_n, pair_r_abs in pair_summaries:
        print_pair_summary(cell_label, total_pairs, n_observed, pair_n, pair_r_abs)

    plot_grid(
        n_computable_cells,
        "Computable pairwise Pearson correlations per benchmark",
        "# other benchmarks with computable r",
        "results/density_corr_count.png",
    )
    plot_grid(
        avg_n_cells,
        "Average n underlying each benchmark's computable correlations",
        "Average pairwise n",
        "results/density_corr_n.png",
    )
    plot_grid(
        pair_n_cells,
        "n per computable benchmark pair",
        "n (shared non-missing observations)",
        "results/density_pair_n.png",
    )


if __name__ == "__main__":
    main()
