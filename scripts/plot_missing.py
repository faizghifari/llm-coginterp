"""Plot KDE density of benchmark observation counts (non-missing scores per
benchmark column) across the raw table and each densifier, for both strategies.
Reads:
  data/combinations/<strategy>/model_benchmark_table.csv
  data/combinations_<C|R|S>/<strategy>/model_benchmark_table.csv
Writes:
  data/density.png   (4 rows x 2 columns, KDE bell curves + histograms)
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import gaussian_kde
REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
RESULTS = REPO / "results"
STRATEGIES = ["all_standard", "all_aggressive"]
KEY = "collapse_key"
SOURCES = [
    ("standard", "combinations"),
    ("C", "combinations_C"),
    ("R", "combinations_R"),
    ("S", "combinations_S"),
]
def benchmark_obs_counts(csv_path: Path):
    df = pl.read_csv(csv_path)
    value_cols = [c for c in df.columns if c != KEY]
    pairs = [(c, df.height - df[c].null_count()) for c in value_cols]
    pairs.sort(key=lambda x: x[1])
    counts = np.array([p[1] for p in pairs], dtype=float)
    return pairs, counts
def main():
    fig, axes = plt.subplots(4, 2, figsize=(12, 14), sharex=False)
    xs = np.linspace(0, 450, 500)
    colors = {"all_standard": "#2196F3", "all_aggressive": "#F44336"}
    for row_idx, (label, src_dir) in enumerate(SOURCES):
        for col_idx, strat in enumerate(STRATEGIES):
            ax = axes[row_idx, col_idx]
            csv_path = DATA / src_dir / strat / "model_benchmark_table.csv"
            _, counts = benchmark_obs_counts(csv_path)
            color = colors[strat]
            x_max = max(counts.max() * 1.05, 10)

            # Histogram on its own y-axis (frequency), drawn behind the KDE.
            ax_hist = ax.twinx()
            n_bins = min(25, max(5, int(np.sqrt(len(counts)) * 2)))
            bin_edges = np.linspace(0, x_max, n_bins + 1)
            ax_hist.hist(
                counts,
                bins=bin_edges,
                color=color,
                alpha=0.25,
                edgecolor="white",
                linewidth=0.5,
            )
            ax_hist.set_ylim(bottom=0)

            # Put the KDE axes on top with a transparent background so the
            # histogram shows through underneath it.
            ax.set_zorder(ax_hist.get_zorder() + 1)
            ax.patch.set_visible(False)

            kde = gaussian_kde(counts)
            ax.plot(xs, kde(xs), color=color, linewidth=2)
            ax.fill_between(xs, kde(xs), color=color, alpha=0.15)
            ax.set_xlim(0, x_max)
            ax.set_ylim(bottom=0)
            mean_val = counts.mean()
            ax.axvline(mean_val, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.text(
                0.98, 0.92,
                f"n={len(counts)}  μ={mean_val:.0f}",
                transform=ax.transAxes,
                fontsize=8,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
            )
            if row_idx == 0:
                ax.set_title(strat.replace("all_", "").title(), fontsize=11, fontweight="bold")
            if row_idx == 3:
                ax.set_xlabel("Observation count per benchmark")
            if col_idx == 1:
                ax_hist.set_ylabel("Frequency", fontsize=9)
            else:
                ax_hist.set_yticklabels([])
        axes[row_idx, 0].set_ylabel(f"{label}\n\nDensity", fontsize=10)
    fig.suptitle("Benchmark observation count distributions", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    out_path = RESULTS / "density_data.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    # ── top 10 / bottom 10 per source+strategy ──────────────────────────
    for label, src_dir in SOURCES:
        for strat in STRATEGIES:
            csv_path = DATA / src_dir / strat / "model_benchmark_table.csv"
            pairs, _ = benchmark_obs_counts(csv_path)
            print(f"\n── {label} / {strat} ({len(pairs)} benchmarks) ──")
            print("  Top 10 most observed:")
            for c, n in pairs[-40:][::-1]:
                print(f"    {c:50s} {n:4d}")
            print("  Bottom 10 least observed:")
            for c, n in pairs[:40]:
                print(f"    {c:50s} {n:4d}")
def parse_args():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default="data",
                    help="input tree, relative to the repo root "
                         "(e.g. data/text_only for the derived text-only copy)")
    ap.add_argument("--results-root", default="results",
                    help="output tree, relative to the repo root")
    return ap.parse_args()


if __name__ == "__main__":
    _a = parse_args()
    # Paths are anchored to the repo root, not the CWD, so this runs from
    # anywhere (matching densify.py / collapse_results.py).
    DATA = Path(_a.data_root) if Path(_a.data_root).is_absolute() else REPO / _a.data_root
    RESULTS = (Path(_a.results_root) if Path(_a.results_root).is_absolute()
               else REPO / _a.results_root)
    main()
