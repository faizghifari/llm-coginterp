#!/usr/bin/env python3
"""Plot per-label cohesion: observed within-distance against its matched null.

Reads the numbers the viewer already computed (viewer/positions.json) and draws
them directly, with no projection -- unlike the UMAP scatter, which preserves
local neighbourhoods and not the global distances cohesion is measured on
(measured on raw|pa: Spearman 0.21 between true and on-screen distance, so a
label that is tighter than chance need not look tighter on screen).

    uv run python scripts/plot_label_cohesion.py --cell raw|pa --axis subject
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
DEFAULT_POSITIONS = REPO / "viewer" / "positions.json"

# dataviz reference palette: diverging pair (blue<->red) on a neutral midpoint.
# Validated with scripts/validate_palette.js --mode light --surface #ffffff:
# CVD dE 21.6 protan, normal-vision dE 32.3, both clear of the gates.
TIGHTER = "#2a78d6"
LOOSER = "#e34948"
NULL_BAND = "#d9d9e0"
INK = "#22222a"
MUTED = "#74747f"
SURFACE = "#ffffff"
FDR_Q = 0.05


def bh_fdr(p: np.ndarray, q: float) -> np.ndarray:
    """Benjamini-Hochberg: True where the label is significant at FDR q."""
    order = np.argsort(p)
    ranked = p[order]
    m = len(p)
    thresh = q * np.arange(1, m + 1) / m
    passed = ranked <= thresh
    keep = np.zeros(m, dtype=bool)
    if passed.any():
        keep[order[: np.flatnonzero(passed)[-1] + 1]] = True
    return keep


def rows_for(payload: dict, cell: str, variant: str, axis: str) -> list[dict]:
    if cell not in payload:
        raise SystemExit(f"cell {cell!r} not in payload; have e.g. {list(payload)[:4]}")
    cohesion = payload[cell].get("category_cohesion")
    if not cohesion:
        raise SystemExit(f"{cell} has no category_cohesion (too few benchmarks to score)")
    prefix = f"{axis}:"
    rows = [r for r in cohesion[variant] if r["category"].startswith(prefix)]
    if not rows:
        raise SystemExit(f"no {axis!r} labels scored in {cell}")
    return sorted(rows, key=lambda r: r["z"])


def plot(rows: list[dict], cell: str, variant: str, axis: str, n_bench: int, out: Path) -> None:
    labels = [r["category"].split(":", 1)[1] for r in rows]
    within = np.array([r["within"] for r in rows])
    null_mean = np.array([r["null_mean"] for r in rows])
    z = np.array([r["z"] for r in rows])
    # p is the Monte Carlo estimate (1 + hits)/(1 + perms), so it is never 0.
    p = np.array([r["p"] for r in rows])
    sd = np.where(z != 0, np.abs((within - null_mean) / np.where(z == 0, 1, z)), np.nan)
    sig = bh_fdr(p, FDR_Q)

    y = np.arange(len(rows))[::-1]  # tightest at the top
    fig, ax = plt.subplots(figsize=(7.6, 0.27 * len(rows) + 2.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    lo, hi = null_mean - 1.96 * sd, null_mean + 1.96 * sd
    ax.hlines(y, lo, hi, color=NULL_BAND, linewidth=5, zorder=1,
              capstyle="round", label="_null")
    ax.plot(null_mean, y, marker="|", markersize=7, linestyle="none",
            color=MUTED, zorder=2)

    for tighter in (True, False):
        for is_sig in (True, False):
            m = (z < 0) == tighter
            m &= sig if is_sig else ~sig
            if not m.any():
                continue
            colour = TIGHTER if tighter else LOOSER
            ax.plot(within[m], y[m], linestyle="none", marker="o", markersize=7,
                    markerfacecolor=colour if is_sig else SURFACE,
                    markeredgecolor=colour, markeredgewidth=1.6, zorder=3)

    for yi, r in zip(y, rows):
        ax.text(1.005, yi, str(r["n"]), transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=7.5, color=MUTED)

    ax.set_yticks(y, labels, fontsize=8.5, color=INK)
    ax.set_ylim(-1.6, len(rows) - 0.2)
    ax.set_xlabel("mean pairwise cosine distance among a label's benchmarks",
                  fontsize=8.5, color=INK)
    ax.tick_params(axis="x", labelsize=8, colors=MUTED)
    ax.xaxis.grid(True, color=NULL_BAND, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(NULL_BAND)

    ax.set_title(
        f"Label cohesion in factor space — {cell} ({variant.replace('_', ' ')}, "
        f"n={n_bench})",
        fontsize=11, color=INK, loc="left", pad=30, fontweight="bold")
    ax.text(0, 1.0,
            "Grey bar: 95% of coverage-matched random sets of the same size; "
            "tick is their mean." + chr(10) + "Filled marker: significant at FDR 5%. "
            "Right column: benchmarks carrying the label.",
            transform=ax.transAxes, va="bottom", ha="left", fontsize=8,
            color=MUTED, linespacing=1.5)

    handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=7, markeredgewidth=1.6,
               markerfacecolor=TIGHTER, markeredgecolor=TIGHTER, label="tighter than chance"),
        Line2D([], [], linestyle="none", marker="o", markersize=7, markeredgewidth=1.6,
               markerfacecolor=SURFACE, markeredgecolor=TIGHTER, label="tighter, not significant"),
        Line2D([], [], linestyle="none", marker="o", markersize=7, markeredgewidth=1.6,
               markerfacecolor=SURFACE, markeredgecolor=LOOSER, label="looser than chance"),
    ]
    if (sig & (z > 0)).any():
        handles.append(Line2D([], [], linestyle="none", marker="o", markersize=7,
                              markeredgewidth=1.6, markerfacecolor=LOOSER,
                              markeredgecolor=LOOSER, label="looser, significant"))
    # bottom-left is empty in tall figures (the loosest labels sit right); a short
    # one has no spare row, so the legend goes under the axis instead
    if len(rows) >= 14:
        ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=8,
                  labelcolor=INK, handletextpad=0.4, borderaxespad=0.6)
    else:
        ax.set_ylim(-0.8, len(rows) - 0.2)
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.12 - 1.1 / len(rows)),
                  frameon=False, fontsize=8, labelcolor=INK, handletextpad=0.4,
                  ncols=min(3, len(handles)), columnspacing=1.4)

    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(out.with_suffix(suffix), dpi=200, facecolor=SURFACE)
        print(f"wrote {out.with_suffix(suffix)}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--positions", default=str(DEFAULT_POSITIONS))
    ap.add_argument("--cell", default="raw|pa", help="payload key, e.g. 'raw|pa'")
    ap.add_argument(
        "--all-pa",
        action="store_true",
        help="every pa cell (all densifiers and imputers, aggregate included), "
        "skipping year cohorts and the retired forced-2f solutions",
    )
    ap.add_argument("--variant", default="with_g", choices=("with_g", "without_g"))
    ap.add_argument("--axis", default="subject", help="label axis to plot")
    ap.add_argument(
        "--out", default=None,
        help="output stem; default results/figures/cohesion/by_cell/<axis>/<cell>",
    )
    args = ap.parse_args()

    payload = json.loads(Path(args.positions).read_text(encoding="utf-8"))
    if args.all_pa:
        # "<dz>|pa" and "<dz>|pa|m<method>"; 2f is retired, year cohorts are separate
        cells = sorted(
            k for k in payload
            if k.split("|")[1:2] == ["pa"] and not any(p.startswith("y") for p in k.split("|")[2:])
        )
        if args.out:
            raise SystemExit("--out names one file; drop it when using --all-pa")
        print(f"{len(cells)} pa cells")
        for cell in cells:
            rows = rows_for(payload, cell, args.variant, args.axis)
            out = REPO / "results" / "figures" / "cohesion" / "by_cell" / args.axis / (
                cell.replace("|", "_") + ("" if args.variant == "with_g" else "_nog")
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            plot(rows, cell, args.variant, args.axis, len(payload[cell]["benchmarks"]), out)
        return

    rows = rows_for(payload, args.cell, args.variant, args.axis)
    stem = args.out or str(
        REPO / "results" / "figures" / "cohesion" / "by_cell" / args.axis / args.cell.replace("|", "_")
    )
    out = Path(stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    plot(rows, args.cell, args.variant, args.axis, len(payload[args.cell]["benchmarks"]), out)


if __name__ == "__main__":
    main()
