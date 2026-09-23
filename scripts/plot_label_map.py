#!/usr/bin/env python3
"""Classical MDS (PCoA) map of benchmark factor space, one facet per label.

The viewer's UMAP optimises local neighbourhoods and discards global distance
(measured on raw|pa: Spearman 0.21 between true and on-screen distance), so a
label that is tighter than chance need not look tighter there. PCoA instead
places points so that plotted distance approximates the real cosine distance,
which is the quantity cohesion is measured on -- so a tight label looks tight.

    uv run python scripts/plot_label_map.py --cell raw|pa

Faceting defaults to the 3 tightest and 3 loosest labels by cohesion z.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "compute_positions", REPO / "viewer" / "compute_positions.py"
)
cp = importlib.util.module_from_spec(_SPEC)
sys.modules["compute_positions"] = cp
_SPEC.loader.exec_module(cp)

HIGHLIGHT = "#2a78d6"   # dataviz categorical slot 1; emphasis = 1 hue + grey
REST = "#c9c9d1"
INK = "#22222a"
MUTED = "#74747f"
SURFACE = "#ffffff"


def cell_distance(cell: str, results_root: Path, *, drop_g: bool = False) -> tuple[np.ndarray, list[str]]:
    """Composite cosine distance for a payload cell key ('raw|pa', 'C|pa|mzeros')."""
    parts = cell.split("|")
    if len(parts) not in (2, 3) or (len(parts) == 3 and not parts[2].startswith("m")):
        raise SystemExit(f"--cell must be 'dz|tag' or 'dz|tag|m<method>', got {cell!r}")
    dz, tag = parts[0], parts[1]
    cells = cp.load_cells(results_root)
    if (dz, tag, None) not in cells:
        raise SystemExit(f"no loadings for {dz}|{tag} under {results_root}")
    cell_list = cells[(dz, tag, None)]
    if len(parts) == 3:
        want = parts[2][1:]
        cell_list = [c for c in cell_list if cp.CELL_KEY_RE.match(c[0]).group(1) == want]
        if not cell_list:
            raise SystemExit(f"no loadings for imputer {want!r} in {dz}|{tag}")
    return cp.composite_distance(cell_list, drop_g=drop_g)


def pick_labels(payload: dict, cell: str, variant: str, axis: str, n: int) -> list[str]:
    rows = [
        r for r in payload[cell]["category_cohesion"][variant]
        if r["category"].startswith(f"{axis}:")
    ]
    rows.sort(key=lambda r: r["z"])
    return [r["category"] for r in rows[:n]] + [r["category"] for r in rows[-n:]]


def plot(xy, bench, payload, cell, variant, labels, explained, out: Path) -> None:
    cats = dict(zip(payload[cell]["benchmarks"], payload[cell]["categories"]))
    zof = {r["category"]: r for r in payload[cell]["category_cohesion"][variant]}
    cols = min(3, len(labels))
    rowsn = -(-len(labels) // cols)
    fig, axes = plt.subplots(rowsn, cols, figsize=(3.1 * cols, 3.25 * rowsn + 0.5),
                             sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, lab in zip(np.atleast_1d(axes).ravel(), labels):
        member = np.array([lab in (cats.get(b) or []) for b in bench])
        ax.set_facecolor(SURFACE)
        ax.scatter(xy[~member, 0], xy[~member, 1], s=7, c=REST, linewidths=0, zorder=1)
        ax.scatter(xy[member, 0], xy[member, 1], s=22, c=HIGHLIGHT,
                   edgecolors=SURFACE, linewidths=0.6, zorder=2)
        r = zof.get(lab)
        stat = f"z={r['z']:+.2f}  n={r['n']}" if r else "no cohesion score"
        ax.set_title(lab.split(":", 1)[1], fontsize=9.5, color=INK, loc="left", pad=9)
        ax.text(0, 1.005, stat, transform=ax.transAxes, va="bottom", ha="left",
                fontsize=8, color=MUTED)
        ax.set_xticks([]); ax.set_yticks([])
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_color("#e6e6ea")
    for ax in np.atleast_1d(axes).ravel()[len(labels):]:
        ax.set_visible(False)
    fig.suptitle(
        f"Benchmarks in factor space (PCoA) — {cell} ({variant.replace('_', ' ')}, "
        f"n={len(bench)})", fontsize=11.5, color=INK, x=0.01, y=0.985, ha="left",
        va="top", fontweight="bold")
    fig.text(0.01, 0.951,
             "Plotted distance approximates true cosine distance; "
             f"2 axes carry {explained:.0%} of it. Highlighted: benchmarks with the label.",
             fontsize=8.5, color=MUTED, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    for suffix in (".png", ".pdf"):
        fig.savefig(out.with_suffix(suffix), dpi=200, facecolor=SURFACE)
        print(f"wrote {out.with_suffix(suffix)}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cell", default="raw|pa")
    ap.add_argument("--variant", default="with_g", choices=("with_g", "without_g"))
    ap.add_argument("--axis", default="subject")
    ap.add_argument("--labels", help="comma-separated label names; default 3 tightest + 3 loosest")
    ap.add_argument("--n-each", type=int, default=3)
    ap.add_argument("--results-root", default="results/text_only")
    ap.add_argument("--positions", default=str(REPO / "viewer" / "positions.json"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    payload = json.loads(Path(args.positions).read_text(encoding="utf-8"))
    if args.cell not in payload:
        raise SystemExit(f"cell {args.cell!r} not in payload")
    root = Path(args.results_root)
    dist, bench = cell_distance(
        args.cell,
        root if root.is_absolute() else REPO / root,
        drop_g=args.variant == "without_g",
    )
    if bench != payload[args.cell]["benchmarks"]:
        raise SystemExit(
            f"{args.cell}: local loadings give a different benchmark set than the payload"
        )
    labels = (
        [f"{args.axis}:{x.strip()}" for x in args.labels.split(",")]
        if args.labels
        else pick_labels(payload, args.cell, args.variant, args.axis, args.n_each)
    )
    xy, explained = cp.pcoa(dist)   # same code path as the viewer payload
    out = Path(args.out or REPO / "results" / "figures" /
               f"map_{args.cell.replace('|', '_')}_{args.axis}")
    out.parent.mkdir(parents=True, exist_ok=True)
    plot(xy, bench, payload, args.cell, args.variant, labels, explained, out)


if __name__ == "__main__":
    main()
