#!/usr/bin/env python3
"""One figure + one table summarising label cohesion across every cell.

Cells are NOT independent replications -- they are re-analyses of the same
results matrix under different densifiers and imputers -- so nothing here pools
them into a single p-value. The summary column is a median effect, and the
"cells significant" count reads as robustness to analysis choices.

Effect is scale-free: 1 - within/null_mean, i.e. how much tighter than a
coverage-matched random set the label's members sit. Unlike z it does not grow
with corpus size, so an effect in a 78-benchmark cell is comparable to one in a
404-benchmark cell.

    uv run python scripts/plot_cohesion_summary.py --all-pa
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

REPO = Path(__file__).resolve().parents[1]

# dataviz diverging pair: blue <-> red poles, neutral grey midpoint.
TIGHT_POLE = "#184f95"
LOOSE_POLE = "#b62d2d"
MIDPOINT = "#f0efec"
MISSING = "#ffffff"
INK = "#22222a"
MUTED = "#74747f"
SURFACE = "#ffffff"
CMAP = LinearSegmentedColormap.from_list(
    "cohesion", [LOOSE_POLE, "#e34948", MIDPOINT, "#2a78d6", TIGHT_POLE]
)
FDR_Q = 0.05
IMPUTER_ABBR = {
    "default": "def",
    "missforest": "mf",
    "onesidedmc": "omc",
    "softimpute": "si",
    "softimpute_corr": "si-corr",
    "zeros": "zero",
}


def bh_fdr(p: np.ndarray, q: float) -> np.ndarray:
    order = np.argsort(p)
    m = len(p)
    passed = p[order] <= q * np.arange(1, m + 1) / m
    keep = np.zeros(m, dtype=bool)
    if passed.any():
        keep[order[: np.flatnonzero(passed)[-1] + 1]] = True
    return keep


def collect(payload: dict, cells: list[str], variant: str, axis: str):
    """{label: {cell: (effect, significant, n)}} with per-cell FDR correction."""
    out: dict[str, dict[str, tuple[float, bool, int]]] = {}
    for cell in cells:
        rows = [
            r for r in (payload[cell].get("category_cohesion") or {}).get(variant, [])
            if r["category"].startswith(f"{axis}:")
        ]
        if not rows:
            continue
        p = np.array([r["p"] for r in rows])
        sig = bh_fdr(p, FDR_Q)
        for r, s in zip(rows, sig):
            lab = r["category"].split(":", 1)[1]
            effect = 1.0 - r["within"] / r["null_mean"]
            out.setdefault(lab, {})[cell] = (effect, bool(s), r["n"])
    return out


def plot(data, cells, variant, axis, out: Path, note: str = "") -> None:
    labels = sorted(data, key=lambda l: -np.median([v[0] for v in data[l].values()]))
    grid = np.full((len(labels), len(cells)), np.nan)
    for i, lab in enumerate(labels):
        for j, cell in enumerate(cells):
            if cell in data[lab]:
                grid[i, j] = data[lab][cell][0]
    med = np.array([np.median([v[0] for v in data[lab].values()]) for lab in labels])
    lim = float(np.nanmax(np.abs(grid)))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)

    fig, (ax, axm) = plt.subplots(
        1, 2, figsize=(0.42 * len(cells) + 7.0, 0.27 * len(labels) + 3.6),
        gridspec_kw={"width_ratios": [len(cells), 2.2], "wspace": 0.04},
    )
    fig.patch.set_facecolor(SURFACE)
    cmap = CMAP.copy()
    cmap.set_bad(MISSING)
    ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    for i, lab in enumerate(labels):
        for j, cell in enumerate(cells):
            if cell in data[lab] and data[lab][cell][1]:
                # significance is marked, never colour-alone
                ax.plot(j, i, marker="o", markersize=2.6, color="#ffffff", zorder=3)
    # "|pa" is constant across columns (the title carries it) and "|m" just marks
    # the imputer, so a bare densifier is its aggregate over every imputer.
    # Imputers are abbreviated: the full names do not fit under a column.
    shown = []
    for c in cells:
        dz, _, rest = c.replace("|pa", "", 1).partition("|m")
        shown.append(f"{dz} · {IMPUTER_ABBR.get(rest, rest)}" if rest else dz)
    ax.set_xticks(range(len(cells)), shown, rotation=90, fontsize=7.5, color=INK)
    ax.set_yticks(range(len(labels)), labels, fontsize=8, color=INK)
    ax.set_xticks(np.arange(-0.5, len(cells)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.2)
    ax.tick_params(which="both", length=0)
    for side in ax.spines.values():
        side.set_visible(False)

    axm.imshow(med[:, None], cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    for i, lab in enumerate(labels):
        vals = data[lab]
        nsig = sum(1 for v in vals.values() if v[1])
        axm.text(0, i, f"{med[i]:+.3f}", ha="center", va="center", fontsize=7.5,
                 color=INK if abs(med[i]) < 0.6 * lim else "#ffffff")
        axm.text(1.25, i, str(int(np.median([v[2] for v in vals.values()]))),
                 transform=axm.get_yaxis_transform(), ha="center", va="center",
                 fontsize=7.5, color=MUTED)
        axm.text(2.15, i, f"{nsig}/{len(vals)}", transform=axm.get_yaxis_transform(),
                 ha="center", va="center", fontsize=7.5, color=MUTED)
    axm.set_xticks([0], ["median A"], fontsize=7.5, color=INK)
    axm.set_yticks([])
    axm.set_xticks([-0.5], minor=True)
    axm.grid(which="minor", color=SURFACE, linewidth=1.2)
    axm.tick_params(which="both", length=0)
    for side in axm.spines.values():
        side.set_visible(False)
    axm.text(1.25, 1.0, "median\nn", transform=axm.transAxes, ha="center",
             va="bottom", fontsize=7.5, color=MUTED, linespacing=1.3)
    axm.text(2.15, 1.0, "sig/\nscored", transform=axm.transAxes, ha="center",
             va="bottom", fontsize=7.5, color=MUTED, linespacing=1.3)

    fig.suptitle(
        f"Chance-corrected within-group agreement (MRPP A) per {axis} label "
        f"({variant.replace('_', ' ')})",
        fontsize=11.5, color=INK, x=0.012, y=0.988, ha="left", va="top", fontweight="bold")
    fig.text(0.012, 0.955,
             "A = 1 - within/null_mean (Mielke et al. 1976): 0 is chance, 1 is identical "
             "members, negative is over-dispersed; A does not grow with corpus size."
             + chr(10) +
             "Blue: tighter. Red: looser. White dot: significant at FDR 5% in that cell."
             + chr(10) +
             "White cell: not scored there (fewer than 4 members, or no local loadings). "
             "Cells are re-analyses of one dataset, not independent replications."
             + chr(10) +
             "Columns are densifier · imputer (def, mf missforest, omc one-sided MC, "
             "si softimpute, si-corr, zero); a bare densifier aggregates over all of them."
             + note,
             fontsize=8, color=MUTED, ha="left", va="top", linespacing=1.5)
    # Explicit margins, in inches converted to figure fractions: tight_layout
    # refuses these imshow axes ("not compatible"), silently leaving the caption
    # on top of the first row. Top clears the title plus caption lines, bottom
    # clears the rotated column labels, right leaves the two text columns room.
    lines = 4 + note.count(chr(10))  # fixed caption is 4 lines
    fig_w, fig_h = fig.get_size_inches()
    fig.subplots_adjust(
        left=1.75 / fig_w,
        right=1 - 1.95 / fig_w,
        top=1 - (0.34 + 0.15 * lines) / fig_h,
        bottom=1.25 / fig_h,
    )
    for suffix in (".png", ".pdf"):
        fig.savefig(out.with_suffix(suffix), dpi=200, facecolor=SURFACE)
        print(f"wrote {out.with_suffix(suffix)}")
    plt.close(fig)


def write_table(data, cells, out: Path) -> None:
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "median_effect", "min_effect", "max_effect",
                    "cells_scored", "cells_significant", "median_n"])
        for lab in sorted(data, key=lambda l: -np.median([v[0] for v in data[l].values()])):
            vals = list(data[lab].values())
            eff = [v[0] for v in vals]
            w.writerow([lab, round(float(np.median(eff)), 4), round(min(eff), 4),
                        round(max(eff), 4), len(vals), sum(1 for v in vals if v[1]),
                        int(np.median([v[2] for v in vals]))])
    print(f"wrote {out}")


def summary_rows(data) -> list[tuple[str, float, float, float, int, int, int]]:
    """(label, median, min, max, n_sig, n_scored, median_n), tightest first."""
    rows = []
    for lab in sorted(data, key=lambda l: -np.median([v[0] for v in data[l].values()])):
        vals = list(data[lab].values())
        eff = [v[0] for v in vals]
        rows.append((lab, float(np.median(eff)), min(eff), max(eff),
                     sum(1 for v in vals if v[1]), len(vals),
                     int(np.median([v[2] for v in vals]))))
    return rows


def write_markdown(data, out: Path, variant: str, axis: str) -> None:
    g = variant.replace("_", " ")
    lines = [
        f"| label | median A | A across cells | significant | median n |",
        "|---|---|---|---|---|",
    ]
    for lab, med, lo, hi, nsig, nscored, n in summary_rows(data):
        lines.append(
            f"| `{lab}` | {med:+.3f} | {lo:+.3f} to {hi:+.3f} | {nsig}/{nscored} | {n} |"
        )
    lines.append("")
    lines.append(
        f"Cohesion of each {axis} label ({g}). A is the chance-corrected within-group "
        "agreement 1 - within/null_mean against a coverage-matched permutation null "
        "(MRPP; Mielke, Berry & Johnson 1976), where 0 is chance and A < 0.1 is "
        "conventionally weak. Cells are densifier x imputer re-analyses of one dataset, so "
        "the count is consistency across analysis choices, not independent replication."
    )
    out.write_text(chr(10).join(lines) + chr(10), encoding="utf-8")
    print(f"wrote {out}")


def write_latex(data, out: Path, variant: str, axis: str) -> None:
    g = variant.replace("_", " ")
    esc = lambda s: s.replace("_", r"\_")
    body = [
        r"\begin{table}[t]", r"\centering",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"label & median $A$ & $A$ across cells & sig. & median $n$ \\", r"\midrule",
    ]
    for lab, med, lo, hi, nsig, nscored, n in summary_rows(data):
        body.append(
            rf"\texttt{{{esc(lab)}}} & ${med:+.3f}$ & "
            rf"${lo:+.3f}$ to ${hi:+.3f}$ & {nsig}/{nscored} & {n} \\"
        )
    body += [
        r"\bottomrule", r"\end{tabular}",
        rf"\caption{{Cohesion of each {axis} label ({g}). $A$ is the chance-corrected "
        r"within-group agreement $1-\bar{d}_{\text{within}}/\bar{d}_{\text{null}}$ against a "
        r"coverage-matched permutation null (MRPP; Mielke, Berry \& Johnson 1976), where $0$ "
        r"is chance and $A<0.1$ is conventionally weak. Cells are densifier $\times$ imputer "
        r"re-analyses of one dataset, so the count is consistency across analysis choices, "
        r"not independent replication.}",
        r"\label{tab:cohesion}", r"\end{table}",
    ]
    out.write_text(chr(10).join(body) + chr(10), encoding="utf-8")
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--positions", default=str(REPO / "viewer" / "positions.json"))
    ap.add_argument("--all-pa", action="store_true",
                    help="every pa cell (default); year cohorts and retired 2f are skipped")
    ap.add_argument("--variant", default="with_g", choices=("with_g", "without_g"))
    ap.add_argument("--axis", default="subject")
    ap.add_argument(
        "--min-n", type=int, default=8,
        help="hide labels whose median member count is below this (default 8): a mean "
        "distance over 4-5 benchmarks turns on a single near-duplicate pair. 0 keeps all.",
    )
    ap.add_argument(
        "--top-k", type=int, default=10,
        help="show only the K labels with the most members (applied after --min-n); "
        "default 10, 0 shows every label that passes --min-n",
    )
    ap.add_argument(
        "--min-cell-n", type=int, default=0,
        help="drop cells with fewer than this many benchmarks before computing "
        "anything (default 0, keep all); the small cells drive the widest ranges",
    )
    ap.add_argument("--table", action="store_true",
                    help="also write the displayed labels as Markdown and LaTeX")
    ap.add_argument("--table-only", action="store_true",
                    help="write the tables and skip the figure")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    payload = json.loads(Path(args.positions).read_text(encoding="utf-8"))
    cells = sorted(
        k for k in payload
        if k.split("|")[1:2] == ["pa"]
        and not any(p.startswith("y") for p in k.split("|")[2:])
    )
    if not cells:
        raise SystemExit("no pa cells in payload")
    cell_note = ""
    if args.min_cell_n:
        small = [c for c in cells if len(payload[c]["benchmarks"]) < args.min_cell_n]
        cells = [c for c in cells if c not in small]
        if not cells:
            raise SystemExit(f"no cell has {args.min_cell_n} or more benchmarks")
        if small:
            print(f"dropping {len(small)} cells under {args.min_cell_n} benchmarks: "
                  + ", ".join(f"{c} ({len(payload[c]['benchmarks'])})" for c in small))
            cell_note = (
                chr(10) + f"{len(small)} cells with fewer than {args.min_cell_n} "
                "benchmarks are excluded; they drove the widest ranges."
            )
    data = collect(payload, cells, args.variant, args.axis)
    if not data:
        raise SystemExit(f"no {args.axis} labels scored in any pa cell")
    everything = data  # the CSV keeps every label; --min-n only thins the figure
    median_n = {lab: float(np.median([v[2] for v in vals.values()])) for lab, vals in data.items()}
    hidden = sorted(lab for lab, n in median_n.items() if n < args.min_n)
    if hidden:
        data = {lab: v for lab, v in data.items() if median_n[lab] >= args.min_n}
        print(f"hiding {len(hidden)} labels with median n < {args.min_n}: {', '.join(hidden)}")
    if not data:
        raise SystemExit(f"no {args.axis} label reaches median n >= {args.min_n}")
    note = cell_note + ((
        chr(10) + f"{len(hidden)} labels with a median of fewer than {args.min_n} "
        "members are not shown; their mean distance turns on one or two pairs."
    ) if hidden else "")
    if args.top_k and args.top_k < len(data):
        kept = sorted(data, key=lambda lab: (-median_n[lab], lab))[: args.top_k]
        dropped = len(data) - len(kept)
        data = {lab: data[lab] for lab in kept}
        smallest = min(median_n[lab] for lab in kept)
        print(f"showing the {args.top_k} largest labels (median n >= {smallest:g}); "
              f"{dropped} smaller ones hidden")
        note += (
            chr(10) + f"Showing the {args.top_k} labels with the most members "
            f"(median n >= {smallest:g}); {dropped} smaller ones are not shown."
        )
    suffix = "" if args.variant == "with_g" else "_nog"
    stem = Path(args.out or REPO / "results" / "figures" /
                f"cohesion_summary_{args.axis}{suffix}")
    stem.parent.mkdir(parents=True, exist_ok=True)
    if not args.table_only:
        plot(data, cells, args.variant, args.axis, stem, note)
    write_table(everything, cells, stem.with_suffix(".csv"))  # every label, unfiltered
    if args.table or args.table_only:
        # the paste-into-the-paper tables cover the displayed labels only
        write_markdown(data, stem.with_suffix(".md"), args.variant, args.axis)
        write_latex(data, stem.with_suffix(".tex"), args.variant, args.axis)


if __name__ == "__main__":
    main()
