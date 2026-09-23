#!/usr/bin/env python3
"""MRPP label cohesion, computed directly -- no viewer pipeline required.

viewer/compute_positions.py's category_cohesion() -- the SAME function called
here, not a reimplementation -- is normally only reachable as a side effect of
building viewer/positions.json: UMAP, HAC/HDBSCAN and writing positions.json/.js
all run just to read off a cohesion table. This script calls
composite_distance() + category_cohesion() directly, for --tag pa cells only
(aggregate + one per imputer; no year cohorts), both g variants, every label on
every axis, in one pass. No embedding or clustering runs.

category_cohesion() is MRPP (Mielke, Berry & Johnson 1976): per label, the
observed within-group mean distance vs. a coverage-matched permutation null.
A = 1 - within/null_mean is the chance-corrected effect size (0 = chance, 1 =
identical members, negative = over-dispersed) -- this is what
scripts/plot_cohesion_summary.py reads from this script's CSV and plots per
axis.

Inherits category_cohesion()'s methodology unmodified, INCLUDING that
never-co-observed pairs are not excluded: composite_distance()'s row-mean fill
feeds straight into the within/null means, same as viewer/positions.json's
own category_cohesion output. Unlike pair_separation.py / label_silhouette.py,
this does not build its own observed-pair mask.

    uv run python scripts/label_cohesion.py
    uv run python scripts/plot_cohesion_summary.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
SEED = 42
DEFAULT_CSV = REPO / "results" / "text_only" / "distance_tests" / "label_cohesion_pa.csv"


def _load(name: str, path: Path):
    """Reuse an already-loaded module; these scripts are not packages."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ps = _load("pair_separation", REPO / "scripts" / "pair_separation.py")
cp = ps.cp


@dataclass(frozen=True)
class Row:
    cell: str
    variant: str
    n_bench: int
    category: str
    n: int
    within: float
    null_mean: float
    z: float | None
    p: float
    A: float | None


def effect(within: float, null_mean: float) -> float | None:
    """Chance-corrected MRPP agreement. None when the null itself is 0."""
    return 1.0 - within / null_mean if null_mean != 0 else None


def analyse(
    key: str,
    cell_list: list,
    labels: dict[str, list[str]],
    coverage: dict[str, int],
    rng: np.random.Generator,
) -> list[Row]:
    dist, bench = cp.composite_distance(cell_list, drop_g=False)
    dist_ng, bench_ng = cp.composite_distance(cell_list, drop_g=True)
    if bench_ng != bench:
        raise RuntimeError(f"{key}: drop_g changed the benchmark ordering")
    strata = cp.coverage_strata(bench, coverage)
    out: list[Row] = []
    for variant, d in (("with_g", dist), ("without_g", dist_ng)):
        for r in cp.category_cohesion(d, bench, labels, strata, rng):
            out.append(
                Row(
                    key, variant, len(bench), r["category"], r["n"],
                    r["within"], r["null_mean"], r["z"], r["p"],
                    effect(r["within"], r["null_mean"]),
                )
            )
    return out


def axes_summary(rows: list[Row]) -> None:
    axes = sorted({r.category.split(cp.LABEL_SEP, 1)[0] for r in rows})
    print(f"axes: {', '.join(axes)}")
    for axis in axes:
        prefix = f"{axis}{cp.LABEL_SEP}"
        sub = [
            r for r in rows
            if r.category.startswith(prefix) and r.variant == "with_g" and "|m" not in r.cell
        ]
        sig = sum(1 for r in sub if r.p < 0.05)
        print(f"  {axis}: {len(sub)} label-cell rows (aggregate, with_g), {sig} at p<0.05")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-root", default="results/text_only")
    ap.add_argument("--data-root", default="data/text_only")
    ap.add_argument("--subject-groups", default=str(cp.DEFAULT_SUBJECT_GROUPS))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tag", action="append", choices=("pa", "2f"),
                    help="factor tag to include; repeatable (default: pa)")
    ap.add_argument("--csv", default=str(DEFAULT_CSV),
                    help="where to write every result row (default: the standard pa "
                    "path plot_cohesion_summary.py reads by default)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    results_root = cp.resolve(args.results_root)
    cells = cp.load_cells(results_root)
    if not cells:
        raise SystemExit(f"no loadings found under {results_root}")
    data_root = cp.resolve(args.data_root)
    groups = cp.load_subject_groups(cp.resolve(args.subject_groups))
    labels, _ = cp.load_labels(data_root, groups)
    coverage = cp.load_coverage(data_root)
    rng = np.random.default_rng(args.seed)

    jobs = list(ps.iter_jobs(cells, tuple(args.tag or ps.DEFAULT_TAGS)))
    if not jobs:
        raise SystemExit(f"no cells for tag(s) {args.tag or ps.DEFAULT_TAGS}")

    rows: list[Row] = []
    for key, cell_list in jobs:
        print(f"{key} ...", file=sys.stderr)
        rows += analyse(key, cell_list, labels, coverage, rng)
    if not rows:
        raise SystemExit("no cell produced a result")

    out_path = cp.resolve(args.csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([asdict(r) for r in rows]).write_csv(out_path)
    print(f"wrote {out_path}: {len(rows)} rows")
    axes_summary(rows)


if __name__ == "__main__":
    main()
