#!/usr/bin/env python3
"""Do the authored labels show up as structure in the benchmark distance matrix?

Two silhouette readouts per cell (aggregate or single imputer), with and
without g, each against a coverage-matched permutation null:

  label: per label, the mean silhouette of its MEMBERS, comparing a member's
    mean distance to other members (a) with its mean distance to non-members
    (b): (b - a) / max(a, b). One-vs-rest, so multi-label benchmarks are fine.
    Non-members are not scored: "the rest" is a diffuse blob that would drag
    every label down for reasons unrelated to the label.
  axis: per label axis, the ordinary multi-class silhouette over benchmarks that
    carry exactly ONE label on that axis. Reports how many qualify; a small n
    means the axis is mostly multi-label and this number says little.

Pairs never co-observed in any cell are excluded rather than trusted:
composite_distance() fills them with a row mean.

Not handled: benchmark families (a parent plus its subdomains) are near-identical
by construction and inflate any label dominated by one. The repo has no family
mapping, so nothing here corrects for it; read a high score on a small label
with that in mind.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
N_PERM = 2000
SEED = 42
ALPHA = 0.05


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
class Geometry:
    """Distance sums and counts with unobserved pairs zeroed out of both."""

    dz: np.ndarray
    cnt: np.ndarray

    @classmethod
    def from_observed(cls, d_obs: np.ndarray) -> Geometry:
        seen = ~np.isnan(d_obs)
        return cls(np.where(seen, d_obs, 0.0), seen.astype(float))


@dataclass(frozen=True)
class Result:
    cell: str
    variant: str
    level: str
    name: str
    n: int
    k: int | None
    sil: float
    null_mean: float
    z: float | None
    p: float


def ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)


def mean_silhouette(a: np.ndarray, b: np.ndarray) -> float:
    """Mean of (b - a) / max(a, b) over points where both are defined."""
    top = np.maximum(a, b)
    s = np.divide(b - a, top, out=np.full(a.shape, np.nan), where=top > 0)
    ok = np.isfinite(s)
    return float(s[ok].mean()) if ok.any() else float("nan")


def member_silhouette(g: Geometry, members: np.ndarray) -> float:
    """Mean silhouette of `members` against everyone else."""
    inside = np.zeros(len(g.dz), dtype=bool)
    inside[members] = True
    dm, cm = g.dz[members], g.cnt[members]
    a = ratio(dm[:, inside].sum(1), cm[:, inside].sum(1))
    b = ratio(dm[:, ~inside].sum(1), cm[:, ~inside].sum(1))
    return mean_silhouette(a, b)


def partition_silhouette(dz: np.ndarray, cnt: np.ndarray, codes: np.ndarray) -> float:
    """Multi-class silhouette. dz/cnt are already restricted to the points, and
    every class has >= 2 members. Self is excluded because the diagonal is
    unobserved (cnt 0)."""
    onehot = np.eye(codes.max() + 1)[codes]
    mean_to = ratio(dz @ onehot, cnt @ onehot)
    rows = np.arange(len(codes))
    a = mean_to[rows, codes]
    other = mean_to.copy()
    other[rows, codes] = np.inf
    other[np.isnan(other)] = np.inf
    b = other.min(axis=1)
    b[np.isinf(b)] = np.nan
    return mean_silhouette(a, b)


def summarise(obs: float, null: np.ndarray) -> tuple[float, float | None, float]:
    """(null mean, z, one-sided p for a higher silhouette)."""
    null = null[np.isfinite(null)]
    sd = float(null.std())
    z = (obs - float(null.mean())) / sd if sd > 0 else None
    return float(null.mean()), z, (1 + int((null >= obs).sum())) / (1 + len(null))


def label_test(
    g: Geometry,
    members: np.ndarray,
    strata: np.ndarray,
    rng: np.random.Generator,
    n_perm: int,
) -> tuple[float, float, float | None, float]:
    """(silhouette, null mean, z, p) against same-size, coverage-matched sets."""
    obs = member_silhouette(g, members)
    pool = {k: np.flatnonzero(strata == k) for k in np.unique(strata)}
    want = {k: int((strata[members] == k).sum()) for k in pool}
    null = np.empty(n_perm)
    for t in range(n_perm):
        draw = np.concatenate(
            [rng.choice(pool[k], w, replace=False) for k, w in want.items() if w]
        )
        null[t] = member_silhouette(g, draw)
    return (obs, *summarise(obs, null))


def label_results(
    key: str,
    variant: str,
    g: Geometry,
    bench: list[str],
    labels: dict[str, list[str]],
    strata: np.ndarray,
    rng: np.random.Generator,
    n_perm: int,
) -> list[Result]:
    pos = {b: i for i, b in enumerate(bench)}
    members: dict[str, list[int]] = defaultdict(list)
    for b, labs in labels.items():
        if b in pos:
            for lab in labs:
                members[lab].append(pos[b])
    out: list[Result] = []
    for lab in sorted(members):
        idx = np.array(sorted(set(members[lab])))
        # same bounds as category_cohesion: too small to score, or everyone
        if len(idx) < cp.COHESION_MIN_N or len(idx) >= len(bench):
            continue
        sil, null_mean, z, p = label_test(g, idx, strata, rng, n_perm)
        if np.isnan(sil):
            continue
        out.append(Result(key, variant, "label", lab, len(idx), None, sil, null_mean, z, p))
    return out


def single_label_codes(
    bench: list[str], labels: dict[str, list[str]], axis: str
) -> tuple[np.ndarray, np.ndarray]:
    """(benchmark indices, class codes) for benchmarks with exactly one label on
    `axis`, dropping classes of fewer than two so every silhouette is defined."""
    prefix = f"{axis}{cp.LABEL_SEP}"
    single: dict[int, str] = {}
    for i, b in enumerate(bench):
        on_axis = [l for l in labels.get(b, []) if l.startswith(prefix)]
        if len(on_axis) == 1:
            single[i] = on_axis[0]
    sizes: dict[str, int] = defaultdict(int)
    for lab in single.values():
        sizes[lab] += 1
    kept = sorted(l for l, n in sizes.items() if n >= 2)
    code = {l: k for k, l in enumerate(kept)}
    idx = np.array([i for i, l in single.items() if l in code], dtype=int)
    return idx, np.array([code[single[i]] for i in idx], dtype=int)


def axis_result(
    key: str,
    variant: str,
    axis: str,
    g: Geometry,
    idx: np.ndarray,
    codes: np.ndarray,
    strata: np.ndarray,
    rng: np.random.Generator,
    n_perm: int,
) -> Result | None:
    """Multi-class silhouette vs the same classes shuffled within coverage strata."""
    if len(np.unique(codes)) < 2:
        return None
    dz, cnt = g.dz[np.ix_(idx, idx)], g.cnt[np.ix_(idx, idx)]
    obs = partition_silhouette(dz, cnt, codes)
    if np.isnan(obs):
        return None
    groups = [np.flatnonzero(strata[idx] == k) for k in np.unique(strata[idx])]
    null = np.empty(n_perm)
    for t in range(n_perm):
        shuffled = codes.copy()
        for grp in groups:
            shuffled[grp] = codes[rng.permutation(grp)]
        null[t] = partition_silhouette(dz, cnt, shuffled)
    null_mean, z, p = summarise(obs, null)
    return Result(
        key, variant, "axis", axis, len(idx), len(np.unique(codes)), obs, null_mean, z, p
    )


def analyse(
    key: str,
    cell_list: list,
    *,
    drop_g: bool,
    labels: dict[str, list[str]],
    axes: list[str],
    coverage: dict[str, int],
    rng: np.random.Generator,
    n_perm: int,
) -> list[Result]:
    variant = "without_g" if drop_g else "with_g"
    try:
        dist, bench = cp.composite_distance(cell_list, drop_g=drop_g)
    except ValueError as e:
        print(f"{key} [{variant}]: skipped, {e}", file=sys.stderr)
        return []
    seen = ps.observed_mask(cell_list, bench, drop_g=drop_g)
    g = Geometry.from_observed(np.where(seen, dist, np.nan))
    strata = cp.coverage_strata(bench, coverage)
    out = label_results(key, variant, g, bench, labels, strata, rng, n_perm)
    for axis in axes:
        idx, codes = single_label_codes(bench, labels, axis)
        res = axis_result(key, variant, axis, g, idx, codes, strata, rng, n_perm)
        if res is None:
            print(f"{key} [{variant}]: axis {axis!r} has too few single-label classes",
                  file=sys.stderr)
        else:
            out.append(res)
    return out


def fmt(x: float | None, spec: str = ".3f") -> str:
    return "-" if x is None else format(x, spec)


def print_tables(results: list[Result], cells: list[str]) -> None:
    print("Axis silhouette, single-label benchmarks only (aggregate cells):")
    print()
    print("| cell | g | axis | n | classes | silhouette | null | z | p |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        if r.level == "axis" and "|m" not in r.cell:
            print(f"| {r.cell} | {r.variant} | {r.name} | {r.n} | {r.k} | {r.sil:.3f} "
                  f"| {r.null_mean:.3f} | {fmt(r.z, '.1f')} | {r.p:.4f} |")

    # imputers agreeing with the aggregate, for the same dz|tag and variant
    robust: dict[tuple[str, str, str], list[bool]] = defaultdict(list)
    for r in results:
        if r.level == "label" and "|m" in r.cell:
            robust[(r.cell.split("|m")[0], r.variant, r.name)].append(r.p < ALPHA)

    for cell in cells:
        print()
        print(f"Label silhouette, cell {cell} (members vs non-members; imputers = "
              f"per-imputer cells with p < {ALPHA}):")
        print()
        print("| g | label | n | silhouette | null | z | p | imputers |")
        print("|---|---|---|---|---|---|---|---|")
        rows = [r for r in results if r.level == "label" and r.cell == cell]
        for r in sorted(rows, key=lambda r: (r.variant, -r.sil)):
            hits = robust.get((cell, r.variant, r.name), [])
            print(f"| {r.variant} | {r.name} | {r.n} | {r.sil:.3f} | {r.null_mean:.3f} "
                  f"| {fmt(r.z, '.1f')} | {r.p:.4f} | {sum(hits)}/{len(hits)} |")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-root", default="results/text_only")
    ap.add_argument("--data-root", default="data/text_only")
    ap.add_argument("--subject-groups", default=str(cp.DEFAULT_SUBJECT_GROUPS))
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tag", action="append", choices=("pa", "2f"),
                    help="factor tag to include; repeatable (default: pa)")
    ap.add_argument(
        "--cell", action="append",
        help="aggregate cell key (e.g. 'C|2f') to print the per-label table for; "
        "repeatable; default is the first aggregate cell",
    )
    ap.add_argument("--csv", help="write every result row here (all cells, all levels)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    results_root = cp.resolve(args.results_root)
    cells = cp.load_cells(results_root)
    if not cells:
        raise SystemExit(f"no loadings found under {results_root}")
    data_root = cp.resolve(args.data_root)
    groups = cp.load_subject_groups(cp.resolve(args.subject_groups))
    labels, axis_labels = cp.load_labels(data_root, groups)
    axes = [a for a in cp.AXIS_ORDER if a in axis_labels]
    coverage = cp.load_coverage(data_root)
    rng = np.random.default_rng(args.seed)

    jobs = list(ps.iter_jobs(cells, tuple(args.tag or ps.DEFAULT_TAGS)))
    if not jobs:
        raise SystemExit(f"no cells for tag(s) {args.tag or ps.DEFAULT_TAGS}")
    aggregate = [k for k, _ in jobs if "|m" not in k]
    shown = args.cell or aggregate[:1]
    unknown = [c for c in shown if c not in aggregate]
    if unknown:
        raise SystemExit(f"unknown aggregate cell(s) {unknown}; have {aggregate}")

    results: list[Result] = []
    for key, cell_list in jobs:
        print(f"{key} ...", file=sys.stderr)
        for drop_g in (False, True):
            results += analyse(
                key, cell_list, drop_g=drop_g, labels=labels, axes=axes,
                coverage=coverage, rng=rng, n_perm=args.n_perm,
            )
    if not results:
        raise SystemExit("no cell produced a result")
    print_tables(results, shown)
    if args.csv:
        pl.DataFrame([asdict(r) for r in results]).write_csv(cp.resolve(args.csv))


if __name__ == "__main__":
    main()
