#!/usr/bin/env python3
"""Quantify whether format-similar benchmarks sit far apart in the distance matrix.

Replaces the visual UMAP check with tests on the n x n benchmark distance matrix
that viewer/compute_positions.py builds. "Similar" is defined by label rule,
BEFORE any distance is looked at: two benchmarks are format-matched when they
share a task label and a language label. Among those, subject-disjoint pairs
are the candidates for "similar in format, different in what they measure".

Three readouts per cell (aggregate or single imputer), with and without g:
  1. separation: mean distance of the subject-disjoint format-matched pairs vs
     a null where benchmarks are shuffled within log-coverage strata.
  2. contrast: AUC of subject-disjoint vs subject-sharing pairs at the same
     format (0.5 = subject does not matter, 1 = disjoint always farther).
  3. named pairs (data/pair_examples.csv): percentile among all observed pairs.
     Controls (near-duplicates such as gpqa/gpqa_diamond) must rank low, or the
     matrix cannot support the claim.

Both nulls permute benchmarks, not pairs, because pairs share benchmarks and a
pair-level null would be badly overconfident. Pairs never co-observed in any
cell are excluded: composite_distance() fills them with a row mean.

If the result is "not farther than chance", that is the result. Do not change
the pair rule after seeing it.

Scoped to --tag pa by default (2f excluded); pass --tag 2f to include it.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[1]
N_PERM = 2000
SEED = 42
DEFAULT_PAIRS = REPO / "data" / "pair_examples.csv"
ROLES = ("control", "example")
DEFAULT_TAGS = ("pa",)


def _load_positions():
    """viewer/ is not a package; reuse an already-loaded module if there is one."""
    if "compute_positions" in sys.modules:
        return sys.modules["compute_positions"]
    spec = importlib.util.spec_from_file_location(
        "compute_positions", REPO / "viewer" / "compute_positions.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["compute_positions"] = mod
    spec.loader.exec_module(mod)
    return mod


cp = _load_positions()

type Cell = cp.Cell


@dataclass(frozen=True)
class NamedPair:
    a: str
    b: str
    role: str


@dataclass(frozen=True)
class Separation:
    mean: float
    null_mean: float
    z: float | None
    p: float


@dataclass(frozen=True)
class Contrast:
    auc: float
    n_disjoint: int
    n_shared: int
    p: float


@dataclass(frozen=True)
class FormatPairs:
    pairs: np.ndarray
    """(P, 2) indices i<j into bench that share a task and a language label,
    and both carry at least one subject label."""
    subject_overlap: np.ndarray
    """(n, n) bool: the two benchmarks share a subject label."""


def incidence(bench: list[str], labels: dict[str, list[str]], axis: str) -> np.ndarray:
    """(n, L) bool matrix of which benchmark carries which label on `axis`."""
    prefix = f"{axis}{cp.LABEL_SEP}"
    vocab = sorted({l for b in bench for l in labels.get(b, []) if l.startswith(prefix)})
    col = {l: k for k, l in enumerate(vocab)}
    m = np.zeros((len(bench), len(vocab)), dtype=bool)
    for i, b in enumerate(bench):
        for l in labels.get(b, []):
            if l in col:
                m[i, col[l]] = True
    return m


def overlap(m: np.ndarray) -> np.ndarray:
    """(n, n) bool: rows share at least one column."""
    x = m.astype(np.int32)
    return (x @ x.T) > 0


def format_pairs(bench: list[str], labels: dict[str, list[str]]) -> FormatPairs:
    """Pairs that match on task and language. Subject is left free for the tests."""
    task = overlap(incidence(bench, labels, "task"))
    lang = overlap(incidence(bench, labels, "language"))
    subj = incidence(bench, labels, "subject")
    has_subject = subj.any(axis=1)
    ok = task & lang & has_subject[:, None] & has_subject[None, :]
    return FormatPairs(np.argwhere(np.triu(ok, k=1)), overlap(subj))


def observed_mask(cell_list: list[Cell], bench: list[str], *, drop_g: bool) -> np.ndarray:
    """(n, n) bool: pair co-observed in a cell that composite_distance() used."""
    idx = {b: i for i, b in enumerate(bench)}
    seen = np.zeros((len(bench), len(bench)), dtype=bool)
    for c in cell_list:
        if drop_g and all(col == "g" for col in c.cols):
            continue
        r = np.array([idx[b] for b in c.bench])
        seen[np.ix_(r, r)] = True
    np.fill_diagonal(seen, False)
    return seen


def strata_permutations(strata: np.ndarray, rng: np.random.Generator, n_perm: int):
    """Vertex permutations that only swap benchmarks within a coverage stratum."""
    groups = [np.flatnonzero(strata == k) for k in np.unique(strata)]
    for _ in range(n_perm):
        perm = np.arange(len(strata))
        for g in groups:
            perm[g] = rng.permutation(g)
        yield perm


def separation_test(
    d_obs: np.ndarray,
    target: np.ndarray,
    strata: np.ndarray,
    rng: np.random.Generator,
    n_perm: int,
) -> Separation:
    """Is the mean distance of `target` pairs larger than for coverage-matched
    random pairs? d_obs holds NaN for unobserved pairs; targets must be observed.
    One-sided p for "farther"."""
    if len(target) == 0:
        raise ValueError("no target pairs")
    i, j = target[:, 0], target[:, 1]
    obs = float(d_obs[i, j].mean())
    null = np.array(
        [np.nanmean(d_obs[p[i], p[j]]) for p in strata_permutations(strata, rng, n_perm)]
    )
    sd = float(null.std())
    return Separation(
        mean=obs,
        null_mean=float(null.mean()),
        z=(obs - float(null.mean())) / sd if sd > 0 else None,
        p=(1 + int((null >= obs).sum())) / (1 + n_perm),
    )


def format_contrast(
    d_obs: np.ndarray,
    fmt: FormatPairs,
    strata: np.ndarray,
    rng: np.random.Generator,
    n_perm: int,
) -> Contrast:
    """AUC that a subject-disjoint pair is farther than a subject-sharing pair,
    among format-matched pairs. Null: shuffle subject sets across benchmarks
    within coverage strata (format stays fixed)."""
    i, j = fmt.pairs[:, 0], fmt.pairs[:, 1]
    ranks = rankdata(d_obs[i, j])

    def auc(shared: np.ndarray) -> float:
        n_b = int(shared.sum())
        n_a = len(shared) - n_b
        if n_a == 0 or n_b == 0:
            return float("nan")
        return float((ranks[~shared].sum() - n_a * (n_a + 1) / 2) / (n_a * n_b))

    shared = fmt.subject_overlap[i, j]
    obs = auc(shared)
    if np.isnan(obs):
        raise ValueError("format-matched pairs are all subject-disjoint or all shared")
    null = np.array(
        [
            auc(fmt.subject_overlap[p[i], p[j]])
            for p in strata_permutations(strata, rng, n_perm)
        ]
    )
    null = null[~np.isnan(null)]
    return Contrast(
        auc=obs,
        n_disjoint=int((~shared).sum()),
        n_shared=int(shared.sum()),
        p=(1 + int((null >= obs).sum())) / (1 + len(null)),
    )


def percentile(sorted_pool: np.ndarray, value: float) -> float:
    """% of observed pairs strictly closer than `value`."""
    return 100.0 * float(np.searchsorted(sorted_pool, value, side="left")) / len(sorted_pool)


def load_named_pairs(path: Path, known: set[str]) -> list[NamedPair]:
    if not path.exists():
        raise SystemExit(f"pairs file not found: {path}")
    df = pl.read_csv(path, infer_schema_length=0)
    if df.columns != ["a", "b", "role"]:
        raise SystemExit(f"{path}: expected columns [a, b, role], got {df.columns}")
    out: list[NamedPair] = []
    seen: set[frozenset[str]] = set()
    for a, b, role in df.iter_rows():
        if role not in ROLES:
            raise SystemExit(f"{path}: role {role!r} not in {ROLES}")
        if a == b:
            raise SystemExit(f"{path}: {a!r} paired with itself")
        unknown = [x for x in (a, b) if x not in known]
        if unknown:
            raise SystemExit(f"{path}: unknown benchmark id(s) {unknown}")
        if frozenset((a, b)) in seen:
            raise SystemExit(f"{path}: duplicate pair {a!r},{b!r}")
        seen.add(frozenset((a, b)))
        out.append(NamedPair(a, b, role))
    return out


@dataclass(frozen=True)
class Row:
    cell: str
    variant: str
    n_bench: int
    n_pairs: int
    n_dropped: int
    sep: Separation
    contrast: Contrast | None
    named: list[tuple[NamedPair, float | None]]


def analyse(
    key: str,
    cell_list: list[Cell],
    *,
    drop_g: bool,
    labels: dict[str, list[str]],
    coverage: dict[str, int],
    named: list[NamedPair],
    rng: np.random.Generator,
    n_perm: int,
) -> Row | None:
    variant = "without_g" if drop_g else "with_g"
    try:
        dist, bench = cp.composite_distance(cell_list, drop_g=drop_g)
    except ValueError as e:
        print(f"{key} [{variant}]: skipped, {e}", file=sys.stderr)
        return None
    seen = observed_mask(cell_list, bench, drop_g=drop_g)
    d_obs = np.where(seen, dist, np.nan)
    strata = cp.coverage_strata(bench, coverage)

    fmt = format_pairs(bench, labels)
    observed = seen[fmt.pairs[:, 0], fmt.pairs[:, 1]]
    kept = FormatPairs(fmt.pairs[observed], fmt.subject_overlap)
    dropped = int((~observed).sum())
    disjoint = kept.pairs[~kept.subject_overlap[kept.pairs[:, 0], kept.pairs[:, 1]]]

    try:
        sep = separation_test(d_obs, disjoint, strata, rng, n_perm)
    except ValueError as e:
        print(f"{key} [{variant}]: skipped, {e}", file=sys.stderr)
        return None
    try:
        contrast = format_contrast(d_obs, kept, strata, rng, n_perm)
    except ValueError as e:
        print(f"{key} [{variant}]: no contrast, {e}", file=sys.stderr)
        contrast = None

    iu = np.triu_indices(len(bench), k=1)
    pool = np.sort(d_obs[iu][seen[iu]])
    pos = {b: i for i, b in enumerate(bench)}
    named_pct: list[tuple[NamedPair, float | None]] = []
    for np_ in named:
        if np_.a in pos and np_.b in pos and seen[pos[np_.a], pos[np_.b]]:
            named_pct.append((np_, percentile(pool, d_obs[pos[np_.a], pos[np_.b]])))
        else:
            named_pct.append((np_, None))
    return Row(key, variant, len(bench), len(disjoint), dropped, sep, contrast, named_pct)


def iter_jobs(cells: dict, tags: tuple[str, ...] = DEFAULT_TAGS):
    """Aggregate (all-years) cells of the given tags: all imputers pooled, then
    one per imputer. Thin filter over cp.build_jobs(), the same job list the
    viewer itself builds -- no separate method-parsing logic here."""
    for j in cp.build_jobs(cells):
        if j.year is None and j.tag in tags:
            yield j.key, list(j.cells)


def role_pcts(row: Row, role: str) -> list[float]:
    return [p for np_, p in row.named if np_.role == role and p is not None]


def fmt(x: float | None, spec: str = ".3f") -> str:
    return "-" if x is None else format(x, spec)


def print_tables(rows: list[Row]) -> None:
    print("| cell | g | bench | pairs (dropped) | mean d | null | z | p | AUC | p(AUC) "
          "| control max %ile | example min %ile |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        ctrl, ex = role_pcts(r, "control"), role_pcts(r, "example")
        print(
            f"| {r.cell} | {r.variant} | {r.n_bench} | {r.n_pairs} ({r.n_dropped}) "
            f"| {r.sep.mean:.3f} | {r.sep.null_mean:.3f} | {fmt(r.sep.z, '.1f')} "
            f"| {r.sep.p:.4f} | {fmt(r.contrast and r.contrast.auc)} "
            f"| {fmt(r.contrast and r.contrast.p, '.4f')} "
            f"| {fmt(max(ctrl) if ctrl else None, '.1f')} "
            f"| {fmt(min(ex) if ex else None, '.1f')} |"
        )
    print()
    print("Named pairs, aggregate cells (percentile of distance among observed pairs; "
          "controls should be low, examples high):")
    print()
    print("| cell | g | a | b | role | %ile |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        if "|m" in r.cell:
            continue
        for np_, p in r.named:
            print(f"| {r.cell} | {r.variant} | {np_.a} | {np_.b} | {np_.role} | {fmt(p, '.1f')} |")


def to_records(rows: list[Row]) -> list[dict]:
    out = []
    for r in rows:
        ctrl, ex = role_pcts(r, "control"), role_pcts(r, "example")
        out.append(
            {
                "cell": r.cell,
                "variant": r.variant,
                "n_bench": r.n_bench,
                "n_pairs": r.n_pairs,
                "n_dropped": r.n_dropped,
                **{f"sep_{k}": v for k, v in asdict(r.sep).items()},
                **{
                    f"contrast_{k}": v
                    for k, v in (asdict(r.contrast) if r.contrast else {}).items()
                },
                "control_max_pct": max(ctrl) if ctrl else None,
                "example_min_pct": min(ex) if ex else None,
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-root", default="results/text_only")
    ap.add_argument("--data-root", default="data/text_only")
    ap.add_argument("--subject-groups", default=str(cp.DEFAULT_SUBJECT_GROUPS))
    ap.add_argument("--pairs", default=str(DEFAULT_PAIRS), help="authored a,b,role CSV")
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tag", action="append", choices=("pa", "2f"),
                    help="factor tag to include; repeatable (default: pa)")
    ap.add_argument("--csv", help="also write the summary rows here")
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
    named = load_named_pairs(Path(args.pairs), set(coverage))
    rng = np.random.default_rng(args.seed)

    rows: list[Row] = []
    for key, cell_list in iter_jobs(cells, tuple(args.tag or DEFAULT_TAGS)):
        print(f"{key} ...", file=sys.stderr)
        for drop_g in (False, True):
            row = analyse(
                key, cell_list, drop_g=drop_g, labels=labels, coverage=coverage,
                named=named, rng=rng, n_perm=args.n_perm,
            )
            if row is not None:
                rows.append(row)
    if not rows:
        raise SystemExit("no cell produced a result")
    print_tables(rows)
    if args.csv:
        pl.DataFrame(to_records(rows)).write_csv(cp.resolve(args.csv))


if __name__ == "__main__":
    main()
