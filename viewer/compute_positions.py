#!/usr/bin/env python3
"""Composite benchmark-similarity embedding and clustering for the viewer.

Reads every *_bifactor_*_loadings.csv under results/text_only/, builds a
consensus cosine-distance matrix between benchmarks per (densifier, tag,
year) by averaging the within-cell cosine distances (rotation/sign invariant
per cell, hence well-defined to average), then clusters and UMAP-embeds each
composite distance matrix and writes viewer/positions.json. Year-less loadings
files are the aggregate embedding (key "dz|tag"); per-release-year cohorts from
factor.R --timed are embedded separately (key "dz|tag|y<year>").

Each benchmark's vector within a cell is [g, F1..Fk] (group + general
factors). Cells where a benchmark is absent simply don't contribute to that
benchmark's pairwise means.

Clustering runs on the composite distance matrix itself, NOT on the 2D UMAP
coordinates: UMAP does not preserve density or global distance, so clusters
found in the embedding are partly artifacts of the embedding. Two algorithms
are emitted per cell, both consuming the same precomputed matrix:

  * HAC, average linkage (UPGMA). Ward/centroid/median are invalid here --
    they are defined only via squared Euclidean distance, and 1 - cos is not
    Euclidean (it is not even a metric; the triangle inequality fails).
    scipy builds the tree once and every k is a cut of that one tree, so the
    cuts are nested and cluster ids can be chained across k.
  * HDBSCAN, which declines to place genuine outliers (label -1) rather than
    forcing every benchmark into a group.

Each is run twice, with and without the general factor g. Every benchmark
loads positively on g (~98%, and g^2 is ~40% of each row's squared norm), so
including it compresses all pairwise distances toward each other; excluding it
asks the narrower question of which *group* factors a benchmark shares. Both
are emitted so the viewer can toggle. The UMAP coordinates are always computed
WITH g, so point positions are unaffected by this choice.

Human labels (see LABEL_COLUMNS) are scored independently of the clustering:
category_cohesion() tests whether each label's members sit closer together in
the distance matrix than a coverage-matched random set. Labels are authored on
benchmark content and never tuned against that score.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "viewer" / "positions.json"

NAME_RE = re.compile(
    r"^(?P<method>.+?)_(?P<dz>C|R|S|raw)_(?P<st>all_standard|all_aggressive)"
    r"_bifactor_(?P<tag>pa|2f|forced2f)(?:_y(?P<year>\d{4}))?_loadings\.csv$"
)
# Year-less files are the aggregate (all-years) embedding; `..._y<year>_` files
# come from factor.R --timed (release-year cohorts). forced2f is the DB run
# name leaked into timed filenames — normalize it to the combined 2f tag.
# Pre-tag runs wrote one bifactor file per cell with no pa/2f distinction.
LEGACY_RE = re.compile(
    r"^(?P<method>.+?)_(?P<dz>C|R|S|raw)_(?P<st>all_standard|all_aggressive)"
    r"_bifactor_loadings\.csv$"
)
# psych::omega() diagnostics, not loadings. Blacklisting rather than
# whitelisting F1*..F31* matters: real cells reach 162 factor columns.
DIAGNOSTIC_COLS = {"h2", "u2", "uq2", "p2", "com"}
FACTOR_RE = re.compile(r"^F\d+\*$")

LINKAGE_METHODS = ("average", "complete", "single", "weighted")
K_MIN, K_MAX = 2, 12
COHESION_PERM = 2000
COHESION_MIN_N = 4
COVERAGE_STRATA = 4

# Labels are AUTHORED in data/benchmarks.csv, one column per axis, and the
# column a label appears in IS its axis. There is deliberately no hardcoded
# vocabulary here: adding a value in the CSV is sufficient to make it appear in
# the viewer, so relabeling never requires a code change.
#
# The legacy single-label `category` column is retained in the data as
# provenance -- it records what the original curation said -- but it is not used
# for labeling. It was split on breadth of subject (general vs specialist),
# which put recall on both sides and so carved nothing real; these axes split on
# what is actually being assessed instead.
LABEL_COLUMNS: dict[str, str] = {
    # What the benchmark is about or measures: capabilities AND content domains,
    # multi-label, so `reading_comprehension` and `medical` coexist on one row.
    # A separate "what is assessed" axis was tried and merged back in: once task
    # carries the mechanism, it only restated subject (math x mathematics,
    # code x programming) or restated task (translation x machine_translation).
    "subject": "subject",
    # HOW the test is administered -- the shape of the interaction, not the
    # capability and not the scoring method (Chatbot Arena is `conversation`;
    # its Elo is just how it is scored).
    "task": "task_labels",
    "language": "language",
}
AXIS_ORDER = tuple(LABEL_COLUMNS)
# Labels are identified as "<axis>:<label>". The same display name can then
# live on several axes -- `miscellaneous` as a catch-all is the motivating case
# -- and stay distinct, which matters because "misc subject" and "misc language"
# are different sets and must be highlighted and scored separately.
LABEL_SEP = ":"
# Broad subject labels are ADDED to rows, never substituted: a row tagged
# `medical` also gets `specialized_domain`. The mapping is data, and deriving
# parents here (not storing them in benchmarks.csv) means they cannot drift out
# of sync with the narrow labels they summarise.
DEFAULT_SUBJECT_GROUPS = REPO / "data" / "subject_groups.csv"


type Cell = tuple[str, np.ndarray, list[str], list[str]]
"""(cell_key, loading matrix, benchmark ids, factor column names)."""

type CellKey = tuple[str, str, int | None]
"""(densifier, tag, release-year cohort or None for the all-years aggregate)."""


CELL_KEY_RE = re.compile(r"^(?P<method>.+)_(?:C|R|S|raw)_(?:all_standard|all_aggressive)$")
"""Recovers the method name from a cell key ("<method>_<dz>_<st>")."""


def load_cells(
    results_root: Path, *, legacy_names: bool = False
) -> dict[CellKey, list[Cell]]:
    """{ (dz, tag, year|None): [ (key, factor_matrix, [benchmarks], [factor_cols]) ] }"""
    cells: dict[CellKey, list[Cell]] = defaultdict(list)
    for path in sorted(results_root.glob("*/*_loadings.csv")):
        m = NAME_RE.match(path.name)
        # forced2f is the DB run name leaking into --timed filenames.
        tag = ("2f" if m["tag"] == "forced2f" else m["tag"]) if m else None
        year = (int(m["year"]) if m["year"] else None) if m else None
        if m is None and legacy_names:
            m = LEGACY_RE.match(path.name)
            tag, year = "legacy", None
        if m is None:
            continue
        df = pl.read_csv(path, infer_schema_length=0)
        # g first, then F1*..Fk* in file order; everything else is a diagnostic.
        cols = [
            c
            for c in df.columns
            if c != "benchmark" and c not in DIAGNOSTIC_COLS and (c == "g" or FACTOR_RE.match(c))
        ]
        if not cols:
            continue
        # skip corrupt cells whose benchmark names were lost upstream
        # (e.g. softimpute_corr surrogate wrote numeric column headers)
        bench_all = df["benchmark"].to_list()
        if any(b.strip().isdigit() for b in bench_all):
            print(f"skipping {path.name}: numeric benchmark names (corrupt)")
            continue
        bench = bench_all
        mat = df.select([pl.col(c).cast(pl.Float64, strict=False) for c in cols]).to_numpy()
        if not np.isfinite(mat).all():
            continue
        key = f"{m['method']}_{m['dz']}_{m['st']}"
        cells[(m["dz"], tag, year)].append((key, mat, bench, cols))
    return cells


def composite_distance(
    cell_list: list[Cell], *, drop_g: bool = False
) -> tuple[np.ndarray, list[str]]:
    """Average pairwise cosine distance across cells (union of benchmarks).

    With drop_g, the general-factor column is excluded so the distance
    reflects only the group factors. A cell whose sole factor column is g
    contributes nothing in that mode and is skipped.
    """
    bench_union: list[str] = []
    seen: set[str] = set()
    for _, _, bench, _ in cell_list:
        for b in bench:
            if b not in seen:
                seen.add(b)
                bench_union.append(b)
    idx = {b: i for i, b in enumerate(bench_union)}
    n = len(bench_union)

    dist_sum = np.zeros((n, n))
    count = np.zeros((n, n))
    for _, mat, bench, cols in cell_list:
        if drop_g:
            keep = [i for i, c in enumerate(cols) if c != "g"]
            if not keep:
                continue
            mat = mat[:, keep]
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        unit = np.where(norms > 0, mat / np.maximum(norms, 1e-12), 0.0)
        cos = unit @ unit.T
        d = np.clip(1.0 - cos, 0.0, 2.0)
        rows = [idx[b] for b in bench]
        r = np.array(rows)
        dist_sum[np.ix_(r, r)] += d
        count[np.ix_(r, r)] += 1.0

    if not count.any():
        raise ValueError("no cell contributed any pairwise distance")

    with np.errstate(invalid="ignore", divide="ignore"):
        dist = np.where(count > 0, dist_sum / np.maximum(count, 1e-12), np.nan)
    # Cells missing one benchmark leave NaNs; fill with the mean of the
    # benchmark's known distances (or global mean) so UMAP gets a full matrix.
    # These fabricated values are indistinguishable from measurements
    # downstream -- pair_coverage() reports how many there are.
    known = ~np.isnan(dist)
    global_mean = dist[known].mean()
    for i in range(n):
        for j in range(i + 1, n):
            if math.isnan(dist[i, j]):
                cand = [dist[i, k] for k in range(n) if known[i, k]] or [
                    dist[k, j] for k in range(n) if known[k, j]
                ]
                v = float(np.mean(cand)) if cand else global_mean
                dist[i, j] = dist[j, i] = v
    np.fill_diagonal(dist, 0.0)
    # symmetrize + PSD-ish for UMAP precomputed use
    dist = (dist + dist.T) / 2.0
    return dist, bench_union


def pair_coverage(cell_list: list[Cell], bench: list[str]) -> tuple[float, int]:
    """(fraction of off-diagonal pairs co-observed in >=1 cell, n fabricated)."""
    idx = {b: i for i, b in enumerate(bench)}
    n = len(bench)
    seen = np.zeros((n, n), dtype=bool)
    for _, _, cell_bench, _ in cell_list:
        r = np.array([idx[b] for b in cell_bench])
        seen[np.ix_(r, r)] = True
    np.fill_diagonal(seen, True)
    total = n * (n - 1)
    if total == 0:
        return 1.0, 0
    missing = int((~seen).sum())
    return round(1.0 - missing / total, 4), missing // 2


def condensed(dist: np.ndarray) -> np.ndarray:
    """Validate the distance-matrix invariants, then squareform.

    Exact comparisons are deliberate: composite_distance() zeroes the diagonal
    before symmetrizing, and (d + d.T)/2 is bit-exactly symmetric because
    IEEE-754 addition commutes. If that ever stops holding, an invariant broke
    and the error is the point -- a tolerance would paper over it.
    """
    if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
        raise ValueError(f"distance matrix must be square, got {dist.shape}")
    if not np.array_equal(dist, dist.T):
        raise ValueError("distance matrix is not exactly symmetric")
    if np.any(np.diag(dist) != 0.0):
        raise ValueError("distance matrix has a non-zero diagonal")
    if not np.isfinite(dist).all() or np.any(dist < 0.0):
        raise ValueError("distance matrix has negative or non-finite entries")
    return squareform(dist, checks=False)


def canonicalize(labels: np.ndarray, bench: list[str]) -> np.ndarray:
    """Relabel 0..K-1 by descending size, ties by smallest member id.

    Makes output independent of row order and of the clusterer's internal
    numbering. Noise (-1) passes through untouched.
    """
    groups: dict[int, list[str]] = defaultdict(list)
    for lab, b in zip(labels, bench):
        if lab >= 0:
            groups[int(lab)].append(b)
    order = sorted(groups, key=lambda g: (-len(groups[g]), min(groups[g])))
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[int(x)] if x >= 0 else -1 for x in labels])


def chain_align(prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
    """Renumber `cur` to maximise id continuity with `prev` (greedy Jaccard).

    Cuts of one linkage tree are nested, so going k -> k+1 splits exactly one
    cluster. Without this, size-descending ids reshuffle every rank below the
    split and the viewer's colors churn on every k change.
    """
    cur_sets = {c: set(np.flatnonzero(cur == c)) for c in np.unique(cur) if c >= 0}
    prev_sets = {p: set(np.flatnonzero(prev == p)) for p in np.unique(prev) if p >= 0}
    pairs = sorted(
        (
            (len(cs & ps) / len(cs | ps), -len(cs), c, p)
            for c, cs in cur_sets.items()
            for p, ps in prev_sets.items()
        ),
        reverse=True,
    )
    remap: dict[int, int] = {}
    taken: set[int] = set()
    for _, _, c, p in pairs:
        if c not in remap and p not in taken:
            remap[c] = p
            taken.add(p)
    nxt = 0
    for c in sorted(cur_sets, key=lambda c: (-len(cur_sets[c]), c)):
        if c not in remap:
            while nxt in taken:
                nxt += 1
            remap[c] = nxt
            taken.add(nxt)
    return np.array([remap[int(x)] if x >= 0 else -1 for x in cur])


def silhouette(dist: np.ndarray, labels: np.ndarray) -> float | None:
    """Precomputed silhouette over non-noise points, or None if degenerate.

    Noise must be excluded: a silhouette that treats -1 as a cluster measures
    the cohesion of "everything the algorithm declined to cluster".
    """
    m = labels >= 0
    lab = labels[m]
    n = int(m.sum())
    if n < 3:
        return None
    k = len(np.unique(lab))
    if k < 2 or k >= n:
        return None
    return round(float(silhouette_score(dist[np.ix_(m, m)], lab, metric="precomputed")), 4)


def hac_labels(
    dist: np.ndarray, bench: list[str], k_values: list[int], method: str
) -> dict[int, np.ndarray]:
    """One linkage tree, one cut per k, ids chained across ascending k."""
    z = linkage(condensed(dist), method=method)
    out: dict[int, np.ndarray] = {}
    prev: np.ndarray | None = None
    for k in k_values:
        lab = canonicalize(fcluster(z, k, criterion="maxclust") - 1, bench)
        if prev is not None:
            lab = chain_align(prev, lab)
        out[k] = lab
        prev = lab
    return out


def hdbscan_labels(
    dist: np.ndarray, min_cluster_size: int, min_samples: int, selection: str
) -> np.ndarray:
    """HDBSCAN on the precomputed matrix. copy=True is mandatory.

    sklearn 1.9's HDBSCAN mutates a precomputed input array in place (measured:
    changes of up to 0.157), and warns that the copy default flips in 1.10.
    Without it every number computed afterwards from the same matrix is wrong.
    """
    return HDBSCAN(
        metric="precomputed",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=selection,
        allow_single_cluster=False,
        copy=True,
    ).fit_predict(dist)


def split_labels(raw: str | None) -> list[str]:
    """Semicolon-separated cell -> label list. Empty cell -> []."""
    return [x.strip() for x in (raw or "").split(";") if x.strip()]


def load_subject_groups(path: Path) -> dict[str, str]:
    """Narrow subject label -> broad subject label. One level only."""
    if not path.exists():
        raise SystemExit(f"subject groups file not found: {path}")
    df = pl.read_csv(path, infer_schema_length=0)
    if df.columns != ["label", "group"]:
        raise SystemExit(f"{path}: expected columns [label, group], got {df.columns}")
    groups: dict[str, str] = {}
    for label, group in zip(df["label"], df["group"]):
        if not label or not group:
            raise SystemExit(f"{path}: blank label or group in row {label!r},{group!r}")
        if LABEL_SEP in label or LABEL_SEP in group:
            raise SystemExit(f"{path}: {label!r}/{group!r} contains {LABEL_SEP!r}")
        if label in groups:
            raise SystemExit(f"{path}: {label!r} is mapped twice")
        groups[label] = group
    chained = sorted(set(groups) & set(groups.values()))
    if chained:
        raise SystemExit(f"{path}: groups must not themselves be grouped: {chained}")
    return groups


def with_groups(subjects: list[str], groups: dict[str, str]) -> list[str]:
    """Narrow labels first, then their broad parents, deduplicated."""
    out = list(dict.fromkeys(subjects))
    for lab in subjects:
        parent = groups.get(lab)
        if parent is not None and parent not in out:
            out.append(parent)
    return out


def load_labels(
    data_root: Path,
    groups: dict[str, str] | None = None,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """(benchmark -> qualified label ids, axis -> ids by descending count).

    Ids are "<axis>:<label>", derived from which column a label sits in, so the
    vocabulary lives entirely in the data and a label name may repeat across axes.
    `groups` adds broad parents to the subject axis alongside the narrow labels.
    """
    path = data_root / "benchmarks.csv"
    if not path.exists():
        raise SystemExit(f"benchmarks.csv not found under {data_root}")
    df = pl.read_csv(path, infer_schema_length=0)
    missing = [c for c in LABEL_COLUMNS.values() if c not in df.columns]
    if missing:
        raise SystemExit(
            f"{path} is missing label columns {missing}; apply the label pass first"
        )

    labels: dict[str, list[str]] = defaultdict(list)
    counts: dict[str, dict[str, int]] = {ax: defaultdict(int) for ax in LABEL_COLUMNS}
    for axis, col in LABEL_COLUMNS.items():
        for bid, cell in zip(df["benchmark_id"], df[col]):
            cell_labels = split_labels(cell)
            if axis == "subject" and groups:
                cell_labels = with_groups(cell_labels, groups)
            for lab in cell_labels:
                if LABEL_SEP in lab:
                    raise SystemExit(
                        f"{bid}: label {lab!r} in `{col}` contains the reserved "
                        f"separator {LABEL_SEP!r}"
                    )
                qid = f"{axis}{LABEL_SEP}{lab}"
                labels[bid].append(qid)
                counts[axis][qid] += 1
    axis_labels = {
        ax: sorted(c, key=lambda l: (-c[l], l)) for ax, c in counts.items() if c
    }
    return dict(labels), axis_labels


def carry_forward(
    prior: dict,
    categories: dict[str, list[str]],
    label_params: dict,
) -> dict:
    """Relabel a prior cell whose loadings are not available locally.

    Cohesion needs the distance matrix, which the payload does not store, so a
    score survives only if its label's member set is unchanged; labels whose
    membership changed (or are new) get no score rather than a stale one.
    """
    entry = copy.deepcopy(prior)
    bench = entry["benchmarks"]
    new_cats = [categories.get(b, []) for b in bench]

    def members(cats: list) -> dict[str, frozenset[int]]:
        m: dict[str, set[int]] = defaultdict(set)
        for i, labs in enumerate(cats):
            for lab in [labs] if isinstance(labs, str) else labs or []:
                m[lab].add(i)
        return {k: frozenset(v) for k, v in m.items()}

    before, after = members(entry.get("categories") or []), members(new_cats)
    if "categories" in entry:
        entry["categories"] = new_cats
    if "category_cohesion" in entry:
        entry["category_cohesion"] = {
            v: [r for r in rows if before.get(r["category"]) == after.get(r["category"])]
            for v, rows in entry["category_cohesion"].items()
        }
    if "clusters" in entry:
        entry["clusters"]["params"].update(label_params)
        entry["clusters"]["diagnostics"]["n_labelled"] = sum(1 for c in new_cats if c)
    return entry


def same_inputs(prior: dict, cell_list: list[Cell]) -> bool:
    """Was this prior cell built from the loadings available now?

    Cohesion is recomputed from the local distance matrix, so a prior whose
    geometry came from a different set of loadings (e.g. one imputer's files are
    missing locally) must be carried forward, not rescored against a different
    matrix than the one that produced its points and clusters.
    """
    if prior.get("n_cells") != len(cell_list):
        return False
    recorded = prior.get("clusters", {}).get("diagnostics", {}).get("n_factors")
    return recorded is None or recorded == [len(cols) for _, _, _, cols in cell_list]


def dump_labels(data_root: Path, out: Path, groups: dict[str, str]) -> None:
    """Write the applied benchmark -> label table. Derived; never hand-edited."""
    df = pl.read_csv(data_root / "benchmarks.csv", infer_schema_length=0)
    col = LABEL_COLUMNS["subject"]
    df = df.with_columns(
        pl.col(col).map_elements(
            lambda c: ";".join(with_groups(split_labels(c), groups)),
            return_dtype=pl.String,
        )
    )
    cols = ["benchmark_id", "benchmark_name", "category"] + list(LABEL_COLUMNS.values())
    table = df.select([c for c in cols if c in df.columns]).sort("benchmark_id")
    table.write_csv(out)
    print(f"wrote {out}: {table.height} benchmarks")


def load_coverage(data_root: Path) -> dict[str, int]:
    """benchmark_id -> number of distinct models evaluated on it."""
    df = pl.read_csv(data_root / "results.csv", infer_schema_length=0)
    return dict(
        df.group_by("benchmark_id")
        .agg(pl.col("model_name").n_unique().alias("n"))
        .iter_rows()
    )


def coverage_strata(bench: list[str], coverage: dict[str, int]) -> np.ndarray:
    """Quantile bins of log10(model coverage), for a matched permutation null."""
    v = np.log10([max(1, coverage.get(b, 1)) for b in bench])
    cuts = np.quantile(v, np.linspace(0.0, 1.0, COVERAGE_STRATA + 1)[1:-1])
    return np.digitize(v, cuts)


def category_cohesion(
    dist: np.ndarray,
    bench: list[str],
    labels: dict[str, list[str]],
    strata: np.ndarray,
    rng: np.random.Generator,
) -> list[dict]:
    """Per label: is its member set tighter in `dist` than a matched random set?

    Answers "does the factor structure know about this label" WITHOUT clustering
    -- no k, no linkage, no cluster assignment. That matters because ARI asks
    whether the clusters partition INTO the labels, which is structurally
    impossible when the silhouette-max solution is k=2; a label can be real and
    tight while sitting nested inside a larger cluster.

    Multi-label safe: each label is tested independently one-vs-rest, so a
    benchmark carrying several labels contributes to several tests. This is why
    it replaced ARI, which requires hard partitions.

    The null is matched on log-coverage quartile, not uniform. Cluster structure
    in this corpus tracks how many models a benchmark was evaluated on, so a
    uniform null would credit that measurement artifact as content signal.

    Negative z means tighter than chance. p is one-sided for tightness.

    NOTE: results are inflated by benchmark families -- a parent plus its own
    subdomains (EWoK's 11) are near-identical vectors by construction, so a
    label dominated by one family scores as hyper-cohesive. Interpret alongside
    family concentration; do not read a single z in isolation.
    """
    pos = {b: i for i, b in enumerate(bench)}
    members: dict[str, list[int]] = defaultdict(list)
    for b, labs in labels.items():
        if b in pos:
            for lab in labs:
                members[lab].append(pos[b])

    n = len(bench)
    pool = {k: np.flatnonzero(strata == k) for k in range(COVERAGE_STRATA)}
    out = []
    for cat in sorted(members):
        idx = np.array(sorted(set(members[cat])))
        m = len(idx)
        if m < COHESION_MIN_N or m >= n:
            continue
        sub = dist.take(idx, 0).take(idx, 1)
        pairs = m * (m - 1) / 2.0
        obs = (sub.sum() / 2.0) / pairs
        want = np.bincount(strata[idx], minlength=COVERAGE_STRATA)
        null = np.empty(COHESION_PERM)
        for t in range(COHESION_PERM):
            r = np.concatenate(
                [
                    rng.choice(pool[k], want[k], replace=False)
                    for k in range(COVERAGE_STRATA)
                    if want[k]
                ]
            )
            rs = dist.take(r, 0).take(r, 1)
            null[t] = (rs.sum() / 2.0) / pairs
        sd = null.std()
        out.append(
            {
                "category": cat,
                "n": m,
                "within": round(float(obs), 4),
                "null_mean": round(float(null.mean()), 4),
                "z": round(float((obs - null.mean()) / sd), 3) if sd > 0 else None,
                "p": round(float((null <= obs).sum() / COHESION_PERM), 4),
            }
        )
    return sorted(out, key=lambda r: (r["z"] is None, r["z"]))


def cluster_records(labels: np.ndarray) -> list[dict]:
    """Per-cluster sizes, descending. Noise (-1) is never a record.

    Deliberately reports only id and size. An earlier version named each
    cluster after the plurality benchmarks.csv category of its members, which
    over-claimed: with no base-rate correction a cluster could be named after a
    bucket it was not enriched for, and a singleton scored 100% "purity" on one
    member. Cluster-vs-category agreement is reported once per cell as ARI
    instead, where it can be read as the weak criterion it is.
    """
    sizes: dict[int, int] = defaultdict(int)
    for lab in labels:
        if lab >= 0:
            sizes[int(lab)] += 1
    return [
        {"id": cid, "size": sizes[cid]}
        for cid in sorted(sizes, key=lambda g: (-sizes[g], g))
    ]


# ARI against the curated labels was removed here deliberately. It requires two
# hard PARTITIONS, and the ground truth is now multi-label -- a benchmark can be
# both `math` and `multilingual`, which no partition represents. It was also the
# wrong instrument even when labels were single: ARI asks whether the clusters
# partition INTO the categories, which is structurally impossible when the
# silhouette-max solution is k=2, so it read ~0 while per-label cohesion showed
# real signal. category_cohesion() is the replacement. If a single
# cluster-vs-taxonomy number is ever wanted, the multi-label-valid form is
# pair-counting over "these two share at least one label", not ARI.


def cluster_variant(
    dist: np.ndarray,
    bench: list[str],
    k_values: list[int],
    args: argparse.Namespace,
) -> dict:
    """HAC (all k) + HDBSCAN for one g-variant of the distance matrix."""
    by_k: dict[str, dict] = {}
    best_k, best_sil = None, -np.inf
    for k, lab in hac_labels(dist, bench, k_values, args.linkage).items():
        sil = silhouette(dist, lab)
        by_k[str(k)] = {"labels": [int(x) for x in lab], "silhouette": sil}
        if sil is not None and sil > best_sil:
            best_k, best_sil = k, sil
    if best_k is None:
        best_k = k_values[0]
    best_labels = np.array(by_k[str(best_k)]["labels"])

    n = len(bench)
    mcs = args.hdbscan_min_cluster_size or max(5, round(math.sqrt(n)))
    hdb = canonicalize(
        hdbscan_labels(dist, mcs, args.hdbscan_min_samples, args.hdbscan_selection), bench
    )
    noise_n = int((hdb < 0).sum())

    return {
        "hac": {
            "best_k": best_k,
            "by_k": by_k,
            "records": cluster_records(best_labels),
        },
        "hdbscan": {
            "labels": [int(x) for x in hdb],
            "n_clusters": int(len(np.unique(hdb[hdb >= 0]))),
            "noise_n": noise_n,
            "noise_frac": round(noise_n / n, 4) if n else 0.0,
            "min_cluster_size": mcs,
            "silhouette": silhouette(dist, hdb),
            "records": cluster_records(hdb),
        },
    }


def embed(dist: np.ndarray) -> np.ndarray:
    # Imported lazily: umap pulls in numba, and clustering must not require an
    # embedding library (it also makes this module importable in tests).
    from umap import UMAP

    return UMAP(
        n_components=2,
        metric="precomputed",
        n_neighbors=10,
        min_dist=0.15,
        random_state=42,
    ).fit_transform(dist)


LEGACY_BANNER = """\
LEGACY MODE: reading untagged bifactor files from the pre-tag, multimodal-
inclusive corpus. Benchmark set, factor counts, and the pa/2f distinction do
not match results/text_only. Code-path verification only."""


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-root", default="results/text_only")
    ap.add_argument("--data-root", default="data/text_only")
    ap.add_argument(
        "--subject-groups",
        default=str(DEFAULT_SUBJECT_GROUPS),
        help="CSV of narrow -> broad subject labels; parents are added, not substituted",
    )
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument(
        "--legacy-names",
        action="store_true",
        help="also match untagged *_bifactor_loadings.csv (requires --out)",
    )
    ap.add_argument("--linkage", choices=LINKAGE_METHODS, default="average")
    ap.add_argument("--k-min", type=int, default=K_MIN)
    ap.add_argument(
        "--k-max",
        type=int,
        default=K_MAX,
        help=f"default {K_MAX}; use 14 to let ARI vs the 14 category buckets reach its ceiling",
    )
    ap.add_argument(
        "--hdbscan-min-cluster-size", type=int, default=0, help="0 = max(5, sqrt(n))"
    )
    ap.add_argument("--hdbscan-min-samples", type=int, default=5)
    ap.add_argument("--hdbscan-selection", choices=("eom", "leaf"), default="eom")
    ap.add_argument(
        "--relabel",
        action="store_true",
        help="reuse points and clusters from the existing --out payload and recompute "
        "only labels + cohesion. Labels feed nothing upstream -- not the distance "
        "matrix, not UMAP, not the clustering -- so re-embedding on a label change is "
        "both wasted work and destabilising (UMAP output shifts across library "
        "versions, which would churn every coordinate on every label edit).",
    )
    ap.add_argument("--no-js", action="store_true")
    ap.add_argument(
        "--dump-labels",
        metavar="PATH",
        help="write the benchmark -> label table (all axes) as CSV and exit",
    )
    return ap.parse_args()


def resolve(root: str) -> Path:
    p = Path(root)
    return p if p.is_absolute() else REPO / p


def main() -> None:
    args = parse_args()
    out_path = resolve(args.out)
    js_path = out_path.with_suffix(".js")
    if args.legacy_names and (
        out_path.resolve() == DEFAULT_OUT
        or js_path.resolve() == DEFAULT_OUT.with_suffix(".js")
    ):
        raise SystemExit(
            "--legacy-names reads the multimodal-inclusive legacy tree; refusing to "
            "overwrite the committed viewer/positions.json. Pass --out."
        )
    if args.legacy_names:
        print(LEGACY_BANNER)

    if args.dump_labels:
        dump_labels(
            resolve(args.data_root),
            resolve(args.dump_labels),
            load_subject_groups(resolve(args.subject_groups)),
        )
        return

    prior: dict = {}
    if args.relabel:
        if not out_path.exists():
            raise SystemExit(f"--relabel needs an existing {out_path}; run without it first")
        prior = json.loads(out_path.read_text())
        print(f"relabel mode: reusing points + clusters from {out_path} ({len(prior)} cells)")

    results_root = resolve(args.results_root)
    cells = load_cells(results_root, legacy_names=args.legacy_names)
    if not cells:
        raise SystemExit(f"no loadings found under {results_root}")
    subject_groups = load_subject_groups(resolve(args.subject_groups))
    categories, axis_labels = load_labels(resolve(args.data_root), subject_groups)
    # qualified child id -> qualified parent id, for the viewer's nested legend
    label_groups = {
        f"subject{LABEL_SEP}{c}": f"subject{LABEL_SEP}{g}" for c, g in subject_groups.items()
    }
    label_params = {
        "label_sep": LABEL_SEP,
        "label_groups": label_groups,
        "axis_order": [a for a in AXIS_ORDER if a in axis_labels],
        "axis_labels": axis_labels,
    }
    model_coverage = load_coverage(resolve(args.data_root))
    rng = np.random.default_rng(42)

    out = {}
    # One job per composite matrix: the aggregate over every imputation method
    # (key "dz|tag", unchanged for backwards compatibility) plus one per method
    # (key "dz|tag|m<method>"). A loadings filename is
    # <method>_<dz>_<st>_bifactor_...; method names may contain underscores, so
    # the trailing "_<dz>_<st>" fields are stripped exactly.
    jobs: list[tuple[str, list[Cell], str | None]] = []
    for (dz, tag, year), cell_list in sorted(
        cells.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] is not None, kv[0][2] or 0)
    ):
        jobs.append((f"{dz}|{tag}" if year is None else f"{dz}|{tag}|y{year}", cell_list, None))
        by_method: dict[str, list[Cell]] = defaultdict(list)
        for cell in cell_list:
            # A cell key is "<method>_<dz>_<st>" where <st> itself contains an
            # underscore (all_standard / all_aggressive), so the method cannot
            # be recovered by counting fields -- anchor on the known tail.
            m = CELL_KEY_RE.match(cell[0])
            if m is None:
                raise RuntimeError(f"unparsable cell key {cell[0]!r}")
            by_method[m.group(1)].append(cell)
        for method, method_cells in sorted(by_method.items()):
            base = f"{dz}|{tag}|m{method}"
            jobs.append((base if year is None else f"{base}|y{year}", method_cells, method))

    for key, cell_list, method in jobs:
        old = prior.get(key) if args.relabel else None
        if old is not None and not same_inputs(old, cell_list):
            out[key] = carry_forward(old, categories, label_params)
            print(f"{key}: local loadings differ from the prior's -- carried forward")
            continue
        dist, bench = composite_distance(cell_list)
        dist_ng, bench_ng = composite_distance(cell_list, drop_g=True)
        pristine, pristine_ng = dist.copy(), dist_ng.copy()
        if bench_ng != bench:
            raise RuntimeError("drop_g changed the benchmark ordering")

        if old is not None and [p["benchmark"] for p in old["points"]] != bench:
            raise SystemExit(
                f"{key}: benchmark set changed since {out_path} was written; "
                "--relabel cannot reuse its geometry -- re-run without it"
            )

        n = len(bench)
        # Year cohorts are far smaller than the aggregate, so k is clamped to n.
        k_values = [k for k in range(args.k_min, args.k_max + 1) if k < n]
        coverage, fabricated = pair_coverage(cell_list, bench)
        cohort = "all years" if year is None else f"cohort {year}"

        xy = (
            [(p["x"], p["y"]) for p in old["points"]] if old is not None else embed(dist)
        )
        entry = {
            "densifier": dz,
            "tag": tag,
            **({} if method is None else {"method": method}),
            **({} if year is None else {"year": year}),
            "n_cells": len(cell_list),
            "benchmarks": bench,
            "points": [
                {"benchmark": b, "x": round(float(x), 5), "y": round(float(y), 5)}
                for b, (x, y) in zip(bench, xy)
            ],
        }
        out[key] = entry

        # A cohort too small to cut into >=2 groups still gets an embedding; the
        # viewer hides the cluster controls per-cell when `clusters` is absent.
        if not k_values:
            print(f"{key}: {n} benchmarks ({cohort}) — too few to cluster, embedding only")
            continue

        if old is not None and "clusters" in old:
            variants = {v: old["clusters"][v] for v in ("with_g", "without_g")}
        else:
            variants = {
                "with_g": cluster_variant(dist, bench, k_values, args),
                "without_g": cluster_variant(dist_ng, bench, k_values, args),
            }
        if not (np.array_equal(dist, pristine) and np.array_equal(dist_ng, pristine_ng)):
            raise RuntimeError(
                "clustering mutated the distance matrix (see sklearn HDBSCAN copy=)"
            )

        entry["categories"] = [categories.get(b, []) for b in bench]
        strata = coverage_strata(bench, model_coverage)
        entry["category_cohesion"] = {
            "with_g": category_cohesion(dist, bench, categories, strata, rng),
            "without_g": category_cohesion(dist_ng, bench, categories, strata, rng),
        }
        entry["clusters"] = {
            "params": {
                "linkage": args.linkage,
                "k_values": k_values,
                "hdbscan": {
                    "min_samples": args.hdbscan_min_samples,
                    "cluster_selection_method": args.hdbscan_selection,
                },
                **label_params,
            },
            "diagnostics": {
                "n_factors": [len(cols) for _, _, _, cols in cell_list],
                "pair_coverage": coverage,
                "n_fabricated_pairs": fabricated,
                "n_labelled": sum(1 for b in bench if categories.get(b)),
            },
            **variants,
        }

        hac = variants["with_g"]["hac"]
        hdb = variants["with_g"]["hdbscan"]
        print(
            f"{key}: {n} benchmarks, {len(cell_list)} cells ({cohort}), "
            f"coverage {coverage:.3f} | HAC best k={hac['best_k']} "
            f"sil={hac['by_k'][str(hac['best_k'])]['silhouette']} | "
            f"HDBSCAN {hdb['n_clusters']} clusters, {hdb['noise_frac']:.0%} noise"
        )

    # --relabel must never shrink the payload: cells whose loadings are not on
    # this machine are carried forward rather than silently dropped.
    for key in [k for k in prior if k not in out]:
        out[key] = carry_forward(prior[key], categories, label_params)
        print(f"{key}: no local loadings -- carried forward; changed labels have no cohesion")

    out_path.write_text(json.dumps(out, indent=1))
    print(f"wrote {out_path}")
    if not args.no_js:
        js_path.write_text("window.POSITIONS = " + json.dumps(out) + ";\n")
        print(f"wrote {js_path}")


if __name__ == "__main__":
    main()
