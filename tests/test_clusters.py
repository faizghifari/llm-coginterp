#!/usr/bin/env python3
"""Unit tests for the viewer's benchmark clustering.

viewer/ is not a package, so the module is loaded by path. The synthetic
block-structured distance matrix lives here and only here -- fixtures must not
leak into the production script.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

_SPEC = importlib.util.spec_from_file_location(
    "compute_positions",
    Path(__file__).resolve().parents[1] / "viewer" / "compute_positions.py",
)
cp = importlib.util.module_from_spec(_SPEC)
sys.modules["compute_positions"] = cp
_SPEC.loader.exec_module(cp)


def blocks(n_per=10, k=4, within=0.15, between=0.9, sigma=0.02, seed=0):
    """Planted-cluster distance matrix + its ground-truth labels."""
    rng = np.random.default_rng(seed)
    truth = np.repeat(np.arange(k), n_per)
    n = k * n_per
    d = np.where(truth[:, None] == truth[None, :], within, between)
    d = d + rng.normal(0, sigma, (n, n))
    d = np.clip((d + d.T) / 2.0, 0.0, 2.0)
    np.fill_diagonal(d, 0.0)
    return d, truth


def test_hac_recovers_planted_blocks():
    d, truth = blocks()
    labels = cp.hac_labels(d, [f"b{i}" for i in range(len(d))], [4], "average")[4]
    assert adjusted_rand_score(truth, labels) == 1.0


def test_canonical_labels_size_descending():
    bench = [f"b{i}" for i in range(10)]
    labels = np.array([2, 2, 2, 2, 2, 0, 0, 0, 1, 1])
    out = cp.canonicalize(labels, bench)
    sizes = [int((out == c).sum()) for c in range(3)]
    assert sizes == sorted(sizes, reverse=True)
    # Renaming the label VALUES must not change the canonical result.
    remap = {2: 7, 0: 4, 1: 9}
    permuted = np.array([remap[int(x)] for x in labels])
    assert np.array_equal(out, cp.canonicalize(permuted, bench))


def test_canonicalize_preserves_noise():
    bench = [f"b{i}" for i in range(6)]
    out = cp.canonicalize(np.array([-1, 0, 0, -1, 1, 1]), bench)
    assert [int(x) for x in out] == [-1, 0, 0, -1, 1, 1]
    assert int((out == -1).sum()) == 2


def test_canonical_labels_chain_stable():
    d, _ = blocks()
    bench = [f"b{i}" for i in range(len(d))]
    got = cp.hac_labels(d, bench, [3, 4], "average")
    prev, cur = got[3], got[4]
    # A nested split keeps every unchanged cluster's id and adds exactly one.
    unchanged = 0
    for c in np.unique(prev):
        members = set(np.flatnonzero(prev == c))
        if any(set(np.flatnonzero(cur == c2)) == members for c2 in np.unique(cur)):
            unchanged += 1
    assert unchanged == len(np.unique(prev)) - 1
    assert len(np.unique(cur)) == len(np.unique(prev)) + 1


def test_condensed_rejects_bad_matrix():
    d, _ = blocks(n_per=3, k=2)
    asym = d.copy()
    asym[0, 1] += 0.5
    with pytest.raises(ValueError, match="symmetric"):
        cp.condensed(asym)

    diag = d.copy()
    diag[0, 0] = 0.3
    with pytest.raises(ValueError, match="diagonal"):
        cp.condensed(diag)

    neg = d.copy()
    neg[0, 1] = neg[1, 0] = -0.1
    with pytest.raises(ValueError, match="negative or non-finite"):
        cp.condensed(neg)

    with pytest.raises(ValueError, match="square"):
        cp.condensed(np.zeros((3, 4)))


def test_silhouette_guards():
    d, _ = blocks(n_per=5, k=2)
    n = len(d)
    assert cp.silhouette(d, np.zeros(n, dtype=int)) is None  # one cluster
    assert cp.silhouette(d, np.arange(n)) is None  # all singletons
    assert cp.silhouette(d, np.full(n, -1)) is None  # all noise


def test_silhouette_excludes_noise():
    d, truth = blocks()
    clean = cp.silhouette(d, truth)
    noisy = truth.copy()
    noisy[:3] = -1
    assert clean is not None and cp.silhouette(d, noisy) is not None


def _write_benchmarks(tmp_path, rows):
    """Minimal benchmarks.csv with the three label columns."""
    cols = ["benchmark_id", "subject", "task_labels", "language"]
    lines = [",".join(cols)] + [",".join(r) for r in rows]
    (tmp_path / "benchmarks.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path


def test_split_labels():
    assert cp.split_labels("math; multilingual ;") == ["math", "multilingual"]
    assert cp.split_labels("") == []
    assert cp.split_labels(None) == []


def test_load_labels_derives_axis_from_column(tmp_path):
    root = _write_benchmarks(
        tmp_path,
        [
            ["mgsm", "math;reasoning", "long_reasoning", "multilingual"],
            ["squad", "reading_comprehension;encyclopedic", "extraction", ""],
        ],
    )
    labels, axis_labels = cp.load_labels(root)
    assert labels["mgsm"] == [
        "subject:math", "subject:reasoning", "task:long_reasoning", "language:multilingual"
    ]
    # a capability and a content domain coexist on the one merged subject axis
    assert labels["squad"] == [
        "subject:reading_comprehension", "subject:encyclopedic", "task:extraction"
    ]
    assert set(axis_labels) == {"subject", "task", "language"}


def test_load_labels_same_name_on_two_axes_stays_distinct(tmp_path):
    """`miscellaneous` as a catch-all on several axes must be separate sets."""
    root = _write_benchmarks(
        tmp_path,
        [["odd", "miscellaneous", "short_qa", "miscellaneous"]],
    )
    labels, axis_labels = cp.load_labels(root)
    assert "subject:miscellaneous" in labels["odd"]
    assert "language:miscellaneous" in labels["odd"]
    assert axis_labels["subject"] != axis_labels["language"]


def test_load_labels_rejects_reserved_separator(tmp_path):
    root = _write_benchmarks(tmp_path, [["x", "a:b", "short_qa", ""]])
    with pytest.raises(SystemExit, match="reserved separator"):
        cp.load_labels(root)


def test_load_labels_requires_label_columns(tmp_path):
    (tmp_path / "benchmarks.csv").write_text("benchmark_id,category\nx,math\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="missing label columns"):
        cp.load_labels(tmp_path)


def test_cluster_records_sizes_descending_no_noise_row():
    records = cp.cluster_records(np.array([0, 0, 0, 1, -1]))
    assert [r["id"] for r in records] == [0, 1]  # noise is never a record
    assert [r["size"] for r in records] == [3, 1]
    assert all(set(r) == {"id", "size"} for r in records)  # no inferred label


def test_cluster_records_all_noise_is_empty():
    assert cp.cluster_records(np.array([-1, -1, -1])) == []


def test_hdbscan_does_not_mutate_distance_matrix():
    """Regression: sklearn 1.9 HDBSCAN(metric='precomputed') mutates in place.

    Without copy=True everything computed after it from the same matrix -- HAC
    labels, silhouettes, ARI, UMAP coordinates -- is silently wrong.
    """
    d, _ = blocks()
    before = d.copy()
    cp.hdbscan_labels(d, 5, 5, "eom")
    assert np.array_equal(d, before)


def test_pair_coverage_counts_fabricated():
    a = ("m1", np.ones((2, 2)), ["a", "b"], ["g", "F1*"])
    b = ("m2", np.ones((2, 2)), ["c", "d"], ["g", "F1*"])
    cov, fabricated = cp.pair_coverage([a, b], ["a", "b", "c", "d"])
    # a/b and c/d are co-observed; the four cross pairs never are.
    assert fabricated == 4
    assert cov == round(1 - 8 / 12, 4)


def test_composite_distance_drop_g_changes_geometry():
    cells = [
        (
            "m1",
            np.array([[0.9, 0.1], [0.9, -0.1], [0.9, 0.8]]),
            ["a", "b", "c"],
            ["g", "F1*"],
        )
    ]
    with_g, bench = cp.composite_distance(cells)
    without_g, bench2 = cp.composite_distance(cells, drop_g=True)
    assert bench == bench2 == ["a", "b", "c"]
    # Dropping a column every row shares must spread the points apart.
    off = ~np.eye(3, dtype=bool)
    assert without_g[off].mean() > with_g[off].mean()


def test_composite_distance_rejects_all_g_cell_when_dropping():
    cells = [("m1", np.array([[0.9], [0.5]]), ["a", "b"], ["g"])]
    with pytest.raises(ValueError, match="no cell contributed"):
        cp.composite_distance(cells, drop_g=True)


def test_coverage_strata_bins_all_benchmarks():
    bench = [f"b{i}" for i in range(20)]
    cov = {b: 10**(i % 4) for i, b in enumerate(bench)}
    st = cp.coverage_strata(bench, cov)
    assert len(st) == 20
    assert st.min() >= 0 and st.max() < cp.COVERAGE_STRATA


def test_category_cohesion_detects_planted_tight_set():
    """A planted block must come out tighter than a matched random set."""
    d, truth = blocks(n_per=12, k=4)
    bench = [f"b{i}" for i in range(len(d))]
    labels = {b: ["tight" if truth[i] == 0 else "spread"] for i, b in enumerate(bench)}
    strata = np.zeros(len(d), dtype=int)  # single stratum => uniform null
    rng = np.random.default_rng(0)
    rows = cp.category_cohesion(d, bench, labels, strata, rng)
    tight = next(r for r in rows if r["category"] == "tight")
    assert tight["z"] < 0
    assert tight["p"] < 0.01
    assert tight["within"] < tight["null_mean"]


def test_category_cohesion_multi_label_counts_once_per_label():
    d, truth = blocks(n_per=6, k=2)
    bench = [f"b{i}" for i in range(len(d))]
    # every member of block 0 carries BOTH labels
    labels = {b: (["a", "b"] if truth[i] == 0 else ["c"]) for i, b in enumerate(bench)}
    rng = np.random.default_rng(0)
    rows = cp.category_cohesion(d, bench, labels, np.zeros(len(d), dtype=int), rng)
    got = {r["category"]: r["n"] for r in rows}
    assert got["a"] == 6 and got["b"] == 6  # same set, scored twice
    assert got["a"] == got["b"]


def test_category_cohesion_skips_tiny_categories():
    d, _ = blocks(n_per=6, k=2)
    bench = [f"b{i}" for i in range(len(d))]
    labels = {bench[0]: ["solo"], bench[1]: ["solo"]}  # n=2 < COHESION_MIN_N
    rows = cp.category_cohesion(
        d, bench, labels, np.zeros(len(d), dtype=int), np.random.default_rng(0)
    )
    assert rows == []

