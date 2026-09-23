#!/usr/bin/env python3
"""Unit tests for scripts/label_silhouette.py.

Synthetic matrices and label dicts live here and only here.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import silhouette_score

_SPEC = importlib.util.spec_from_file_location(
    "label_silhouette",
    Path(__file__).resolve().parents[1] / "scripts" / "label_silhouette.py",
)
ls = importlib.util.module_from_spec(_SPEC)
sys.modules["label_silhouette"] = ls
_SPEC.loader.exec_module(ls)

N_PERM = 300


def planted(n_per=10, k=3, within=0.2, between=0.9, sigma=0.02, seed=0):
    rng = np.random.default_rng(seed)
    truth = np.repeat(np.arange(k), n_per)
    n = k * n_per
    d = np.where(truth[:, None] == truth[None, :], within, between)
    d = d + rng.normal(0, sigma, (n, n))
    d = np.clip((d + d.T) / 2.0, 0.0, 2.0)
    np.fill_diagonal(d, 0.0)
    return d, truth


def geometry(d):
    obs = d.copy()
    np.fill_diagonal(obs, np.nan)
    return ls.Geometry.from_observed(obs)


def test_partition_silhouette_matches_sklearn():
    d, truth = planted()
    g = geometry(d)
    ours = ls.partition_silhouette(g.dz, g.cnt, truth)
    assert abs(ours - silhouette_score(d, truth, metric="precomputed")) < 1e-9


def test_member_silhouette_hand_computed():
    # members {0,1}: a = 1 for both; non-member 2 is 3 away from each -> b = 3
    d = np.array([[0.0, 1.0, 3.0], [1.0, 0.0, 3.0], [3.0, 3.0, 0.0]])
    assert abs(ls.member_silhouette(geometry(d), np.array([0, 1])) - (3 - 1) / 3) < 1e-12


def test_true_label_scores_high_and_random_label_does_not():
    d, truth = planted()
    g = geometry(d)
    strata = np.zeros(len(d), dtype=int)
    rng = np.random.default_rng(1)
    sil, null_mean, _, p = ls.label_test(g, np.flatnonzero(truth == 0), strata, rng, N_PERM)
    assert sil > 0.7 and sil > null_mean and p < 0.05
    rand = np.random.default_rng(5).choice(len(d), 10, replace=False)
    _, _, _, p_rand = ls.label_test(g, rand, strata, rng, N_PERM)
    assert p_rand > 0.05


def test_unobserved_pairs_are_ignored_not_zeroed():
    d, truth = planted()
    full = ls.member_silhouette(geometry(d), np.flatnonzero(truth == 0))
    obs = d.copy()
    np.fill_diagonal(obs, np.nan)
    obs[0, 1] = obs[1, 0] = np.nan
    holed = ls.member_silhouette(ls.Geometry.from_observed(obs), np.flatnonzero(truth == 0))
    assert np.isfinite(holed) and abs(holed - full) < 0.05


def test_small_and_universal_labels_are_skipped():
    d, _ = planted(n_per=4, k=2)
    bench = [f"b{i}" for i in range(len(d))]
    labels = {
        b: ["l:all", "l:tiny"] if i < 2 else ["l:all"] for i, b in enumerate(bench)
    }
    labels["b0"].append("l:half")
    labels["b1"].append("l:half")
    labels["b2"].append("l:half")
    labels["b3"].append("l:half")
    out = ls.label_results(
        "k", "with_g", geometry(d), bench, labels,
        np.zeros(len(d), dtype=int), np.random.default_rng(0), 50,
    )
    assert [r.name for r in out] == ["l:half"]  # tiny: n=2 < min; all: n == total


def test_single_label_codes_drop_multi_label_and_singletons():
    bench = ["a", "b", "c", "d", "e", "f"]
    labels = {
        "a": ["language:en"],
        "b": ["language:en"],
        "c": ["language:fr"],
        "d": ["language:fr"],
        "e": ["language:en", "language:fr"],  # multi-label: excluded
        "f": ["language:de"],  # singleton class: dropped
    }
    idx, codes = ls.single_label_codes(bench, labels, "language")
    assert list(idx) == [0, 1, 2, 3]
    assert list(codes) == [0, 0, 1, 1]


def test_axis_result_detects_planted_and_rejects_shuffled():
    d, truth = planted()
    g = geometry(d)
    strata = np.zeros(len(d), dtype=int)
    idx = np.arange(len(d))
    rng = np.random.default_rng(2)
    good = ls.axis_result("k", "with_g", "language", g, idx, truth, strata, rng, N_PERM)
    assert good.sil > 0.7 and good.p < 0.05 and good.k == 3
    shuffled = np.random.default_rng(9).permutation(truth)
    bad = ls.axis_result("k", "with_g", "language", g, idx, shuffled, strata, rng, N_PERM)
    assert bad.p > 0.05


def test_axis_result_needs_two_classes():
    d, _ = planted()
    res = ls.axis_result(
        "k", "with_g", "language", geometry(d), np.arange(len(d)),
        np.zeros(len(d), dtype=int), np.zeros(len(d), dtype=int),
        np.random.default_rng(0), 10,
    )
    assert res is None
