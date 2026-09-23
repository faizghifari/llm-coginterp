#!/usr/bin/env python3
"""Unit tests for scripts/label_cohesion.py.

Synthetic matrices and label dicts live here and only here.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "label_cohesion",
    Path(__file__).resolve().parents[1] / "scripts" / "label_cohesion.py",
)
lc = importlib.util.module_from_spec(_SPEC)
sys.modules["label_cohesion"] = lc
_SPEC.loader.exec_module(lc)

cp = lc.cp


def planted(n_per=8, k=2, within=0.15, between=0.9, sigma=0.02, seed=0):
    rng = np.random.default_rng(seed)
    truth = np.repeat(np.arange(k), n_per)
    n = k * n_per
    d = np.where(truth[:, None] == truth[None, :], within, between)
    d = d + rng.normal(0, sigma, (n, n))
    d = np.clip((d + d.T) / 2.0, 0.0, 2.0)
    np.fill_diagonal(d, 0.0)
    return d, truth


def test_effect_is_chance_corrected():
    assert lc.effect(0.2, 0.8) == pytest.approx(0.75)
    assert lc.effect(0.8, 0.8) == pytest.approx(0.0)
    assert lc.effect(1.2, 0.8) < 0
    assert lc.effect(0.5, 0.0) is None


def test_analyse_matches_category_cohesion_directly():
    """analyse() must be a thin pass-through: same rows as calling
    composite_distance() + category_cohesion() by hand, given the same rng
    draw order. composite_distance() is monkeypatched to return the planted
    matrix directly -- building a loading matrix whose cosine distances equal
    a chosen `d` exactly is not generally possible, and analyse()'s own
    correctness (not composite_distance()'s) is what's under test."""
    dist, truth = planted()
    bench = [f"b{i}" for i in range(len(dist))]
    labels = {b: [f"subject:g{t}"] for b, t in zip(bench, truth)}
    coverage = {b: 10 for b in bench}
    cell_list = [("k_C_all_standard", dist[:, :0], bench, [])]

    strata = cp.coverage_strata(bench, coverage)
    direct = cp.category_cohesion(dist, bench, labels, strata, np.random.default_rng(1))

    orig = cp.composite_distance
    cp.composite_distance = lambda cell_list, drop_g=False: (dist, bench)
    try:
        rows = lc.analyse("k", cell_list, labels, coverage, np.random.default_rng(1))
    finally:
        cp.composite_distance = orig

    with_g = [r for r in rows if r.variant == "with_g"]
    assert [r.category for r in with_g] == [row["category"] for row in direct]
    assert [r.within for r in with_g] == [row["within"] for row in direct]
    assert [r.p for r in with_g] == [row["p"] for row in direct]
    assert all(r.n_bench == len(bench) for r in rows)
    assert {r.variant for r in rows} == {"with_g", "without_g"}
    for r in with_g:
        assert r.A == pytest.approx(lc.effect(r.within, r.null_mean))


def test_analyse_raises_when_drop_g_reorders_benchmarks():
    d, truth = planted()
    bench = [f"b{i}" for i in range(len(d))]
    labels = {b: [f"subject:g{t}"] for b, t in zip(bench, truth)}
    coverage = {b: 10 for b in bench}
    cell_list = [("k", d[:, :0], bench, [])]

    orig = cp.composite_distance

    def fake(cell_list, drop_g=False):
        return (d, list(reversed(bench))) if drop_g else (d, bench)

    cp.composite_distance = fake
    try:
        with pytest.raises(RuntimeError, match="reordered|ordering"):
            lc.analyse("k", cell_list, labels, coverage, np.random.default_rng(0))
    finally:
        cp.composite_distance = orig


def test_axes_summary_counts_aggregate_with_g_rows(capsys):
    rows = [
        lc.Row("C|pa", "with_g", 10, "subject:code", 5, 0.2, 0.8, -3.0, 0.001, 0.75),
        lc.Row("C|pa", "without_g", 10, "subject:code", 5, 0.2, 0.8, -3.0, 0.001, 0.75),
        lc.Row("C|pa|mzeros", "with_g", 10, "subject:code", 5, 0.2, 0.8, -3.0, 0.001, 0.75),
        lc.Row("C|pa", "with_g", 10, "task:mc", 5, 0.7, 0.7, 0.0, 0.9, 0.0),
    ]
    lc.axes_summary(rows)
    out = capsys.readouterr().out
    assert "axes: subject, task" in out
    assert "subject: 1 label-cell rows (aggregate, with_g), 1 at p<0.05" in out
    assert "task: 1 label-cell rows (aggregate, with_g), 0 at p<0.05" in out
