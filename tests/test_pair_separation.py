#!/usr/bin/env python3
"""Unit tests for scripts/pair_separation.py.

Synthetic matrices and label dicts live here and only here.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "pair_separation",
    Path(__file__).resolve().parents[1] / "scripts" / "pair_separation.py",
)
ps = importlib.util.module_from_spec(_SPEC)
sys.modules["pair_separation"] = ps
_SPEC.loader.exec_module(ps)

N_PERM = 500


def planted(n_per=12, far=0.9, near=0.2, sigma=0.02, seed=0):
    """Two subject blocks; every benchmark shares format. Cross-subject is far."""
    rng = np.random.default_rng(seed)
    subj = np.repeat([0, 1], n_per)
    n = 2 * n_per
    d = np.where(subj[:, None] == subj[None, :], near, far) + rng.normal(0, sigma, (n, n))
    d = np.clip((d + d.T) / 2.0, 0.0, 2.0)
    np.fill_diagonal(d, 0.0)
    bench = [f"b{i}" for i in range(n)]
    labels = {
        b: ["task:mc", "language:en", f"subject:s{s}"] for b, s in zip(bench, subj)
    }
    return d, bench, labels


def setup(d, bench, labels):
    fmt = ps.format_pairs(bench, labels)
    strata = np.zeros(len(bench), dtype=int)
    return fmt, strata, d.copy()


def disjoint_pairs(fmt):
    i, j = fmt.pairs[:, 0], fmt.pairs[:, 1]
    return fmt.pairs[~fmt.subject_overlap[i, j]]


def test_format_pairs_rule():
    bench = ["a", "b", "c", "d", "e"]
    labels = {
        "a": ["task:mc", "language:en", "subject:math"],
        "b": ["task:mc", "language:en", "subject:law"],  # disjoint subject -> pair
        "c": ["task:mc", "language:fr", "subject:law"],  # other language
        "d": ["task:chat", "language:en", "subject:law"],  # other task
        "e": ["task:mc", "language:en"],  # no subject: excluded
    }
    fmt = ps.format_pairs(bench, labels)
    assert [tuple(p) for p in fmt.pairs] == [(0, 1)]
    assert not fmt.subject_overlap[0, 1]


def test_format_pairs_parent_group_counts_as_shared():
    labels = {
        "a": ["task:mc", "language:en", "subject:medical", "subject:specialized_domain"],
        "b": ["task:mc", "language:en", "subject:law", "subject:specialized_domain"],
    }
    fmt = ps.format_pairs(["a", "b"], labels)
    assert fmt.subject_overlap[0, 1]
    assert len(disjoint_pairs(fmt)) == 0


def test_far_pairs_are_detected():
    d, bench, labels = planted()
    fmt, strata, d_obs = setup(d, bench, labels)
    rng = np.random.default_rng(1)
    sep = ps.separation_test(d_obs, disjoint_pairs(fmt), strata, rng, N_PERM)
    con = ps.format_contrast(d_obs, fmt, strata, rng, N_PERM)
    assert sep.mean > sep.null_mean
    assert sep.p < 0.05
    assert con.auc > 0.99
    assert con.p < 0.05


def test_random_matrix_shows_no_separation():
    rng = np.random.default_rng(3)
    n = 24
    d = rng.uniform(0.2, 0.9, (n, n))
    d = (d + d.T) / 2.0
    np.fill_diagonal(d, 0.0)
    bench = [f"b{i}" for i in range(n)]
    labels = {
        b: ["task:mc", "language:en", f"subject:s{i % 2}"] for i, b in enumerate(bench)
    }
    fmt, strata, d_obs = setup(d, bench, labels)
    con = ps.format_contrast(d_obs, fmt, strata, rng, N_PERM)
    assert 0.35 < con.auc < 0.65
    assert con.p > 0.05


def test_close_pairs_do_not_pass_as_far():
    # Subjects differ but the geometry is flat: disjoint pairs are not farther.
    d, bench, labels = planted(far=0.2, near=0.2)
    fmt, strata, d_obs = setup(d, bench, labels)
    sep = ps.separation_test(
        d_obs, disjoint_pairs(fmt), strata, np.random.default_rng(2), N_PERM
    )
    assert sep.p > 0.05


def test_observed_mask_excludes_unobserved_and_g_only_cells():
    mat = np.ones((2, 1))
    cells = [
        ("k1", np.ones((2, 2)), ["a", "b"], ["g", "F1*"]),
        ("k2", mat, ["b", "c"], ["g"]),
    ]
    bench = ["a", "b", "c"]
    seen = ps.observed_mask(cells, bench, drop_g=False)
    assert seen[1, 2] and seen[0, 1] and not seen[0, 2]
    seen_ng = ps.observed_mask(cells, bench, drop_g=True)
    assert seen_ng[0, 1] and not seen_ng[1, 2]
    assert not seen.diagonal().any()


def test_unobserved_pairs_do_not_leak_into_the_test():
    d, bench, labels = planted()
    fmt, strata, d_obs = setup(d, bench, labels)
    d_obs[0, 15] = d_obs[15, 0] = np.nan
    keep = ~np.isnan(d_obs[fmt.pairs[:, 0], fmt.pairs[:, 1]])
    kept = ps.FormatPairs(fmt.pairs[keep], fmt.subject_overlap)
    assert len(kept.pairs) == len(fmt.pairs) - 1
    ps.separation_test(d_obs, disjoint_pairs(kept), strata, np.random.default_rng(0), 50)


def test_iter_jobs_filters_tag_and_year():
    cell = lambda name: (name, np.ones((2, 1)), ["a", "b"], ["g"])
    cells = {
        ("C", "pa", None): [cell("softimpute_C_all_standard")],
        ("C", "2f", None): [cell("softimpute_C_all_standard")],
        ("C", "pa", 2020): [cell("softimpute_C_all_standard")],
    }
    assert [k for k, _ in ps.iter_jobs(cells)] == ["C|pa", "C|pa|msoftimpute"]
    assert [k for k, _ in ps.iter_jobs(cells, ("2f",))] == ["C|2f", "C|2f|msoftimpute"]


def test_percentile():
    pool = np.array([0.1, 0.2, 0.3, 0.4])
    assert ps.percentile(pool, 0.1) == 0.0
    assert ps.percentile(pool, 0.4) == 75.0
    assert ps.percentile(pool, 0.5) == 100.0


def test_load_named_pairs_validation(tmp_path):
    known = {"a", "b", "c"}

    def write(text):
        p = tmp_path / "pairs.csv"
        p.write_text(text)
        return p

    ok = ps.load_named_pairs(write("a,b,role\na,b,control\nb,c,example\n"), known)
    assert [(p.a, p.b, p.role) for p in ok] == [("a", "b", "control"), ("b", "c", "example")]
    with pytest.raises(SystemExit, match="unknown benchmark"):
        ps.load_named_pairs(write("a,b,role\na,zzz,control\n"), known)
    with pytest.raises(SystemExit, match="role"):
        ps.load_named_pairs(write("a,b,role\na,b,friend\n"), known)
    with pytest.raises(SystemExit, match="duplicate"):
        ps.load_named_pairs(write("a,b,role\na,b,control\nb,a,control\n"), known)
    with pytest.raises(SystemExit, match="itself"):
        ps.load_named_pairs(write("a,b,role\na,a,control\n"), known)
    with pytest.raises(SystemExit, match="expected columns"):
        ps.load_named_pairs(write("x,y\na,b\n"), known)
