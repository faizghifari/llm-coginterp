#!/usr/bin/env python3
"""Unit tests for scripts/release_date.py.

Synthetic loadings and masks live here and only here; the full run is exercised
by `make release-date` against real data.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "release_date",
    Path(__file__).resolve().parents[1] / "scripts" / "release_date.py",
)
rd = importlib.util.module_from_spec(_SPEC)
sys.modules["release_date"] = rd
_SPEC.loader.exec_module(rd)


@pytest.fixture(autouse=True)
def fewer_permutations(monkeypatch):
    monkeypatch.setattr(rd, "ERA_PERM", 300)
    monkeypatch.setattr(rd, "CONFOUND_PERM", 200)


def test_cohort_bins_match_the_r_worker():
    assert [rd.cohort_of(y) for y in (2019, 2022, 2023, 2024, 2025, 2026)] == \
        [2022, 2022, 2023, 2024, 2025, 2025]
    assert rd.cohort_of(np.nan) is None
    r_src = rd.COHORT_R.read_text()
    assert "ifelse(y <= 2022, 2022L, ifelse(y >= 2025, 2025L" in r_src


def test_parse_year_takes_first_plausible_year():
    got = rd.parse_year(pd.Series(["2023-05", "2021", "code-002-175B", None, "v2 2024-01"]))
    assert got.tolist()[:2] == [2023, 2021]
    assert got.isna().tolist()[2:4] == [True, True]
    assert got.iloc[4] == 2024


def test_gate_is_read_from_factor_r():
    assert 0 < rd.r2_gate() < 1


def test_seed_is_stable_and_cell_specific():
    assert rd.seed_of("era", "C", "knn") == rd.seed_of("era", "C", "knn")
    assert rd.seed_of("era", "C", "knn") != rd.seed_of("era", "S", "knn")


def test_era_cohesion_detects_planted_eras_and_not_noise():
    rng = np.random.default_rng(0)
    eras = np.repeat(["a", "b", "c"], 8)
    centres = {"a": [1, 0], "b": [0, 1], "c": [-1, 0]}
    planted = np.array([centres[e] for e in eras]) + rng.normal(0, .05, (24, 2))
    z, p = rd.era_cohesion(planted, eras, seed=1)
    assert z < -5 and p < .01
    z0, p0 = rd.era_cohesion(rng.normal(size=(24, 2)), eras, seed=1)
    assert abs(z0) < 3 and p0 > .01


def test_year_gap_vs_coobservation_attributes_to_the_real_driver():
    rng = np.random.default_rng(0)
    n, models = 20, 200
    groups = np.repeat([0, 1], n // 2)
    # each group spans the same years, so the year gap carries no group signal;
    # observation sets and loadings both follow the group
    years = np.tile(np.arange(2015, 2025), 2).astype(float)
    O = np.zeros((models, n))
    for j in range(n):
        rows = np.arange(models // 2) + (models // 2) * groups[j]
        O[rng.choice(rows, 60, replace=False), j] = 1
    L = np.column_stack([groups, 1 - groups]) + rng.normal(0, .05, (n, 2))
    r = rd.gap_vs_coobs(L, years, O, seed=2)
    assert r["beta_coobs"] > 0.5
    assert abs(r["beta_year_gap"]) < 0.2
    assert r["p_year_gap"] > .05


def test_cohort_vs_random_flags_outside_band():
    rows = [dict(dz="C", method="knn", cohort="2023", rep=0, n=50, p=10, nf=2,
                 omega_h=0.9, tucker_g=0.8)]
    rows += [dict(dz="C", method="knn", cohort="2023", rep=b, n=50, p=10, nf=2,
                  omega_h=0.4 + b / 1000, tucker_g=0.95) for b in range(1, 51)]
    t = rd.cohort_vs_random(pd.DataFrame(rows))
    assert t.flag.iloc[0] == "above"
    assert t.tucker_random.iloc[0] == pytest.approx(0.95)
