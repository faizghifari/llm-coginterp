#!/usr/bin/env python3
"""Latent factor scores per model, from the bifactor loadings + data matrices.

# ─────────────────────────────────────────────────────────────────────────────
# STATISTICAL METHODOLOGY
# ─────────────────────────────────────────────────────────────────────────────
#
# Goal. Each cell of the pipeline cross-product (imputation method × densifier
# C/S/R × strategy) yields a Schmid–Leiman bifactor loading matrix Λ
# (benchmarks × {g, F1*, ..., Fk*}, orthogonal by construction) plus a
# correlation matrix R and a model×benchmark data matrix. This script produces
# the per-model LATENT VARIABLE SCORES: each model's estimated position on the
# general factor g and on every group factor, in ~standard-deviation units.
# The general recipe is regression (Thomson) scoring:
#
#   W = R⁻¹ Λ                     (scoring weights, variables × factors)
#   F = Z W                       (scores, models × factors)
#
# where Z is the model×benchmark matrix with each benchmark column z-scored
# (columns only — rows are models and row-scaling would erase the general
# factor; see DATAFLOW gotchas). Λ contains *structure* coefficients (how
# strongly each benchmark reflects each factor, ignoring collinearity);
# multiplying by R⁻¹ converts them into *beta* weights that correct for
# benchmark intercorrelation — the standard fa()/regression method, which
# maximizes the correlation between estimated and true factor scores
# (Thomson, 1939). Because the Schmid–Leiman factors are mutually orthogonal
# (Φ = I), no Φ correction term is needed: the g score is the
# "weighted-average ability" analogue and group scores are residualized.
#
# Sign and scale. Factor scores inherit the sign of the estimated loadings
# (psych's sign indeterminacy is therefore consistent within a cell, since
# loadings and scores multiply the same Z). Scores come out in z units:
# E[F] ≈ 0 per factor; group factors are residualized components, so their
# scores are also approximately standardized.
#
# all_sum. The rowwise sum of g + all group-factor scores. The Schmid–Leiman
# group factors are orthogonal, so the sum is a well-defined aggregate index
# in SD units ("total common-variance position"); it is monotone in average
# standardized benchmark performance and equivalent (same ranking) to the
# mean. Diagnostic columns of om$schmid$sl (h2, u2/uq2, p2, com) are never
# scored.
#
# ── Complete-data cells (imputed methods incl. onesidedmc surrogates) ────────
#
# Data: data/imputed/<method>/<dz>/<st>/imputed_model_benchmark_table.csv —
# the exact completed matrix the loadings were factored from. R is the
# empirical correlation of the scored columns. The imputed/surrogate matrices
# are often low-rank, so R can be singular and a plain solve() would explode;
# we use the Moore–Penrose pseudoinverse (numpy.linalg.pinv): when R is
# well-conditioned it is numerically identical to R⁻¹; when R is
# rank-deficient it is the least-squares ridge analogue. Output:
#   results/<method>/<method>_<dz>_<st>_bifactor_<tag>_scores.csv
#
# ── Raw-method cells (default/zeros: factored directly on sparse tables) ────
#
# These cells were factored on the pairwise-complete correlation of the
# sparse model×benchmark table (see prepare_raw_default/zeros in
# src/factor/factoring.R), so Z has holes. The smoothed correlation matrix R
# that factoring ran on is persisted by the orchestrators as
#   results/<method>/<method>_<dz>_<st>_correlation.csv
# and reused here verbatim (no reimplementation of the fill/PSD recipe).
#
# Why this works despite "a correlation matrix alone cannot recover cell-level
# values": conditional scoring never needs cell-level data it doesn't have.
# It needs (a) R — exactly the covariance-level object that IS recoverable —
# and (b) the observed cells themselves, taken straight from the sparse table.
# The surrogate-fabrication step that OSMC-style scoring would need simply
# drops out. (Caveat retained: OSMC surrogates fabricate plausible cells
# consistent with the global covariance; they are valid as factor *inputs*,
# not as "data".)
#
# Two estimators are computed for raw cells, as separate files, so the choice
# of missing-data handling can be triangulated (if they agree on
# well-observed models, the scores are not an artifact of the estimator):
#
# 1. CONDITIONAL regression scoring (…_scores_conditional.csv). Per model i
#    with observed benchmark subset Oᵢ, the posterior-mean factor score under
#    the fitted Gaussian model:
#
#        F̂ᵢ = Λ_Oᵢᵀ · R_OᵢOᵢ⁻¹ · zᵢ,Oᵢ
#
#    where Λ_Oᵢ and R_OᵢOᵢ are the loading matrix and correlation restricted
#    to the benchmarks model i actually has. This is the exact multivariate-
#    normal conditional expectation E[factor | observed cells] — i.e. it is
#    mathematically the same as "impute the missing benchmarks by regressing
#    them on the observed ones, then score" (the Gaussian conditioning does
#    the imputation implicitly and in closed form). It is the same machinery
#    as predict_cell_from_corr in src/impute/corr_common.R, predicting the
#    factor instead of the cell. R_OᵢOᵢ can be near-singular for small Oᵢ →
#    pinv fallback. Scored for every model with ≥1 observation (n_obs column
#    reports coverage; scores with tiny n_obs are noisy).
#
# 2. PRORATED weighted-sum scoring (…_scores_prorated.csv). Global weights
#    W = pinv(R) Λ are computed once, then each model's score is the weighted
#    mean of its observed cells with missing benchmarks treated as z = 0
#    (column mean), renormalized by the observed weight mass:
#
#        F̂ᵢj = Σ_{v ∈ Oᵢ} w_vj · zᵢv / Σ_{v ∈ Oᵢ} |w_vj|
#
#    Both estimators USE the loadings — they differ only in how they cope
#    with the holes in Z: per-model restricted regression (conditional) vs
#    one global weighting scheme with mean-fill (prorated). The prorated
#    variant ignores covariance among observed benchmarks and therefore
#    degrades gracefully-to-crudely as coverage drops; it is the cheap
#    sanity check against the conditional estimator.
#
# Raw output naming (bare _scores.csv is reserved for complete-data cells):
#   results/<method>/<method>_<dz>_<st>_bifactor_<tag>_scores_conditional.csv
#   results/<method>/<method>_<dz>_<st>_bifactor_<tag>_scores_prorated.csv
# columns: collapse_key, n_obs (observed benchmark count), all_sum, g, F1*...
#
# Column alignment. prep_matrix dropped degenerate benchmark columns before
# imputation/factoring, so loading rows are a subset of the data columns; we
# intersect to the common set and score on that (dropped columns carry no
# loading and cannot contribute). For raw cells the data table and R share
# the same benchmark set, likewise intersected with the loading rows.
#
# Factor-count note (context only — no PA here): raw cells pick nf from Horn
# parallel analysis on the raw (unsmoothed) pairwise-complete eigenvalues
# vs cutoffs at n_eff = number of models, floored at 2 and capped by rank
# and dimensions (factor_raw in factoring.R). Both pa and 2f solutions are
# scored per cell; each loadings file is scored independently.
# ─────────────────────────────────────────────────────────────────────────────

Usage:
    python3 scripts/latent_scores.py
        [--data-root data/text_only] [--results-root results/text_only]
        [--method <name>] [--tag <pa|2f>]
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]

DENSIFIERS = ("raw", "C", "S", "R")
TAGS = ("pa", "2f")
RAW_METHODS = {"default", "zeros"}
DIAGNOSTIC_COLS = {"h2", "u2", "uq2", "p2", "com"}
FNAME_RE = re.compile(
    r"^(?P<method>.+?)_(?P<densifier>" + "|".join(DENSIFIERS) +
    r")_(?P<strategy>.+?)_bifactor_(?P<tag>" + "|".join(TAGS) + r")_loadings\.csv$"
)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default="data/text_only",
                    help="tree holding imputed/ and combinations[_dz]/ matrices")
    ap.add_argument("--results-root", default="results/text_only",
                    help="tree holding *_loadings.csv files (relative to repo)")
    ap.add_argument("--method", default=None, help="only process this method")
    ap.add_argument("--tag", choices=TAGS, default=None, help="only process this run tag")
    return ap.parse_args()


def completed_matrix_path(data_root: Path, method: str, dz: str, st: str) -> Path:
    return data_root / "imputed" / method / dz / st / "imputed_model_benchmark_table.csv"


def sparse_table_path(data_root: Path, dz: str, st: str) -> Path:
    return (data_root / ("combinations" if dz == "raw" else f"combinations_{dz}")
            / st / "model_benchmark_table.csv")


def correlation_csv_path(results_root: Path, method: str, dz: str, st: str) -> Path:
    return results_root / method / f"{method}_{dz}_{st}_correlation.csv"


def load_loadings(path: Path):
    """Return (benchmark names, factor names, Λ). None on failure."""
    df = pl.read_csv(path, infer_schema_length=None)
    if "benchmark" not in df.columns:
        return None
    factor_cols = [c for c in df.columns
                   if c != "benchmark" and c not in DIAGNOSTIC_COLS]
    if not factor_cols:
        return None
    names = df["benchmark"].to_list()
    L = df.select(factor_cols).to_numpy().astype(float)
    return names, factor_cols, L


def load_matrix(path: Path):
    """Return (row keys, column names, M). None on failure."""
    df = pl.read_csv(path, infer_schema_length=None)
    keys = df[df.columns[0]].to_list()
    cols = df.columns[1:]
    M = df.select(cols).to_numpy().astype(float)
    return keys, cols, M


def zscore_cols(Z: np.ndarray):
    """Column-standardize on finite entries (missing-aware). Returns (Z, ok mask)."""
    mu = np.nanmean(Z, axis=0)
    sd = np.nanstd(Z, axis=0, ddof=1)
    ok = np.isfinite(sd) & (sd > 0)
    Z = (Z[:, ok] - mu[ok]) / sd[ok]
    return Z, ok


def scores_output_path(loadings_path: Path, suffix: str) -> Path:
    return loadings_path.with_name(
        loadings_path.name.replace("_loadings.csv", f"_scores{suffix}.csv"))


def write_scores(path: Path, keys: list, F: np.ndarray, factor_cols: list,
                 n_obs: list | None = None):
    cols = {"collapse_key": keys}
    if n_obs is not None:
        cols["n_obs"] = n_obs
    cols["all_sum"] = np.nansum(F, axis=1) if n_obs is None else F.sum(axis=1)
    cols.update({c: F[:, j] for j, c in enumerate(factor_cols)})
    pl.DataFrame(cols).write_csv(path)


# ── Complete-data cells ──────────────────────────────────────────────────────

def score_cell_complete(loadings_path: Path, mat_path: Path) -> bool:
    loaded = load_loadings(loadings_path)
    if loaded is None:
        print(f"  skip {loadings_path.name} — no factor columns")
        return False
    bench_names, factor_cols, L = loaded

    mat = load_matrix(mat_path)
    if mat is None:
        print(f"  skip {loadings_path.name} — unreadable matrix {mat_path.name}")
        return False
    keys, cols, M = mat

    # Align benchmarks: loadings rows ⊆ matrix columns; reorder both to match.
    col_idx = {c: i for i, c in enumerate(cols)}
    row_idx = {b: i for i, b in enumerate(bench_names)}
    common = [b for b in bench_names if b in col_idx]
    if len(common) < 2:
        print(f"  skip {loadings_path.name} — fewer than 2 shared benchmarks")
        return False
    Z = M[:, [col_idx[b] for b in common]]
    L = L[[row_idx[b] for b in common], :]

    Z, ok = zscore_cols(Z)
    L = L[ok, :]
    if Z.shape[1] < 2:
        print(f"  skip {loadings_path.name} — fewer than 2 nonzero-variance benchmarks")
        return False

    R = np.corrcoef(Z, rowvar=False)
    R[~np.isfinite(R)] = 0.0
    np.fill_diagonal(R, 1.0)
    F = Z @ (np.linalg.pinv(R) @ L)

    out = scores_output_path(loadings_path, "")
    write_scores(out, keys, F, factor_cols)
    print(f"  {loadings_path.stem}: {F.shape[0]} models × {F.shape[1]} factors "
          f"({Z.shape[1]} benchmarks) -> {out.name}")
    return True


# ── Raw-method cells ─────────────────────────────────────────────────────────

def score_cell_raw(loadings_path: Path, sparse_path: Path, corr_path: Path) -> bool:
    loaded = load_loadings(loadings_path)
    if loaded is None:
        print(f"  skip {loadings_path.name} — no factor columns")
        return False
    bench_names, factor_cols, L = loaded

    corr = load_matrix(corr_path)
    if corr is None:
        print(f"  skip {loadings_path.name} — unreadable correlation {corr_path.name}")
        return False
    r_bench, r_cols, R_full = corr

    mat = load_matrix(sparse_path)
    if mat is None:
        print(f"  skip {loadings_path.name} — unreadable sparse table {sparse_path.name}")
        return False
    keys, cols, M = mat

    # Intersect loadings rows with the R/correlation and sparse-table columns.
    common = [b for b in bench_names if b in r_cols and b in set(cols)]
    if len(common) < 2:
        print(f"  skip {loadings_path.name} — fewer than 2 shared benchmarks")
        return False
    r_idx = {c: i for i, c in enumerate(r_cols)}
    m_idx = {c: i for i, c in enumerate(cols)}
    l_idx = {b: i for i, b in enumerate(bench_names)}
    R = R_full[np.ix_([r_idx[b] for b in common], [r_idx[b] for b in common])]
    Zall = M[:, [m_idx[b] for b in common]]
    L = L[[l_idx[b] for b in common], :]

    Z, ok = zscore_cols(Zall)
    R, L = R[np.ix_(ok, ok)], L[ok, :]
    if Z.shape[1] < 2:
        print(f"  skip {loadings_path.name} — fewer than 2 nonzero-variance benchmarks")
        return False

    n, p = Z.shape
    obs = np.isfinite(Z)
    n_obs = obs.sum(axis=1)

    # 1) Conditional regression: F̂ᵢ = Λ_Oᵢᵀ R_OᵢOᵢ⁻¹ zᵢ,Oᵢ per model.
    F_cond = np.full((n, L.shape[1]), np.nan)
    W_global = np.linalg.pinv(R) @ L  # p×k, reused by both estimators
    for i in range(n):
        oi = np.where(obs[i])[0]
        if len(oi) == 0:
            continue
        if len(oi) == p:
            F_cond[i] = Z[i] @ W_global
            continue
        Ri = R[np.ix_(oi, oi)]
        Li = L[oi, :]
        F_cond[i] = Li.T @ (np.linalg.pinv(Ri) @ Z[i, oi])

    # 2) Prorated weighted sum: global weights, missing cells = 0, renormalize
    #    by observed weight mass (weighted mean over observed benchmarks).
    Wp = np.abs(W_global)
    denom = obs @ Wp
    F_pror = np.where(obs, np.nan_to_num(Z), 0.0) @ W_global
    F_pror = np.where(denom > 0, F_pror / np.where(denom > 0, denom, 1.0), np.nan)

    out_c = scores_output_path(loadings_path, "_conditional")
    write_scores(out_c, keys, F_cond, factor_cols, n_obs=n_obs.tolist())
    print(f"  {loadings_path.stem}: {n} models × {F_cond.shape[1]} factors "
          f"({p} benchmarks) -> {out_c.name}")
    out_p = scores_output_path(loadings_path, "_prorated")
    write_scores(out_p, keys, F_pror, factor_cols, n_obs=n_obs.tolist())
    print(f"  {loadings_path.stem}: -> {out_p.name}")
    return True


def main():
    args = parse_args()
    results_root = REPO / args.results_root
    data_root = REPO / args.data_root
    if not results_root.exists():
        print(f"No results tree at {results_root} — run factoring first.", file=sys.stderr)
        sys.exit(1)

    paths = sorted(results_root.rglob("*_loadings.csv"))
    print(f"latent_scores: {len(paths)} loadings file(s) under {results_root}, "
          f"data from {data_root}")
    done = 0
    for p in paths:
        m = FNAME_RE.match(p.name)
        if not m:
            continue
        method, dz, st, tag = m.group("method", "densifier", "strategy", "tag")
        if args.method and method != args.method:
            continue
        if args.tag and tag != args.tag:
            continue
        try:
            if method in RAW_METHODS:
                ok = score_cell_raw(p, sparse_table_path(data_root, dz, st),
                                    correlation_csv_path(results_root, method, dz, st))
            else:
                mat_path = completed_matrix_path(data_root, method, dz, st)
                if not mat_path.exists():
                    print(f"  skip {method}/{dz}/{st}/{tag} — missing {mat_path}")
                    continue
                ok = score_cell_complete(p, mat_path)
        except Exception as e:  # keep sweeping on per-cell failures
            print(f"  FAILED {p.name}: {e}")
            continue
        if ok:
            done += 1
    print(f"DONE. {done} cells scored.")


if __name__ == "__main__":
    main()
