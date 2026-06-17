# OneSidedMC

One-sided matrix completion (Cao, Liang & Valiant 2023): recover the **right
singular vectors** of a low-rank matrix `X` from very few observations per row,
even when full cell-by-cell completion is impossible. Used in this project as an
imputation method — it estimates `Θ̂ = (1/m) XᵀX` (the benchmark covariance) and
synthesizes a covariance-matched surrogate for downstream factor analysis. See
the repo root `README.md` for how it plugs into the pipeline.

### Rank-selection metric (cell-level)

Rank `r` is chosen by held-out **cell** reconstruction, so the RMSE/R² are on the
same scale as softimpute/iterativepca and directly comparable. We hold out
observed cells, fit `Θ̂` on the rest, then predict each held-out cell from that
row's surviving cells via the conditional-Gaussian (best linear) predictor

```
ẑ_j = Θ̂[j, S] · pinv(Θ̂[S, S]) · z_S
```

(`S` = the row's training cells; `pinv` handles `Θ̂[S,S]` being rank-≤r singular
when `|S| > r`, no ridge). R² baseline is the train-cell column mean.

> Note: this is an *off-label* use of OSMC — the paper recovers the covariance,
> not cells; we predict cells *via* the recovered covariance to get a number
> comparable to the other methods. OSMC's **native** product-reconstruction
> metric (reconstructing held-out pairwise products = entries of Θ*) is kept as a
> dead branch in `pipeline.jl` behind `OSMC_CELL_METRIC = false` — it's the right
> diagnostic for "is Θ̂ itself good" vs "does Θ̂ predict cells".

## Layout

```
src/
  data.jl         # synthetic data + observation sampling (paper §5.1)
  loss.jl         # fixed-k pairwise-product loss + gradient
  algorithm.jl    # fit_onesided (Adam) + right_singular_vectors
  baselines.jl    # direct factorization, factorization-without-diag, full MC
  metrics.jl      # rowspace_error, mean_principal_angle_deg
  realdata.jl     # ragged loss/fit for real (variable-obs-per-row) tables  [exported]
  pipeline.jl     # CSV -> Θ̂ -> covariance surrogate; rank selection; sensitivity
run.jl            # real-data driver (loops densifier × strategy)
test/             # test suite (see below)
```

`src/realdata.jl` is part of the module and exported; `src/pipeline.jl` is a
script-level driver (`include`d by `run.jl`), not part of the module.

## Running the tests

The suite covers the paper core (synthetic recovery) plus an MNAR stress suite.
Run from this directory (`impute/OneSidedMC/`):

```bash
# everything (core recovery + MNAR) — the default
julia --project=. -e 'using Pkg; Pkg.test()'
```

Two env knobs select scope and cost:

- `TEST` — which groups run: `all` (default) | `core` (recovery, no MNAR) |
  `mnar` (only MNAR + the lightweight unit tests).
- `NSEEDS` — seeds per parameter set in the MNAR suite (default `10`); higher =
  slower but less noisy.

```bash
# core recovery only (faster; skips the MNAR stress suite)
TEST=core julia --project=. -e 'using Pkg; Pkg.test()'

# MNAR suite only, with more seeds for a tighter check
TEST=mnar NSEEDS=20 julia --project=. -e 'using Pkg; Pkg.test()'
```

The always-on lightweight unit tests (`test_metrics.jl`, `test_data.jl`,
`test_loss.jl`) run in every mode. `test_recovery.jl` runs under `all`/`core`;
`test_mnar.jl` runs under `all`/`mnar`.

> Note: the recovery and MNAR suites sweep over problem sizes and take a few
> minutes. Use `TEST=core` or a small `NSEEDS` for a quick smoke check.

## Running on real data

```bash
# from the repo root, via the orchestrator (recommended):
Rscript main/run.R --method onesidedmc

# or the Julia driver directly (uses all cores for the sensitivity sweep):
julia --threads=auto --project=impute/OneSidedMC impute/OneSidedMC/run.jl
```

Env knobs honored by `run.jl`: `OSMC_DATA_ROOT`, `OSMC_RESULTS_ROOT`,
`OSMC_DENSIFIERS`, `OSMC_STRATEGIES`, `OSMC_SENSITIVITY` (set by the orchestrator;
see repo root `README.md`).
