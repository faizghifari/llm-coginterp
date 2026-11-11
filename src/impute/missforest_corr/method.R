# ─────────────────────────────────────────────────────────────────────────────
# missforest_corr method: MissForest (iterative random-forest) imputation applied
# to the observed PAIRWISE CORRELATION matrix (not the raw dataset). Entries that
# are NA (pairs never co-observed) are the imputation target; after imputation we
# symmetrize + nearPD the result into a valid correlation matrix.
#
# Because this operates on a correlation matrix it loses scale and cannot
# reconstruct specific cells/rows, so — mirroring OneSidedMC — effectivity is
# tested by predicting held-out CELLS via the conditional-Gaussian predictor,
# and the emitted "completed" matrix is a covariance-matched surrogate (see
# impute/corr_common.R). No factoring here.
#
# Hyperparameter selection follows the regular missforest method exactly: sweep
# ntree, pick the CV-best by held-out cell RMSE (column-balanced score_holdout).
# ─────────────────────────────────────────────────────────────────────────────

suppressMessages(library(missForest))

# Fill the NA entries of the correlation matrix with missForest at a given ntree
# (rows are benchmarks; features are the other benchmarks' correlations).
# missForest returns a data.frame — the corr pipeline (nearPD, eigen, sweep)
# downstream needs a matrix.
fit_missforest_corr <- function(R_obs, ntree) {
  out <- missForest::missForest(as.data.frame(R_obs), ntree = ntree,
                                maxiter = 10)
  as.matrix(out$ximp)
}

# Main entry point. Sweeps ntree, picks CV-best by held-out cell RMSE, returns
# the uniform method contract with M = the covariance-matched surrogate at the
# best ntree.
impute_missforest_corr <- function(x, ntrees = c(50L, 100L, 200L, 400L),
                                   seed = 1L) {
  run_corr_sweep(x, fit_missforest_corr, ntrees, "ntree", seed = seed)
}
