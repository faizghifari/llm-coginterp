# ─────────────────────────────────────────────────────────────────────────────
# knn_corr method: k-nearest-neighbour imputation applied to the observed
# PAIRWISE CORRELATION matrix (not the raw dataset). Each benchmark's
# correlation profile across the other benchmarks is a row; entries that are NA
# (pairs never co-observed) are filled from the k most similar benchmarks' rows.
# After imputation we symmetrize + nearPD the result into a valid correlation
# matrix.
#
# Because this operates on a correlation matrix it loses scale and cannot
# reconstruct specific cells/rows, so — mirroring OneSidedMC — effectivity is
# tested by predicting held-out CELLS via the conditional-Gaussian predictor,
# and the emitted "completed" matrix is a covariance-matched surrogate (see
# impute/corr_common.R). No factoring here.
#
# Hyperparameter selection follows the regular knn method exactly: sweep k = 1..10,
# pick the CV-best by held-out cell RMSE (column-balanced score_holdout).
# ─────────────────────────────────────────────────────────────────────────────

suppressMessages(library(VIM))

# Fill the NA entries of the correlation matrix with VIM::kNN at a given k (rows
# are benchmarks; Gower distance over the other benchmarks' correlations).
fit_knn_corr <- function(R_obs, k) {
  imp <- VIM::kNN(as.data.frame(R_obs), k = k, imp_var = FALSE,
                  numFun = weighted.mean)
  as.matrix(imp)
}

# Main entry point. Sweeps k, picks CV-best by held-out cell RMSE, returns the
# uniform method contract with M = the covariance-matched surrogate at best k.
impute_knn_corr <- function(x, ks = 1:10, seed = 1L) {
  # rows of the correlation matrix are benchmarks, so k is capped by ncol(x)
  ks <- ks[ks < ncol(x)]
  run_corr_sweep(x, fit_knn_corr, ks, "k", seed = seed)
}
