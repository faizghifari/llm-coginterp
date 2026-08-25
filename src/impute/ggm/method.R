# ─────────────────────────────────────────────────────────────────────────────
# GGM method: graphical-model MLE completion via ggm::fitConGraph (iterative
# conditional fitting) on the observed pairwise correlation matrix. Handles
# non-chordal observed patterns, not just decomposable ones.
#
# No hyperparameter sweep (single fixed fit).
#
# Uses the shared correlation imputer driver (corr_common.R).
# ─────────────────────────────────────────────────────────────────────────────

suppressMessages(library(ggm))

impute_ggm <- function(x, min_n = 10L, seed = 1L) {
  n_obs <- nrow(x)

  fit_fn <- function(R_obs, .unused) {
    R <- R_obs
    mask <- is.finite(R)
    diag(mask) <- TRUE
    R[!is.finite(R)] <- 0

    amat <- mask
    diag(amat) <- FALSE

    fit <- ggm::fitConGraph(amat, R, n = n_obs)
    fit$Shat
  }

  run_corr_single(x, fit_fn, "min_n", min_n, seed = seed)
}
