# ─────────────────────────────────────────────────────────────────────────────
# OptSpace method: OptSpace matrix completion (Keshavan-Montanari-Oh 2010) on the
# observed pairwise correlation matrix. Non-computable off-diagonal pairs are NA
# and are the imputation target. After completion we symmetrize + nearPD.
#
# Uses the filling package's fill.OptSpace, which returns the completed matrix in
# res$X directly (unlike raw ROptSpace::OptSpace which returns separate factors).
# ropt = NA lets OptSpace estimate the rank automatically (no rank search).
#
# Effectivity is tested by predicting held-out cells via the conditional-Gaussian
# predictor; the emitted M is a covariance-matched surrogate (corr_common.R).
# ─────────────────────────────────────────────────────────────────────────────

suppressMessages(library(filling))

# Main entry point. Single fixed fit (ropt = NA -> auto rank), returns the uniform
# method contract.
impute_optspace <- function(x, niter = 50L, tol = 1e-6, seed = 1L) {
  fit_fn <- function(R_obs, ropt) {
    r <- if (identical(ropt, "auto")) NA else ropt
    fill.OptSpace(R_obs, ropt = r, niter = niter, tol = tol)$X
  }

  run_corr_single(x, fit_fn, "ropt", "auto", seed = seed)
}