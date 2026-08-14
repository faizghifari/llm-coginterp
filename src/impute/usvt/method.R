# ─────────────────────────────────────────────────────────────────────────────
# USVT method: Universal Singular Value Thresholding (Chatterjee 2015) on the
# observed pairwise correlation matrix. Non-computable off-diagonal pairs are NA
# and are the imputation target. After completion we symmetrize + nearPD.
#
# Uses the filling package's fill.USVT, which hard-thresholds singular values at
# `eta` to separate signal from noise. eta is FIXED (no search).
#
# Effectivity is tested by predicting held-out cells via the conditional-Gaussian
# predictor; the emitted M is a covariance-matched surrogate (corr_common.R).
# ─────────────────────────────────────────────────────────────────────────────

suppressMessages(library(filling))

# Main entry point. Single fixed fit (eta = 0.01), returns the uniform method
# contract.
impute_usvt <- function(x, eta = 0.01, seed = 1L) {
  fit_fn <- function(R_obs, e) {
    fill.USVT(R_obs, eta = e)$X
  }

  run_corr_single(x, fit_fn, "eta", eta, seed = seed)
}