# ─────────────────────────────────────────────────────────────────────────────
# Method "zeros" — fill-smooth imputation like `default`, but the never-co-
# observed entries of the observed pairwise correlation matrix are filled with
# 0 (absent co-observation means "no association") before the PSD smoothing
# (psych::cor.smooth). A correlation/reduced-matrix imputer in the same family
# as softimpute_corr/optspace/usvt/cvxr/ggm.
#
# No hyperparameter sweep (single fixed recipe; the reported param is the fill
# rule "zero").
#
# Beyond the uniform contract, the exact smoothed correlation is returned as
# `R`; the orchestrator persists it (..._correlation.csv) and it is the matrix
# the EFA and the conditional/prorated latent scoring run on. M is the usual
# covariance-matched surrogate for the completed-CSV hand-off.
#
# Uses the shared correlation imputer driver (corr_common.R).
# ─────────────────────────────────────────────────────────────────────────────

impute_zeros <- function(x, seed = 1L) {
  run_fill_smooth(x, fill = "zero", seed = seed)
}
