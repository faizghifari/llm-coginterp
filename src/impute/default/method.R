# ─────────────────────────────────────────────────────────────────────────────
# Method "default" — fill-smooth imputation: fill the never-co-observed entries
# of the observed pairwise correlation matrix with the mean finite off-diagonal
# correlation (absent co-observation gets the typical association), then
# PSD-smooth (psych::cor.smooth). A correlation/reduced-matrix imputer in the
# same family as softimpute_corr/optspace/usvt/cvxr/ggm.
#
# No hyperparameter sweep (single fixed recipe; the reported param is the fill
# rule "mean").
#
# Beyond the uniform contract, the exact smoothed correlation is returned as
# `R`; the orchestrator persists it (..._correlation.csv) and it is the matrix
# the EFA and the conditional/prorated latent scoring run on. M is the usual
# covariance-matched surrogate for the completed-CSV hand-off.
#
# Uses the shared correlation imputer driver (corr_common.R).
# ─────────────────────────────────────────────────────────────────────────────

impute_default <- function(x, seed = 1L) {
  run_fill_smooth(x, fill = "mean", seed = seed)
}
