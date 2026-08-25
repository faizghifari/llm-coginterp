# ─────────────────────────────────────────────────────────────────────────────
# CVXR method: SDP completion (Vandenberghe/Boyd/Wu MAXDET) on the observed
# pairwise correlation matrix. Maximize log-det(Sigma) subject to Sigma PSD
# and matching observed entries within per-pair Fisher-z confidence bands
# (scaled by each pair's co-observation count). Diagonal matched exactly.
#
# No hyperparameter sweep (single fixed fit). ci_mult controls the confidence
# band width (default 2 ~ 95%).
#
# Uses the shared correlation imputer driver (corr_common.R).
# ─────────────────────────────────────────────────────────────────────────────

suppressMessages(library(CVXR))

impute_cvxr <- function(x, min_n = 10L, ci_mult = 2, seed = 1L) {
  present <- !is.na(x)
  n_pair <- crossprod(present)

  fit_fn <- function(R_obs, cm) {
    R <- R_obs
    mask <- is.finite(R)
    diag(mask) <- TRUE
    R[!is.finite(R)] <- 0

    p <- ncol(R)
    Sigma_var <- CVXR::Variable(c(p, p), symmetric = TRUE)

    obs <- which(upper.tri(mask, diag = TRUE) & mask, arr.ind = TRUE)
    constraints <- list(Sigma_var %>>% 0)
    for (k in seq_len(nrow(obs))) {
      i <- obs[k, 1]; j <- obs[k, 2]
      if (i == j) {
        constraints[[length(constraints) + 1]] <- Sigma_var[i, j] == R[i, j]
        next
      }
      se <- 1 / sqrt(max(n_pair[i, j] - 3, 1))
      z  <- atanh(min(max(R[i, j], -0.999), 0.999))
      lo <- tanh(z - cm * se)
      hi <- tanh(z + cm * se)
      constraints[[length(constraints) + 1]] <- Sigma_var[i, j] >= lo
      constraints[[length(constraints) + 1]] <- Sigma_var[i, j] <= hi
    }

    prob <- CVXR::Problem(CVXR::Maximize(CVXR::log_det(Sigma_var)), constraints)
    CVXR::psolve(prob, solver = "SCS")
    st <- CVXR::status(prob)
    if (!st %in% c("optimal", "optimal_inaccurate"))
      stop("cvxr_maxdet_complete: solver status = ", st,
           " (raise ci_mult -- the trusted pairwise correlations may not be jointly PSD-consistent even at their sampling uncertainty)")

    out <- CVXR::value(Sigma_var)
    dimnames(out) <- dimnames(R_obs)
    out
  }

  run_corr_single(x, fit_fn, "ci_mult", ci_mult, seed = seed)
}
