# ─────────────────────────────────────────────────────────────────────────────
# softimpute_corr method: softImpute applied to the observed PAIRWISE CORRELATION
# matrix (not the raw dataset). Pairs that are never co-observed have no computable
# correlation, so those NA entries are the imputation target. After imputation we
# symmetrize + nearPD the result into a valid correlation matrix.
#
# Because this operates on a correlation matrix it loses scale and cannot
# reconstruct specific cells/rows, so — mirroring OneSidedMC — effectivity is
# tested by predicting held-out CELLS via the conditional-Gaussian predictor, and
# the emitted "completed" matrix is a covariance-matched surrogate (see
# impute/corr_common.R). No factoring here.
#
# Hyperparameter selection follows the regular softimpute method exactly: sweep
# rank 1..max_rank; at each rank sweep a geometric lambda grid and pick lambda by
# held-out cell RMSE (cell-weighted internal minimisation); report the column-
# balanced score_holdout RMSE/R².
# ─────────────────────────────────────────────────────────────────────────────

suppressMessages(library(softImpute))

# Fit softImpute at a rank cap on the train correlation matrix, sweeping lambda
# and scoring held-out cells via the conditional-Gaussian predictor. Returns the
# FULL-data refit (no holdout) correlation matrix at the chosen lambda, plus the
# held-out score. R_full is used only for the final refit.
fit_at_rank_corr <- function(R_train, R_full, test_cells, holdout, zt, nrow_x,
                             rank_cap) {
  lam_max <- lambda0(R_train)
  lambdas <- exp(seq(log(lam_max), log(lam_max / 100), length.out = 30))

  best_rmse <- Inf; best_lambda <- NA; best_pred <- NULL; warm <- NULL
  for (lam in lambdas) {
    fit <- softImpute(R_train, rank.max = rank_cap, lambda = lam,
                      type = "als", warm.start = warm, maxit = 300)
    warm <- fit
    R_hat <- fit$u %*% (fit$d * t(fit$v))
    R_corr <- symmetrize_nearpd(R_hat)
    zh <- predict_cells_from_corr(R_corr, test_cells)
    rmse <- sqrt(mean((zh - zt)^2))          # cell-weighted lambda selection only
    if (rmse < best_rmse) {
      best_rmse <- rmse; best_lambda <- lam; best_pred <- zh
    }
  }
  # column-balanced score of the held-out cells (masked-train predictions).
  sc <- score_corr_holdout(zt, best_pred, holdout, nrow_x)
  # refit on the FULL correlation matrix (no holdout) at the chosen lambda.
  fit_full <- softImpute(R_full, rank.max = rank_cap, lambda = best_lambda,
                         type = "als", maxit = 500)
  R_final <- symmetrize_nearpd(fit_full$u %*% (fit_full$d * t(fit_full$v)))
  list(R_corr = R_final, rmse = unname(sc["rmse"]), r2 = unname(sc["r2"]),
       lambda = best_lambda)
}

# Main entry point. Sweeps rank, picks CV-best by held-out cell RMSE, returns the
# uniform method contract with M = the covariance-matched surrogate at best rank.
impute_softimpute_corr <- function(x, max_rank = 10L, seed = 1L) {
  set.seed(seed)
  holdout <- make_holdout(x, frac = 0.2)

  # standardize by TRAIN-cell moments (mask holdout first, like the other R
  # methods) so the held-out baseline is the train column mean = 0 in z-space.
  x_train <- x; x_train[holdout] <- NA
  mu  <- colMeans(x_train, na.rm = TRUE)
  sdv <- apply(x_train, 2, sd, na.rm = TRUE)
  sdv[!is.finite(sdv) | sdv == 0] <- 1
  z       <- corr_zscore(x, mu, sdv)          # full standardized
  z_train <- z; z_train[holdout] <- NA

  R_train <- observed_corr(z_train)           # NA where pair never co-observed
  R_full  <- observed_corr(z)                 # full (no holdout) for final refit

  test_cells <- make_holdout_cells(z, holdout)
  zt <- vapply(test_cells, function(tc) tc$held_val, numeric(1))

  nrow_x <- nrow(x)
  ranks <- 1:min(max_rank, ncol(x) - 1L)

  ok_ranks <- integer(0); rmse_v <- numeric(0); r2_v <- numeric(0)
  fits <- list()
  for (k in seq_along(ranks)) {
    fr <- tryCatch(
      fit_at_rank_corr(R_train, R_full, test_cells, holdout, zt, nrow_x,
                       ranks[k]),
      error = function(e) {
        cat(sprintf("  rank %2d | FAILED (%s) - skipping\n",
                    ranks[k], conditionMessage(e)))
        NULL
      })
    if (is.null(fr)) next
    ok_ranks <- c(ok_ranks, ranks[k]); fits[[length(fits) + 1L]] <- fr
    rmse_v <- c(rmse_v, fr$rmse); r2_v <- c(r2_v, fr$r2)
    cat(sprintf("  rank %2d | RMSE %.4f | R2 %.3f\n", ranks[k], fr$rmse, fr$r2))
  }
  if (!length(ok_ranks))
    stop(sprintf("softimpute_corr: every rank failed (%s), no valid imputation;",
                 paste(ranks, collapse = ",")),
         " try other methods for this cell")
  best_k <- which.min(rmse_v); best_r <- ok_ranks[best_k]
  cat(sprintf("  >> CV-best imputation rank = %d (RMSE %.4f, R2 %.3f)\n",
              best_r, rmse_v[best_k], r2_v[best_k]))

  # final surrogate on original scale (observed-cell moments), covariance = best
  # rank's imputed correlation matrix.
  mom <- corr_column_moments(x)
  M <- generate_surrogate(fits[[best_k]]$R_corr, nrow(x), mom$mu, mom$sd,
                          seed = seed)

  list(M = M,
       best_param = best_r, params = ok_ranks, curve = rmse_v, curve_r2 = r2_v,
       param_name = "rank", metric_name = "Held-out RMSE")
}
