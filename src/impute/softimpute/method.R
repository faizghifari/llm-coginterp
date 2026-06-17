# ─────────────────────────────────────────────────────────────────────────────
# SoftImpute method: imputation + held-out-RMSE rank selection + sensitivity.
# No factoring here — this emits a completed matrix and diagnostics; factoring
# is done by the shared factor/ module via the orchestrator.
# ─────────────────────────────────────────────────────────────────────────────

suppressMessages(library(softImpute))

# make_holdout (column-stratified) lives in impute/common.R, shared by all methods.

# Fit softImpute at a rank cap, sweeping lambda and scoring on the holdout.
# lambda is selected by cell-weighted RMSE (just an internal minimisation), but
# the REPORTED rmse/r2 go through score_holdout so they honour the column-balance
# flag and the train-mean baseline, consistent with every other method.
# Note: xs is already column z-scored (biScale), so xs[holdout] / xhat[holdout]
# are standardized true / predicted values, and the train mean ~ 0 in z-space.
fit_at_rank <- function(xs, holdout, rank_cap) {
  x_train <- xs
  x_train[holdout] <- NA
  lam_max <- lambda0(x_train)
  lambdas <- exp(seq(log(lam_max), log(lam_max / 100), length.out = 30))

  best_rmse <- Inf; best_lambda <- NA; best_pred <- NULL; warm <- NULL
  for (lam in lambdas) {
    fit <- softImpute(x_train, rank.max = rank_cap, lambda = lam,
                      type = "als", warm.start = warm, maxit = 300)
    warm <- fit
    xhat <- fit$u %*% (fit$d * t(fit$v))   # predictions from the MASKED-train fit
    rmse <- sqrt(mean((xhat[holdout] - xs[holdout])^2))   # lambda selection only
    if (rmse < best_rmse) {
      best_rmse <- rmse; best_lambda <- lam
      best_pred <- xhat[holdout]           # hold the masked-fit's held-out preds
    }
  }
  # Score the held-out cells with the MASKED-train predictions (best_pred) — NOT a
  # fit on the full xs, which would leak the holdout and give RMSE 0 / R^2 1.
  sc <- score_holdout(xs[holdout], best_pred, holdout, nrow(xs))
  # The returned completed matrix uses the full data at the chosen lambda.
  fit <- softImpute(xs, rank.max = rank_cap, lambda = best_lambda,
                    type = "als", maxit = 500)
  list(fit = fit, rmse = unname(sc["rmse"]), r2 = unname(sc["r2"]),
       lambda = best_lambda, eff_rank = sum(fit$d > 1e-8))
}

# Main entry point. Sweeps rank, picks CV-best by holdout RMSE, returns the
# uniform method contract (see README): the completed matrix at best rank, the
# predictive curve, axis metadata, and a complete_at(v) closure so the
# orchestrator can factor at any rank WITHOUT this module ever factoring itself.
impute_softimpute <- function(x, max_rank = 15L, seed = 1L) {
  xs <- biScale(x, row.center = FALSE, row.scale = FALSE,
                col.center = TRUE, col.scale = TRUE, maxit = 100)
  set.seed(seed)
  holdout <- make_holdout(xs, frac = 0.2)   # min_keep = 2 (shared default)
  ranks <- 1:min(max_rank, nrow(x) - 1L, ncol(x) - 1L)

  rmse_v <- numeric(length(ranks)); r2_v <- numeric(length(ranks))
  eff_v  <- integer(length(ranks)); fits <- vector("list", length(ranks))
  for (k in seq_along(ranks)) {
    fr <- fit_at_rank(xs, holdout, rank_cap = ranks[k])
    fits[[k]] <- fr; rmse_v[k] <- fr$rmse; r2_v[k] <- fr$r2; eff_v[k] <- fr$eff_rank
    cat(sprintf("  rank %2d | RMSE %.4f | R2 %.3f | eff_rank %2d\n",
                ranks[k], fr$rmse, fr$r2, fr$eff_rank))
  }
  best_k <- which.min(rmse_v); best_r <- ranks[best_k]
  cat(sprintf("  >> CV-best imputation rank = %d (RMSE %.4f, R2 %.3f)\n",
              best_r, rmse_v[best_k], r2_v[best_k]))

  # Completed matrix at any swept rank (reuses that rank's fit; no re-fit).
  complete_at <- function(v) complete(x, fits[[which(ranks == v)]]$fit)

  list(M = complete(x, fits[[best_k]]$fit),
       best_param = best_r, params = ranks, curve = rmse_v, curve_r2 = r2_v,
       eff_rank = eff_v, param_name = "rank", metric_name = "Held-out RMSE",
       complete_at = complete_at)
}

# Repeated-holdout sensitivity of the held-out RMSE curve (parallelised), plus an
# omega_h distribution (bifactor at the fixed `nf` from the main run).
sensitivity_softimpute <- function(x, max_rank = 15L, n_seeds = 50L,
                                   holdout_frac = 0.2, nf = NA_integer_) {
  suppressMessages({ library(parallel); library(doParallel); library(foreach) })
  xs <- biScale(x, row.center = FALSE, row.scale = FALSE,
                col.center = TRUE, col.scale = TRUE, maxit = 100)
  ranks <- 1:min(max_rank, nrow(x) - 1L, ncol(x) - 1L)

  n_cores <- max(1L, detectCores() - 2L)
  cat(sprintf("  sensitivity using %d cores, %d seeds\n", n_cores, n_seeds))
  cl <- makeCluster(n_cores); registerDoParallel(cl)
  on.exit({ stopCluster(cl); registerDoSEQ() })

  # each seed returns (rmse, r2) per rank + omega_h of the best-rank completion
  res_list <- foreach(s = seq_len(n_seeds), .packages = c("softImpute", "psych"),
                      .export = c("fit_at_rank", "make_holdout",
                                  "score_holdout", "BALANCE_HOLDOUT",
                                  "omega_h_only")) %dopar% {
    set.seed(s)
    holdout <- make_holdout(xs, frac = holdout_frac)   # min_keep = 2 (shared default)
    rmse <- numeric(length(ranks)); r2 <- numeric(length(ranks))
    fits <- vector("list", length(ranks))
    for (k in seq_along(ranks)) {
      fr <- fit_at_rank(xs, holdout, rank_cap = ranks[k])
      rmse[k] <- fr$rmse; r2[k] <- fr$r2; fits[[k]] <- fr$fit
    }
    oh <- NA_real_
    if (!is.na(nf)) {
      bk <- which.min(replace(rmse, is.na(rmse), Inf))
      Mc <- tryCatch(complete(x, fits[[bk]]), error = function(e) NULL)
      if (!is.null(Mc)) oh <- omega_h_only(Mc, nf)
    }
    list(rmse = rmse, r2 = r2, omega_h = oh)
  }
  rmse_mat <- do.call(rbind, lapply(res_list, `[[`, "rmse"))
  r2_mat   <- do.call(rbind, lapply(res_list, `[[`, "r2"))
  colnames(rmse_mat) <- colnames(r2_mat) <- paste0("rank_", ranks)
  best_ranks <- ranks[apply(rmse_mat, 1, which.min)]
  list(rmse_mat = rmse_mat, r2_mat = r2_mat, best_ranks = best_ranks,
       omega_h = vapply(res_list, `[[`, numeric(1), "omega_h"),
       ranks = ranks, param = "rank")
}
