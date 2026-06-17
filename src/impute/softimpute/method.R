# ─────────────────────────────────────────────────────────────────────────────
# SoftImpute method: imputation + held-out-RMSE rank selection + sensitivity.
# No factoring here — this emits a completed matrix and diagnostics; factoring
# is done by the shared factor/ module via the orchestrator.
# ─────────────────────────────────────────────────────────────────────────────

suppressMessages(library(softImpute))

# Column-stratified holdout: sample ~frac of observed cells, but never mask so
# many cells in a column that fewer than min_keep observed cells remain. Without
# this, sparse columns can be emptied and softImpute has no in-column signal.
make_holdout <- function(xs, frac = 0.2, min_keep = 1L) {
  nr <- nrow(xs)
  holdout <- integer(0)
  for (j in seq_len(ncol(xs))) {
    rows_obs <- which(!is.na(xs[, j]))
    n_obs <- length(rows_obs)
    if (n_obs == 0L) next
    n_hold <- min(floor(frac * n_obs), n_obs - min_keep)
    if (n_hold > 0L) {
      picked <- rows_obs[sample.int(n_obs, size = n_hold)]
      holdout <- c(holdout, (j - 1L) * nr + picked)
    }
  }
  holdout
}

# Baseline SS for held-out R^2: predict each held-out cell with its column's
# TRAIN mean (observed cells minus the holdout). Using the train mean -- not the
# holdout mean -- keeps the baseline honest (it sees only what the model sees).
holdout_baseline_ss <- function(xs, holdout) {
  x_train <- xs; x_train[holdout] <- NA
  col_mean <- colMeans(x_train, na.rm = TRUE)          # train-only column means
  cidx <- ((holdout - 1L) %/% nrow(xs)) + 1L           # column of each held-out cell
  base_pred <- col_mean[cidx]
  base_pred[is.na(base_pred)] <- 0                     # column with no train cells
  sum((base_pred - xs[holdout])^2)
}

# Fit softImpute at a rank cap, sweeping lambda and scoring on the holdout.
# Returns held-out RMSE and held-out R^2 (1 - SS_resid / SS_train-mean-baseline).
fit_at_rank <- function(xs, holdout, rank_cap) {
  x_train <- xs
  x_train[holdout] <- NA
  lam_max <- lambda0(x_train)
  lambdas <- exp(seq(log(lam_max), log(lam_max / 100), length.out = 30))

  ss_base <- holdout_baseline_ss(xs, holdout)
  n_hold  <- length(holdout)

  best_rmse <- Inf; best_lambda <- NA; warm <- NULL
  for (lam in lambdas) {
    fit <- softImpute(x_train, rank.max = rank_cap, lambda = lam,
                      type = "als", warm.start = warm, maxit = 300)
    warm <- fit
    xhat <- fit$u %*% (fit$d * t(fit$v))
    rmse <- sqrt(mean((xhat[holdout] - xs[holdout])^2))
    if (rmse < best_rmse) { best_rmse <- rmse; best_lambda <- lam }
  }
  fit <- softImpute(xs, rank.max = rank_cap, lambda = best_lambda,
                    type = "als", maxit = 500)
  ss_resid <- best_rmse^2 * n_hold
  list(fit = fit, rmse = best_rmse, r2 = 1 - ss_resid / ss_base,
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
  holdout <- make_holdout(xs, frac = 0.2, min_keep = 1L)
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

# Repeated-holdout sensitivity of the held-out RMSE curve (parallelised).
sensitivity_softimpute <- function(x, max_rank = 15L, n_seeds = 50L,
                                   holdout_frac = 0.2) {
  suppressMessages({ library(parallel); library(doParallel); library(foreach) })
  xs <- biScale(x, row.center = FALSE, row.scale = FALSE,
                col.center = TRUE, col.scale = TRUE, maxit = 100)
  ranks <- 1:min(max_rank, nrow(x) - 1L, ncol(x) - 1L)

  n_cores <- max(1L, detectCores() - 2L)
  cat(sprintf("  sensitivity using %d cores, %d seeds\n", n_cores, n_seeds))
  cl <- makeCluster(n_cores); registerDoParallel(cl)
  on.exit({ stopCluster(cl); registerDoSEQ() })

  # each seed returns a (rmse, r2) pair per rank
  res_list <- foreach(s = seq_len(n_seeds), .packages = "softImpute",
                      .export = c("fit_at_rank", "make_holdout",
                                  "holdout_baseline_ss")) %dopar% {
    set.seed(s)
    holdout <- make_holdout(xs, frac = holdout_frac, min_keep = 1L)
    rmse <- numeric(length(ranks)); r2 <- numeric(length(ranks))
    for (k in seq_along(ranks)) {
      fr <- fit_at_rank(xs, holdout, rank_cap = ranks[k])
      rmse[k] <- fr$rmse; r2[k] <- fr$r2
    }
    list(rmse = rmse, r2 = r2)
  }
  rmse_mat <- do.call(rbind, lapply(res_list, `[[`, "rmse"))
  r2_mat   <- do.call(rbind, lapply(res_list, `[[`, "r2"))
  colnames(rmse_mat) <- colnames(r2_mat) <- paste0("rank_", ranks)
  best_ranks <- ranks[apply(rmse_mat, 1, which.min)]
  list(rmse_mat = rmse_mat, r2_mat = r2_mat, best_ranks = best_ranks,
       ranks = ranks, param = "rank")
}
