# ─────────────────────────────────────────────────────────────────────────────
# MissForest imputation: iterative random-forest imputation, sweeping ntree.
# Nonparametric — captures nonlinear structure softImpute's low-rank model can't,
# no normality assumption. Impute-only; emits completed matrix + held-out RMSE/R²
# sweep, no factoring here.
#
# Held-out RMSE/R² computed the SAME way as the other methods (mask, impute,
# score in z-units, train-mean baseline) so all are comparable.
# ─────────────────────────────────────────────────────────────────────────────

suppressMessages(library(missForest))

# Impute with missForest at a given ntree; returns the completed matrix.
fit_missforest <- function(x, ntree) {
  out <- missForest::missForest(x, ntree = ntree, maxiter = 10)
  out$ximp
}

# Held-out RMSE + R^2 at a given ntree (z-scored, train-mean baseline).
holdout_rmse_r2_mf <- function(x, ntree, holdout) {
  x_train <- x; x_train[holdout] <- NA
  mu <- colMeans(x_train, na.rm = TRUE)
  sdv <- apply(x_train, 2, sd, na.rm = TRUE); sdv[!is.finite(sdv) | sdv == 0] <- 1
  z <- function(M) sweep(sweep(M, 2, mu, "-"), 2, sdv, "/")
  Xc <- tryCatch(fit_missforest(x_train, ntree), error = function(e) NULL)
  if (is.null(Xc)) return(c(rmse = NA, r2 = NA))
  zt <- z(x)[holdout]; zh <- z(Xc)[holdout]
  score_holdout(zt, zh, holdout, nrow(x))
}

# Main entry point. Sweeps ntree, picks CV-best by held-out RMSE, returns the
# uniform contract. complete_at(ntree) re-imputes at that ntree.
impute_missforest <- function(x, ntrees = c(50L, 100L, 200L, 400L), seed = 1L) {
  set.seed(seed)
  holdout <- make_holdout(x, frac = 0.2)

  rmse_v <- numeric(length(ntrees)); r2_v <- numeric(length(ntrees))
  for (i in seq_along(ntrees)) {
    rr <- holdout_rmse_r2_mf(x, ntrees[i], holdout)
    rmse_v[i] <- rr["rmse"]; r2_v[i] <- rr["r2"]
    cat(sprintf("  ntree %4d | RMSE %.4f | R2 %.3f\n", ntrees[i], rr["rmse"], rr["r2"]))
  }
  best_i <- which.min(replace(rmse_v, is.na(rmse_v), Inf))
  best_nt <- ntrees[best_i]
  cat(sprintf("  >> CV-best ntree = %d (RMSE %.4f, R2 %.3f)\n",
              best_nt, rmse_v[best_i], r2_v[best_i]))

  complete_at <- function(v) fit_missforest(x, v)
  list(M = complete_at(best_nt),
       best_param = best_nt, params = ntrees, curve = rmse_v, curve_r2 = r2_v,
       param_name = "ntree", metric_name = "Held-out RMSE",
       complete_at = complete_at)
}

# Seed-sweep sensitivity: repeated random holdouts, RMSE + R^2 distribution per ntree.
sensitivity_missforest <- function(x, ntrees = c(50L, 100L, 200L, 400L),
                                   n_seeds = 20L, holdout_frac = 0.2,
                                   nf = NA_integer_) {
  suppressMessages({ library(parallel); library(doParallel); library(foreach) })
  n_cores <- max(1L, detectCores() - 2L)
  cat(sprintf("  sensitivity using %d cores, %d seeds\n", n_cores, n_seeds))
  cl <- makeCluster(n_cores); registerDoParallel(cl)
  on.exit({ stopCluster(cl); registerDoSEQ() })

  res_list <- foreach(s = seq_len(n_seeds), .packages = c("missForest", "psych"),
                      .export = c("fit_missforest", "holdout_rmse_r2_mf",
                                  "make_holdout", "score_holdout", "BALANCE_HOLDOUT",
                                  "omega_h_only")) %dopar% {
    set.seed(s)
    holdout <- make_holdout(x, frac = holdout_frac)
    rmse <- numeric(length(ntrees)); r2 <- numeric(length(ntrees))
    for (i in seq_along(ntrees)) {
      rr <- holdout_rmse_r2_mf(x, ntrees[i], holdout)
      rmse[i] <- rr["rmse"]; r2[i] <- rr["r2"]
    }
    oh <- NA_real_
    if (!is.na(nf)) {
      best_nt <- ntrees[which.min(replace(rmse, is.na(rmse), Inf))]
      Mc <- tryCatch(fit_missforest(x, best_nt), error = function(e) NULL)
      if (!is.null(Mc)) oh <- omega_h_only(Mc, nf)
    }
    list(rmse = rmse, r2 = r2, omega_h = oh)
  }
  rmse_mat <- do.call(rbind, lapply(res_list, `[[`, "rmse"))
  r2_mat   <- do.call(rbind, lapply(res_list, `[[`, "r2"))
  colnames(rmse_mat) <- colnames(r2_mat) <- paste0("ntree_", ntrees)
  best_ranks <- ntrees[apply(rmse_mat, 1, which.min)]
  list(rmse_mat = rmse_mat, r2_mat = r2_mat, best_ranks = best_ranks,
       omega_h = vapply(res_list, `[[`, numeric(1), "omega_h"),
       ranks = ntrees, param = "ntree")
}
