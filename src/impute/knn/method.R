# ─────────────────────────────────────────────────────────────────────────────
# KNN imputation: fill each missing cell from the k most similar models, sweeping
# k. Assumption-light baseline (no normality / low-rank assumption). Impute-only
# — emits the completed matrix + held-out RMSE/R² sweep; no factoring here.
#
# Uses VIM::kNN on the standardized matrix (Gower distance over benchmarks). The
# held-out RMSE/R² is computed the SAME way as softimpute/iterativepca (mask
# observed cells, impute, score in z-units, train-mean baseline) so all methods
# are directly comparable.
# ─────────────────────────────────────────────────────────────────────────────

suppressMessages(library(VIM))

# Standardize columns by train moments, kNN-impute, return completed matrix on
# the ORIGINAL scale. `train_only` masks the holdout before fitting.
fit_knn <- function(x, k) {
  df <- as.data.frame(x)
  imp <- VIM::kNN(df, k = k, imp_var = FALSE, numFun = weighted.mean)
  as.matrix(imp)
}

# Held-out RMSE + R^2 at a given k (z-scored, train-mean baseline).
holdout_rmse_r2_knn <- function(x, k, holdout) {
  x_train <- x; x_train[holdout] <- NA
  mu <- colMeans(x_train, na.rm = TRUE)
  sdv <- apply(x_train, 2, sd, na.rm = TRUE); sdv[!is.finite(sdv) | sdv == 0] <- 1
  z <- function(M) sweep(sweep(M, 2, mu, "-"), 2, sdv, "/")
  Xc <- tryCatch(fit_knn(x_train, k), error = function(e) NULL)
  if (is.null(Xc)) return(c(rmse = NA, r2 = NA))
  zt <- z(x)[holdout]; zh <- z(Xc)[holdout]
  score_holdout(zt, zh, holdout, nrow(x))
}

# Main entry point. Sweeps k, picks the CV-best by held-out RMSE, returns the
# uniform method contract. complete_at(k) re-imputes at that k.
impute_knn <- function(x, ks = 1:10, seed = 1L) {
  ks <- ks[ks < nrow(x)]
  set.seed(seed)
  holdout <- make_holdout(x, frac = 0.2)

  rmse_v <- numeric(length(ks)); r2_v <- numeric(length(ks))
  for (i in seq_along(ks)) {
    rr <- holdout_rmse_r2_knn(x, ks[i], holdout)
    rmse_v[i] <- rr["rmse"]; r2_v[i] <- rr["r2"]
    cat(sprintf("  k %2d | RMSE %.4f | R2 %.3f\n", ks[i], rr["rmse"], rr["r2"]))
  }
  best_i <- which.min(replace(rmse_v, is.na(rmse_v), Inf))
  best_k <- ks[best_i]
  cat(sprintf("  >> CV-best k = %d (RMSE %.4f, R2 %.3f)\n",
              best_k, rmse_v[best_i], r2_v[best_i]))

  complete_at <- function(v) fit_knn(x, v)
  list(M = complete_at(best_k),
       best_param = best_k, params = ks, curve = rmse_v, curve_r2 = r2_v,
       param_name = "k", metric_name = "Held-out RMSE",
       complete_at = complete_at)
}

# Seed-sweep sensitivity: repeated random holdouts, RMSE + R^2 distribution per k,
# plus an omega_h distribution (bifactor at the fixed `nf` from the main run).
sensitivity_knn <- function(x, ks = 1:10, n_seeds = 20L, holdout_frac = 0.2,
                            nf = NA_integer_) {
  suppressMessages({ library(parallel); library(doParallel); library(foreach) })
  ks <- ks[ks < nrow(x)]
  n_cores <- max(1L, detectCores() - 2L)
  cat(sprintf("  sensitivity using %d cores, %d seeds\n", n_cores, n_seeds))
  cl <- makeCluster(n_cores); registerDoParallel(cl)
  on.exit({ stopCluster(cl); registerDoSEQ() })

  # nf is fixed (from the main run's factoring) so the worker needs no parallel
  # analysis — just impute at the seed's best k and run bifactor at that nf.
  res_list <- foreach(s = seq_len(n_seeds), .packages = c("VIM", "psych"),
                      .export = c("fit_knn", "holdout_rmse_r2_knn",
                                  "make_holdout", "score_holdout", "BALANCE_HOLDOUT",
                                  "omega_h_only")) %dopar% {
    set.seed(s)
    holdout <- make_holdout(x, frac = holdout_frac)
    rmse <- numeric(length(ks)); r2 <- numeric(length(ks))
    for (i in seq_along(ks)) {
      rr <- holdout_rmse_r2_knn(x, ks[i], holdout)
      rmse[i] <- rr["rmse"]; r2[i] <- rr["r2"]
    }
    oh <- NA_real_
    if (!is.na(nf)) {
      best_k <- ks[which.min(replace(rmse, is.na(rmse), Inf))]
      Mc <- tryCatch(fit_knn(x, best_k), error = function(e) NULL)
      if (!is.null(Mc)) oh <- omega_h_only(Mc, nf)
    }
    list(rmse = rmse, r2 = r2, omega_h = oh)
  }
  rmse_mat <- do.call(rbind, lapply(res_list, `[[`, "rmse"))
  r2_mat   <- do.call(rbind, lapply(res_list, `[[`, "r2"))
  colnames(rmse_mat) <- colnames(r2_mat) <- paste0("k_", ks)
  best_ranks <- ks[apply(rmse_mat, 1, which.min)]
  list(rmse_mat = rmse_mat, r2_mat = r2_mat, best_ranks = best_ranks,
       omega_h = vapply(res_list, `[[`, numeric(1), "omega_h"),
       ranks = ks, param = "k")
}
