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
  c(rmse = sqrt(mean((zh - zt)^2)),
    r2 = 1 - sum((zh - zt)^2) / sum(zt^2))
}

# Main entry point. Sweeps k, picks the CV-best by held-out RMSE, returns the
# uniform method contract. complete_at(k) re-imputes at that k.
impute_knn <- function(x, ks = 1:10, seed = 1L) {
  ks <- ks[ks < nrow(x)]
  set.seed(seed)
  holdout <- sample(which(!is.na(x)), size = floor(0.2 * sum(!is.na(x))))

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

# Seed-sweep sensitivity: repeated random holdouts, RMSE + R^2 distribution per k.
sensitivity_knn <- function(x, ks = 1:10, n_seeds = 50L, holdout_frac = 0.2) {
  suppressMessages({ library(parallel); library(doParallel); library(foreach) })
  ks <- ks[ks < nrow(x)]
  n_cores <- max(1L, detectCores() - 2L)
  cat(sprintf("  sensitivity using %d cores, %d seeds\n", n_cores, n_seeds))
  cl <- makeCluster(n_cores); registerDoParallel(cl)
  on.exit({ stopCluster(cl); registerDoSEQ() })

  res_list <- foreach(s = seq_len(n_seeds), .packages = "VIM",
                      .export = c("fit_knn", "holdout_rmse_r2_knn")) %dopar% {
    set.seed(s)
    holdout <- sample(which(!is.na(x)), size = floor(holdout_frac * sum(!is.na(x))))
    rmse <- numeric(length(ks)); r2 <- numeric(length(ks))
    for (i in seq_along(ks)) {
      rr <- holdout_rmse_r2_knn(x, ks[i], holdout)
      rmse[i] <- rr["rmse"]; r2[i] <- rr["r2"]
    }
    list(rmse = rmse, r2 = r2)
  }
  rmse_mat <- do.call(rbind, lapply(res_list, `[[`, "rmse"))
  r2_mat   <- do.call(rbind, lapply(res_list, `[[`, "r2"))
  colnames(rmse_mat) <- colnames(r2_mat) <- paste0("k_", ks)
  best_ranks <- ks[apply(rmse_mat, 1, which.min)]
  list(rmse_mat = rmse_mat, r2_mat = r2_mat, best_ranks = best_ranks,
       ranks = ks, param = "k")
}
