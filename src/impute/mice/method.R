# ─────────────────────────────────────────────────────────────────────────────
# MICE (multiple imputation by chained equations), sweeping m = number of
# imputations. Unlike the other methods, MICE produces m completed matrices; we
# hand factoring the MEAN across them (standard averaged multiple imputation).
# The m imputations also carry uncertainty, surfaced via the seed-sweep
# sensitivity. Impute-only; no factoring here.
#
# Held-out RMSE/R² computed the SAME way as the other methods (mask, impute,
# score in z-units, train-mean baseline). At a given m we score the mean of the
# m imputations, for comparability with the single-matrix methods.
# ─────────────────────────────────────────────────────────────────────────────

suppressMessages(library(mice))

# Mean of m MICE completions; returns the averaged completed matrix.
#
# This benchmark matrix is wide and highly collinear (~56 cols, ~13% observed),
# so mice's default pmm regresses each column on a rank-deficient predictor set
# and DROPS most columns as "collinear", leaving them NA. We:
#   - ridge-regularize each per-column regression (ridge) so the (X'X + ridge*I)
#     inverse exists and coefficients don't explode, and
#   - remove.collinear = FALSE so mice imputes collinear columns instead of
#     skipping them.
# Any cell mice still can't fill (e.g. a fully-degenerate column) is backfilled
# with the column mean so the completed matrix has no NAs (never silent NA).
fit_mice <- function(x, m, seed = 1L, ridge = 1e-3) {
  df <- as.data.frame(x)
  pred <- mice::quickpred(df)  # predictor matrix; mice scales ridge per-column
  imp <- mice::mice(df, m = m, printFlag = FALSE, seed = seed,
                    method = "pmm", maxit = 5, ridge = ridge,
                    remove.collinear = FALSE, predictorMatrix = pred)
  comps <- lapply(seq_len(m), function(i) as.matrix(mice::complete(imp, i)))
  M <- Reduce(`+`, comps) / m
  # backfill any cells mice left NA with the column mean (last-resort).
  if (anyNA(M)) {
    cm <- colMeans(M, na.rm = TRUE); cm[!is.finite(cm)] <- 0
    na_idx <- which(is.na(M), arr.ind = TRUE)
    M[na_idx] <- cm[na_idx[, "col"]]
  }
  M
}

# Held-out RMSE + R^2 at a given m (z-scored, train-mean baseline).
holdout_rmse_r2_mice <- function(x, m, holdout, seed = 1L) {
  x_train <- x; x_train[holdout] <- NA
  mu <- colMeans(x_train, na.rm = TRUE)
  sdv <- apply(x_train, 2, sd, na.rm = TRUE); sdv[!is.finite(sdv) | sdv == 0] <- 1
  z <- function(M) sweep(sweep(M, 2, mu, "-"), 2, sdv, "/")
  Xc <- tryCatch(fit_mice(x_train, m, seed), error = function(e) NULL)
  if (is.null(Xc)) return(c(rmse = NA, r2 = NA))
  zt <- z(x)[holdout]; zh <- z(Xc)[holdout]
  score_holdout(zt, zh, holdout, nrow(x))
}

# Main entry point. Sweeps m, picks CV-best by held-out RMSE, returns the uniform
# contract. complete_at(m) re-imputes (mean of m completions) at that m.
impute_mice <- function(x, ms = c(5L, 10L, 20L), seed = 1L) {
  set.seed(seed)
  holdout <- make_holdout(x, frac = 0.2)

  rmse_v <- numeric(length(ms)); r2_v <- numeric(length(ms))
  for (i in seq_along(ms)) {
    rr <- holdout_rmse_r2_mice(x, ms[i], holdout, seed)
    rmse_v[i] <- rr["rmse"]; r2_v[i] <- rr["r2"]
    cat(sprintf("  m %3d | RMSE %.4f | R2 %.3f\n", ms[i], rr["rmse"], rr["r2"]))
  }
  best_i <- which.min(replace(rmse_v, is.na(rmse_v), Inf))
  best_m <- ms[best_i]
  cat(sprintf("  >> CV-best m = %d (RMSE %.4f, R2 %.3f)\n",
              best_m, rmse_v[best_i], r2_v[best_i]))

  complete_at <- function(v) fit_mice(x, v, seed)
  list(M = complete_at(best_m),
       best_param = best_m, params = ms, curve = rmse_v, curve_r2 = r2_v,
       param_name = "m", metric_name = "Held-out RMSE",
       complete_at = complete_at)
}

# Seed-sweep sensitivity: repeated random holdouts, RMSE + R^2 distribution per m.
sensitivity_mice <- function(x, ms = c(5L, 10L, 20L), n_seeds = 20L,
                             holdout_frac = 0.2, nf = NA_integer_) {
  suppressMessages({ library(parallel); library(doParallel); library(foreach) })
  n_cores <- max(1L, detectCores() - 2L)
  cat(sprintf("  sensitivity using %d cores, %d seeds\n", n_cores, n_seeds))
  cl <- makeCluster(n_cores); registerDoParallel(cl)
  on.exit({ stopCluster(cl); registerDoSEQ() })

  res_list <- foreach(s = seq_len(n_seeds), .packages = c("mice", "psych"),
                      .export = c("fit_mice", "holdout_rmse_r2_mice",
                                  "make_holdout", "score_holdout", "BALANCE_HOLDOUT",
                                  "omega_h_only")) %dopar% {
    set.seed(s)
    holdout <- make_holdout(x, frac = holdout_frac)
    rmse <- numeric(length(ms)); r2 <- numeric(length(ms))
    for (i in seq_along(ms)) {
      rr <- holdout_rmse_r2_mice(x, ms[i], holdout, seed = s)
      rmse[i] <- rr["rmse"]; r2[i] <- rr["r2"]
    }
    oh <- NA_real_
    if (!is.na(nf)) {
      best_m <- ms[which.min(replace(rmse, is.na(rmse), Inf))]
      Mc <- tryCatch(fit_mice(x, best_m, seed = s), error = function(e) NULL)
      if (!is.null(Mc)) oh <- omega_h_only(Mc, nf)
    }
    list(rmse = rmse, r2 = r2, omega_h = oh)
  }
  rmse_mat <- do.call(rbind, lapply(res_list, `[[`, "rmse"))
  r2_mat   <- do.call(rbind, lapply(res_list, `[[`, "r2"))
  colnames(rmse_mat) <- colnames(r2_mat) <- paste0("m_", ms)
  best_ranks <- ms[apply(rmse_mat, 1, which.min)]
  list(rmse_mat = rmse_mat, r2_mat = r2_mat, best_ranks = best_ranks,
       omega_h = vapply(res_list, `[[`, numeric(1), "omega_h"),
       ranks = ms, param = "m")
}
