# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers for correlation-matrix imputers (softimpute_corr, and future
# ones). Unlike the cell-filling methods, these operate on the observed pairwise
# CORRELATION matrix rather than the raw dataset, so they lose scale and cannot
# reconstruct specific cells/rows. We reuse the OneSidedMC (Julia) strategy to
# work around this:
#
#   - compute the observed correlation matrix (NA where a pair is not computable
#     because it is never co-observed);
#   - impute those NA entries (softImpute, then symmetrize + nearPD to a valid
#     correlation matrix);
#   - test effectivity by predicting held-out CELLS via the conditional-Gaussian
#     (best-linear) predictor from the imputed correlation matrix;
#   - emit an n x p surrogate dataset whose covariance matches the imputed
#     correlation matrix, handed to the shared factoring like any other method.
#
# No min_n co-observation threshold here: the co-observation floor is already
# enforced upstream (densifier MIN_OBS = 2 and prep_matrix both require >= 2 obs
# per column). A pair that is never co-observed yields NA via cor(..., use =
# "pairwise.complete.obs") and that is exactly the imputation target.
# ─────────────────────────────────────────────────────────────────────────────

# Column mean/sd over OBSERVED cells (used to un-standardize the final surrogate
# back to the original column scale, mirroring OSMC's column_moments).
corr_column_moments <- function(x) {
  mu <- colMeans(x, na.rm = TRUE)
  sd <- apply(x, 2, stats::sd, na.rm = TRUE)
  sd[!is.finite(sd) | sd == 0] <- 1
  list(mu = mu, sd = sd)
}

# Standardize columns by the given moments (returns a matrix with NAs preserved).
corr_zscore <- function(x, mu, sd) {
  sweep(sweep(x, 2, mu, "-"), 2, sd, "/")
}

# Observed pairwise correlation matrix from a (standardized or raw) matrix with
# NAs. Non-finite entries (pairs never co-observed, or with <2 complete obs) are
# set to NA — those are what the imputer fills. Diagonal stays 1 (observed).
observed_corr <- function(z) {
  R <- suppressWarnings(cor(z, use = "pairwise.complete.obs"))
  R[!is.finite(R)] <- NA
  diag(R) <- 1
  R
}

# Symmetrize then project to the nearest valid correlation matrix (PD, unit
# diagonal). The final eigenvalue projection (do2eigen) guarantees the result is
# safely PD, not merely "close to" PD, so R[S,S] is invertible for any S.
symmetrize_nearpd <- function(R, maxit = 100L, conv.tol = 1e-8) {
  X_sym <- (R + t(R)) / 2
  res <- Matrix::nearPD(X_sym, corr = TRUE, keepDiag = FALSE,
                        do2eigen = TRUE, conv.tol = conv.tol, maxit = maxit)
  as.matrix(res$mat)
}

# Conditional-Gaussian (best-linear) prediction of held-out column `j` from a
# row's surviving observed cells S, using a valid correlation matrix R:
#   ẑ_j = R[j, S] · R[S, S]^-1 · z_S
# Because R is PD (post-nearPD), R[S,S] is a PD principal submatrix and always
# invertible — no rank floor needed (unlike OSMC's rank-deficient Θ̂). Empty S
# degenerates to the z-mean (0).
predict_cell_from_corr <- function(R, kept_cols, kept_vals, j) {
  k <- length(kept_cols)
  if (k == 0L) return(0)
  if (k == 1L) return(R[kept_cols, j] * kept_vals)
  coeff <- qr.solve(R[kept_cols, kept_cols, drop = FALSE], R[kept_cols, j])
  sum(coeff * kept_vals)
}

# Predict every held-out cell given the test-cell structure produced by
# make_holdout_cells. Returns the vector of predicted z-values aligned with the
# test cells (same order).
predict_cells_from_corr <- function(R, test_cells) {
  vapply(test_cells, function(tc)
    predict_cell_from_corr(R, tc$kept_cols, tc$kept_vals, tc$held_col),
    numeric(1))
}

# Build the held-out test-cell structure from an existing holdout mask (produced
# by make_holdout, column-stratified) PLUS the per-row surviving-cells info needed
# by the conditional-Gaussian predictor. A held-out cell carries:
#   kept_cols, kept_vals  -> the row's other OBSERVED (train) cells
#   held_col, held_val    -> the held-out cell's column and standardized value
# Returns a list of per-cell lists (held_val is the TRUE standardized value),
# aligned with `holdout` order. `z` is the fully-standardized matrix (train
# moments), so held_val = the true held-out z-score.
make_holdout_cells <- function(z, holdout) {
  if (length(holdout) == 0L) return(list())
  z_train <- z
  z_train[holdout] <- NA

  idx <- arrayInd(holdout, .dim = dim(z))  # (linear -> [row, col])
  held_rows <- idx[, 1L]
  held_cols <- idx[, 2L]

  lapply(seq_along(holdout), function(k) {
    i <- held_rows[k]; j <- held_cols[k]
    obs <- which(!is.na(z_train[i, ]))
    list(kept_cols = obs, kept_vals = as.numeric(z_train[i, obs]),
         held_col = j, held_val = as.numeric(z[i, j]))
  })
}

# Score held-out cells predicted from a correlation matrix. `test_cells` is the
# structure from make_holdout_cells; `zt`/`zh` are the true/predicted standardized
# held-out values (same length/order). Reuses score_holdout for the column-balanced
# RMSE/R². `holdout` are the linear indices for column attribution.
score_corr_holdout <- function(zt, zh, holdout, nrow_x, balance = BALANCE_HOLDOUT) {
  score_holdout(zt, zh, holdout, nrow_x, balance = balance)
}

# Synthesize an n x p surrogate with covariance ~ R (a valid correlation matrix),
# then un-standardize back to original column scale (× sd, + mu). Z ~ N(0, I_p)
# so cov(Z W') = W W' = R where W = Q sqrt(Λ). Mirrors OSMC synthesize_surrogate.
generate_surrogate <- function(R, n, mu, sd, seed = 1L) {
  set.seed(seed)
  Rm <- R
  Rm[!is.finite(Rm)] <- 0
  e <- eigen((Rm + t(Rm)) / 2, symmetric = TRUE)
  lam <- pmax(e$values, 0)                 # clamp tiny negatives from fp
  W <- e$vectors %*% diag(sqrt(lam), nrow = length(lam))
  p <- length(mu)
  Z <- matrix(rnorm(n * p), n, p)
  Xz <- Z %*% t(W)
  sweep(sweep(Xz, 2, sd, "*"), 2, mu, "+")
}

# Generic correlation-matrix imputation driver, shared by the simple corr
# imputers (optspace, usvt, future ones). `fit_fn(R_obs, param)` returns the
# completed correlation matrix for a single FIXED hyperparameter value (no
# sweep): we impute once, predict held-out cells via the conditional-Gaussian
# predictor, score column-balanced RMSE/R², refit on the FULL correlation matrix,
# and emit a covariance-matched surrogate as M.
run_corr_single <- function(x, fit_fn, param_name, param_value, seed = 1L) {
  set.seed(seed)
  holdout <- make_holdout(x, frac = 0.2)

  # standardize by TRAIN-cell moments (mask holdout first), like the other R
  # methods, so the held-out baseline is the train column mean = 0 in z-space.
  x_train <- x; x_train[holdout] <- NA
  mu  <- colMeans(x_train, na.rm = TRUE)
  sdv <- apply(x_train, 2, sd, na.rm = TRUE)
  sdv[!is.finite(sdv) | sdv == 0] <- 1
  z       <- corr_zscore(x, mu, sdv)
  z_train <- z; z_train[holdout] <- NA

  R_train <- observed_corr(z_train)
  R_full  <- observed_corr(z)

  test_cells <- make_holdout_cells(z, holdout)
  zt <- vapply(test_cells, function(tc) tc$held_val, numeric(1))
  nrow_x <- nrow(x)

  R_hat  <- fit_fn(R_train, param_value)
  R_corr <- symmetrize_nearpd(R_hat)
  zh     <- predict_cells_from_corr(R_corr, test_cells)
  sc     <- score_corr_holdout(zt, zh, holdout, nrow_x)
  rmse   <- unname(sc["rmse"]); r2 <- unname(sc["r2"])
  cat(sprintf("  %s=%s | RMSE %.4f | R2 %.3f\n", param_name,
              as.character(param_value), rmse, r2))

  # final surrogate on original scale (observed-cell moments), covariance = the
  # FULL (no-holdout) refit.
  R_final <- symmetrize_nearpd(fit_fn(R_full, param_value))
  mom <- corr_column_moments(x)
  M <- generate_surrogate(R_final, nrow(x), mom$mu, mom$sd, seed = seed)

  list(M = M,
       best_param = param_value, params = param_value,
       curve = rmse, curve_r2 = r2,
       param_name = param_name, metric_name = "Held-out RMSE")
}
