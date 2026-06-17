# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers for the R imputation methods (softimpute, iterativepca).
#
# prep_matrix is identical across methods so both see the same input matrix and
# the imputer is the only difference. Output-path helpers keep every method
# writing to data/imputed/<method>/<densifier>/<strategy>/ consistently.
# ─────────────────────────────────────────────────────────────────────────────

# Read a densified model x benchmark table into a numeric matrix + keys. Drops
# columns that are useless for correlation/imputation:
#   - fewer than `min_obs` observed values, AND
#   - zero variance among observed values (all observed entries identical) —
#     these carry no correlational signal AND break softImpute's biScale
#     (col.scale divides by a zero scale -> NaN -> "missing value where
#     TRUE/FALSE needed" in its convergence loop).
prep_matrix <- function(path, min_obs = 2L) {
  df <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  num_cols <- setdiff(names(df), "collapse_key")
  x <- as.matrix(df[, num_cols, drop = FALSE])
  storage.mode(x) <- "double"

  sparse_cols <- colSums(!is.na(x)) < min_obs
  col_sd <- apply(x, 2, function(col) stats::sd(col[!is.na(col)]))
  const_cols <- is.na(col_sd) | col_sd == 0
  drop_cols <- sparse_cols | const_cols
  if (any(drop_cols)) {
    cat("  dropping", sum(drop_cols), "cols",
        sprintf("(%d <%d obs, %d zero-variance):", sum(sparse_cols), min_obs,
                sum(const_cols & !sparse_cols)),
        paste(names(which(drop_cols)), collapse = ", "), "\n")
    x <- x[, !drop_cols, drop = FALSE]
  }
  list(x = x, keys = df[["collapse_key"]])
}

# Column-stratified holdout, SHARED by every method so all use the same masking.
# Samples ~frac of observed cells, but never masks so many in a column that fewer
# than min_keep remain in TRAINING. min_keep = 2 matches the pipeline-wide floor
# (densifier MIN_OBS, prep_matrix, OSMC all require >=2 obs/col); leaving a column
# with <2 training cells breaks softImpute's biScale and starves the imputers.
# Returns linear indices into the matrix (column-major), like which(is.na(x)).
make_holdout <- function(x, frac = 0.2, min_keep = 2L) {
  nr <- nrow(x)
  holdout <- integer(0)
  for (j in seq_len(ncol(x))) {
    rows_obs <- which(!is.na(x[, j]))
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

# Whether held-out RMSE/R^2 is column-balanced (default) or cell-weighted. With
# cell-weighting (the naive global mean), high-frequency columns donate far more
# held-out cells and dominate the metric — so RMSE reflects the dense "famous"
# core and is largely insensitive to densification. Column-balancing averages
# the per-column RMSE/R^2 (equal weight per benchmark) to remove that bias.
# Set BALANCE_HOLDOUT <- FALSE (via --no-balance) for the old cell-weighted score.
BALANCE_HOLDOUT <- TRUE

# Score held-out cells given true (zt) and predicted (zh) standardized values and
# the held-out linear indices (column-major). Returns c(rmse, r2). R^2 baseline
# is the train column mean, which is 0 in z-space, so per-cell baseline = zt^2.
#
# Both modes derive R^2 from the SAME aggregated MSE/baseline as the RMSE (a
# single final ratio), never by averaging per-column R^2 — averaging R^2 over
# columns is unstable: a thin column with a tiny baseline yields R^2 of -10..-50
# and a few of those wreck the mean.
#   cell-weighted: aggregate over all held-out cells equally.
#   balanced:      aggregate each column's MEAN squared error / MEAN baseline,
#                  then average those across columns (equal weight per column).
score_holdout <- function(zt, zh, holdout, nrow_x, balance = BALANCE_HOLDOUT) {
  resid2 <- (zh - zt)^2
  base2  <- zt^2
  if (!balance) {
    return(c(rmse = sqrt(mean(resid2)),
             r2 = 1 - sum(resid2) / sum(base2)))
  }
  col <- ((holdout - 1L) %/% nrow_x) + 1L           # column of each held-out cell
  # RMSE: mean of per-column RMSE (unchanged).
  rmse <- mean(sqrt(tapply(resid2, col, mean)), na.rm = TRUE)
  # R^2: single pooled ratio of column-balanced MSE / baseline-MSE — NOT a mean
  # of per-column R^2 (which blows up to -10..-50 on thin columns).
  mse  <- mean(tapply(resid2, col, mean))
  base <- mean(tapply(base2,  col, mean))
  c(rmse = rmse, r2 = if (base > 0) 1 - mse / base else NA_real_)
}

# data/imputed/<method>/<densifier>/<strategy>/  — created if missing.
imputed_dir <- function(method, densifier, strategy,
                        root = "data/imputed") {
  d <- file.path(root, method, densifier, strategy)
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
  d
}

# Write a completed matrix (+ collapse_key) to the standard location.
write_completed <- function(out_dir, keys, M) {
  p <- file.path(out_dir, "imputed_model_benchmark_table.csv")
  write.csv(data.frame(collapse_key = keys, M, check.names = FALSE),
            p, row.names = FALSE)
  cat("  wrote", p, "\n")
  p
}
