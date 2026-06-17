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
