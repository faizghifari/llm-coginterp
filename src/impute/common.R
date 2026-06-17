# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers for the R imputation methods (softimpute, iterativepca).
#
# prep_matrix is identical across methods so both see the same input matrix and
# the imputer is the only difference. Output-path helpers keep every method
# writing to data/imputed/<method>/<densifier>/<strategy>/ consistently.
# ─────────────────────────────────────────────────────────────────────────────

# Read a densified model x benchmark table into a numeric matrix + keys, dropping
# columns with fewer than `min_obs` observations (correlation/imputation can't
# use them).
prep_matrix <- function(path, min_obs = 2L) {
  df <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  num_cols <- setdiff(names(df), "collapse_key")
  x <- as.matrix(df[, num_cols, drop = FALSE])
  storage.mode(x) <- "double"
  sparse_cols <- colSums(!is.na(x)) < min_obs
  if (any(sparse_cols)) {
    cat("  dropping", sum(sparse_cols), "cols with <", min_obs, "obs:",
        paste(names(which(sparse_cols)), collapse = ", "), "\n")
    x <- x[, !sparse_cols, drop = FALSE]
  }
  list(x = x, keys = df[["collapse_key"]])
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
