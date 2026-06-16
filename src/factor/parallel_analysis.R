# ─────────────────────────────────────────────────────────────────────────────
# Parallel analysis, decomposed and cached.
#
# Horn's PA decides the factor count by comparing the OBSERVED eigenvalues to a
# RANDOM baseline. The random baseline (the per-position eigenvalue cutoff)
# depends only on the matrix SHAPE (n rows, p cols) and the PA settings — not on
# the data values. So we compute it once per (n, p, n.iter, quantile) and cache
# it as JSON; every dataset of that shape reuses the cutoffs.
#
# We use the PC (principal-component) flavor: cutoffs are quantiles of the
# eigenvalues of random N(0,1) data's CORRELATION matrix, and the observed side
# uses the eigenvalues of the imputed data's correlation matrix. Factor count =
# number of observed eigenvalues exceeding the cutoff at the same position.
# ─────────────────────────────────────────────────────────────────────────────

suppressMessages(library(jsonlite))

PA_CACHE_DIR <- file.path("factor", "pa_cache")

# Random-baseline PC eigenvalue cutoffs for a given shape. Generates `n.iter`
# random n x p Gaussian matrices, takes the descending eigenvalues of each one's
# correlation matrix, and returns the per-position `quantile`. Cached by shape.
pa_cutoffs <- function(n, p, n.iter = 100L, quantile = 0.95, seed = 1L,
                       cache_dir = PA_CACHE_DIR, verbose = TRUE) {
  dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
  key <- sprintf("n%d_p%d_iter%d_q%02d.json",
                 n, p, n.iter, round(quantile * 100))
  fp <- file.path(cache_dir, key)
  if (file.exists(fp)) {
    if (verbose) cat("  PA cutoffs (cached):", fp, "\n")
    return(as.numeric(fromJSON(fp)$cutoffs))
  }

  if (verbose)
    cat(sprintf("  PA cutoffs (computing %d iters for %dx%d)...\n", n.iter, n, p))
  set.seed(seed)
  eig_mat <- matrix(NA_real_, nrow = n.iter, ncol = p)
  for (i in seq_len(n.iter)) {
    R <- cor(matrix(rnorm(n * p), n, p))
    eig_mat[i, ] <- sort(eigen(R, symmetric = TRUE, only.values = TRUE)$values,
                         decreasing = TRUE)
  }
  cutoffs <- apply(eig_mat, 2, stats::quantile, probs = quantile, names = FALSE)

  write_json(list(n = n, p = p, n.iter = n.iter, quantile = quantile,
                  cutoffs = cutoffs), fp, auto_unbox = TRUE, digits = 8)
  if (verbose) cat("  wrote", fp, "\n")
  cutoffs
}

# Observed PC eigenvalues (descending) of a completed matrix's correlation.
# Constant columns make cor() return NA, so coerce any NA correlation to 0 (an
# uncorrelated, zero-variance dimension) before the eigendecomposition.
observed_pc_eigenvalues <- function(M) {
  R <- suppressWarnings(cor(M))
  R[!is.finite(R)] <- 0
  diag(R) <- 1
  sort(eigen(R, symmetric = TRUE, only.values = TRUE)$values, decreasing = TRUE)
}

# Horn factor count: number of observed eigenvalues above the random cutoff at
# the same position. Always >= 1. NA comparisons (degenerate eigenvalues) are
# treated as "not exceeding" so nf never becomes NA.
choose_nfactors <- function(M, n.iter = 100L, quantile = 0.95,
                            cache_dir = PA_CACHE_DIR, verbose = TRUE) {
  n <- nrow(M); p <- ncol(M)
  obs <- observed_pc_eigenvalues(M)
  cut <- pa_cutoffs(n, p, n.iter, quantile, cache_dir = cache_dir,
                    verbose = verbose)
  nf <- max(1L, sum(obs > cut, na.rm = TRUE))
  list(nf = nf, observed = obs, cutoffs = cut)
}
