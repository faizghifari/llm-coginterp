# ─────────────────────────────────────────────────────────────────────────────
# Shared factoring module.
#
# Every imputation method (softimpute, iterativepca, onesidedmc) hands a single
# COMPLETED data matrix to this module so the factor analysis is identical across
# methods and the only thing that varies is the imputed/surrogate input. For
# onesidedmc the "completed" matrix is a synthetic surrogate whose covariance
# equals the recovered Theta-hat = V V' (see impute/onesidedmc) — psych never
# learns it is synthetic; it just factors a data matrix like any other.
#
# Factoring choices (held constant across all methods):
#   - minimum-residual factoring (fm = "minres")
#   - promax rotation when >1 factor, else none
#   - number of factors from Horn's parallel analysis (fa.parallel)
# ─────────────────────────────────────────────────────────────────────────────

suppressMessages(library(psych))
suppressMessages({ library(doParallel); library(foreach) })
suppressMessages({ library(CVXR); library(ggm) })

# Resolve this file's own directory so the sibling source + PA cache work from
# any CWD, whether run via Rscript (--file=) or source()'d (ofile in a frame).
.factor_dir <- local({
  f <- NA_character_
  # 1) source()'d: walk frames for the ofile set by source().
  for (i in seq_len(sys.nframe())) {
    of <- tryCatch(get("ofile", envir = sys.frame(i)), error = function(e) NULL)
    if (!is.null(of) && is.character(of) && nzchar(of)) { f <- of; break }
  }
  # 2) Rscript src/factor/factoring.R: --file= arg.
  if (is.na(f)) {
    a <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
    if (length(a) && nzchar(a[1])) f <- a[1]
  }
  if (is.na(f) || !nzchar(f)) normalizePath("src/factor", mustWork = FALSE)
  else dirname(normalizePath(f))
})
source(file.path(.factor_dir, "parallel_analysis.R"))
PA_CACHE_DIR <- file.path(.factor_dir, "pa_cache")  # anchor cache to factor/ dir

# Principal-axis EFA at a fixed factor count. Caps nf at ncol-1 and returns NULL
# (with a note) on failure so callers can degrade gracefully.
# Largest factor count the data can actually support: bounded by matrix dims AND
# the numeric rank of the correlation matrix (the imputed/surrogate matrices are
# often low-rank or p >> n, so cor(M) is singular and fa() would error if asked
# for more factors than there are non-trivial eigenvalues).
safe_nf <- function(M, nf) {
  R <- suppressWarnings(cor(M))
  R[!is.finite(R)] <- 0; diag(R) <- 1
  eig <- eigen(R, symmetric = TRUE, only.values = TRUE)$values
  rk <- sum(eig > 1e-8, na.rm = TRUE)
  max(2L, min(nf, ncol(M) - 1L, nrow(M) - 1L, rk - 1L))
}

# Minres at a fixed nf, with the psych-recommended escape for singular correlation
# matrices: if the default (SMC communalities) errors, retry with SMC = FALSE
# (unity diagonal). Only ERRORS trigger the fallback / NULL — psych's benign
# warnings (smc<0, singular pseudo-inverse) still return a usable fa object.
fa_try <- function(M, nf, n_obs = NA) {
  rot <- if (nf > 1) "promax" else "none"
  efa <- tryCatch(suppressWarnings(fa(M, nfactors = nf, n.obs = n_obs, fm = "minres", rotate = rot)),
                  error = function(e) NULL)
  if (!is.null(efa)) return(efa)
  cat(sprintf("    fa_try(nf=%d, rot=%s) failed, retrying SMC=FALSE\n", nf, rot))
  efa <- tryCatch(suppressWarnings(fa(M, nfactors = nf, n.obs = n_obs, fm = "minres", rotate = rot,
                               SMC = FALSE)),
           error = function(e) NULL)
  if (is.null(efa)) cat(sprintf("    fa_try(nf=%d, SMC=FALSE) also failed\n", nf))
  efa
}

run_efa <- function(M, nf) {
  nf <- safe_nf(M, nf)
  efa <- fa_try(M, nf)
  if (is.null(efa)) cat("    EFA failed at nf", nf, "\n")
  efa
}

# Cumulative variance explained from an fa object (handles the 1-factor case
# where only "Proportion Var" is present).
extract_variance <- function(efa) {
  if (is.null(efa)) return(NA)
  va <- efa$Vaccounted
  rn <- rownames(va)
  cum_row <- if ("Cumulative Var" %in% rn) "Cumulative Var" else "Proportion Var"
  va[cum_row, ncol(va)]
}

# Per-factor variance explained and inter-factor correlations from an fa object.
# Returns list(phi_avg, phi, var_factors, var_avg) suitable for db_insert_factoring.
efa_stats <- function(efa) {
  phi <- efa$Phi
  phi_avg <- if (!is.null(phi) && ncol(phi) > 1)
    mean(phi[upper.tri(phi)]) else NA_real_

  va <- efa$Vaccounted
  nf_e <- ncol(unclass(efa$loadings))
  pv <- if (!is.null(va) && "Proportion Var" %in% rownames(va))
    as.numeric(va["Proportion Var", 1:nf_e]) else numeric(0)
  var_avg <- if (length(pv) > 0) mean(pv) else NA_real_

  list(phi_avg = phi_avg, phi = phi, var_factors = pv, var_avg = var_avg)
}

# Pairwise-complete correlation from a sparse matrix (NAs allowed), PSD-smoothed.
# Returns the smoothed correlation R, the raw dataset's row count as the effective
# sample size (N), and eigenvalues of the RAW correlation (unsmoothed).
#
# We use nrow(M) directly rather than the harmonic mean of pairwise complete-case
# counts. The harmonic mean is unreliable with zero-imputation sparse data because
# a single column pair with zero overlapping non-NA observations sends 1/0 = Inf
# and collapses the entire effective sample size to zero. The raw row count is a
# stable, conservative estimate that avoids this singularity and is appropriate
# for no-imputation pairwise-complete correlation analysis where every row
# contributes at least some pairwise information.
prepare_raw_default <- function(M) {
  R <- cor(M, use = "pairwise.complete.obs")
  off_diag <- R[upper.tri(R)]
  mu <- mean(off_diag[is.finite(off_diag)])
  R[!is.finite(R)] <- mu
  diag(R) <- 1

  n_eff <- nrow(M)

  eig_raw <- sort(eigen(R, symmetric = TRUE, only.values = TRUE)$values,
                  decreasing = TRUE)
  R_smooth <- psych::cor.smooth(R)

  list(R = R_smooth, n_eff = as.integer(n_eff), eig_raw = eig_raw)
}

# Method "zeros": like "default" (pairwise-complete correlation) but fills
# unobserved / non-finite off-diagonal pairs with 0 instead of the average
# off-diagonal correlation. This treats absent co-observation as "no
# association" rather than imputing the typical pairwise correlation.
prepare_raw_zeros <- function(M) {
  R <- cor(M, use = "pairwise.complete.obs")
  R[!is.finite(R)] <- 0
  diag(R) <- 1

  n_eff <- nrow(M)

  eig_raw <- sort(eigen(R, symmetric = TRUE, only.values = TRUE)$values,
                  decreasing = TRUE)
  R_smooth <- psych::cor.smooth(R)

  list(R = R_smooth, n_eff = as.integer(n_eff), eig_raw = eig_raw)
}

# Shared by prepare_raw_cvxr / prepare_raw_ggm: pairwise-complete correlations,
# plus a mask of which entries are "trustworthy" (co-observed row count >=
# min_n) vs which should be left for the completion method to fill rather than
# taken at face value. Untrusted/non-finite entries are zeroed as placeholders
# only -- they're never read by the completion methods, only the mask is.
partial_cor_mask <- function(M, min_n = 4L) {
  present <- !is.na(M)
  n_pair <- crossprod(present)  # p x p co-observed-row counts, vectorized
  R <- suppressWarnings(cor(M, use = "pairwise.complete.obs"))

  mask <- (n_pair >= min_n) & is.finite(R)
  diag(mask) <- TRUE

  R[!is.finite(R)] <- 0
  diag(R) <- 1

  list(R = R, mask = mask, n_pair = n_pair)
}

# Direct SDP completion (Vandenberghe/Boyd/Wu MAXDET): maximize log-det(Sigma)
# subject to Sigma PSD and Sigma matching R on `mask` entries within a
# per-pair Fisher-z confidence band (diagonal matched exactly, since it must
# stay 1). A single flat tolerance can't serve both a pair estimated from
# n=10 and one from n=200 -- their sampling uncertainty differs by a lot --
# so the band is scaled to each pair's actual n via the Fisher-z SE
# (1/sqrt(n-3)). Even so, no PSD matrix may exist within band for badly
# inconsistent pairs; that's a genuine infeasibility, not a bug. No
# decomposability assumption, so this works regardless of whether the
# observed pattern is chordal. Used by prepare_raw_cvxr only -- prepare_raw_ggm
# has no fallback.
cvxr_maxdet_complete <- function(R, mask, n_pair, ci_mult = 2) {
  p <- ncol(R)
  Sigma_var <- CVXR::Variable(c(p, p), symmetric = TRUE)

  obs <- which(upper.tri(mask, diag = TRUE) & mask, arr.ind = TRUE)
  constraints <- list(Sigma_var %>>% 0)
  for (k in seq_len(nrow(obs))) {
    i <- obs[k, 1]; j <- obs[k, 2]
    if (i == j) {
      constraints[[length(constraints) + 1]] <- Sigma_var[i, j] == R[i, j]
      next
    }
    se <- 1 / sqrt(max(n_pair[i, j] - 3, 1))
    z  <- atanh(min(max(R[i, j], -0.999), 0.999))
    lo <- tanh(z - ci_mult * se)
    hi <- tanh(z + ci_mult * se)
    constraints[[length(constraints) + 1]] <- Sigma_var[i, j] >= lo
    constraints[[length(constraints) + 1]] <- Sigma_var[i, j] <= hi
  }

  prob <- CVXR::Problem(CVXR::Maximize(CVXR::log_det(Sigma_var)), constraints)
  CVXR::psolve(prob, solver = "SCS")
  st <- CVXR::status(prob)
  if (!st %in% c("optimal", "optimal_inaccurate"))
    stop("cvxr_maxdet_complete: solver status = ", st,
         " (raise ci_mult -- the trusted pairwise correlations may not be jointly PSD-consistent even at their sampling uncertainty)")

  out <- CVXR::value(Sigma_var)
  dimnames(out) <- dimnames(R)
  out
}

# Method "cvxr": pairwise-complete correlations trusted where co-observed
# n >= min_n; everything else filled via MAXDET completion instead of
# mean-imputation. cor.smooth is still applied as a cheap safety net against
# solver numerical slack, not because the completion is expected to be
# non-PSD.
prepare_raw_cvxr <- function(M, min_n = 10L, ci_mult = 2) {
  pc <- partial_cor_mask(M, min_n = min_n)
  R_complete <- cvxr_maxdet_complete(pc$R, pc$mask, pc$n_pair, ci_mult = ci_mult)

  n_eff <- nrow(M)
  eig_raw <- sort(eigen(R_complete, symmetric = TRUE, only.values = TRUE)$values,
                  decreasing = TRUE)
  R_smooth <- psych::cor.smooth(R_complete)

  list(R = R_smooth, n_eff = as.integer(n_eff), eig_raw = eig_raw)
}

# Method "ggm": same trusted/untrusted split as prepare_raw_cvxr, but
# completion via ggm::fitConGraph (graphical-model MLE, iterative conditional
# fitting -- handles non-chordal patterns, not just decomposable ones). No
# fallback -- if fitConGraph fails, this method fails, full stop.
prepare_raw_ggm <- function(M, min_n = 10L) {
  pc <- partial_cor_mask(M, min_n = min_n)
  amat <- pc$mask
  diag(amat) <- FALSE

  fit <- ggm::fitConGraph(amat, pc$R, n = nrow(M))
  R_complete <- fit$Shat

  n_eff <- nrow(M)
  eig_raw <- sort(eigen(R_complete, symmetric = TRUE, only.values = TRUE)$values,
                  decreasing = TRUE)
  R_smooth <- psych::cor.smooth(R_complete)

  list(R = R_smooth, n_eff = as.integer(n_eff), eig_raw = eig_raw)
}

# Factoring of sparse data via pairwise-complete correlation + PSD smoothing.
# PA uses raw eigenvalues vs cutoffs at effective N. Returns the same shape
# as factor_matrix plus R (smoothed correlation) and n_eff for downstream use.
factor_raw <- function(M, pa_iter = 100L, pa_quantile = 0.95,
                       method = c("default", "zeros", "cvxr", "ggm"), min_n = 10L) {
  method <- match.arg(method)
  prep <- switch(method,
    default = prepare_raw_default(M),
    zeros   = prepare_raw_zeros(M),
    cvxr    = prepare_raw_cvxr(M, min_n = min_n),
    ggm     = prepare_raw_ggm(M, min_n = min_n))

  cut <- pa_cutoffs(prep$n_eff, ncol(M), n.iter = pa_iter,
                    quantile = pa_quantile)
  nf_req <- max(2L, sum(prep$eig_raw > cut, na.rm = TRUE))
  nf_req <- min(nf_req, 20L)

  rk <- sum(prep$eig_raw > 1e-8, na.rm = TRUE)
  nf <- max(2L, min(nf_req, ncol(M) - 1L, prep$n_eff - 1L, rk - 1L))
  cat("  raw pa$nf =", nf_req, " capped nf =", nf, "\n")

  efa <- NULL
  for (k in seq.int(nf, 2L)) {
    cat("  trying nf =", k, "...\n"); t0 <- Sys.time()
    efa <- fa_try(prep$R, k, n_obs = prep$n_eff)
    cat("  nf =", k, "took", Sys.time() - t0, "\n")
    if (!is.null(efa)) { nf <- k; break }
  }
  if (is.null(efa))
    stop("raw factoring failed at every nf down to 2")

  list(efa = efa, eig = prep$eig_raw, cutoffs = cut,
       nf = ncol(unclass(efa$loadings)),
       R = prep$R, n_eff = prep$n_eff)
}

# One-shot factoring of a completed matrix: cached parallel analysis picks the
# factor count, then minres + promax.
factor_matrix <- function(M, pa_iter = 100L, pa_quantile = 0.95,
                          nf_override = NULL) {
  pa <- choose_nfactors(M, n.iter = pa_iter, quantile = pa_quantile)
  nf_req <- if (!is.null(nf_override)) nf_override else pa$nf
  nf_req <- min(nf_req, 20L)
  nf <- safe_nf(M, nf_req)
  cat("  pa$nf =", pa$nf, " capped nf =", nf, "\n")

  efa <- NULL
  for (k in seq.int(nf, 1L)) {
    cat("  trying nf =", k, "...\n"); t0 <- Sys.time()
    efa <- fa_try(M, k)
    cat("  nf =", k, "took", Sys.time() - t0, "\n")
    if (!is.null(efa)) { nf <- k; break }
  }
  if (is.null(efa))
    stop("factoring failed at every nf down to 1 (matrix too degenerate)")

  list(efa = efa, eig = pa$observed, cutoffs = pa$cutoffs,
       nf = ncol(unclass(efa$loadings)))
}

# Higher-order factor analysis on a completed matrix M with `nf` first-order
# factors. Schmid-Leiman bifactor via psych::omega: every benchmark loads
# directly on a general factor g + its group factor; yields omega_h (proportion
# of variance from g) and omega_total.
higher_order <- function(M, nf, n_obs = NA) {
  out <- list(bifactor_loadings = NULL, omega_h = NA_real_,
              omega_h_asymptotic = NA_real_, omega_total = NA_real_,
              omega_group = NULL, nf = nf)

  om <- tryCatch(
    suppressWarnings(psych::omega(M, nfactors = nf, n.obs = n_obs, fm = "minres",
                                  flip = FALSE, plot = FALSE)),
    error = function(e) NULL)
  if (!is.null(om)) {
    out$bifactor_loadings <- unclass(om$schmid$sl)
    out$omega_h <- tryCatch(as.numeric(om$omega_h), error = function(e) NA_real_)
    out$omega_h_asymptotic <- tryCatch(as.numeric(om$omega.lim), error = function(e) NA_real_)
    out$omega_total <- tryCatch(as.numeric(om$omega.tot), error = function(e) NA_real_)
    out$omega_group <- tryCatch(as.matrix(om$omega.group), error = function(e) NULL)
  }
  out
}

# omega_h only (cheap): for the sensitivity seed-sweep. Returns NA only on failure.
omega_h_only <- function(M, nf) {
  om <- tryCatch(
    suppressWarnings(psych::omega(M, nfactors = nf, fm = "minres", flip = FALSE,
                                  plot = FALSE)),
    error = function(e) NULL)
  if (is.null(om)) NA_real_ else tryCatch(as.numeric(om$omega_h),
                                          error = function(e) NA_real_)
}

# Leave-One-Covariate-Out delta omega_h: for each benchmark (column index i),
# drop the i-th row and column from the correlation matrix R, run a promax EFA
# followed by a bifactor omega_h on the reduced matrix, and return
# omega_h_full - omega_h_{-i} for every benchmark. Workers = detectCores() - 2
# (embarrassingly parallel).
loco_delta <- function(R, n_obs, nf) {
  p <- ncol(R)

  omega_full <- tryCatch(
    suppressWarnings(psych::omega(R, nfactors = nf, n.obs = n_obs,
                                   fm = "minres", flip = FALSE, plot = FALSE)$omega_h),
    error = function(e) NA_real_)
  if (is.na(omega_full)) return(rep(NA_real_, p))

  n_workers <- max(1L, detectCores() - 2L)
  cl <- makeCluster(n_workers)
  registerDoParallel(cl)
  on.exit({ stopCluster(cl); registerDoSEQ() })

  omega_i <- foreach(i = seq_len(p), .packages = "psych",
                     .export = "fa_try",
                     .combine = c) %dopar% {
    R_i <- R[-i, -i, drop = FALSE]
    fa_try(R_i, nf, n_obs = n_obs)
    tryCatch(
      suppressWarnings(psych::omega(R_i, nfactors = nf, n.obs = n_obs,
                                     fm = "minres", flip = FALSE, plot = FALSE)$omega_h),
      error = function(e) NA_real_)
  }

  omega_full - unlist(omega_i)
}

# Generic loading-matrix -> markdown table (rowname column + numeric cols).
# Loadings with |value| >= 0.4 are bolded. Rows are sorted by primary factor
# assignment then by absolute loading descending.
matrix_to_markdown <- function(L, path, rowname = "row", note = "") {
  ord <- order(apply(abs(L), 1, which.max), -apply(abs(L), 1, max))
  L <- L[ord, , drop = FALSE]
  disp <- formatC(L, format = "f", digits = 2)
  bold <- abs(L) >= 0.4
  disp[bold] <- paste0("**", disp[bold], "**")
  header <- paste0("| ", rowname, " | ", paste(colnames(L), collapse = " | "), " |")
  sep    <- paste0("|", paste(rep("---", ncol(L) + 1), collapse = "|"), "|")
  rows   <- vapply(seq_len(nrow(L)), function(i)
    paste0("| ", rownames(L)[i], " | ", paste(disp[i, ], collapse = " | "), " |"),
    character(1))
  writeLines(c(if (nzchar(note)) paste0(note, "\n") else character(0),
               header, sep, rows), path)
  cat("  wrote", path, "\n")
}

# Write bifactor outputs: Schmid-Leiman loadings (CSV + MD), scalar omega
# coefficients, and per-group omega_hs breakdown.
write_higher_order <- function(ho, bifactor_csv, scalar_csv, group_csv,
                               bifactor_md = NULL) {
  if (!is.null(ho$bifactor_loadings)) {
    L <- ho$bifactor_loadings
    write.csv(data.frame(benchmark = rownames(L), L, check.names = FALSE),
              bifactor_csv, row.names = FALSE)
    cat("  wrote", bifactor_csv, "\n")
    if (!is.null(bifactor_md))
      matrix_to_markdown(L, bifactor_md, rowname = "benchmark",
                         note = "Bifactor (Schmid-Leiman) loadings: g + group factors; |.|>=0.4 bolded.")
  }
  write.csv(data.frame(omega_h = ho$omega_h,
                       omega_h_asymptotic = ho$omega_h_asymptotic,
                       omega_total = ho$omega_total, n_first_order = ho$nf),
            scalar_csv, row.names = FALSE)
  cat("  wrote", scalar_csv, "\n")
  if (!is.null(ho$omega_group)) {
    g <- ho$omega_group
    write.csv(data.frame(factor = rownames(g), g, check.names = FALSE),
              group_csv, row.names = FALSE)
    cat("  wrote", group_csv, "\n")
  }
}
