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
fa_try <- function(M, nf) {
  rot <- if (nf > 1) "promax" else "none"
  efa <- tryCatch(suppressWarnings(fa(M, nfactors = nf, fm = "minres", rotate = rot)),
                  error = function(e) NULL)
  if (!is.null(efa)) return(efa)
  cat(sprintf("    fa_try(nf=%d, rot=%s) failed, retrying SMC=FALSE\n", nf, rot))
  efa <- tryCatch(suppressWarnings(fa(M, nfactors = nf, fm = "minres", rotate = rot,
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
higher_order <- function(M, nf) {
  out <- list(bifactor_loadings = NULL, omega_h = NA_real_,
              omega_h_asymptotic = NA_real_, omega_total = NA_real_,
              omega_group = NULL, nf = nf)

  om <- tryCatch(
    suppressWarnings(psych::omega(M, nfactors = nf, fm = "minres", flip = FALSE,
                                  plot = FALSE)),
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
