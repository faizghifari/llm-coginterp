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
#   - principal-axis factoring (fm = "pa")
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
  max(1L, min(nf, ncol(M) - 1L, nrow(M) - 1L, rk - 1L))
}

# PAF at a fixed nf, with the psych-recommended escape for singular correlation
# matrices: if the default (SMC communalities) errors, retry with SMC = FALSE
# (unity diagonal). Only ERRORS trigger the fallback / NULL — psych's benign
# warnings (smc<0, singular pseudo-inverse) still return a usable fa object.
fa_try <- function(M, nf) {
  rot <- if (nf > 1) "promax" else "none"
  efa <- tryCatch(suppressWarnings(fa(M, nfactors = nf, fm = "pa", rotate = rot)),
                  error = function(e) NULL)
  if (!is.null(efa)) return(efa)
  tryCatch(suppressWarnings(fa(M, nfactors = nf, fm = "pa", rotate = rot,
                               SMC = FALSE)),
           error = function(e) NULL)
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
# factor count, then PAF + promax. Returns everything downstream plots/reports
# need (the observed eigenvalues and random cutoffs go to the scree plot).
factor_matrix <- function(M, pa_iter = 100L, pa_quantile = 0.95,
                          nf_override = NULL) {
  pa <- choose_nfactors(M, n.iter = pa_iter, quantile = pa_quantile)
  nf_req <- if (!is.null(nf_override)) nf_override else pa$nf
  nf <- safe_nf(M, nf_req)

  # Singular / low-rank correlation matrices (p >> n, or low-rank surrogates)
  # can still make PAF fail; step down the factor count until it succeeds
  # (fa_try also retries each k with SMC = FALSE).
  efa <- NULL
  for (k in seq.int(nf, 1L)) {
    efa <- fa_try(M, k)
    if (!is.null(efa)) { nf <- k; break }
  }
  if (is.null(efa))
    stop("factoring failed at every nf down to 1 (matrix too degenerate)")

  list(efa = efa, eig = pa$observed, cutoffs = pa$cutoffs,
       nf = ncol(unclass(efa$loadings)))
}

# Higher-order factor analysis on a completed matrix M with `nf` first-order
# factors. Two complementary views of the SAME hierarchy:
#   - second-order: FA of the first-order promax factor-correlation matrix (Phi);
#     loadings are of the first-order factors on a single general factor.
#   - bifactor (Schmid-Leiman via psych::omega): re-expresses the hierarchy so
#     every benchmark loads directly on a general factor g + its group factor;
#     also yields omega_h (proportion of variance from g) and omega_total.
#
# Requires >= 3 first-order factors (2nd-order is unidentified below that) and a
# meaningfully non-identity Phi (no correlated first-order factors -> no general
# factor to find). Returns NULL components that can't be computed.
higher_order <- function(M, nf) {
  out <- list(second = NULL, second_loadings = NULL,
              bifactor_loadings = NULL, omega_h = NA_real_,
              omega_h_asymptotic = NA_real_, omega_total = NA_real_,
              omega_group = NULL, nf = nf)

  # ── second-order: factor the first-order factor-correlation matrix ──────────
  efa1 <- fa_try(M, nf)
  if (!is.null(efa1) && !is.null(efa1$Phi)) {
    so <- tryCatch(
      suppressWarnings(fa(efa1$Phi, nfactors = 1, fm = "pa",
                          n.obs = nrow(M), rotate = "none")),
      error = function(e) NULL)
    if (!is.null(so)) {
      out$second <- so
      out$second_loadings <- unclass(so$loadings)  # first-order factors x g
    }
  }

  # ── bifactor / Schmid-Leiman via psych::omega ───────────────────────────────
  om <- tryCatch(
    suppressWarnings(psych::omega(M, nfactors = nf, fm = "pa", flip = FALSE,
                                  plot = FALSE)),
    error = function(e) NULL)
  if (!is.null(om)) {
    out$bifactor_loadings <- unclass(om$schmid$sl)   # benchmarks x (g + groups + h2/u2/p2)
    out$omega_h <- tryCatch(as.numeric(om$omega_h), error = function(e) NA_real_)
    out$omega_h_asymptotic <- tryCatch(as.numeric(om$omega.lim), error = function(e) NA_real_)
    out$omega_total <- tryCatch(as.numeric(om$omega.tot), error = function(e) NA_real_)
    # per-group breakdown: rows g/F1*/F2*..., cols total/general/group; the
    # `group` column is omega_hs per factor (needed to interpret omega_h).
    out$omega_group <- tryCatch(as.matrix(om$omega.group), error = function(e) NULL)
  }
  out
}

# omega_h only (cheap): for the sensitivity seed-sweep. Returns NA only on failure.
omega_h_only <- function(M, nf) {
  om <- tryCatch(
    suppressWarnings(psych::omega(M, nfactors = nf, fm = "pa", flip = FALSE,
                                  plot = FALSE)),
    error = function(e) NULL)
  if (is.null(om)) NA_real_ else tryCatch(as.numeric(om$omega_h),
                                          error = function(e) NA_real_)
}

# Loadings -> tidy CSV (benchmark x factor), kept verbatim from the old report.R.
write_loadings_csv <- function(efa, path) {
  L <- unclass(efa$loadings)
  write.csv(data.frame(benchmark = rownames(L), L, check.names = FALSE),
            path, row.names = FALSE)
  cat("  wrote", path, "\n")
}

# Loadings -> human-readable markdown table, blanking |loading| below cutoff.
write_loadings_markdown <- function(efa, path, cutoff = 0.30) {
  L <- unclass(efa$loadings)
  ord <- order(apply(abs(L), 1, which.max), -apply(abs(L), 1, max))
  L <- L[ord, , drop = FALSE]
  disp <- formatC(L, format = "f", digits = 2)
  disp[abs(L) < cutoff] <- ""
  header <- paste0("| benchmark | ", paste(colnames(L), collapse = " | "), " |")
  sep    <- paste0("|", paste(rep("---", ncol(L) + 1), collapse = "|"), "|")
  rows   <- vapply(seq_len(nrow(L)), function(i)
    paste0("| ", rownames(L)[i], " | ", paste(disp[i, ], collapse = " | "), " |"),
    character(1))
  writeLines(c(paste0("Loadings blanked below |", cutoff, "|; PAF + promax.\n"),
               header, sep, rows), path)
  cat("  wrote", path, "\n")
}

# Generic loading-matrix -> markdown table (rowname column + numeric cols),
# blanking |value| below cutoff. Reused for first- and higher-order tables.
matrix_to_markdown <- function(L, path, rowname = "row", cutoff = 0.30,
                               note = "") {
  disp <- formatC(L, format = "f", digits = 2)
  disp[abs(L) < cutoff] <- ""
  header <- paste0("| ", rowname, " | ", paste(colnames(L), collapse = " | "), " |")
  sep    <- paste0("|", paste(rep("---", ncol(L) + 1), collapse = "|"), "|")
  rows   <- vapply(seq_len(nrow(L)), function(i)
    paste0("| ", rownames(L)[i], " | ", paste(disp[i, ], collapse = " | "), " |"),
    character(1))
  writeLines(c(if (nzchar(note)) paste0(note, "\n") else character(0),
               header, sep, rows), path)
  cat("  wrote", path, "\n")
}

# Write higher-order outputs from higher_order(): second-order loadings, bifactor
# (Schmid-Leiman) loadings, the scalar omega coefficients, and the per-group
# omega breakdown (omega_hs). Each loadings table is written as both CSV and MD
# (IDE-friendly). Skips any component that is NULL.
write_higher_order <- function(ho, second_csv, bifactor_csv, scalar_csv,
                               group_csv, second_md = NULL, bifactor_md = NULL) {
  if (!is.null(ho$second_loadings)) {
    L <- ho$second_loadings
    write.csv(data.frame(first_order_factor = rownames(L), L, check.names = FALSE),
              second_csv, row.names = FALSE)
    cat("  wrote", second_csv, "\n")
    if (!is.null(second_md))
      matrix_to_markdown(L, second_md, rowname = "first_order_factor",
                         note = "Second-order loadings (first-order factors -> g); |.|<0.3 blanked.")
  }
  if (!is.null(ho$bifactor_loadings)) {
    L <- ho$bifactor_loadings
    write.csv(data.frame(benchmark = rownames(L), L, check.names = FALSE),
              bifactor_csv, row.names = FALSE)
    cat("  wrote", bifactor_csv, "\n")
    if (!is.null(bifactor_md))
      matrix_to_markdown(L, bifactor_md, rowname = "benchmark",
                         note = "Bifactor (Schmid-Leiman) loadings: g + group factors; |.|<0.3 blanked.")
  }
  # scalar coefficients
  write.csv(data.frame(omega_h = ho$omega_h,
                       omega_h_asymptotic = ho$omega_h_asymptotic,
                       omega_total = ho$omega_total, n_first_order = ho$nf),
            scalar_csv, row.names = FALSE)
  cat("  wrote", scalar_csv, "\n")
  # per-group breakdown (omega_hs is the `group` column)
  if (!is.null(ho$omega_group)) {
    g <- ho$omega_group
    write.csv(data.frame(factor = rownames(g), g, check.names = FALSE),
              group_csv, row.names = FALSE)
    cat("  wrote", group_csv, "\n")
  }
}

# Shared scree + parallel-analysis plot (identical across all methods). Takes
# the observed eigenvalues and the cached random cutoffs from factor_matrix().
plot_scree <- function(eig, cutoffs, path, max_k = 15L, title = "") {
  ne <- min(length(eig), max_k + 5L)
  png(path, width = 800, height = 600, res = 120)
  par(mar = c(4.5, 4.5, 3, 1))
  plot(seq_len(ne), eig[seq_len(ne)], type = "b", pch = 19, lwd = 2,
       xlab = "Component", ylab = "Eigenvalue",
       main = paste0("Scree + parallel analysis: ", title))
  if (!is.null(cutoffs))
    lines(seq_len(min(ne, length(cutoffs))), cutoffs[seq_len(min(ne, length(cutoffs)))],
          col = "red", lty = 2, lwd = 2)
  legend("topright", c("observed", "random cutoff (PA)"),
         col = c("black", "red"), lty = c(1, 2), lwd = 2, bty = "n")
  dev.off()
  cat("  wrote", path, "\n")
}
