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
