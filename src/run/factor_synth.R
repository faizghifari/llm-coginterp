# ─────────────────────────────────────────────────────────────────────────────
# Factor-level recovery on the synthetic MNAR corpus (scripts/make_mnar.py).
#
# impute.R --synth imputes every data/synth/<cell>/<mask>/s0 root and scores
# CELL recovery; this script closes the loop at the FACTOR level, which is what
# the analysis actually reports: it runs the exact factor.R recipe (Horn PA
# factor count -> minres/promax -> Schmid-Leiman via psych::omega, flip=FALSE)
# on each completed synthetic matrix, plus an ORACLE arm that factors the
# complete, unmasked truth matrix (separates EFA/SL estimator bias from
# missingness damage). Scoring against the planted truth (omega_h, factor
# count, g-loading congruence, domain recovery) is done in Python by
# scripts/score_synth_factors.py.
#
# No R² gate here: the gate is applied at scoring time (imputation_synth R²),
# so the effect of the gate itself can be reported.
#
# Run (inside the container, after `impute.R --synth --density <d>`):
#   Rscript src/run/factor_synth.R --density 0.1 --method softimpute [--method missforest ...]
#   Rscript src/run/factor_synth.R --density 0.1 --oracle
# Outputs:
#   results/synth/<density>/factor/<cell>/<mask>/<method>_<dz>_loadings.csv  (SL: g, F*, h2, u2)
#   results/synth/<density>/factor/rows_<method>.csv   one row per (cell, mask, dz)
# ─────────────────────────────────────────────────────────────────────────────

.script_path <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
SRC  <- if (length(.script_path)) dirname(dirname(normalizePath(.script_path))) else normalizePath("src")
REPO <- dirname(SRC)

source(file.path(SRC, "factor", "factoring.R"))
source(file.path(SRC, "impute", "common.R"))

args <- commandArgs(trailingOnly = TRUE)
methods <- character(0); oracle <- FALSE; density <- NULL
i <- 1L
while (i <= length(args)) {
  if (args[[i]] == "--method") { methods <- c(methods, args[[i + 1L]]); i <- i + 2L }
  else if (args[[i]] == "--density") { density <- args[[i + 1L]]; i <- i + 2L }
  else if (args[[i]] == "--oracle") { oracle <- TRUE; i <- i + 1L }
  else stop("unknown arg: ", args[[i]])
}
if (!oracle && !length(methods)) stop("give --method <name> (repeatable) and/or --oracle")
if (is.null(density)) stop("--density <fraction> is required (the synth corpus is density-nested)")

RAW_METHODS <- c("default", "zeros")   # cached fill+smooth correlation, as in factor.R
# --density 0 is the "factor the hierarchical matrix" slice: it aliases the
# SPARSEST density (the final step of the impute.R --hier --synth descent) and
# reads that density's s0/hierarchical/imputed/... matrices. Outputs still land
# in results/synth/0/factor/ so score_synth_factors.py --density 0 keeps working.
HIER <- identical(density, "0")
if (HIER) {
  dens <- suppressWarnings(as.numeric(list.dirs(file.path(REPO, "data", "synth"),
                                                recursive = FALSE, full.names = FALSE)))
  dens <- dens[is.finite(dens)]
  if (!length(dens)) stop("density 0: no numeric density slices under data/synth/")
  density <- sprintf("%g", min(dens))
  cat(sprintf("density 0 -> hierarchical end result: data/synth/%s/ (sparsest descent step)\n",
              density))
}
SYNTH  <- file.path(REPO, "data", "synth", density)
OUTDIR <- file.path(REPO, "results", "synth", "0", "factor")
dir.create(OUTDIR, recursive = TRUE, showWarnings = FALSE)
DZS <- c("C", "S"); ST <- "all_standard"

cells <- sort(setdiff(list.dirs(SYNTH, recursive = FALSE, full.names = FALSE), "truth"))

read_completed <- function(path) {
  df <- read.csv(path, check.names = FALSE)
  M <- as.matrix(df[, setdiff(names(df), "collapse_key")])
  storage.mode(M) <- "double"
  M
}

# One factoring, factor.R's recipe. Returns a one-row data.frame + loadings.
factor_one <- function(M = NULL, R = NULL, n = NA) {
  if (!is.null(R)) {
    fr <- factor_cached_R(R, n, pa_iter = 100L)
    ho <- higher_order(fr$R, fr$nf, n_obs = fr$n_eff)
  } else {
    fr <- factor_matrix(M, pa_iter = 100L)
    ho <- higher_order(M, fr$nf)
  }
  list(nf = fr$nf, omega_h = ho$omega_h, omega_total = ho$omega_total,
       L = ho$bifactor_loadings)
}

run_and_write <- function(tag_method, cell, mask, dz, M = NULL, R = NULL, n = NA) {
  t0 <- Sys.time()
  res <- tryCatch(factor_one(M, R, n), error = function(e) {
    cat(sprintf("  FAILED %s %s/%s/%s: %s\n", tag_method, cell, mask, dz, conditionMessage(e)))
    NULL })
  if (is.null(res)) return(NULL)
  d <- file.path(OUTDIR, cell, mask); dir.create(d, recursive = TRUE, showWarnings = FALSE)
  if (!is.null(res$L)) {
    L <- as.data.frame(res$L); L <- cbind(benchmark = rownames(res$L), L)
    write.csv(L, file.path(d, sprintf("%s_%s_loadings.csv", tag_method, dz)), row.names = FALSE)
  }
  cat(sprintf("  %-15s %-9s %-10s %s  nf=%2d omega_h=%.3f omega_t=%.3f (%.1fs)\n",
              tag_method, cell, mask, dz, res$nf, res$omega_h, res$omega_total,
              as.numeric(difftime(Sys.time(), t0, units = "secs"))))
  data.frame(method = tag_method, cell = cell, mask = mask, dz = dz,
             nf = res$nf, omega_h = res$omega_h, omega_total = res$omega_total)
}

flush_rows <- function(rows, name) {
  if (!length(rows)) return(invisible())
  out <- do.call(rbind, rows)
  write.csv(out, file.path(OUTDIR, sprintf("rows_%s.csv", name)), row.names = FALSE)
  cat(sprintf("wrote %d rows -> %s\n", nrow(out), file.path(OUTDIR, sprintf("rows_%s.csv", name))))
}

if (oracle) {
  rows <- list()
  for (cell in cells) for (dz in DZS) {
    p <- file.path(SYNTH, "truth", sprintf("%s_%s_%s_complete.csv", dz, ST, cell))
    if (!file.exists(p)) { cat("  missing", p, "\n"); next }
    r <- run_and_write("oracle", cell, "complete", dz, M = read_completed(p))
    if (!is.null(r)) rows[[length(rows) + 1L]] <- r
  }
  flush_rows(rows, "oracle")
}

for (method in methods) {
  rows <- list()
  for (cell in cells) {
    masks <- sort(list.dirs(file.path(SYNTH, cell), recursive = FALSE, full.names = FALSE))
    for (mask in masks) for (dz in DZS) {
      root <- file.path(SYNTH, cell, mask, "s0")
      p <- if (HIER) file.path(root, "hierarchical", "imputed", method, dz, ST,
                               "imputed_model_benchmark_table.csv")
            else file.path(root, "imputed", method, dz, ST,
                           "imputed_model_benchmark_table.csv")
      if (!file.exists(p)) { cat("  missing", p, "\n"); next }
      M <- read_completed(p)
      r <- if (method %in% RAW_METHODS) {
        cp <- if (HIER)
          file.path(REPO, "results", "synth", "hierarchical", cell, mask, density, method,
                    sprintf("%s_%s_%s_correlation.csv", method, dz, ST))
        else file.path(REPO, "results", "synth", density, cell, mask, method,
                       sprintf("%s_%s_%s_correlation.csv", method, dz, ST))
        if (!file.exists(cp)) { cat("  missing", cp, "\n"); NULL }
        else run_and_write(method, cell, mask, dz, R = read_correlation_csv(cp), n = nrow(M))
      } else run_and_write(method, cell, mask, dz, M = M)
      if (!is.null(r)) rows[[length(rows) + 1L]] <- r
    }
  }
  flush_rows(rows, method)
}
