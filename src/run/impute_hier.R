#!/usr/bin/env Rscript
# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENTAL: hierarchical (density-chained) imputation orchestrator.
#
# Instead of imputing each --density slice independently, this descends the
# density ladder for one imputer method, carrying the previous (denser) run's
# completed COLUMNS down as already-filled data:
#
#   densest density  -> reuse the standard run's cached imputed CSV as the
#                       seed (if missing: abort the cell — the chain cannot
#                       start)
#   each next density -> build the union grid (rows by collapse_key, columns
#                       by benchmark name), overlay this density's observed
#                       cells on top of the previous completed matrix, and
#                       impute. The previously-filled cells stay in TRAINING
#                       but are masked OUT of the held-out CV
#                       (HOLDOUT_MASK in impute/common.R), so the sweep RMSE/
#                       R2 only grade cells genuinely observed at this
#                       density and stay comparable to the standard runs.
#
# Densities are discovered by globbing numeric directories under --data-root
# (e.g. 0.3, 0.2, 0.1) and sorted DESCENDING. No --density argument: the
# hierarchy spans slices.
#
# Output:
#   <data-root>/hierarchical/<method>/<dz>/<st>/<density>/
#       imputed_model_benchmark_table.csv   completed matrix per step
#       filled_model_benchmark_table.csv    the union-grid input (auditable)
#   <results-root>/hierarchical/<density>/<method>/<method>_<dz>_<st>_rank_sweep.csv
#   <results-root>/hierarchical/<density>/<method>/<method>_<dz>_<st>_correlation.csv
#                                           (default/zeros only)
#   <results-root>/hierarchical/database.db  table imputation_hier
#                                           (PK density, dataset, method)
#
# The standard per-density outputs (data/<density>/imputed/, results/
# <density>/) are never touched, so factor.R cannot pick these up.
#
# Run from anywhere:
#   Rscript src/run/impute_hier.R [--method <name>] [--reimpute] [--raw]
#     --method       an R imputer name or "all" (onesidedmc is rejected: it
#                    is a Julia subprocess with no per-step hook)
#     --reimpute     force fresh imputation even if a hierarchical CSV exists
#     --raw          run ONLY the undensified "raw" level (default: C,S,R)
#     --data-root    input tree, relative to repo root (default data/text_only)
#     --results-root output tree, relative to repo root (default results/text_only)
# ─────────────────────────────────────────────────────────────────────────────

.script_path <- sub("^--file=", "",
                    grep("^--file=", commandArgs(FALSE), value = TRUE))[1]
SRC_DIR <- if (length(.script_path) && nzchar(.script_path))
  dirname(normalizePath(.script_path)) else normalizePath("src/run")
SRC  <- dirname(SRC_DIR)
REPO <- dirname(SRC)

.renv_activate <- file.path(REPO, "renv", "activate.R")
if (file.exists(.renv_activate)) source(.renv_activate)

source(file.path(SRC, "impute", "common.R"))
source(file.path(SRC, "impute", "db.R"))
source(file.path(SRC, "impute", "dispatch.R"))

MAX_RANK  <- 10L
REIMPUTE  <- FALSE

R_METHODS <- c("softimpute", "softimpute_corr", "iterativepca",
               "knn", "missforest", "mice",
               "knn_corr", "missforest_corr",
               "optspace", "usvt", "cvxr", "ggm",
               "default", "zeros")
# Fill-smooth methods: their contract carries the extra smoothed correlation R,
# persisted as ..._correlation.csv next to the flat results (same as impute.R).
RAW_METHODS <- c("default", "zeros")

parse_args <- function(args) {
  method <- "all"; raw <- FALSE; reimpute <- FALSE
  data_root <- "data/text_only"; results_root <- "results/text_only"
  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]
    if (a == "--method")        { method     <- args[[i + 1L]]; i <- i + 2L }
    else if (a == "--data-root")    { data_root    <- args[[i + 1L]]; i <- i + 2L }
    else if (a == "--results-root") { results_root <- args[[i + 1L]]; i <- i + 2L }
    else if (a == "--raw")         { raw      <- TRUE;     i <- i + 1L }
    else if (a == "--reimpute")    { reimpute <- TRUE;     i <- i + 1L }
    else stop("unknown arg: ", a)
  }
  if (method != "all" && !(method %in% R_METHODS)) {
    if (method == "onesidedmc")
      stop("onesidedmc is not supported hierarchically (Julia subprocess, ",
           "no per-step hook)")
    stop("--method must be one of: ", paste(c("all", R_METHODS), collapse = ", "))
  }
  list(methods = if (method == "all") R_METHODS else method,
       raw = raw, reimpute = reimpute,
       data_root = data_root, results_root = results_root)
}
opt <- parse_args(commandArgs(trailingOnly = TRUE))

METHODS    <- opt$methods
DENSIFIERS <- if (opt$raw) "raw" else c("C", "S", "R")
STRATEGIES <- c("all_standard", "all_aggressive")
REIMPUTE   <- opt$reimpute

# Density discovery: numeric directories under the data root, DESCENDING
# (0.3 -> 0.2 -> 0.1). The corpus-root files and combinations/ live alongside
# them and are filtered out by the numeric-name pattern.
DATA_BASE <- file.path(REPO, opt$data_root)
densities <- list.dirs(DATA_BASE, recursive = FALSE, full.names = FALSE)
densities <- densities[grepl("^[0-9]+(\\.[0-9]+)?$", densities)]
densities <- densities[order(as.numeric(densities), decreasing = TRUE)]
if (!length(densities))
  stop("no numeric density directories under ", DATA_BASE)

# Hierarchical outputs: data lives OUTSIDE every density slice (the artifact
# spans densities), so factor.R can never read it by accident.
HIER_DATA_ROOT <- file.path(DATA_BASE, "hierarchical")
RESULTS_ROOT   <- file.path(REPO, opt$results_root, "hierarchical")
DB_FILE        <- file.path(RESULTS_ROOT, "database.db")
dir.create(RESULTS_ROOT, recursive = TRUE, showWarnings = FALSE)
cat(sprintf("impute-hier: methods=[%s]  data_root=%s  densities=[%s]  results=%s  reimpute=%s\n",
            paste(METHODS, collapse = ","),
            file.path(opt$data_root), paste(densities, collapse = " -> "),
            file.path(opt$results_root, "hierarchical"), REIMPUTE))

# Inputs/outputs for one density slice of the data root.
combos_path_at <- function(density, dz, st) {
  sub <- if (dz == "raw") "combinations" else sprintf("combinations_%s", dz)
  file.path(DATA_BASE, density, sub, st, "model_benchmark_table.csv")
}

hier_step_dir <- function(density, method, dz, st) {
  d <- file.path(HIER_DATA_ROOT, method, dz, st, density)
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
  d
}

res_path_hier <- function(density, method, dz, st, suffix) {
  d <- file.path(RESULTS_ROOT, density, method)
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
  file.path(d, sprintf("%s_%s_%s_%s", method, dz, st, suffix))
}

db_record_hier <- function(method, dataset, density, rmse, r2, desc) {
  db_insert_imputation(method, dataset, rmse, r2, desc, DB_FILE, REIMPUTE,
                       table = "imputation_hier", density = density)
}

# One descent step: seed the union grid with P (previous completed matrix),
# overlay this density's observed cells, impute, persist. Returns the new
# running matrix P (list(M, keys)), P unchanged when the density's input is
# missing, or NULL when imputation failed (chain broken -> abort the cell).
descend_step <- function(method, dz, st, density, P) {
  src <- combos_path_at(density, dz, st)
  if (!file.exists(src)) {
    cat(sprintf("  [%s] missing input: %s — carrying P forward unchanged\n",
                density, src))
    return(P)
  }
  tdf <- read.csv(src, check.names = FALSE)
  tk  <- tdf$collapse_key
  t_m <- as.matrix(tdf[, setdiff(names(tdf), "collapse_key"), drop = FALSE])
  storage.mode(t_m) <- "double"

  # Union grid: this density's rows/columns first, previous run's extras
  # appended. Rows match on collapse_key, columns on benchmark name — never
  # position (densify's peel + prep_matrix's column drops make the column
  # sets non-nested across densities).
  keys_out <- union(tk, P$keys)
  cols_out <- union(colnames(t_m), colnames(P$M))
  X <- matrix(NA_real_, length(keys_out), length(cols_out),
              dimnames = list(NULL, cols_out))
  ti <- match(tk, keys_out)
  tj <- match(colnames(t_m), cols_out)
  X[ti, tj] <- t_m
  # real-observed mask: TRUE exactly where THIS density observed a value.
  real <- matrix(FALSE, length(keys_out), length(cols_out),
                 dimnames = list(NULL, cols_out))
  real[ti, tj] <- !is.na(t_m)

  # Fill the rest from the previous completed matrix (P-only rows/columns and
  # cells T is missing on). Observed cells of T already won above.
  pi <- match(P$keys, keys_out)
  pj <- match(colnames(P$M), cols_out)
  block <- X[pi, pj, drop = FALSE]
  fill  <- is.na(block) & !is.na(P$M)
  block[fill] <- P$M[fill]
  X[pi, pj] <- block

  cat(sprintf("  [%s] union grid: %d x %d | real obs: %d (%.1f%%) | filled from previous: %d | prev-only cols: %d | new cols: %d\n",
              density, length(keys_out), length(cols_out),
              sum(real), 100 * mean(real), sum(fill),
              length(setdiff(colnames(P$M), colnames(t_m))),
              length(setdiff(colnames(t_m), colnames(P$M)))))

  step_dir    <- hier_step_dir(density, method, dz, st)
  imputed_csv <- file.path(step_dir, "imputed_model_benchmark_table.csv")
  if (!REIMPUTE && file.exists(imputed_csv)) {
    cat("  reusing existing hierarchical CSV (skip imputation; use --reimpute to force)\n")
    return(read_matrix(imputed_csv))
  }

  # Persist the union-grid input for auditability, then run the unchanged
  # prep + imputer on it. Previously-filled columns trivially survive
  # prep_matrix's <2-obs drop (they are complete).
  filled_csv <- file.path(step_dir, "filled_model_benchmark_table.csv")
  write.csv(data.frame(collapse_key = keys_out, X, check.names = FALSE),
            filled_csv, row.names = FALSE)
  cat("  wrote", filled_csv, "\n")

  pm <- prep_matrix(filled_csv)
  if (ncol(pm$x) == 0L) {
    cat("  IMPUTE FAILED: prep_matrix dropped every column\n")
    return(NULL)
  }

  # Holdouts are sampled ONLY from this density's real observations — filled
  # cells train but are never graded (see HOLDOUT_MASK in impute/common.R).
  assign("HOLDOUT_MASK", real[, colnames(pm$x), drop = FALSE],
         envir = globalenv())
  on.exit(assign("HOLDOUT_MASK", NULL, envir = globalenv()), add = TRUE)

  res <- tryCatch(impute_R(method, pm$x), error = function(e) {
    cat("  IMPUTE FAILED:", conditionMessage(e), "\n"); NULL })
  if (is.null(res)) return(NULL)

  write_completed(step_dir, pm$keys, res$M)
  if (method %in% RAW_METHODS && !is.null(res$R))
    write_correlation_csv(res$R,
                          res_path_hier(density, method, dz, st,
                                        "correlation.csv"))
  sweep_csv <- res_path_hier(density, method, dz, st, "rank_sweep.csv")
  write.csv(data.frame(param = res$params, param_name = res$param_name,
                       rmse = res$curve,
                       r2 = if (!is.null(res$curve_r2)) res$curve_r2 else NA),
            sweep_csv, row.names = FALSE)
  best_idx  <- which(res$params == res$best_param)
  best_rmse <- res$curve[best_idx]
  best_r2   <- if (!is.null(res$curve_r2)) res$curve_r2[best_idx] else NA
  desc <- build_desc(res$params, res$param_name, res$best_param)
  db_record_hier(method, paste0(dz, "_", st), density, best_rmse, best_r2,
                 desc)
  list(M = res$M, keys = pm$keys)
}

run_hier_cell <- function(method, dz, st) {
  tag <- sprintf("%s/%s/%s", method, dz, st)
  cat("\n======== hier ", tag, " ========\n", sep = "")

  # Seed: the standard run's cached completed matrix at the DENSEST density.
  # No cache -> the chain cannot start -> abort the cell.
  d1 <- densities[1]
  cache_csv <- file.path(DATA_BASE, d1, "imputed", method, dz, st,
                         "imputed_model_benchmark_table.csv")
  if (!file.exists(cache_csv)) {
    cat("  no cached imputed CSV at the densest density (", d1,
        ") — aborting cell. Run the standard imputation first:\n",
        "    Rscript src/run/impute.R --method ", method,
        " --density ", d1, "\n", sep = "")
    return(invisible())
  }
  P <- read_matrix(cache_csv)
  write_completed(hier_step_dir(d1, method, dz, st), P$keys, P$M)
  # Parity for the fill-smooth pair: carry their correlation cache too.
  if (method %in% RAW_METHODS) {
    seed_corr <- file.path(REPO, opt$results_root, d1, method,
                           sprintf("%s_%s_%s_%s", method, dz, st,
                                   "correlation.csv"))
    if (file.exists(seed_corr))
      file.copy(seed_corr,
                res_path_hier(d1, method, dz, st, "correlation.csv"),
                overwrite = TRUE)
  }
  if (length(densities) == 1L) {
    cat("  only one density available — nothing to descend\n")
    return(invisible())
  }

  for (d in densities[-1]) {
    P <- descend_step(method, dz, st, d, P)
    if (is.null(P)) {
      cat("  chain broken at density ", d, " — aborting cell\n", sep = "")
      return(invisible())
    }
  }
}

main <- function() {
  for (method in METHODS) for (st in STRATEGIES) {
    for (dz in DENSIFIERS) {
      run_hier_cell(method, dz, st)
    }
  }
  cat("\nDONE.\n  imputed CSVs -> ", file.path(opt$data_root, "hierarchical"),
      "/<method>/<dz>/<st>/<density>/\n  results       -> ",
      file.path(opt$results_root, "hierarchical"), "/ (flat + database.db)\n",
      sep = "")
}

main()
