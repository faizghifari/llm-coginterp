#!/usr/bin/env Rscript
# ─────────────────────────────────────────────────────────────────────────────
# Pipeline orchestrator.
#
# Cross-product: {densifier C,R,S} x {strategy all_standard, all_aggressive} x
# {method softimpute, iterativepca, onesidedmc}.
#
# For each cell:
#   1. impute     -> completed (or, for OSMC, covariance-surrogate) matrix
#   2. factor     -> shared parallel analysis (cached) + PAF + promax loadings
#   3. dashboard  -> a single 6-panel PNG (orchestrator drives sweep + factoring)
#
# Output split:
#   data/imputed/<method>/<densifier>/<strategy>/  -> ONLY the imputed CSV
#   results/                                       -> everything else, FLAT, e.g.
#       softimpute_C_all_standard_dashboard.png
#       softimpute_C_all_standard_loadings.{csv,md}
#       softimpute_C_all_standard_sensitivity_*.png
#
# OSMC is implemented in Julia; we shell out to its run.jl to produce per-r
# surrogate CSVs, then factor them through the SAME shared R path as the others.
#
# Run from anywhere:
#   Rscript src/run/main.R [--method <name>] [--raw] [--smoke] [--sensitivity]
#     --method       softimpute | iterativepca | onesidedmc | all   (default all)
#     --raw          run ONLY the slow undensified "raw" level (default: C,S,R)
#     --smoke        use the data/smoke fixture instead of data/
#     --sensitivity  also run the (slow) seed-sweep sensitivity analysis
# Strategies (all_standard, all_aggressive) always run.
#
# Inputs read from <repo>/data/, outputs written to <repo>/data/imputed and
# <repo>/results/ — all anchored to the repo root, not the current directory.
# ─────────────────────────────────────────────────────────────────────────────

# Locate this script -> src/run, so SRC = src and REPO = repo root, regardless
# of the working directory the orchestrator is invoked from.
.script_path <- sub("^--file=", "",
                    grep("^--file=", commandArgs(FALSE), value = TRUE))[1]
SRC_DIR <- if (length(.script_path) && nzchar(.script_path))
  dirname(normalizePath(.script_path)) else normalizePath("src/run")
SRC  <- dirname(SRC_DIR)            # .../src
REPO <- dirname(SRC)               # repo root

# Activate the project-scoped renv library (if set up) so the right package
# versions load regardless of the working directory Rscript was started in.
.renv_activate <- file.path(REPO, "renv", "activate.R")
if (file.exists(.renv_activate)) source(.renv_activate)

source(file.path(SRC, "impute", "common.R"))
source(file.path(SRC, "factor", "factoring.R"))
source(file.path(SRC, "run", "plots.R"))
source(file.path(SRC, "run", "dashboard.R"))

# ── Argument parsing ─────────────────────────────────────────────────────────
ALL_METHODS <- c("softimpute", "iterativepca", "onesidedmc",
                 "knn", "missforest", "mice")
parse_args <- function(args) {
  method <- "all"; smoke <- FALSE; sens <- FALSE; raw <- FALSE
  reimpute <- FALSE; no_balance <- FALSE
  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]
    if (a == "--method") { method <- args[[i + 1L]]; i <- i + 2L }
    else if (a == "--smoke")       { smoke      <- TRUE; i <- i + 1L }
    else if (a == "--sensitivity") { sens       <- TRUE; i <- i + 1L }
    else if (a == "--raw")         { raw        <- TRUE; i <- i + 1L }
    else if (a == "--reimpute")    { reimpute   <- TRUE; i <- i + 1L }
    else if (a == "--no-balance")  { no_balance <- TRUE; i <- i + 1L }
    else stop("unknown arg: ", a)
  }
  if (method != "all" && !(method %in% ALL_METHODS))
    stop("--method must be one of: ", paste(c("all", ALL_METHODS), collapse = ", "))
  list(methods = if (method == "all") ALL_METHODS else method,
       smoke = smoke, sensitivity = sens, raw = raw, reimpute = reimpute,
       no_balance = no_balance)
}
opt <- parse_args(commandArgs(trailingOnly = TRUE))

METHODS    <- opt$methods
# "raw" = the undensified aggregated table (slow). --raw runs ONLY raw; without
# it, only the densified levels C/S/R run, so raw can be run separately.
DENSIFIERS <- if (opt$raw) "raw" else c("C", "S", "R")
STRATEGIES <- c("all_standard", "all_aggressive")
DO_SENS    <- opt$sensitivity
REIMPUTE   <- opt$reimpute   # force fresh imputation even if an imputed CSV exists
# --no-balance: revert held-out RMSE/R^2 to the old cell-weighted score (high-
# frequency columns dominate). Default is column-balanced (mean of per-column
# scores). Set in common.R's BALANCE_HOLDOUT, which the scorers read.
BALANCE_HOLDOUT <- !opt$no_balance
MAX_RANK   <- 10L
DATA_ROOT  <- file.path(REPO, if (opt$smoke) "data/smoke" else "data")
RESULTS_ROOT <- file.path(REPO, if (opt$smoke) "results/smoke" else "results")
dir.create(RESULTS_ROOT, recursive = TRUE, showWarnings = FALSE)
cat(sprintf("methods=[%s]  data_root=%s  results=%s  sensitivity=%s  reimpute=%s\n",
            paste(METHODS, collapse = ","), DATA_ROOT, RESULTS_ROOT, DO_SENS, REIMPUTE))

# Input table for a densifier level. "raw" reads the undensified aggregated
# table under data/combinations/; C/S/R read the densified copies.
combos_path <- function(dz, st) {
  sub <- if (dz == "raw") "combinations" else sprintf("combinations_%s", dz)
  file.path(DATA_ROOT, sub, st, "model_benchmark_table.csv")
}

# Results path: results/<method>/<method>_<densifier>_<strategy>_<suffix>
# (flat filenames, nested one level under a per-method subdir).
res_path <- function(method, dz, st, suffix) {
  d <- file.path(RESULTS_ROOT, method)
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
  file.path(d, sprintf("%s_%s_%s_%s", method, dz, st, suffix))
}

# ── Imputation dispatch ──────────────────────────────────────────────────────
# softimpute / iterativepca run in-process and return (M, sweep meta).
# onesidedmc runs as a Julia subprocess (once, up front) writing surrogate CSVs;
# here we just load the surrogate it produced.
impute_R <- function(method, x) {
  if (method == "softimpute") {
    source(file.path(SRC, "impute", "softimpute", "method.R"))
    impute_softimpute(x, max_rank = MAX_RANK)
  } else if (method == "iterativepca") {
    source(file.path(SRC, "impute", "iterativepca", "method.R"))
    impute_iterativepca(x, max_ncp = MAX_RANK)
  } else if (method == "knn") {
    source(file.path(SRC, "impute", "knn", "method.R"))
    impute_knn(x)
  } else if (method == "missforest") {
    source(file.path(SRC, "impute", "missforest", "method.R"))
    impute_missforest(x)
  } else if (method == "mice") {
    source(file.path(SRC, "impute", "mice", "method.R"))
    impute_mice(x)
  } else stop("not an R imputer: ", method)
}

# `nf` is the first-order factor count from the cell's main-run factoring; passed
# so each seed's omega_h is computed at a fixed nf (no per-seed parallel analysis).
sensitivity_R <- function(method, x, nf = NA_integer_) {
  if (method == "softimpute") {
    source(file.path(SRC, "impute", "softimpute", "method.R"))
    sensitivity_softimpute(x, max_rank = MAX_RANK, nf = nf)
  } else if (method == "iterativepca") {
    source(file.path(SRC, "impute", "iterativepca", "method.R"))
    sensitivity_iterativepca(x, max_ncp = MAX_RANK)  # HO not wired (deferred)
  } else if (method == "knn") {
    source(file.path(SRC, "impute", "knn", "method.R"))
    sensitivity_knn(x, nf = nf)
  } else if (method == "missforest") {
    source(file.path(SRC, "impute", "missforest", "method.R"))
    sensitivity_missforest(x, nf = nf)
  } else if (method == "mice") {
    source(file.path(SRC, "impute", "mice", "method.R"))
    sensitivity_mice(x, nf = nf)
  } else stop("no R sensitivity for: ", method)
}

run_osmc_subprocess <- function() {
  cat("\n##### OneSidedMC (Julia subprocess) #####\n")
  Sys.setenv(OSMC_DENSIFIERS   = paste(DENSIFIERS, collapse = ","),
             OSMC_STRATEGIES   = paste(STRATEGIES, collapse = ","),
             OSMC_DATA_ROOT    = normalizePath(DATA_ROOT, mustWork = FALSE),
             OSMC_RESULTS_ROOT = normalizePath(RESULTS_ROOT, mustWork = FALSE),
             OSMC_SENSITIVITY  = if (DO_SENS) "1" else "",
             OSMC_BALANCE      = if (BALANCE_HOLDOUT) "1" else "0")
  # --threads=auto so the OSMC seed-sweep sensitivity uses all cores. Paths are
  # absolute (anchored to SRC) so this works regardless of the current directory.
  osmc <- file.path(SRC, "impute", "OneSidedMC")
  status <- system2("julia",
    args = c("--threads=auto", paste0("--project=", osmc),
             file.path(osmc, "run.jl")))
  if (status != 0) cat("  WARNING: OSMC subprocess exited with status", status, "\n")
}

# Read a surrogate/completed CSV into a numeric matrix + keys.
read_matrix <- function(path) {
  df <- read.csv(path, check.names = FALSE)
  M <- as.matrix(df[, setdiff(names(df), "collapse_key")])
  storage.mode(M) <- "double"
  list(M = M, keys = df$collapse_key)
}

# Build the uniform imputer contract for OSMC from the Julia subprocess outputs:
# best-r surrogate (in data/imputed) + per-r surrogates & rank curve (in the
# results/_osmc_sweep staging dir). complete_at(r) reads surrogate_r<r>.csv.
osmc_contract <- function(dz, st, imputed_csv) {
  sweep_dir <- file.path(RESULTS_ROOT, "_osmc_sweep", sprintf("%s_%s", dz, st))
  curve_csv <- file.path(sweep_dir, "rank_sweep.csv")
  if (!file.exists(imputed_csv) || !file.exists(curve_csv)) return(NULL)
  sw <- read.csv(curve_csv)
  # column is `rmse` (cell-level by default); fall back to the old `pairwise_rmse`
  # name if a stale dead-branch CSV is encountered.
  rmse_col <- if ("rmse" %in% names(sw)) sw$rmse else sw$pairwise_rmse
  best_r <- sw$r[which.min(replace(rmse_col, is.na(rmse_col), Inf))]
  mb <- read_matrix(imputed_csv)
  list(M = mb$M, keys = mb$keys,
       best_param = best_r, params = sw$r, curve = rmse_col,
       curve_r2 = if ("r2" %in% names(sw)) sw$r2 else NULL,
       param_name = "r", metric_name = "Held-out RMSE",
       complete_at = function(v)
         read_matrix(file.path(sweep_dir, sprintf("surrogate_r%d.csv", v)))$M)
}

# Read OSMC's Julia seed-sweep sensitivity.csv (seed, chosen_r, rmse_r*, r2_r*)
# into the record plot_sensitivity_grid expects. NULL if not produced.
osmc_sensitivity <- function(dz, st) {
  f <- file.path(RESULTS_ROOT, "_osmc_sweep", sprintf("%s_%s", dz, st),
                 "sensitivity.csv")
  if (!file.exists(f)) return(NULL)
  d <- read.csv(f, check.names = FALSE)
  rmse_cols <- grep("^rmse_r", names(d), value = TRUE)
  r2_cols   <- grep("^r2_r", names(d), value = TRUE)
  ranks <- as.integer(sub("rmse_r", "", rmse_cols))
  list(rmse_mat = as.matrix(d[, rmse_cols]),
       r2_mat   = if (length(r2_cols)) as.matrix(d[, r2_cols]) else NULL,
       best_ranks = d$chosen_r, ranks = ranks, param = "r")
}

# Factor at best param, write loadings + higher-order + the single dashboard.
factor_and_report <- function(method, dz, st, res, keys) {
  tag <- sprintf("%s/%s/%s", method, dz, st)
  fr <- factor_matrix(res$M, pa_iter = 100L)
  cat(sprintf("  factored: nf = %d\n", fr$nf))

  write_loadings_csv(fr$efa, res_path(method, dz, st, "loadings.csv"))
  write_loadings_markdown(fr$efa, res_path(method, dz, st, "loadings.md"))

  # higher-order: second-order FA + bifactor (Schmid-Leiman) + omega_h
  ho <- tryCatch(higher_order(res$M, fr$nf), error = function(e) {
    cat("  higher-order failed:", conditionMessage(e), "\n"); NULL })
  if (!is.null(ho)) {
    write_higher_order(ho,
      second_csv   = res_path(method, dz, st, "secondorder_loadings.csv"),
      bifactor_csv = res_path(method, dz, st, "bifactor_loadings.csv"),
      scalar_csv   = res_path(method, dz, st, "bifactor_scalars.csv"),
      group_csv    = res_path(method, dz, st, "bifactor_omega_group.csv"),
      second_md    = res_path(method, dz, st, "secondorder_loadings.md"),
      bifactor_md  = res_path(method, dz, st, "bifactor_loadings.md"))
    omega_hs <- if (!is.null(ho$omega_group) && "group" %in% colnames(ho$omega_group))
      ho$omega_group[rownames(ho$omega_group) != "g", "group"] else numeric(0)
    cat(sprintf("  higher-order: omega_h = %.3f, omega_total = %.3f, omega_hs = %s\n",
                ho$omega_h, ho$omega_total,
                if (length(omega_hs)) paste(sprintf("%.3f", omega_hs), collapse = ",") else "NA"))
  }

  dims <- c(nrow(res$M), ncol(res$M), 100)  # completed matrix is fully observed
  sw <- tryCatch(sweep_factor_curve(res, pa_iter = 100L),
                 error = function(e) {
                   cat("  sweep-factor failed:", conditionMessage(e), "\n")
                   list(cumvar = rep(NA, length(res$params)),
                        pa_nf = rep(NA, length(res$params))) })
  plot_dashboard(res_path(method, dz, st, "dashboard.png"), res, fr, sw,
                 max_k = MAX_RANK, title = tag, dims = dims, ho = ho)
  invisible(fr$nf)  # so the caller can fix nf for the sensitivity omega_h
}

# Impute + factor + dashboard for one cell. Returns the prepped matrix `x` so
# the caller can run sensitivity on the same input (sensitivity is collected per
# method x strategy and plotted as one grid, so it's not done here).
run_cell <- function(method, dz, st) {
  tag <- sprintf("%s/%s/%s", method, dz, st)
  cat("\n======== ", tag, " ========\n", sep = "")
  out_dir <- imputed_dir(method, dz, st, root = file.path(DATA_ROOT, "imputed"))

  if (method == "onesidedmc") {
    # surrogates already written by the Julia subprocess; adapt + factor.
    imputed_csv <- file.path(out_dir, "imputed_model_benchmark_table.csv")
    res <- osmc_contract(dz, st, imputed_csv)
    if (is.null(res)) { cat("  no OSMC outputs, skipping\n"); return(NULL) }
    tryCatch(factor_and_report(method, dz, st, res, res$keys),
             error = function(e) cat("  FACTOR FAILED:", conditionMessage(e), "\n"))
    return(NULL)  # OSMC has no R-side seed-sweep sensitivity
  }

  # R imputers
  src <- combos_path(dz, st)
  if (!file.exists(src)) { cat("  missing input:", src, "\n"); return(NULL) }
  pm <- prep_matrix(src)
  x <- pm$x
  cat(sprintf("  matrix: %d x %d, %.1f%% observed\n",
              nrow(x), ncol(x), 100 * mean(!is.na(x))))

  imputed_csv <- file.path(out_dir, "imputed_model_benchmark_table.csv")
  sweep_csv   <- res_path(method, dz, st, "rank_sweep.csv")

  # --reimpute default OFF: reuse an existing imputed CSV (skip the slow impute),
  # rebuild a partial contract from disk, and just re-factor + re-plot. The
  # per-param dashboard panels (3 & 6) need complete_at (the in-memory fits), so
  # they show "no data" on a reused run — the curve panels still render.
  if (!REIMPUTE && file.exists(imputed_csv)) {
    cat("  reusing existing imputed CSV (skip imputation; use --reimpute to force)\n")
    mb <- read_matrix(imputed_csv)
    res <- list(M = mb$M, complete_at = NULL)
    if (file.exists(sweep_csv)) {
      sw <- read.csv(sweep_csv)
      res$params <- sw$param; res$curve <- sw$rmse
      res$curve_r2 <- if ("r2" %in% names(sw)) sw$r2 else NULL
      res$best_param <- sw$param[which.min(replace(sw$rmse, is.na(sw$rmse), Inf))]
      res$param_name <- sw$param_name[1]; res$metric_name <- "Held-out RMSE"
    } else {
      res$params <- 1L; res$curve <- NA; res$best_param <- 1L
      res$param_name <- "rank"; res$metric_name <- "Held-out RMSE"
    }
    nf <- tryCatch(factor_and_report(method, dz, st, res, mb$keys),
             error = function(e) { cat("  FACTOR FAILED:", conditionMessage(e), "\n"); NA_integer_ })
    return(list(x = x, nf = nf))
  }

  res <- tryCatch(impute_R(method, x), error = function(e) {
    cat("  IMPUTE FAILED:", conditionMessage(e), "\n"); NULL })
  if (is.null(res)) return(NULL)

  write_completed(out_dir, pm$keys, res$M)   # data/imputed: CSV only
  # persist the sweep curve so a later --reimpute-off run can rebuild the dashboard.
  write.csv(data.frame(param = res$params, param_name = res$param_name,
                       rmse = res$curve,
                       r2 = if (!is.null(res$curve_r2)) res$curve_r2 else NA),
            sweep_csv, row.names = FALSE)
  nf <- tryCatch(factor_and_report(method, dz, st, res, pm$keys),
           error = function(e) { cat("  FACTOR FAILED:", conditionMessage(e), "\n"); NA_integer_ })
  list(x = x, nf = nf)  # x for sensitivity, nf to fix omega_h
}

# ── Main loop ────────────────────────────────────────────────────────────────
# Order: method -> strategy -> densifier, so all densifiers' sensitivity results
# for one method x strategy are in hand before drawing the single grid (one row
# per densifier: C/S/R by default, or just raw under --raw).
main <- function() {
  if ("onesidedmc" %in% METHODS) run_osmc_subprocess()

  for (method in METHODS) for (st in STRATEGIES) {
    sens_by_dz <- list()
    for (dz in DENSIFIERS) {
      cell <- run_cell(method, dz, st)   # list(x, nf) for R methods, NULL otherwise
      if (DO_SENS) {
        if (method == "onesidedmc") {
          # OSMC's seed-sweep ran in Julia; read its CSV for the grid.
          sens_by_dz[[dz]] <- osmc_sensitivity(dz, st)
        } else if (!is.null(cell)) {
          cat("  --- sensitivity (", dz, ") ---\n", sep = "")
          sens_by_dz[[dz]] <- tryCatch(
            sensitivity_R(method, cell$x, nf = cell$nf),
            error = function(e) { cat("  SENSITIVITY FAILED:", conditionMessage(e), "\n"); NULL })
        }
      }
    }
    if (DO_SENS && length(sens_by_dz) > 0) {
      # name by densifier set so a separate --raw run doesn't clobber the C/S/R one.
      set_tag <- if (identical(DENSIFIERS, "raw")) "raw" else "csr"
      mdir <- file.path(RESULTS_ROOT, method)
      dir.create(mdir, recursive = TRUE, showWarnings = FALSE)
      plot_sensitivity_grid(sens_by_dz,
        file.path(mdir, sprintf("%s_%s_%s_sensitivity.png", method, st, set_tag)),
        title = sprintf("%s / %s", method, st))
    }
  }
  # Stack raw + csr aggregates from whatever PNGs now exist in results/ (so a
  # separate --raw run and a default C/S/R run can be viewed together).
  cat("\n-- combining aggregates --\n")
  for (method in METHODS) for (st in STRATEGIES) {
    tryCatch(combine_dashboards(method, st, RESULTS_ROOT),
             error = function(e) cat("  combine dashboards failed:", conditionMessage(e), "\n"))
    if (DO_SENS)
      tryCatch(combine_sensitivity(method, st, RESULTS_ROOT),
               error = function(e) cat("  combine sensitivity failed:", conditionMessage(e), "\n"))
  }

  cat("\nDONE.\n  imputed CSVs -> ", file.path(DATA_ROOT, "imputed"),
      "/<method>/<densifier>/<strategy>/\n  results       -> ", RESULTS_ROOT,
      "/ (flat)\n", sep = "")
}

main()
