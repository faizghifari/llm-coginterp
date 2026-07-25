#!/usr/bin/env Rscript
# ─────────────────────────────────────────────────────────────────────────────
# Imputation-only orchestrator.
#
# Runs imputation + held-out sweep + optional sensitivity for every
# (method × densifier × strategy) cell. DOES NOT factor — that's factor.R.
#
# Output:
#   data/imputed/<method>/<densifier>/<strategy>/  -> completed CSV + keys
#   results/<method>/<method>_<dz>_<st>_rank_sweep.csv
#   results/<method>/<method>_<st>_<set>_sensitivity.png  (if --sensitivity)
#
# Run from anywhere:
#   Rscript src/run/impute.R [--method <name>] [--raw] [--reimpute] [--sensitivity]
#     --method       softimpute | iterativepca | onesidedmc | knn | missforest | mice | all
#     --raw          run ONLY the undensified "raw" level (default: C,S,R)
#     --reimpute     force fresh imputation even if an imputed CSV exists
#     --sensitivity  also run the (slow) seed-sweep sensitivity analysis
#     --data-root    input tree, relative to repo root (default data)
#     --results-root output tree, relative to repo root (default results)
#     --smoke        use data/smoke fixture
#     --no-balance   revert to cell-weighted holdout scores
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
source(file.path(SRC, "run", "plots.R"))

ALL_METHODS <- c("softimpute", "iterativepca", "onesidedmc",
                 "knn", "missforest", "mice")
parse_args <- function(args) {
  method <- "all"; smoke <- FALSE; sens <- FALSE; raw <- FALSE
  reimpute <- FALSE; no_balance <- FALSE
  data_root <- "data"; results_root <- "results"
  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]
    if (a == "--method") { method <- args[[i + 1L]]; i <- i + 2L }
    else if (a == "--data-root")    { data_root    <- args[[i + 1L]]; i <- i + 2L }
    else if (a == "--results-root") { results_root <- args[[i + 1L]]; i <- i + 2L }
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
       no_balance = no_balance,
       data_root = data_root, results_root = results_root)
}
opt <- parse_args(commandArgs(trailingOnly = TRUE))

METHODS    <- opt$methods
DENSIFIERS <- if (opt$raw) "raw" else c("C", "S", "R")
STRATEGIES <- c("all_standard", "all_aggressive")
DO_SENS    <- opt$sensitivity
REIMPUTE   <- opt$reimpute
BALANCE_HOLDOUT <- !opt$no_balance
MAX_RANK   <- 10L
DATA_ROOT  <- file.path(REPO, if (opt$smoke) "data/smoke" else opt$data_root)
RESULTS_ROOT <- file.path(REPO, if (opt$smoke) "results/smoke" else opt$results_root)
dir.create(RESULTS_ROOT, recursive = TRUE, showWarnings = FALSE)
cat(sprintf("impute: methods=[%s]  data_root=%s  results=%s  sensitivity=%s  reimpute=%s\n",
            paste(METHODS, collapse = ","), DATA_ROOT, RESULTS_ROOT, DO_SENS, REIMPUTE))

combos_path <- function(dz, st) {
  sub <- if (dz == "raw") "combinations" else sprintf("combinations_%s", dz)
  file.path(DATA_ROOT, sub, st, "model_benchmark_table.csv")
}

res_path <- function(method, dz, st, suffix) {
  d <- file.path(RESULTS_ROOT, method)
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
  file.path(d, sprintf("%s_%s_%s_%s", method, dz, st, suffix))
}

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

sensitivity_R <- function(method, x, nf = NA_integer_) {
  if (method == "softimpute") {
    source(file.path(SRC, "impute", "softimpute", "method.R"))
    sensitivity_softimpute(x, max_rank = MAX_RANK, nf = nf)
  } else if (method == "iterativepca") {
    source(file.path(SRC, "impute", "iterativepca", "method.R"))
    sensitivity_iterativepca(x, max_ncp = MAX_RANK)
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
  osmc <- file.path(SRC, "impute", "OneSidedMC")
  status <- system2("julia",
    args = c("--threads=auto", paste0("--project=", osmc),
             file.path(osmc, "run.jl")))
  if (status != 0) cat("  WARNING: OSMC subprocess exited with status", status, "\n")
}

read_matrix <- function(path) {
  df <- read.csv(path, check.names = FALSE)
  M <- as.matrix(df[, setdiff(names(df), "collapse_key")])
  storage.mode(M) <- "double"
  list(M = M, keys = df$collapse_key)
}

osmc_contract <- function(dz, st, imputed_csv) {
  sweep_dir <- file.path(RESULTS_ROOT, "_osmc_sweep", sprintf("%s_%s", dz, st))
  curve_csv <- file.path(sweep_dir, "rank_sweep.csv")
  if (!file.exists(imputed_csv) || !file.exists(curve_csv)) return(NULL)
  sw <- read.csv(curve_csv)
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

run_cell <- function(method, dz, st) {
  tag <- sprintf("%s/%s/%s", method, dz, st)
  cat("\n======== ", tag, " ========\n", sep = "")
  out_dir <- imputed_dir(method, dz, st, root = file.path(DATA_ROOT, "imputed"))

  if (method == "onesidedmc") {
    imputed_csv <- file.path(out_dir, "imputed_model_benchmark_table.csv")
    res <- osmc_contract(dz, st, imputed_csv)
    if (is.null(res)) { cat("  no OSMC outputs, skipping\n"); return(NULL) }
    best_idx  <- which(res$params == res$best_param)
    best_rmse <- res$curve[best_idx]
    best_r2   <- if (!is.null(res$curve_r2)) res$curve_r2[best_idx] else NA
    desc <- build_desc(res$params, res$param_name, res$best_param)
    db_insert_imputation(method, paste0(dz, "_", st),
                         best_rmse, best_r2, desc,
                         file.path(RESULTS_ROOT, "database.db"), REIMPUTE)
    return(NULL)  # OSMC has no R-side seed-sweep sensitivity
  }

  src <- combos_path(dz, st)
  if (!file.exists(src)) { cat("  missing input:", src, "\n"); return(NULL) }
  pm <- prep_matrix(src)
  x <- pm$x
  cat(sprintf("  matrix: %d x %d, %.1f%% observed\n",
              nrow(x), ncol(x), 100 * mean(!is.na(x))))

  imputed_csv <- file.path(out_dir, "imputed_model_benchmark_table.csv")
  sweep_csv   <- res_path(method, dz, st, "rank_sweep.csv")

  if (!REIMPUTE && file.exists(imputed_csv)) {
    cat("  reusing existing imputed CSV (skip imputation; use --reimpute to force)\n")
    return(list(x = x, nf = NA_integer_))
  }

  res <- tryCatch(impute_R(method, x), error = function(e) {
    cat("  IMPUTE FAILED:", conditionMessage(e), "\n"); NULL })
  if (is.null(res)) return(NULL)

  write_completed(out_dir, pm$keys, res$M)
  write.csv(data.frame(param = res$params, param_name = res$param_name,
                       rmse = res$curve,
                       r2 = if (!is.null(res$curve_r2)) res$curve_r2 else NA),
            sweep_csv, row.names = FALSE)
  best_idx  <- which(res$params == res$best_param)
  best_rmse <- res$curve[best_idx]
  best_r2   <- if (!is.null(res$curve_r2)) res$curve_r2[best_idx] else NA
  desc <- build_desc(res$params, res$param_name, res$best_param)
  db_insert_imputation(method, paste0(dz, "_", st),
                       best_rmse, best_r2, desc,
                       file.path(RESULTS_ROOT, "database.db"), REIMPUTE)
  list(x = x, nf = NA_integer_)
}

main <- function() {
  if ("onesidedmc" %in% METHODS) run_osmc_subprocess()

  for (method in METHODS) for (st in STRATEGIES) {
    sens_by_dz <- list()
    for (dz in DENSIFIERS) {
      cell <- run_cell(method, dz, st)
      if (DO_SENS) {
        if (method == "onesidedmc") {
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
      set_tag <- if (identical(DENSIFIERS, "raw")) "raw" else "csr"
      mdir <- file.path(RESULTS_ROOT, method)
      dir.create(mdir, recursive = TRUE, showWarnings = FALSE)
      plot_sensitivity_grid(sens_by_dz,
        file.path(mdir, sprintf("%s_%s_%s_sensitivity.png", method, st, set_tag)),
        title = sprintf("%s / %s", method, st))
    }
  }

  cat("\n-- combining sensitivity aggregates --\n")
  for (method in METHODS) for (st in STRATEGIES) {
    if (DO_SENS)
      tryCatch(combine_sensitivity(method, st, RESULTS_ROOT),
               error = function(e) cat("  combine sensitivity failed:", conditionMessage(e), "\n"))
  }

  cat("\nDONE.\n  imputed CSVs -> ", file.path(DATA_ROOT, "imputed"),
      "/<method>/<densifier>/<strategy>/\n  results       -> ", RESULTS_ROOT,
      "/ (flat)\n", sep = "")
}

main()