#!/usr/bin/env Rscript
# ─────────────────────────────────────────────────────────────────────────────
# Factor-analysis-only orchestrator.
#
# Reads COMPLETED matrices (from data/imputed/) and SWEEP CURVES (from results/)
# written by the imputation stage, then runs parallel analysis, PAF + promax,
# higher-order factor analysis, and dashboard generation. No imputation here.
#
# Output (per cell):
#   results/<method>/<method>_<dz>_<st>_loadings.csv
#   results/<method>/<method>_<dz>_<st>_loadings.md
#   results/<method>/<method>_<dz>_<st>_secondorder_loadings.csv
#   results/<method>/<method>_<dz>_<st>_secondorder_loadings.md
#   results/<method>/<method>_<dz>_<st>_bifactor_loadings.csv
#   results/<method>/<method>_<dz>_<st>_bifactor_loadings.md
#   results/<method>/<method>_<dz>_<st>_bifactor_scalars.csv
#   results/<method>/<method>_<dz>_<st>_bifactor_omega_group.csv
#   results/<method>/<method>_<dz>_<st>_dashboard.png
#
# Dashboard panels 3 (cumulative variance vs param) and 6 (PA nf vs param) show
# "no data" because the per-param complete_at() closure is only available during
# in-memory imputation — they are populated only in the combined main.R pipeline.
#
# Run from anywhere:
#   Rscript src/run/factor.R [--method <name>] [--raw]
#     --method       softimpute | iterativepca | onesidedmc | knn | missforest | mice | all
#     --raw          run ONLY the "raw" densifier level (default: C,S,R)
#     --data-root    input tree, relative to repo root (default data)
#     --results-root output tree, relative to repo root (default results)
#     --smoke        use data/smoke fixture
# ─────────────────────────────────────────────────────────────────────────────

.script_path <- sub("^--file=", "",
                    grep("^--file=", commandArgs(FALSE), value = TRUE))[1]
SRC_DIR <- if (length(.script_path) && nzchar(.script_path))
  dirname(normalizePath(.script_path)) else normalizePath("src/run")
SRC  <- dirname(SRC_DIR)
REPO <- dirname(SRC)

.renv_activate <- file.path(REPO, "renv", "activate.R")
if (file.exists(.renv_activate)) source(.renv_activate)

source(file.path(SRC, "factor", "factoring.R"))
source(file.path(SRC, "run", "dashboard.R"))
source(file.path(SRC, "run", "plots.R"))

ALL_METHODS <- c("softimpute", "iterativepca", "onesidedmc",
                 "knn", "missforest", "mice")
parse_args <- function(args) {
  method <- "all"; raw <- FALSE; smoke <- FALSE
  data_root <- "data"; results_root <- "results"
  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]
    if (a == "--method") { method <- args[[i + 1L]]; i <- i + 2L }
    else if (a == "--data-root")    { data_root    <- args[[i + 1L]]; i <- i + 2L }
    else if (a == "--results-root") { results_root <- args[[i + 1L]]; i <- i + 2L }
    else if (a == "--raw")   { raw   <- TRUE; i <- i + 1L }
    else if (a == "--smoke") { smoke <- TRUE; i <- i + 1L }
    else stop("unknown arg: ", a)
  }
  if (method != "all" && !(method %in% ALL_METHODS))
    stop("--method must be one of: ", paste(c("all", ALL_METHODS), collapse = ", "))
  list(methods = if (method == "all") ALL_METHODS else method,
       raw = raw, smoke = smoke,
       data_root = data_root, results_root = results_root)
}
opt <- parse_args(commandArgs(trailingOnly = TRUE))

METHODS    <- opt$methods
DENSIFIERS <- if (opt$raw) "raw" else c("C", "S", "R")
STRATEGIES <- c("all_standard", "all_aggressive")
MAX_RANK   <- 10L
DATA_ROOT  <- file.path(REPO, if (opt$smoke) "data/smoke" else opt$data_root)
RESULTS_ROOT <- file.path(REPO, if (opt$smoke) "results/smoke" else opt$results_root)
dir.create(RESULTS_ROOT, recursive = TRUE, showWarnings = FALSE)
cat(sprintf("factor: methods=[%s]  data_root=%s  results=%s\n",
            paste(METHODS, collapse = ","), DATA_ROOT, RESULTS_ROOT))

res_path <- function(method, dz, st, suffix) {
  d <- file.path(RESULTS_ROOT, method)
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
  file.path(d, sprintf("%s_%s_%s_%s", method, dz, st, suffix))
}

read_matrix <- function(path) {
  df <- read.csv(path, check.names = FALSE)
  M <- as.matrix(df[, setdiff(names(df), "collapse_key")])
  storage.mode(M) <- "double"
  list(M = M, keys = df$collapse_key)
}

build_contract_from_disk <- function(method, dz, st) {
  completed_csv <- file.path(DATA_ROOT, "imputed", method, dz, st,
                             "imputed_model_benchmark_table.csv")
  if (!file.exists(completed_csv)) {
    cat("  missing completed matrix:", completed_csv, "\n")
    return(NULL)
  }
  mb <- read_matrix(completed_csv)

  sweep_csv <- res_path(method, dz, st, "rank_sweep.csv")
  if (!file.exists(sweep_csv) && method == "onesidedmc") {
    sweep_csv <- file.path(RESULTS_ROOT, "_osmc_sweep",
                           sprintf("%s_%s", dz, st), "rank_sweep.csv")
  }
  if (!file.exists(sweep_csv)) {
    cat("  missing sweep CSV (no curve panels in dashboard):", sweep_csv, "\n")
    return(list(M = mb$M, params = 1L, curve = NA_real_,
                best_param = 1L, param_name = "rank",
                metric_name = "Held-out RMSE", complete_at = NULL))
  }

  sw <- read.csv(sweep_csv)
  if (method == "onesidedmc") {
    rmse_col <- if ("rmse" %in% names(sw)) sw$rmse else sw$pairwise_rmse
    params <- sw$r
    curve <- rmse_col
    curve_r2 <- if ("r2" %in% names(sw)) sw$r2 else NULL
    param_name <- "r"
  } else {
    params <- sw$param
    curve <- sw$rmse
    curve_r2 <- if ("r2" %in% names(sw)) sw$r2 else NULL
    param_name <- sw$param_name[1]
  }
  best_param <- params[which.min(replace(curve, is.na(curve), Inf))]

  list(M = mb$M, params = params, curve = curve, curve_r2 = curve_r2,
       best_param = best_param, param_name = param_name,
       metric_name = "Held-out RMSE", complete_at = NULL)
}

factor_and_report <- function(method, dz, st, res) {
  tag <- sprintf("%s/%s/%s", method, dz, st)
  fr <- factor_matrix(res$M, pa_iter = 100L)
  cat(sprintf("  factored: nf = %d\n", fr$nf))

  write_loadings_csv(fr$efa, res_path(method, dz, st, "loadings.csv"))
  write_loadings_markdown(fr$efa, res_path(method, dz, st, "loadings.md"))

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

  dims <- c(nrow(res$M), ncol(res$M), 100)
  sw <- tryCatch(sweep_factor_curve(res, pa_iter = 100L),
                 error = function(e) {
                   cat("  sweep-factor failed:", conditionMessage(e), "\n")
                   list(cumvar = rep(NA, length(res$params)),
                        pa_nf = rep(NA, length(res$params))) })
  plot_dashboard(res_path(method, dz, st, "dashboard.png"), res, fr, sw,
                 max_k = MAX_RANK, title = tag, dims = dims, ho = ho)
}

main <- function() {
  for (method in METHODS) for (st in STRATEGIES) {
    for (dz in DENSIFIERS) {
      cat("\n======== ", method, "/", dz, "/", st, " ========\n", sep = "")
      res <- build_contract_from_disk(method, dz, st)
      if (is.null(res)) next
      tryCatch(factor_and_report(method, dz, st, res),
               error = function(e) cat("  FACTOR FAILED:", conditionMessage(e), "\n"))
    }
  }

  cat("\n-- combining dashboards --\n")
  for (method in METHODS) for (st in STRATEGIES) {
    tryCatch(combine_dashboards(method, st, RESULTS_ROOT),
             error = function(e) cat("  combine dashboards failed:", conditionMessage(e), "\n"))
  }

  cat("\nDONE.\n  loadings + dashboards -> ", RESULTS_ROOT, "/\n", sep = "")
}

main()