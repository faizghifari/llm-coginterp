#!/usr/bin/env Rscript
# ─────────────────────────────────────────────────────────────────────────────
# Factor-analysis-only orchestrator.
#
# Reads COMPLETED matrices (from data/imputed/) written by the imputation stage,
# gates on imputation R² >= 0.4 (SQLite), then runs two bifactor analyses per
# cell: one at the PA-based factor count (min 2) and one forced to 2 factors.
#
# Output (per cell):
#   results/<method>/<method>_<dz>_<st>_bifactor_pa_loadings.csv
#   results/<method>/<method>_<dz>_<st>_bifactor_pa_loadings.md
#   results/<method>/<method>_<dz>_<st>_bifactor_pa_scalars.csv
#   results/<method>/<method>_<dz>_<st>_bifactor_pa_omega_group.csv
#   results/<method>/<method>_<dz>_<st>_bifactor_2f_loadings.csv
#   results/<method>/<method>_<dz>_<st>_bifactor_2f_loadings.md
#   results/<method>/<method>_<dz>_<st>_bifactor_2f_scalars.csv
#   results/<method>/<method>_<dz>_<st>_bifactor_2f_omega_group.csv
#
# Factoring results are also persisted to results/<prefix>/database.db, table
# `factoring`.
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
source(file.path(SRC, "factor", "db.R"))

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
DATA_ROOT  <- file.path(REPO, if (opt$smoke) "data/smoke" else opt$data_root)
RESULTS_ROOT <- file.path(REPO, if (opt$smoke) "results/smoke" else opt$results_root)
dir.create(RESULTS_ROOT, recursive = TRUE, showWarnings = FALSE)
cat(sprintf("factor: methods=[%s]  data_root=%s  results=%s\n",
            paste(METHODS, collapse = ","), DATA_ROOT, RESULTS_ROOT))

DB_FILE <- file.path(RESULTS_ROOT, "database.db")

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
  read_matrix(completed_csv)
}

# Run a single bifactor analysis and write outputs for one (method, dz, st, run_tag).
do_bifactor <- function(M, nf, method, dz, st, run_tag) {
  ho <- tryCatch(higher_order(M, nf), error = function(e) {
    cat(sprintf("  higher_order(nf=%d, %s) failed: %s\n", nf, run_tag,
                conditionMessage(e))); NULL })
  if (is.null(ho)) return(NULL)

  tag <- paste0("bifactor_", run_tag)
  write_higher_order(ho,
    bifactor_csv = res_path(method, dz, st, paste0(tag, "_loadings.csv")),
    scalar_csv   = res_path(method, dz, st, paste0(tag, "_scalars.csv")),
    group_csv    = res_path(method, dz, st, paste0(tag, "_omega_group.csv")),
    bifactor_md  = res_path(method, dz, st, paste0(tag, "_loadings.md")))

  omega_hs <- if (!is.null(ho$omega_group) && "group" %in% colnames(ho$omega_group))
    ho$omega_group[rownames(ho$omega_group) != "g", "group"] else numeric(0)
  cat(sprintf("  %s: nf=%d omega_h=%.3f omega_total=%.3f omega_hs=%s\n",
              run_tag, ho$nf, ho$omega_h, ho$omega_total,
              if (length(omega_hs)) paste(sprintf("%.3f", omega_hs), collapse = ",") else "NA"))
  ho
}

factor_and_report <- function(method, dz, st, M) {
  tag <- sprintf("%s/%s/%s", method, dz, st)
  dataset <- paste0(dz, "_", st)

  r2 <- tryCatch(db_read_r2(method, dataset, DB_FILE),
                 error = function(e) { cat("  db read failed:", conditionMessage(e), "\n"); NA_real_ })
  if (is.na(r2) || r2 < 0.4) {
    cat(sprintf("  skipping (%s) — imputation R² = %s < 0.4\n", tag,
                if (is.na(r2)) "NA" else sprintf("%.3f", r2)))
    return(invisible())
  }
  cat(sprintf("  R² = %.3f >= 0.4, proceeding\n", r2))

  fr <- factor_matrix(M, pa_iter = 100L)
  pa_nf <- fr$nf
  var_explained <- extract_variance(fr$efa)
  cat(sprintf("  factored: nf = %d  cumvar = %.3f\n", pa_nf, var_explained))

  # PA-based bifactor
  ho_pa <- do_bifactor(M, pa_nf, method, dz, st, "pa")
  if (!is.null(ho_pa)) {
    omega_hs_pa <- if (!is.null(ho_pa$omega_group) && "group" %in% colnames(ho_pa$omega_group))
      ho_pa$omega_group[rownames(ho_pa$omega_group) != "g", "group"] else numeric(0)
    db_insert_factoring(method, dataset, "pa", pa_nf, var_explained,
                        ho_pa$omega_total, ho_pa$omega_h, omega_hs_pa, DB_FILE)
  }

  # Forced 2-factor bifactor
  ho_2f <- do_bifactor(M, 2L, method, dz, st, "2f")
  if (!is.null(ho_2f)) {
    omega_hs_2f <- if (!is.null(ho_2f$omega_group) && "group" %in% colnames(ho_2f$omega_group))
      ho_2f$omega_group[rownames(ho_2f$omega_group) != "g", "group"] else numeric(0)
    db_insert_factoring(method, dataset, "forced2f", 2L, var_explained,
                        ho_2f$omega_total, ho_2f$omega_h, omega_hs_2f, DB_FILE)
  }
}

main <- function() {
  for (method in METHODS) for (st in STRATEGIES) {
    for (dz in DENSIFIERS) {
      cat("\n======== ", method, "/", dz, "/", st, " ========\n", sep = "")
      res <- build_contract_from_disk(method, dz, st)
      if (is.null(res)) next
      tryCatch(factor_and_report(method, dz, st, res$M),
               error = function(e) cat("  FACTOR FAILED:", conditionMessage(e), "\n"))
    }
  }
  cat("\nDONE.\n  results -> ", RESULTS_ROOT, "/\n", sep = "")
}

main()
