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
#     --method       softimpute | softimpute_corr | optspace | usvt | iterativepca | onesidedmc | knn | missforest | mice | default | zeros | cvxr | ggm | all
#     --raw          run ONLY the "raw" densifier level (default: C,S,R)
#     --data-root    input tree, relative to repo root
#                    (default data/text_only -- the analysis corpus; pass
#                     `--data-root data` for the multimodal-inclusive one)
#     --results-root output tree, relative to repo root
#                    (default results/text_only)
#     --smoke        use data/smoke fixture
#     --loco         run leave-one-covariate-out delta omega_h instead of the
#                    standard bifactor outputs (writes database.db table `loco`)
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
source(file.path(SRC, "impute", "common.R"))

ALL_METHODS <- c("softimpute", "softimpute_corr", "iterativepca",
                 "onesidedmc", "knn", "missforest", "mice",
                 "optspace", "usvt",
                 "default", "zeros", "cvxr", "ggm")
RAW_METHODS <- c("default", "zeros")
parse_args <- function(args) {
  method <- "all"; raw <- FALSE; smoke <- FALSE; loco <- FALSE
  data_root <- "data/text_only"; results_root <- "results/text_only"
  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]
    if (a == "--method") { method <- args[[i + 1L]]; i <- i + 2L }
    else if (a == "--data-root")    { data_root    <- args[[i + 1L]]; i <- i + 2L }
    else if (a == "--results-root") { results_root <- args[[i + 1L]]; i <- i + 2L }
    else if (a == "--raw")   { raw   <- TRUE; i <- i + 1L }
    else if (a == "--smoke") { smoke <- TRUE; i <- i + 1L }
    else if (a == "--loco")  { loco  <- TRUE; i <- i + 1L }
    else stop("unknown arg: ", a)
  }
  if (method != "all" && !(method %in% ALL_METHODS))
    stop("--method must be one of: ", paste(c("all", ALL_METHODS), collapse = ", "))
  list(methods = if (method == "all") ALL_METHODS else method,
       raw = raw, smoke = smoke, loco = loco,
       data_root = data_root, results_root = results_root)
}
opt <- parse_args(commandArgs(trailingOnly = TRUE))

METHODS    <- opt$methods
DENSIFIERS <- if (opt$raw) "raw" else c("C", "S", "R")
STRATEGIES <- c("all_standard", "all_aggressive")
LOCO       <- opt$loco       # leave-one-covariate-out delta omega_h mode
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
  if (method %in% RAW_METHODS) {
    sparse_csv <- file.path(DATA_ROOT,
      if (dz == "raw") "combinations" else sprintf("combinations_%s", dz),
      st, "model_benchmark_table.csv")
    if (!file.exists(sparse_csv)) {
      cat("  missing sparse input:", sparse_csv, "\n")
      return(NULL)
    }
    pm <- prep_matrix(sparse_csv)
    return(list(M = pm$x, keys = pm$keys))
  }
  completed_csv <- file.path(DATA_ROOT, "imputed", method, dz, st,
                             "imputed_model_benchmark_table.csv")
  if (!file.exists(completed_csv)) {
    cat("  missing completed matrix:", completed_csv, "\n")
    return(NULL)
  }
  read_matrix(completed_csv)
}

# Run a single bifactor analysis and write outputs for one (method, dz, st, run_tag).
do_bifactor <- function(M, nf, method, dz, st, run_tag, n_obs = NA) {
  ho <- tryCatch(higher_order(M, nf, n_obs = n_obs), error = function(e) {
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

  if (LOCO) {
    if (method %in% RAW_METHODS) {
      prep <- switch(method,
        default = prepare_raw_default(M),
        zeros   = prepare_raw_zeros(M))
      R       <- prep$R
      n_obs   <- prep$n_eff
      cut     <- pa_cutoffs(n_obs, ncol(M))
      nf_pa   <- max(2L, sum(prep$eig_raw > cut, na.rm = TRUE))
      nf_pa   <- min(nf_pa, 20L)
      nf_pa   <- max(2L, min(nf_pa, ncol(M) - 1L, n_obs - 1L,
                             sum(prep$eig_raw > 1e-8, na.rm = TRUE) - 1L))
    } else {
      r2 <- tryCatch(db_read_r2(method, dataset, DB_FILE),
                     error = function(e) { cat("  db read failed:", conditionMessage(e), "\n"); NA_real_ })
      if (is.na(r2) || r2 < 0.4) {
        cat(sprintf("  skipping LOCO (%s) — imputation R² = %s < 0.4\n", tag,
                    if (is.na(r2)) "NA" else sprintf("%.3f", r2)))
        return(invisible())
      }
      cat(sprintf("  R² = %.3f >= 0.4, proceeding\n", r2))
      R     <- cor(M)
      n_obs <- nrow(M)
      pa    <- choose_nfactors(M)
      nf_pa <- min(pa$nf, 20L)
      nf_pa <- safe_nf(M, nf_pa)
    }

    cat(sprintf("  LOCO pa nf=%d\n", nf_pa))
    deltas_pa <- loco_delta(R, n_obs, nf_pa)
    db_insert_loco(method, dataset, "pa", nf_pa, deltas_pa, DB_FILE)

    cat("  LOCO forced2f nf=2\n")
    deltas_2f <- loco_delta(R, n_obs, 2L)
    db_insert_loco(method, dataset, "forced2f", 2L, deltas_2f, DB_FILE)

    return(invisible())
  }

  if (method %in% RAW_METHODS) {
    cat(sprintf("  %s factoring — pairwise-complete correlation (no imputation R² gate)\n", method))
    fr <- factor_raw(M, pa_iter = 100L, method = method)
    pa_nf <- fr$nf
    var_explained <- extract_variance(fr$efa)
    st_pa <- efa_stats(fr$efa)
    cat(sprintf("  factored: nf = %d  cumvar = %.3f  n = %d  phi_avg = %.3f\n",
                pa_nf, var_explained, fr$n_eff, st_pa$phi_avg))

    ho_pa <- do_bifactor(fr$R, pa_nf, method, dz, st, "pa", n_obs = fr$n_eff)
    if (!is.null(ho_pa)) {
      omega_hs_pa <- if (!is.null(ho_pa$omega_group) && "group" %in% colnames(ho_pa$omega_group))
        ho_pa$omega_group[rownames(ho_pa$omega_group) != "g", "group"] else numeric(0)
      db_insert_factoring(method, dataset, "pa", pa_nf, var_explained,
                          st_pa$var_factors, st_pa$var_avg,
                          ho_pa$omega_total, ho_pa$omega_h, omega_hs_pa,
                          st_pa$phi_avg, st_pa$phi, DB_FILE)
    }

    efa_2f <- fa_try(fr$R, 2L, n_obs = fr$n_eff)
    st_2f <- if (!is.null(efa_2f)) efa_stats(efa_2f) else list(phi_avg = NA_real_, phi = NULL, var_factors = numeric(0), var_avg = NA_real_)
    ho_2f <- do_bifactor(fr$R, 2L, method, dz, st, "2f", n_obs = fr$n_eff)
    if (!is.null(ho_2f)) {
      omega_hs_2f <- if (!is.null(ho_2f$omega_group) && "group" %in% colnames(ho_2f$omega_group))
        ho_2f$omega_group[rownames(ho_2f$omega_group) != "g", "group"] else numeric(0)
      db_insert_factoring(method, dataset, "forced2f", 2L, var_explained,
                          st_2f$var_factors, st_2f$var_avg,
                          ho_2f$omega_total, ho_2f$omega_h, omega_hs_2f,
                          st_2f$phi_avg, st_2f$phi, DB_FILE)
    }
    return(invisible())
  }

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
  st_pa <- efa_stats(fr$efa)
  cat(sprintf("  factored: nf = %d  cumvar = %.3f  phi_avg = %.3f\n",
              pa_nf, var_explained, st_pa$phi_avg))

  ho_pa <- do_bifactor(M, pa_nf, method, dz, st, "pa")
  if (!is.null(ho_pa)) {
    omega_hs_pa <- if (!is.null(ho_pa$omega_group) && "group" %in% colnames(ho_pa$omega_group))
      ho_pa$omega_group[rownames(ho_pa$omega_group) != "g", "group"] else numeric(0)
    db_insert_factoring(method, dataset, "pa", pa_nf, var_explained,
                        st_pa$var_factors, st_pa$var_avg,
                        ho_pa$omega_total, ho_pa$omega_h, omega_hs_pa,
                        st_pa$phi_avg, st_pa$phi, DB_FILE)
  }

  efa_2f <- fa_try(M, 2L)
  st_2f <- if (!is.null(efa_2f)) efa_stats(efa_2f) else list(phi_avg = NA_real_, phi = NULL, var_factors = numeric(0), var_avg = NA_real_)
  ho_2f <- do_bifactor(M, 2L, method, dz, st, "2f")
  if (!is.null(ho_2f)) {
    omega_hs_2f <- if (!is.null(ho_2f$omega_group) && "group" %in% colnames(ho_2f$omega_group))
      ho_2f$omega_group[rownames(ho_2f$omega_group) != "g", "group"] else numeric(0)
    db_insert_factoring(method, dataset, "forced2f", 2L, var_explained,
                        st_2f$var_factors, st_2f$var_avg,
                        ho_2f$omega_total, ho_2f$omega_h, omega_hs_2f,
                        st_2f$phi_avg, st_2f$phi, DB_FILE)
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
