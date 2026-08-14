#!/usr/bin/env Rscript
# ─────────────────────────────────────────────────────────────────────────────
# Pipeline orchestrator.
#
# Cross-product: {densifier C,R,S} x {strategy all_standard, all_aggressive} x
# {method softimpute, iterativepca, onesidedmc, knn, missforest, mice, raw}.
#
# For each cell:
#   1. impute     -> completed (or, for OSMC, covariance-surrogate) matrix
#   2. factor     -> shared parallel analysis (cached) + two bifactor runs
#                     (PA-based and forced 2-factor)
#
# Output split:
#   data/imputed/<method>/<densifier>/<strategy>/  -> ONLY the imputed CSV
#   results/                                       -> bifactor CSVs/MDs + database
#
# OSMC is implemented in Julia; we shell out to its run.jl to produce per-r
# surrogate CSVs, then factor them through the SAME shared R path as the others.
#
# Run from anywhere:
#   Rscript src/run/main.R [--method <name>] [--raw] [--smoke]
#     --method       softimpute | softimpute_corr | optspace | usvt | iterativepca | onesidedmc | raw | all   (default all)
#     --raw          run ONLY the slow undensified "raw" level (default: C,S,R)
#     --smoke        use the data/smoke fixture instead of data/
#     --data-root    input tree, relative to the repo root (default data; e.g.
#                    data/text_only for the derived text-only copy)
#     --results-root output tree, relative to the repo root (default results)
#     --reimpute     force fresh imputation even if an imputed CSV exists
#     --no-balance   revert to cell-weighted holdout scores
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
source(file.path(SRC, "impute", "db.R"))
source(file.path(SRC, "factor", "factoring.R"))
source(file.path(SRC, "factor", "db.R"))

# ── Argument parsing ─────────────────────────────────────────────────────────
ALL_METHODS <- c("softimpute", "softimpute_corr", "iterativepca",
                 "onesidedmc", "knn", "missforest", "mice",
                 "optspace", "usvt", "raw")
parse_args <- function(args) {
  method <- "all"; smoke <- FALSE; raw <- FALSE
  reimpute <- FALSE; no_balance <- FALSE; loco <- FALSE
  data_root <- "data"; results_root <- "results"
  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]
    if (a == "--method") { method <- args[[i + 1L]]; i <- i + 2L }
    else if (a == "--data-root")    { data_root    <- args[[i + 1L]]; i <- i + 2L }
    else if (a == "--results-root") { results_root <- args[[i + 1L]]; i <- i + 2L }
    else if (a == "--smoke")       { smoke      <- TRUE; i <- i + 1L }
    else if (a == "--raw")         { raw        <- TRUE; i <- i + 1L }
    else if (a == "--reimpute")    { reimpute   <- TRUE; i <- i + 1L }
    else if (a == "--no-balance")  { no_balance <- TRUE; i <- i + 1L }
    else if (a == "--loco")        { loco       <- TRUE; i <- i + 1L }
    else stop("unknown arg: ", a)
  }
  if (method != "all" && !(method %in% ALL_METHODS))
    stop("--method must be one of: ", paste(c("all", ALL_METHODS), collapse = ", "))
  list(methods = if (method == "all") ALL_METHODS else method,
       smoke = smoke, raw = raw, reimpute = reimpute,
       no_balance = no_balance, loco = loco,
       data_root = data_root, results_root = results_root)
}
opt <- parse_args(commandArgs(trailingOnly = TRUE))

METHODS    <- opt$methods
# "raw" = the undensified aggregated table (slow). --raw runs ONLY raw; without
# it, only the densified levels C/S/R run, so raw can be run separately.
DENSIFIERS <- if (opt$raw) "raw" else c("C", "S", "R")
STRATEGIES <- c("all_standard", "all_aggressive")
REIMPUTE   <- opt$reimpute   # force fresh imputation even if an imputed CSV exists
LOCO       <- opt$loco       # leave-one-covariate-out delta omega_h mode
# --no-balance: revert held-out RMSE/R^2 to the old cell-weighted score (high-
# frequency columns dominate). Default is column-balanced (mean of per-column
# scores). Set in common.R's BALANCE_HOLDOUT, which the scorers read.
BALANCE_HOLDOUT <- !opt$no_balance
MAX_RANK   <- 10L
DATA_ROOT  <- file.path(REPO, if (opt$smoke) "data/smoke" else opt$data_root)
RESULTS_ROOT <- file.path(REPO, if (opt$smoke) "results/smoke" else opt$results_root)
dir.create(RESULTS_ROOT, recursive = TRUE, showWarnings = FALSE)
cat(sprintf("methods=[%s]  data_root=%s  results=%s  reimpute=%s\n",
            paste(METHODS, collapse = ","), DATA_ROOT, RESULTS_ROOT, REIMPUTE))

DB_FILE <- file.path(RESULTS_ROOT, "database.db")

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
  } else if (method == "softimpute_corr") {
    source(file.path(SRC, "impute", "corr_common.R"))
    source(file.path(SRC, "impute", "softimpute_corr", "method.R"))
    impute_softimpute_corr(x, max_rank = MAX_RANK)
  } else if (method == "optspace") {
    source(file.path(SRC, "impute", "corr_common.R"))
    source(file.path(SRC, "impute", "optspace", "method.R"))
    impute_optspace(x)
  } else if (method == "usvt") {
    source(file.path(SRC, "impute", "corr_common.R"))
    source(file.path(SRC, "impute", "usvt", "method.R"))
    impute_usvt(x)
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

run_osmc_subprocess <- function() {
  cat("\n##### OneSidedMC (Julia subprocess) #####\n")
  Sys.setenv(OSMC_DENSIFIERS   = paste(DENSIFIERS, collapse = ","),
             OSMC_STRATEGIES   = paste(STRATEGIES, collapse = ","),
             OSMC_DATA_ROOT    = normalizePath(DATA_ROOT, mustWork = FALSE),
             OSMC_RESULTS_ROOT = normalizePath(RESULTS_ROOT, mustWork = FALSE),
             OSMC_SENSITIVITY  = "",
             OSMC_BALANCE      = if (BALANCE_HOLDOUT) "1" else "0",
             OSMC_ALLCOLHOLDOUT = if (ALLCOLHOLDOUT) "1" else "0")
  # --threads=auto so OSMC uses all cores. Paths are
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
# best-r surrogate from data/imputed + sweep curve from results/_osmc_sweep.
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
       param_name = "r", metric_name = "Held-out RMSE")
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

# Factor at best param: PA-based bifactor + forced 2-factor bifactor.
factor_and_report <- function(method, dz, st, M) {
  tag <- sprintf("%s/%s/%s", method, dz, st)
  dataset <- paste0(dz, "_", st)

  if (LOCO) {
    if (method == "raw") {
      prep    <- prepare_raw_cor(M)
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

  if (method == "raw") {
    cat(sprintf("  raw factoring — pairwise-complete correlation (no imputation R² gate)\n"))
    fr <- factor_raw(M, pa_iter = 100L)
    pa_nf <- fr$nf
    var_explained <- extract_variance(fr$efa)
    st_pa <- efa_stats(fr$efa)
    cat(sprintf("  factored: nf = %d  cumvar = %.3f  n_eff = %d  phi_avg = %.3f\n",
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

# Impute + factor for one cell.
run_cell <- function(method, dz, st) {
  tag <- sprintf("%s/%s/%s", method, dz, st)
  cat("\n======== ", tag, " ========\n", sep = "")

  if (method == "raw") {
    src <- combos_path(dz, st)
    if (!file.exists(src)) { cat("  missing input:", src, "\n"); return() }
    pm <- prep_matrix(src)
    cat(sprintf("  matrix: %d x %d, %.1f%% observed  (no imputation)\n",
                nrow(pm$x), ncol(pm$x), 100 * mean(!is.na(pm$x))))
    tryCatch(factor_and_report(method, dz, st, pm$x),
             error = function(e) cat("  FACTOR FAILED:", conditionMessage(e), "\n"))
    return()
  }

  out_dir <- imputed_dir(method, dz, st, root = file.path(DATA_ROOT, "imputed"))

  if (method == "onesidedmc") {
    # surrogates already written by the Julia subprocess; adapt + factor.
    imputed_csv <- file.path(out_dir, "imputed_model_benchmark_table.csv")
    res <- osmc_contract(dz, st, imputed_csv)
    if (is.null(res)) { cat("  no OSMC outputs, skipping\n"); return() }
    best_idx  <- which(res$params == res$best_param)
    best_rmse <- res$curve[best_idx]
    best_r2   <- if (!is.null(res$curve_r2)) res$curve_r2[best_idx] else NA
    desc <- build_desc(res$params, res$param_name, res$best_param)
    db_insert_imputation(method, paste0(dz, "_", st),
                         best_rmse, best_r2, desc, DB_FILE, REIMPUTE)
    tryCatch(factor_and_report(method, dz, st, res$M),
             error = function(e) cat("  FACTOR FAILED:", conditionMessage(e), "\n"))
    return()
  }

  # R imputers
  src <- combos_path(dz, st)
  if (!file.exists(src)) { cat("  missing input:", src, "\n"); return() }
  pm <- prep_matrix(src)
  x <- pm$x
  cat(sprintf("  matrix: %d x %d, %.1f%% observed\n",
              nrow(x), ncol(x), 100 * mean(!is.na(x))))

  imputed_csv <- file.path(out_dir, "imputed_model_benchmark_table.csv")
  sweep_csv   <- res_path(method, dz, st, "rank_sweep.csv")

  # --reimpute default OFF: reuse an existing imputed CSV (skip the slow impute),
  # rebuild a partial contract from disk, and just re-factor.
  if (!REIMPUTE && file.exists(imputed_csv)) {
    cat("  reusing existing imputed CSV (skip imputation; use --reimpute to force)\n")
    mb <- read_matrix(imputed_csv)
    tryCatch(factor_and_report(method, dz, st, mb$M),
             error = function(e) cat("  FACTOR FAILED:", conditionMessage(e), "\n"))
    return()
  }

  res <- tryCatch(impute_R(method, x), error = function(e) {
    cat("  IMPUTE FAILED:", conditionMessage(e), "\n"); NULL })
  if (is.null(res)) return()

  write_completed(out_dir, pm$keys, res$M)   # data/imputed: CSV only
  # persist the sweep curve so a later --reimpute-off run can rebuild the
  # the sweep CSV.
  write.csv(data.frame(param = res$params, param_name = res$param_name,
                       rmse = res$curve,
                       r2 = if (!is.null(res$curve_r2)) res$curve_r2 else NA),
            sweep_csv, row.names = FALSE)
  best_idx  <- which(res$params == res$best_param)
  best_rmse <- res$curve[best_idx]
  best_r2   <- if (!is.null(res$curve_r2)) res$curve_r2[best_idx] else NA
  desc <- build_desc(res$params, res$param_name, res$best_param)
  db_insert_imputation(method, paste0(dz, "_", st),
                       best_rmse, best_r2, desc, DB_FILE, REIMPUTE)
  tryCatch(factor_and_report(method, dz, st, res$M),
           error = function(e) cat("  FACTOR FAILED:", conditionMessage(e), "\n"))
}

# ── Main loop ────────────────────────────────────────────────────────────────
main <- function() {
  if ("onesidedmc" %in% METHODS) run_osmc_subprocess()

  for (method in METHODS) for (st in STRATEGIES) {
    for (dz in DENSIFIERS) {
      run_cell(method, dz, st)
    }
  }

  cat("\nDONE.\n  imputed CSVs -> ", file.path(DATA_ROOT, "imputed"),
      "/<method>/<densifier>/<strategy>/\n  results       -> ", RESULTS_ROOT,
      "/ (flat)\n", sep = "")
}

main()