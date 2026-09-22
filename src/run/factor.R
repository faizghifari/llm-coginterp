#!/usr/bin/env Rscript
# ─────────────────────────────────────────────────────────────────────────────
# Factor-analysis-only orchestrator.
#
# Reads COMPLETED matrices (from data/imputed/) written by the imputation stage,
# gates on imputation R² >= 0.3 (SQLite), then runs two bifactor analyses per
# cell: one at the PA-based factor count (min 2) and one forced to 2 factors.
# The fill-smooth methods (default/zeros) are gated like every other imputer but
# factor the smoothed correlation their imputation persisted
# (results/<method>/<method>_<dz>_<st>_correlation.csv) instead of the completed
# surrogate — that cached matrix is the exact fill+PSD recipe output.
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
#     --timed        year-separated factoring: partition rows by release year
#                    (collapse_mapping.csv, written by scripts/collapse_results.py)
#                    and run the full PA -> EFA -> bifactor pipeline per cohort.
#                    Skips the fill-smooth methods (default/zeros: their cached
#                    correlation is global, not per cohort); incompatible
#                    with --loco. Output suffixes get y<year>, DB runs become
#                    pa_y<year> / forced2f_y<year> (dataset stays <dz>_<st>).
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
RAW_METHODS <- c("default", "zeros")   # fill-smooth: factor the cached correlation
parse_args <- function(args) {
  method <- "all"; raw <- FALSE; smoke <- FALSE; loco <- FALSE; timed <- FALSE
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
    else if (a == "--timed") { timed <- TRUE; i <- i + 1L }
    else stop("unknown arg: ", a)
  }
  if (method != "all" && !(method %in% ALL_METHODS))
    stop("--method must be one of: ", paste(c("all", ALL_METHODS), collapse = ", "))
  if (timed && loco)
    stop("--timed and --loco are mutually exclusive: per-year cohorts are too small for a LOCO sweep")
  list(methods = if (method == "all") ALL_METHODS else method,
       raw = raw, smoke = smoke, loco = loco, timed = timed,
       data_root = data_root, results_root = results_root)
}
opt <- parse_args(commandArgs(trailingOnly = TRUE))

METHODS    <- opt$methods
DENSIFIERS <- if (opt$raw) "raw" else c("C", "S", "R")
STRATEGIES <- c("all_standard", "all_aggressive")
LOCO       <- opt$loco       # leave-one-covariate-out delta omega_h mode
TIMED      <- opt$timed      # year-partitioned factoring mode
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

# Imputation-R² gate, applied to every method (fill-smooth included: their
# held-out R² lives in the same `imputation` table). Cells whose R² is missing
# or below R2_GATE are skipped.
R2_GATE <- 0.3

gate_r2 <- function(method, dataset, tag, what = "factoring") {
  r2 <- tryCatch(db_read_r2(method, dataset, DB_FILE),
                 error = function(e) { cat("  db read failed:", conditionMessage(e), "\n"); NA_real_ })
  if (is.na(r2) || r2 < R2_GATE) {
    cat(sprintf("  skipping %s (%s) — imputation R² = %s < %.1f\n", what, tag,
                if (is.na(r2)) "NA" else sprintf("%.3f", r2), R2_GATE))
    return(FALSE)
  }
  cat(sprintf("  R² = %.3f >= %.1f, proceeding\n", r2, R2_GATE))
  TRUE
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

# ── timed mode: release-year mapping ─────────────────────────────────────────
# collapse_key -> release year, read from the strategy's collapse_mapping.csv
# (written by scripts/collapse_results.py). The mapping is densifier-independent
# (same key universe across raw/C/S/R), so one file per strategy serves every dz.
# release_date is "YYYY-MM" or bare "YYYY"; the first 19xx/20xx run wins and
# anything unparseable ("chatgpt", "code-002-175B", blank) maps to NA and is
# excluded from timed runs.
YEAR_RX <- "(19|20)[0-9]{2}"
YEAR_MAP_CACHE <- new.env(parent = emptyenv())

year_map_for <- function(st) {
  if (!is.null(YEAR_MAP_CACHE[[st]])) return(YEAR_MAP_CACHE[[st]])
  path <- file.path(DATA_ROOT, "combinations", st, "collapse_mapping.csv")
  if (!file.exists(path))
    stop("timed mode requires ", path, " — run `make preproc` first")
  df <- read.csv(path, colClasses = "character")
  m <- regexpr(YEAR_RX, df$release_date)
  yr <- rep(NA_integer_, nrow(df))
  hit <- !is.na(m) & m > 0
  yr[hit] <- as.integer(substr(df$release_date[hit], m[hit],
                               m[hit] + attr(m, "match.length")[hit] - 1L))
  ym <- setNames(yr, df$collapse_key)
  ym <- ym[!is.na(ym)]
  ym <- ym[!duplicated(names(ym))]
  cat(sprintf("timed: %d collapse keys with a release year (%s)\n", length(ym), st))
  assign(st, ym, envir = YEAR_MAP_CACHE)
  ym
}

# Shared imputed-method factoring core: R² gate, PA factor count, bifactor at
# the PA count and at forced 2f. run_suffix tags the DB run names and output
# filenames: "" for the combined run ("pa" / "forced2f"), "_y<year>" for timed
# per-cohort runs ("pa_y2023" / "forced2f_y2023"). The R² gate is per cell —
# it describes the imputation quality of the full matrix and is reused as-is
# for every year cohort factored from that matrix.
factor_imputed_core <- function(method, dz, st, M, run_suffix = "") {
  dataset <- paste0(dz, "_", st)
  tag <- sprintf("%s/%s/%s", method, dz, st)

  if (!gate_r2(method, dataset, tag)) return(invisible())

  run_pa <- paste0("pa", run_suffix)
  run_2f <- paste0("forced2f", run_suffix)

  fr <- factor_matrix(M, pa_iter = 100L)
  pa_nf <- fr$nf
  var_explained <- extract_variance(fr$efa)
  st_pa <- efa_stats(fr$efa)
  cat(sprintf("  factored: nf = %d  cumvar = %.3f  phi_avg = %.3f\n",
              pa_nf, var_explained, st_pa$phi_avg))

  ho_pa <- do_bifactor(M, pa_nf, method, dz, st, run_pa)
  if (!is.null(ho_pa)) {
    omega_hs_pa <- if (!is.null(ho_pa$omega_group) && "group" %in% colnames(ho_pa$omega_group))
      ho_pa$omega_group[rownames(ho_pa$omega_group) != "g", "group"] else numeric(0)
    db_insert_factoring(method, dataset, run_pa, pa_nf, var_explained,
                        st_pa$var_factors, st_pa$var_avg,
                        ho_pa$omega_total, ho_pa$omega_h, omega_hs_pa,
                        st_pa$phi_avg, st_pa$phi, DB_FILE)
  }

  efa_2f <- fa_try(M, 2L)
  st_2f <- if (!is.null(efa_2f)) efa_stats(efa_2f) else list(phi_avg = NA_real_, phi = NULL, var_factors = numeric(0), var_avg = NA_real_)
  ho_2f <- do_bifactor(M, 2L, method, dz, st, run_2f)
  if (!is.null(ho_2f)) {
    omega_hs_2f <- if (!is.null(ho_2f$omega_group) && "group" %in% colnames(ho_2f$omega_group))
      ho_2f$omega_group[rownames(ho_2f$omega_group) != "g", "group"] else numeric(0)
    db_insert_factoring(method, dataset, run_2f, 2L, var_explained,
                        st_2f$var_factors, st_2f$var_avg,
                        ho_2f$omega_total, ho_2f$omega_h, omega_hs_2f,
                        st_2f$phi_avg, st_2f$phi, DB_FILE)
  }
  invisible()
}

# Timed-mode driver for one cell: skip the fill-smooth methods (their cached
# correlation is built from the whole sparse table — a year subset would be far
# too thin to mean anything), then partition the completed matrix's rows into
# release-year cohorts and factor each one independently.
# There is deliberately no minimum cohort size: degenerate years fail inside
# factor_matrix, log FACTOR FAILED, and the loop moves on.
factor_timed_cell <- function(method, dz, st, M, keys) {
  tag <- sprintf("%s/%s/%s", method, dz, st)
  if (method %in% RAW_METHODS) {
    cat(sprintf("  skipping timed (%s) — fill-smooth methods factor a global cached correlation, not per-cohort data\n", tag))
    return(invisible())
  }
  if (is.null(keys)) {
    cat(sprintf("  skipping timed (%s) — no row keys\n", tag))
    return(invisible())
  }
  ym <- year_map_for(st)
  yr <- unname(ym[as.character(keys)])
  have <- !is.na(yr)
  cat(sprintf("  timed: %d/%d rows have a release year (%d cohorts)\n",
              sum(have), length(yr), length(unique(yr[have]))))
  for (y in sort(unique(yr[have]))) {
    idx <- which(yr == y)
    cat(sprintf("  ---- year %d (n = %d) ----\n", y, length(idx)))
    tryCatch(
      factor_imputed_core(method, dz, st, M[idx, , drop = FALSE],
                          run_suffix = sprintf("_y%d", y)),
      error = function(e)
        cat(sprintf("  FACTOR FAILED (year %d): %s\n", y, conditionMessage(e))))
  }
  invisible()
}

factor_and_report <- function(method, dz, st, M, keys = NULL) {
  tag <- sprintf("%s/%s/%s", method, dz, st)
  dataset <- paste0(dz, "_", st)

  if (TIMED) return(factor_timed_cell(method, dz, st, M, keys))

  if (LOCO) {
    if (!gate_r2(method, dataset, tag, what = "LOCO")) return(invisible())
    if (method %in% RAW_METHODS) {
      # Fill-smooth methods never re-derive the correlation matrix: they load
      # the one the imputation stage persisted (correlation.csv) and peel
      # row/column i per covariate inside loco_delta.
      R     <- read_correlation_csv(res_path(method, dz, st, "correlation.csv"))
      n_obs <- nrow(M)
      nf_pa <- choose_nfactors_cached_R(R, n_obs)$nf
    } else {
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
    if (!gate_r2(method, dataset, tag)) return(invisible())
    cat(sprintf("  %s factoring — cached fill+smooth correlation from imputation\n", method))
    R <- read_correlation_csv(res_path(method, dz, st, "correlation.csv"))
    fr <- factor_cached_R(R, nrow(M), pa_iter = 100L)
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

  factor_imputed_core(method, dz, st, M)
}

main <- function() {
  for (method in METHODS) for (st in STRATEGIES) {
    for (dz in DENSIFIERS) {
      cat("\n======== ", method, "/", dz, "/", st, " ========\n", sep = "")
      res <- build_contract_from_disk(method, dz, st)
      if (is.null(res)) next
      tryCatch(factor_and_report(method, dz, st, res$M, res$keys),
               error = function(e) cat("  FACTOR FAILED:", conditionMessage(e), "\n"))
    }
  }
  cat("\nDONE.\n  results -> ", RESULTS_ROOT, "/\n", sep = "")
}

main()
