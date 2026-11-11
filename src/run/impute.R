#!/usr/bin/env Rscript
# ─────────────────────────────────────────────────────────────────────────────
# Imputation-only orchestrator.
#
# Runs imputation + held-out sweep for every (method × densifier × strategy)
# cell. Saves completed CSVs and sweep curves. DOES NOT factor. The fill-smooth
# methods (default/zeros) additionally persist the exact smoothed correlation
# their recipe produced (..._correlation.csv) — the matrix the EFA and the
# conditional/prorated latent scoring run on.
#
# Output:
#   data/imputed/<method>/<densifier>/<strategy>/  -> completed CSV + keys
#   results/<method>/<method>_<dz>_<st>_rank_sweep.csv
#   results/<method>/<method>_<dz>_<st>_correlation.csv   (default/zeros only)
#   results/<prefix>/database.db -> imputation table rows
#
# Run from anywhere:
#   Rscript src/run/impute.R [--method <name>] [--raw] [--reimpute]
#     --method       softimpute | softimpute_corr | optspace | usvt | iterativepca | onesidedmc | knn | missforest | mice | cvxr | ggm | default | zeros | all
#     --raw          run ONLY the undensified "raw" level (default: C,S,R)
#     --reimpute     force fresh imputation even if an imputed CSV exists
#     --data-root    input tree, relative to repo root
#                    (default data/text_only -- the analysis corpus; pass
#                     `--data-root data` for the multimodal-inclusive one)
#     --results-root output tree, relative to repo root
#                    (default results/text_only)
#     --density      REQUIRED. Density slice, e.g. 0.1. Roots gain a density
#                    segment: <data-root>/<density>/ and <results-root>/<density>/.
#                    --synth and --smoke fix their own layouts and reject it.
#     --smoke        use data/smoke fixture
#     --no-balance   revert to cell-weighted holdout scores
#     --synth        run the synthetic stress corpus instead: loops over every
#                    data/synth/<cell>/<mask>/s0 drop-in root (C/S all_standard
#                    only). Completed CSVs land inside each root as usual; flat
#                    results go to results/synth/<cell>/<mask>/ and DB rows to
#                    results/synth/database.db, table imputation_synth
#                    (keyed by cell, mask, dataset, method). Mutually exclusive
#                    with --smoke/--raw/--data-root-style overrides.
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

ALL_METHODS <- c("softimpute", "softimpute_corr", "iterativepca",
                 "onesidedmc", "knn", "missforest", "mice",
                 "optspace", "usvt", "cvxr", "ggm",
                 "default", "zeros")
# Fill-smooth methods: their contract carries the extra smoothed correlation R,
# persisted as ..._correlation.csv next to the flat results (the completed CSV
# alone cannot recover it).
RAW_METHODS <- c("default", "zeros")

# Enumerate the synthetic stress corpus (scripts/make_mnar.py output): every
# data/synth/<cell>/<mask>/s0 drop-in data root. `truth/` and `summary.csv`
# carry no s0/ combinations tree and are excluded by the check below.
synth_roots <- function(base) {
  roots <- list()
  for (cell in sort(list.files(base))) {
    cell_dir <- file.path(base, cell)
    if (!dir.exists(cell_dir)) next
    for (mask in sort(list.files(cell_dir))) {
      root <- file.path(cell_dir, mask, "s0")
      if (dir.exists(file.path(root, "combinations_C")) ||
            dir.exists(file.path(root, "combinations_S")))
        roots[[length(roots) + 1L]] <-
          list(cell = cell, mask = mask, root = root)
    }
  }
  roots
}

parse_args <- function(args) {
  method <- "all"; raw <- FALSE; smoke <- FALSE
  reimpute <- FALSE; no_balance <- FALSE; synth <- FALSE
  data_root <- "data/text_only"; results_root <- "results/text_only"
  density <- NULL
  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]
    if (a == "--method") { method <- args[[i + 1L]]; i <- i + 2L }
    else if (a == "--data-root")    { data_root    <- args[[i + 1L]]; i <- i + 2L }
    else if (a == "--results-root") { results_root <- args[[i + 1L]]; i <- i + 2L }
    else if (a == "--density")      { density      <- args[[i + 1L]]; i <- i + 2L }
    else if (a == "--smoke")       { smoke      <- TRUE; i <- i + 1L }
    else if (a == "--raw")         { raw        <- TRUE; i <- i + 1L }
    else if (a == "--reimpute")    { reimpute   <- TRUE; i <- i + 1L }
    else if (a == "--no-balance")  { no_balance <- TRUE; i <- i + 1L }
    else if (a == "--synth")       { synth      <- TRUE; i <- i + 1L }
    else stop("unknown arg: ", a)
  }
  if (method != "all" && !(method %in% ALL_METHODS))
    stop("--method must be one of: ", paste(c("all", ALL_METHODS), collapse = ", "))
  if (is.null(density)) stop("--density <fraction> is required (e.g. --density 0.1)")
  if (synth && smoke) stop("--synth and --smoke are mutually exclusive")
  if (synth && raw)   stop("--synth covers only the C/S all_standard cells (no raw level)")
  if (synth && (data_root != "data/text_only" || results_root != "results/text_only"))
    stop("--synth fixes its own data roots and results layout; drop --data-root/--results-root")
  if (synth)
    stop("--synth fixes its own data roots and results layout; drop --density")
  if (smoke)
    stop("--smoke uses the fixed data/smoke fixture (no density dimension); drop --density")
  list(methods = if (method == "all") ALL_METHODS else method,
       smoke = smoke, raw = raw, reimpute = reimpute,
       no_balance = no_balance, synth = synth, density = density,
       data_root = data_root, results_root = results_root)
}
opt <- parse_args(commandArgs(trailingOnly = TRUE))

METHODS    <- opt$methods
DENSIFIERS <- if (opt$raw) "raw" else c("C", "S", "R")
STRATEGIES <- c("all_standard", "all_aggressive")
REIMPUTE   <- opt$reimpute
BALANCE_HOLDOUT <- !opt$no_balance
MAX_RANK   <- 10L
SYNTH      <- opt$synth
# Active synth cell/mask; NULL (and unused) outside --synth mode. --synth
# re-points DATA_ROOT/RESULTS_ROOT per (cell, mask) inside main().
SYNTH_CELL <- NULL
SYNTH_MASK <- NULL
# --density nests a density segment into both roots: inputs read from
# <data-root>/<density>/ (combinations_<dz>, imputed/), outputs go to
# <results-root>/<density>/ (flat artifacts + database.db).
DATA_ROOT  <- file.path(REPO, if (opt$smoke) "data/smoke"
                        else file.path(opt$data_root, opt$density))
RESULTS_ROOT <- file.path(REPO, if (opt$smoke) "results/smoke"
                          else file.path(opt$results_root, opt$density))
if (SYNTH) {
  # The generator writes only combinations_C/S at all_standard; nothing else
  # exists to skip.
  DENSIFIERS <- c("C", "S")
  DB_FILE    <- file.path(REPO, "results", "synth", "database.db")
} else {
  dir.create(RESULTS_ROOT, recursive = TRUE, showWarnings = FALSE)
  DB_FILE <- file.path(RESULTS_ROOT, "database.db")
  cat(sprintf("impute: methods=[%s]  data_root=%s  results=%s  reimpute=%s\n",
              paste(METHODS, collapse = ","), DATA_ROOT, RESULTS_ROOT, REIMPUTE))
}
if (SYNTH) {
  .synth_roots <- synth_roots(file.path(REPO, "data", "synth"))
  if (!length(.synth_roots))
    stop("no data/synth/<cell>/<mask>/s0 roots found — run scripts/make_mnar.py first")
  cat(sprintf("impute --synth: methods=[%s]  roots=%d  results=results/synth  reimpute=%s\n",
              paste(METHODS, collapse = ","), length(.synth_roots), REIMPUTE))
}

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
  } else if (method == "cvxr") {
    source(file.path(SRC, "impute", "corr_common.R"))
    source(file.path(SRC, "impute", "cvxr", "method.R"))
    impute_cvxr(x)
  } else if (method == "ggm") {
    source(file.path(SRC, "impute", "corr_common.R"))
    source(file.path(SRC, "impute", "ggm", "method.R"))
    impute_ggm(x)
  } else if (method == "default") {
    source(file.path(SRC, "impute", "corr_common.R"))
    source(file.path(SRC, "impute", "default", "method.R"))
    impute_default(x)
  } else if (method == "zeros") {
    source(file.path(SRC, "impute", "corr_common.R"))
    source(file.path(SRC, "impute", "zeros", "method.R"))
    impute_zeros(x)
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
             OSMC_ALLCOLHOLDOUT = "1")
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
       param_name = "r", metric_name = "Held-out RMSE")
}

# One imputation row: the standard `imputation` table, or `imputation_synth`
# (keyed by cell/mask, gt_r2 reserved for truth scoring) in --synth mode.
db_record <- function(method, dataset, rmse, r2, desc) {
  if (!SYNTH)
    return(db_insert_imputation(method, dataset, rmse, r2, desc,
                                DB_FILE, REIMPUTE))
  db_insert_imputation(method, dataset, rmse, r2, desc, DB_FILE, REIMPUTE,
                       table = "imputation_synth",
                       cell = SYNTH_CELL, mask = SYNTH_MASK)
}

# Ground-truth recovery R^2 for one --synth cell: score the completed matrix
# against the planted truth (data/synth/truth/<tag>_<cell>_complete.csv) on
# exactly the cells the masked input was MISSING — the imputation target.
# (Observed cells pass through every imputer verbatim, so scoring them would
# only re-measure observation frequency.) Per-column z-stats come from the
# masked input's observed cells — the same baseline as the held-out metric
# (train column mean = 0 in z-space) — and the aggregation is column-balanced,
# matching score_holdout's shape: missing-cell counts differ wildly per
# column (that IS the MNAR mechanism), and cell-weighting would let the
# dense columns dominate the score. Returns NA when nothing can be scored.
gt_r2_score <- function(masked_csv, imputed_csv, truth_csv) {
  mask_df <- read.csv(masked_csv, check.names = FALSE)
  imp_df  <- read.csv(imputed_csv,  check.names = FALSE)
  tru_df  <- read.csv(truth_csv,    check.names = FALSE)
  keys    <- intersect(imp_df$collapse_key, tru_df$collapse_key)
  cols    <- intersect(names(imp_df)[-1], names(tru_df)[-1])
  if (!length(keys) || !length(cols)) return(NA_real_)
  ri <- match(keys, imp_df$collapse_key)
  rt <- match(keys, tru_df$collapse_key)
  rm <- match(keys, mask_df$collapse_key)
  M   <- as.matrix(imp_df[ri, cols, drop = FALSE])
  tru <- as.matrix(tru_df[rt, cols, drop = FALSE])
  msk <- as.matrix(mask_df[rm, cols, drop = FALSE])
  storage.mode(M) <- storage.mode(tru) <- storage.mode(msk) <- "double"

  # Columns whose z-stats are unusable (no observed cells / zero variance)
  # cannot be scored — prep_matrix normally drops these upstream anyway.
  mu  <- colMeans(msk, na.rm = TRUE)
  sds <- apply(msk, 2L, function(c) stats::sd(c[!is.na(c)]))
  ok  <- is.finite(mu) & is.finite(sds) & sds > 0
  if (!any(ok)) return(NA_real_)
  M   <- M[, ok, drop = FALSE]
  tru <- tru[, ok, drop = FALSE]
  msk <- msk[, ok, drop = FALSE]
  mu  <- mu[ok]; sds <- sds[ok]
  miss <- is.na(msk)
  if (!any(miss)) return(NA_real_)

  zt <- sweep(sweep(tru, 2L, mu, "-"), 2L, sds, "/")
  zi <- sweep(sweep(M,   2L, mu, "-"), 2L, sds, "/")
  resid2 <- (zi - zt)^2
  base2  <- zt^2
  resid2[!miss] <- NA_real_
  base2[!miss]  <- NA_real_
  colmean <- function(X) apply(X, 2L, function(c) mean(c[is.finite(c)]))
  mse  <- colmean(resid2)   # per-column mean squared error (missing cells)
  base <- colmean(base2)    # per-column mean baseline
  keep <- is.finite(mse) & is.finite(base) & base > 0
  if (!any(keep)) return(NA_real_)
  1 - mean(mse[keep]) / mean(base[keep])  # single pooled ratio, as in score_holdout
}

# Score every completed matrix of the current synth root against the planted
# truth and refresh imputation_synth.gt_r2. Runs AFTER the imputation pass
# and on EVERY invocation — even when the completed CSVs were reused without
# --reimpute — because it only reads artifacts from disk (cheap).
score_gt_r2 <- function(cell) {
  for (method in METHODS) for (st in STRATEGIES) {
    for (dz in DENSIFIERS) {
      dataset   <- paste0(dz, "_", st)
      truth_csv <- file.path(REPO, "data", "synth", "truth",
                             paste0(dataset, "_", cell, "_complete.csv"))
      imp_csv   <- file.path(DATA_ROOT, "imputed", method, dz, st,
                             "imputed_model_benchmark_table.csv")
      if (!file.exists(imp_csv)) next
      if (!file.exists(truth_csv)) {
        cat("  gt: missing truth file:", truth_csv, "\n"); next
      }
      gt <- gt_r2_score(combos_path(dz, st), imp_csv, truth_csv)
      db_update_gt_r2(cell, SYNTH_MASK, dataset, method, gt, DB_FILE)
      cat(sprintf("  gt: %s/%s gt_r2=%s\n", dataset, method,
                  if (is.na(gt)) "NA" else sprintf("%.4f", gt)))
    }
  }
}

run_cell <- function(method, dz, st) {
  tag <- sprintf("%s/%s/%s", method, dz, st)
  cat("\n======== ", tag, " ========\n", sep = "")
  out_dir <- imputed_dir(method, dz, st, root = file.path(DATA_ROOT, "imputed"))

  if (method == "onesidedmc") {
    imputed_csv <- file.path(out_dir, "imputed_model_benchmark_table.csv")
    res <- osmc_contract(dz, st, imputed_csv)
    if (is.null(res)) { cat("  no OSMC outputs, skipping\n"); return() }
    best_idx  <- which(res$params == res$best_param)
    best_rmse <- res$curve[best_idx]
    best_r2   <- if (!is.null(res$curve_r2)) res$curve_r2[best_idx] else NA
    desc <- build_desc(res$params, res$param_name, res$best_param)
    db_record(method, paste0(dz, "_", st), best_rmse, best_r2, desc)
    return()
  }

  src <- combos_path(dz, st)
  if (!file.exists(src)) { cat("  missing input:", src, "\n"); return() }
  pm <- prep_matrix(src)
  x <- pm$x
  cat(sprintf("  matrix: %d x %d, %.1f%% observed\n",
              nrow(x), ncol(x), 100 * mean(!is.na(x))))

  imputed_csv <- file.path(out_dir, "imputed_model_benchmark_table.csv")
  sweep_csv   <- res_path(method, dz, st, "rank_sweep.csv")
  # fill-smooth methods also persist the correlation cache; both artifacts must
  # exist before imputation can be skipped.
  cache_csv   <- res_path(method, dz, st, "correlation.csv")

  if (!REIMPUTE && file.exists(imputed_csv) &&
      (!(method %in% RAW_METHODS) || file.exists(cache_csv))) {
    cat("  reusing existing imputed CSV (skip imputation; use --reimpute to force)\n")
    return()
  }

  res <- tryCatch(impute_R(method, x), error = function(e) {
    cat("  IMPUTE FAILED:", conditionMessage(e), "\n"); NULL })
  if (is.null(res)) return()

  write_completed(out_dir, pm$keys, res$M)
  if (method %in% RAW_METHODS && !is.null(res$R))
    write_correlation_csv(res$R, cache_csv)
  write.csv(data.frame(param = res$params, param_name = res$param_name,
                       rmse = res$curve,
                       r2 = if (!is.null(res$curve_r2)) res$curve_r2 else NA),
            sweep_csv, row.names = FALSE)
  best_idx  <- which(res$params == res$best_param)
  best_rmse <- res$curve[best_idx]
  best_r2   <- if (!is.null(res$curve_r2)) res$curve_r2[best_idx] else NA
  desc <- build_desc(res$params, res$param_name, res$best_param)
  db_record(method, paste0(dz, "_", st), best_rmse, best_r2, desc)
}

main <- function() {
  if (SYNTH) {
    for (root in .synth_roots) {
      SYNTH_CELL   <<- root$cell
      SYNTH_MASK   <<- root$mask
      DATA_ROOT    <<- root$root
      RESULTS_ROOT <<- file.path(REPO, "results", "synth", root$cell, root$mask)
      dir.create(RESULTS_ROOT, recursive = TRUE, showWarnings = FALSE)
      cat(sprintf("\n#### synth root %s/%s ####\n", root$cell, root$mask))
      if ("onesidedmc" %in% METHODS) run_osmc_subprocess()
      for (method in METHODS) for (st in STRATEGIES) {
        for (dz in DENSIFIERS) {
          run_cell(method, dz, st)
        }
      }
      score_gt_r2(root$cell)
    }
    cat("\nDONE.\n  imputed CSVs -> data/synth/<cell>/<mask>/s0/imputed/<method>/...\n",
        "  results      -> results/synth/<cell>/<mask>/ (flat)\n",
        "  db           -> results/synth/database.db (table imputation_synth)\n",
        sep = "")
    return()
  }

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
