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
#     --method       softimpute | softimpute_corr | optspace | usvt | iterativepca | onesidedmc | knn | missforest | mice | knn_corr | missforest_corr | cvxr | ggm | default | zeros | all
#     --raw          run ONLY the undensified "raw" level (default: C,S,R)
#     --reimpute     force fresh imputation even if an imputed CSV exists
#     --hier         EXPERIMENTAL hierarchical imputation. Descends the density
#                    slices under <data-root> DENSEST-FIRST: the seed is the
#                    standard run's cached completed matrix at the densest
#                    density (abort if absent), then each sparser density is
#                    imputed on the union grid with the previous step's columns
#                    filled in. Holdouts are drawn only from each density's real
#                    observations. Outputs: <data-root>/hierarchical/<density>/
#                    imputed/... and <results-root>/hierarchical/. --density is
#                    accepted (the Makefile always passes it) but IGNORED here —
#                    densities are discovered and sorted descending. The FINAL
#                    (sparsest) step's artifacts are also copied into the
#                    density-"0" slice (<data-root>/0/, <results-root>/0/),
#                    reserved for factoring the hierarchical matrix; that
#                    slice's imputation row carries wr2 (the cumulative
#                    weighted R2) and the factoring gate reads it.
#     --data-root    input tree, relative to repo root
#                    (default data/text_only -- the analysis corpus; pass
#                     `--data-root data` for the multimodal-inclusive one)
#     --results-root output tree, relative to repo root
#                    (default results/text_only)
#     --density      REQUIRED. Density slice, e.g. 0.1. Roots gain a density
#                    segment: <data-root>/<density>/ and <results-root>/<density>/.
#                    --synth requires it too (it names the data/synth/<density>
#                    slice); --smoke rejects it, --hier accepts and ignores it.
#     --smoke        use data/smoke fixture
#     --no-balance   revert to cell-weighted holdout scores
#     --synth        run the synthetic stress corpus instead: loops over every
#                    data/synth/<density>/<cell>/<mask>/s0 drop-in root (C/S
#                    all_standard only). Completed CSVs land inside each root
#                    as usual; flat results go to
#                    results/synth/<density>/<cell>/<mask>/ and DB rows to
#                    results/synth/<density>/database.db, table
#                    imputation_synth (keyed by cell, mask, dataset, method).
#                    Composable with --hier: the hier descent then runs over
#                    the SYNTH density slices, per (cell, mask) — see the hier
#                    block below. Mutually exclusive with
#                    --smoke/--raw/--data-root-style overrides.
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
                 "knn_corr", "missforest_corr",
                 "optspace", "usvt", "cvxr", "ggm",
                 "default", "zeros")
# Fill-smooth methods: their contract carries the extra smoothed correlation R,
# persisted as ..._correlation.csv next to the flat results (the completed CSV
# alone cannot recover it).
RAW_METHODS <- c("default", "zeros")

# Enumerate the synthetic stress corpus (scripts/make_mnar.py output): every
# data/synth/<density>/<cell>/<mask>/s0 drop-in data root under the given
# per-density base. `truth/` and `summary.csv` carry no s0/ combinations tree
# and are excluded by the check below.
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
  reimpute <- FALSE; no_balance <- FALSE; synth <- FALSE; hier <- FALSE
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
    else if (a == "--hier")        { hier       <- TRUE; i <- i + 1L }
    else stop("unknown arg: ", a)
  }
  if (method != "all" && !(method %in% ALL_METHODS))
    stop("--method must be one of: ", paste(c("all", ALL_METHODS), collapse = ", "))
  # --density is required for the real corpus only; --smoke rejects it and
  # --hier discovers the densities itself (it accepts and ignores --density).
  # --synth requires it too: it names the data/synth/<density> slice (in
  # --hier --synth mode the descent discovers the densities and ignores it).
  if (is.null(density) && !smoke && !hier)
    stop("--density <fraction> is required (e.g. --density 0.1)")
  if (synth && smoke) stop("--synth and --smoke are mutually exclusive")
  if (synth && raw)   stop("--synth covers only the C/S all_standard cells (no raw level)")
  if (synth && (data_root != "data/text_only" || results_root != "results/text_only"))
    stop("--synth fixes its own data roots and results layout; drop --data-root/--results-root")
  if (hier && smoke) stop("--hier descends real density slices; drop --smoke")
  if (smoke && !is.null(density))
    stop("--smoke uses the fixed data/smoke fixture (no density dimension); drop --density")
  list(methods = if (method == "all") ALL_METHODS else method,
       smoke = smoke, raw = raw, reimpute = reimpute,
       no_balance = no_balance, synth = synth, hier = hier, density = density,
       data_root = data_root, results_root = results_root)
}
opt <- parse_args(commandArgs(trailingOnly = TRUE))

METHODS    <- opt$methods
DENSIFIERS <- if (opt$raw) "raw" else c("C", "S", "R")
STRATEGIES <- c("all_standard", "all_aggressive")
REIMPUTE   <- opt$reimpute
HIER       <- opt$hier
BALANCE_HOLDOUT <- !opt$no_balance
MAX_RANK   <- 10L
SYNTH      <- opt$synth
# Active synth slice/cell/mask; NULL (and unused) outside --synth mode. --synth
# re-points DATA_ROOT/RESULTS_ROOT per (cell, mask) inside main().
SYNTH_DENSITY <- NULL
SYNTH_CELL <- NULL
SYNTH_MASK <- NULL
# --density nests a density segment into both roots: inputs read from
# <data-root>/<density>/ (combinations_<dz>, imputed/), outputs go to
# <results-root>/<density>/ (flat artifacts + database.db).
# --hier is the exception: it spans densities, so neither root gains a density
# segment. Inputs resolve per density inside the descent; outputs go to
# <data-root>/hierarchical/ and <results-root>/hierarchical/. The --density
# flag the Makefile always passes is parsed and ignored.
DATA_ROOT  <- file.path(REPO, if (opt$smoke) "data/smoke"
                        else if (HIER) opt$data_root
                        else file.path(opt$data_root, opt$density))
RESULTS_ROOT <- file.path(REPO, if (opt$smoke) "results/smoke"
                          else if (HIER) file.path(opt$results_root, "hierarchical")
                          else file.path(opt$results_root, opt$density))
if (SYNTH) {
  # The generator writes only combinations_C/S at all_standard; nothing else
  # exists to skip.
  DENSIFIERS <- c("C", "S")
  if (HIER) {
    # hier-over-synth re-points DB_FILE per density step (hier_db_at).
    DB_FILE <- NULL
  } else {
    DB_FILE <- file.path(REPO, "results", "synth", opt$density, "database.db")
  }
} else {
  dir.create(RESULTS_ROOT, recursive = TRUE, showWarnings = FALSE)
  DB_FILE <- file.path(RESULTS_ROOT, "database.db")
  cat(sprintf("impute: methods=[%s]  data_root=%s  results=%s  reimpute=%s  hier=%s\n",
              paste(METHODS, collapse = ","), DATA_ROOT, RESULTS_ROOT, REIMPUTE,
              HIER))
}
if (SYNTH && !HIER) {
  SYNTH_DENSITY <<- opt$density
  .synth_roots <- synth_roots(file.path(REPO, "data", "synth", opt$density))
  if (!length(.synth_roots))
    stop("no data/synth/", opt$density, "/<cell>/<mask>/s0 roots found — run ",
         "scripts/make_mnar.py --density ", opt$density, " first")
  cat(sprintf("impute --synth: methods=[%s]  density=%s  roots=%d  results=results/synth/%s  reimpute=%s\n",
              paste(METHODS, collapse = ","), opt$density, length(.synth_roots),
              opt$density, REIMPUTE))
}

# density = NULL resolves inside the (density-nested) DATA_ROOT as before;
# a density string (hier mode) prefixes that density segment explicitly —
# except in hier-over-synth mode, where the density selects the s0 root
# data/synth/<density>/<cell>/<mask>/s0 (which carries no inner density level).
combos_path <- function(dz, st, density = NULL) {
  sub <- if (dz == "raw") "combinations" else sprintf("combinations_%s", dz)
  root <- if (is.null(density)) DATA_ROOT
          else if (SYNTH) synth_s0_root(density)
          else file.path(DATA_ROOT, density)
  file.path(root, sub, st, "model_benchmark_table.csv")
}

# density = NULL resolves inside RESULTS_ROOT as before; a density string
# (hier mode) nests that density segment under the hierarchical results tree —
# results/synth/hierarchical/<cell>/<mask>/<density>/ in hier-over-synth mode.
res_path <- function(method, dz, st, suffix, density = NULL) {
  base <- if (is.null(density)) RESULTS_ROOT
          else if (SYNTH) file.path(REPO, "results", "synth", "hierarchical",
                                    SYNTH_CELL, SYNTH_MASK, density)
          else file.path(RESULTS_ROOT, density)
  d <- file.path(base, method)
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
  } else if (method == "knn_corr") {
    source(file.path(SRC, "impute", "corr_common.R"))
    source(file.path(SRC, "impute", "knn_corr", "method.R"))
    impute_knn_corr(x)
  } else if (method == "missforest_corr") {
    source(file.path(SRC, "impute", "corr_common.R"))
    source(file.path(SRC, "impute", "missforest_corr", "method.R"))
    impute_missforest_corr(x)
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

# Hierarchical mode: OSMC runs once PER DESCENT STEP, on that step's filled
# input, with a holdout-mask sidecar (0/1 grid, same header/keys as the filled
# CSV) marking the density's real observations — Julia must never grade on
# seeded cells. run.jl's single-input mode (OSMC_SINGLE_INPUT) handles exactly
# one table and returns; the dz/st loop is skipped.
run_osmc_step <- function(filled_csv, mask_csv, out_dir, sweep_dir) {
  cat("\n##### OneSidedMC hier step (Julia subprocess) #####\n")
  Sys.setenv(OSMC_SINGLE_INPUT  = normalizePath(filled_csv, mustWork = FALSE),
             OSMC_OUT_DIR       = normalizePath(out_dir, mustWork = FALSE),
             OSMC_SWEEP_DIR     = normalizePath(sweep_dir, mustWork = FALSE),
             OSMC_HOLDOUT_MASK  = normalizePath(mask_csv, mustWork = FALSE),
             OSMC_BALANCE       = if (BALANCE_HOLDOUT) "1" else "0",
             OSMC_ALLCOLHOLDOUT = "1")
  osmc <- file.path(SRC, "impute", "OneSidedMC")
  status <- system2("julia",
    args = c("--threads=auto", paste0("--project=", osmc),
             file.path(osmc, "run.jl")))
  # Empty string = "not set" for run.jl's get(ENV, ...) defaults.
  Sys.setenv(OSMC_SINGLE_INPUT = "", OSMC_HOLDOUT_MASK = "")
  if (status != 0) cat("  WARNING: OSMC step exited with status", status, "\n")
}

read_matrix <- function(path) {
  df <- read.csv(path, check.names = FALSE)
  M <- as.matrix(df[, setdiff(names(df), "collapse_key")])
  storage.mode(M) <- "double"
  list(M = M, keys = df$collapse_key)
}

osmc_contract <- function(dz, st, imputed_csv, sweep_dir = file.path(
                            RESULTS_ROOT, "_osmc_sweep", sprintf("%s_%s", dz, st))) {
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
  # The sparse input table may lack rows (dropped models) or columns
  # (carried forward from a denser hier step) the completed matrix has:
  # those cells are missing by construction, so the mask is NA there.
  mcols <- intersect(cols, names(mask_df)[-1])
  msk <- matrix(NA_real_, nrow = length(keys), ncol = length(cols),
                dimnames = list(NULL, cols))
  msk[, mcols] <- as.matrix(mask_df[rm, mcols, drop = FALSE])
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
      truth_csv <- file.path(REPO, "data", "synth", SYNTH_DENSITY, "truth",
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

# ── Hierarchical mode (--hier) ────────────────────────────────────────────────
# Descends the density slices under the data root DENSEST-FIRST, for a given
# imputer method:
#
#   seed    = the standard run's cached completed matrix at the densest density
#             (data/<d1>/imputed/<method>/<dz>/<st>/...); abort the cell if
#             absent (fill-smooth seeds also require the correlation cache)
#   step d  = impute on the UNION grid of the chain (P) and this density's
#             sparse table T: T's observed cells win, P's completed values fill
#             the rest, NAs elsewhere. Holdouts come ONLY from T's real
#             observations (HOLDOUT_MASK in R, OSMC_HOLDOUT_MASK in Julia), so
#             sweeps stay comparable to the standard run at density d
#   ...     = repeat until the densities are exhausted; each step's completed
#             matrix becomes P for the next
#
# Matching across densities: rows by exact `collapse_key`, columns by exact
# benchmark name — never positional (column sets need not nest).
#
# Outputs: <DATA_ROOT>/hierarchical/<density>/imputed/<method>/<dz>/<st>/
#            imputed_model_benchmark_table.csv (+ filled + holdout_mask CSVs)
#          <RESULTS_ROOT>/<density>/<method>/<method>_<dz>_<st>_rank_sweep.csv
#            (+ ..._correlation.csv for fill-smooth)
#          <RESULTS_ROOT>/<density>/database.db — the REGULAR per-density
#            layout: table `imputation`, dataset = "<dz>_<st>" (DB_FILE is
#            re-pointed per density, as --synth re-points it per root). Every
#            row also carries wr2 — the cumulative weighted R2 (each step's
#            held-out R2 weighted by the columns it introduces, divided by the
#            running matrix's column count).
#
# Hand-off: after the descent, the FINAL (sparsest) step's imputed CSV + sweep
# (+ correlation cache) are copied into the density-"0" slice
# (<data-root>/0/, <results-root>/0/) and its `imputation` row is written with
# wr2 = the final weighted R2. `factor.R --density 0` then factors the
# hierarchical matrix through the completely standard path, gated on wr2.
#
# Synth variant (--hier --synth): the descent runs over the SYNTH density
# slices, for each (cell, mask) in turn — data/synth/<d1>/<cell>/<mask>/s0 ->
# data/synth/<d2>/... densest-first, same union-grid algorithm. The seed is
# the standard --synth run's cached completed matrix inside the densest s0
# root. Per-step outputs mirror the real layout per s0 root:
#   data/synth/<d>/<cell>/<mask>/s0/hierarchical/imputed/...   (the s0 root
#     already names the density, so no inner density level)
#   results/synth/hierarchical/<cell>/<mask>/<d>/ + database.db there
#     (table imputation_synth, rows carry wr2)
#   hand-off: data/synth/<cell>/<mask>/0/ +
#     results/synth/hierarchical/<cell>/<mask>/0/
hier_densities <- function() {
  base <- if (SYNTH) file.path(REPO, "data", "synth") else DATA_ROOT
  d <- list.dirs(base, full.names = FALSE, recursive = FALSE)
  d <- d[grepl("^[0-9]+(\\.[0-9]+)?$", d)]
  # Density "0" is reserved for the final-step hand-off slice (below) — never
  # a descent step itself.
  d <- d[d != "0"]
  as.character(sort(as.numeric(d), decreasing = TRUE))
}

# The s0 drop-in root of the active (cell, mask) at one synth density step.
synth_s0_root <- function(density) {
  file.path(REPO, "data", "synth", density, SYNTH_CELL, SYNTH_MASK, "s0")
}

# <DATA_ROOT>/hierarchical/<density>/imputed/<method>/<dz>/<st>/  (real corpus)
# data/synth/<density>/<cell>/<mask>/s0/hierarchical/imputed/...  (synth)
hier_imputed_dir <- function(density, method, dz, st) {
  d <- if (SYNTH)
    file.path(synth_s0_root(density), "hierarchical", "imputed",
              method, dz, st)
  else
    file.path(DATA_ROOT, "hierarchical", density, "imputed",
              method, dz, st)
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
  d
}

# Standard-run cache the seed step reuses (densest density's own imputation):
# the real corpus's <DATA_ROOT>/<density>/imputed/..., or the standard --synth
# run's cache inside the densest s0 root.
hier_seed_csv <- function(density, method, dz, st) {
  if (SYNTH)
    file.path(synth_s0_root(density), "imputed", method, dz, st,
              "imputed_model_benchmark_table.csv")
  else
    file.path(DATA_ROOT, density, "imputed", method, dz, st,
              "imputed_model_benchmark_table.csv")
}

# Fill-smooth seeds additionally require the standard run's correlation cache,
# which lives in the standard results tree (results/<density>/<method>/...,
# results/synth/<density>/<cell>/<mask>/<method>/... for synth).
hier_seed_cache <- function(density, method, dz, st) {
  if (SYNTH)
    file.path(REPO, "results", "synth", density, SYNTH_CELL, SYNTH_MASK,
              method, sprintf("%s_%s_%s_correlation.csv", method, dz, st))
  else
    file.path(REPO, opt$results_root, density, method,
              sprintf("%s_%s_%s_correlation.csv", method, dz, st))
}

# Union-grid fill: rows = union(P keys, T keys), cols = union(P cols, T cols).
# T's observed cells win; P's completed values fill the rest; NAs elsewhere.
# Column order: T's columns first, P-only columns appended (deterministic).
# Returns list(M, mask, keys) where mask marks T's REAL observations — exactly
# the cells hierarchical holdouts may draw from.
hier_fill_grid <- function(P, t_csv) {
  t_df <- read.csv(t_csv, check.names = FALSE)
  t_keys <- as.character(t_df$collapse_key)
  t_cols <- setdiff(names(t_df), "collapse_key")
  t_mat <- as.matrix(t_df[, t_cols, drop = FALSE])
  storage.mode(t_mat) <- "double"

  keys <- c(t_keys, setdiff(P$keys, t_keys))
  cols <- c(t_cols, setdiff(P$cols, t_cols))

  # NA in a match index yields an NA row/col — "absent" for that source.
  Tg <- t_mat[match(keys, t_keys), match(cols, t_cols), drop = FALSE]
  Pg <- P$M[match(keys, P$keys), match(cols, P$cols), drop = FALSE]
  M <- Tg
  needs <- is.na(M)
  M[needs] <- Pg[needs]
  colnames(M) <- cols
  mask <- !is.na(Tg)
  dimnames(mask) <- dimnames(M)
  list(M = M, mask = mask, keys = keys)
}

# Per-density DB for hier mode: the REGULAR impute.R layout, rooted under the
# hierarchical results tree. DB_FILE is re-pointed per density step (the same
# global-reassignment pattern --synth uses per synth root). Synth variant:
# results/synth/hierarchical/<cell>/<mask>/<density>/database.db.
hier_db_at <- function(density) {
  d <- if (SYNTH)
    file.path(REPO, "results", "synth", "hierarchical",
              SYNTH_CELL, SYNTH_MASK, density)
  else
    file.path(RESULTS_ROOT, density)
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
  file.path(d, "database.db")
}

# Density "0" slice — reserved for the hierarchical hand-off. The FINAL
# (sparsest) step's imputation + results are copied here at the end of every
# hier cell, so the standard factoring path (`factor.R --density 0`) runs on
# the hierarchical matrix. Its imputation row carries wr2 (the weighted R2),
# which is what the R2 gate reads there.
hier0_data_root <- function() {
  if (SYNTH) file.path(REPO, "data", "synth", SYNTH_CELL, SYNTH_MASK, "0")
  else file.path(REPO, opt$data_root, "0")
}
hier0_results_root <- function() {
  if (SYNTH) file.path(REPO, "results", "synth", "hierarchical",
                       SYNTH_CELL, SYNTH_MASK, "0")
  else file.path(REPO, opt$results_root, "0")
}

run_hier_cell <- function(method, dz, st, densities) {
  tag <- sprintf("%s/%s/%s", method, dz, st)
  cat("\n======== hier ", tag, " ========\n", sep = "")

  # Seed: the chain starts from the standard run's cached output verbatim.
  d1 <- densities[1]
  seed_csv <- hier_seed_csv(d1, method, dz, st)
  if (!file.exists(seed_csv)) {
    cat("  no seed at densest density ", d1, " (", seed_csv,
        ") — run the standard imputation first; aborting cell\n", sep = "")
    return()
  }
  if (method %in% RAW_METHODS &&
      !file.exists(hier_seed_cache(d1, method, dz, st))) {
    cat("  fill-smooth seed correlation cache missing; aborting cell\n")
    return()
  }
  seed_dir <- hier_imputed_dir(d1, method, dz, st)
  mb <- read_matrix(seed_csv)
  write_completed(seed_dir, mb$keys, mb$M)
  if (method %in% RAW_METHODS)
    file.copy(hier_seed_cache(d1, method, dz, st),
              res_path(method, dz, st, "correlation.csv", density = d1),
              overwrite = TRUE)
  # Parity: mirror the seed's standard-run sweep + DB row. No imputation runs
  # at the seed — its numbers ARE the standard run's.
  DB_FILE <<- hier_db_at(d1)
  seed_sweep_src <- if (SYNTH)
    file.path(REPO, "results", "synth", d1, SYNTH_CELL, SYNTH_MASK, method,
              sprintf("%s_%s_%s_rank_sweep.csv", method, dz, st))
  else
    file.path(REPO, opt$results_root, d1, method,
              sprintf("%s_%s_%s_rank_sweep.csv", method, dz, st))
  if (file.exists(seed_sweep_src)) {
    sw <- read.csv(seed_sweep_src)
    write.csv(sw, res_path(method, dz, st, "rank_sweep.csv", density = d1),
              row.names = FALSE)
    best_idx <- which.min(replace(sw$rmse, is.na(sw$rmse), Inf))
    best_r2 <- if ("r2" %in% names(sw)) sw$r2[best_idx] else NA
    desc <- build_desc(sw$param, sw$param_name[1L], sw$param[best_idx])
    db_record(method, paste0(dz, "_", st), sw$rmse[best_idx], best_r2, desc)
  }
  P <- list(M = mb$M, keys = mb$keys, cols = colnames(mb$M))

  for (d in densities[-1]) {
    cat("\n-------- hier ", tag, " @ ", d, " --------\n", sep = "")
    DB_FILE <<- hier_db_at(d)
    out_dir <- hier_imputed_dir(d, method, dz, st)
    imputed_csv <- file.path(out_dir, "imputed_model_benchmark_table.csv")
    sweep_csv   <- res_path(method, dz, st, "rank_sweep.csv", density = d)
    cache_csv   <- res_path(method, dz, st, "correlation.csv", density = d)

    # Same reuse semantics as the standard loop (--reimpute to force). The
    # chain continues from the reused CSV: P must track it either way.
    if (!REIMPUTE && file.exists(imputed_csv) &&
        (!(method %in% RAW_METHODS) || file.exists(cache_csv))) {
      cat("  reusing existing hier imputed CSV (skip step)\n")
      mb <- read_matrix(imputed_csv)
      P <- list(M = mb$M, keys = mb$keys, cols = colnames(mb$M))
      next
    }

    src <- combos_path(dz, st, density = d)
    if (!file.exists(src)) {
      cat("  missing input:", src, " — P carries forward unchanged\n")
      next
    }
    grid <- hier_fill_grid(P, src)
    filled_csv <- file.path(out_dir, "filled_model_benchmark_table.csv")
    mask_csv   <- file.path(out_dir, "holdout_mask.csv")
    write.csv(data.frame(collapse_key = grid$keys, grid$M, check.names = FALSE),
              filled_csv, row.names = FALSE)
    # 0/1 numerics, not TRUE/FALSE — run.jl's read_mask parses Float64.
    write.csv(data.frame(collapse_key = grid$keys, grid$mask + 0L, check.names = FALSE),
              mask_csv, row.names = FALSE)
    cat(sprintf("  input: %d x %d — %d real cells, %d seeded from %s\n",
                nrow(grid$M), ncol(grid$M), sum(grid$mask),
                sum(!is.na(grid$M)) - sum(grid$mask), d1))

    pm <- prep_matrix(filled_csv)
    x <- pm$x
    # Align the mask with the prepped matrix (prep may drop degenerate cols;
    # it never drops rows).
    mask <- grid$mask[, pm$cols, drop = FALSE]

    if (method == "onesidedmc") {
      sweep_dir <- if (SYNTH)
        file.path(REPO, "results", "synth", "hierarchical",
                  SYNTH_CELL, SYNTH_MASK, d, "_osmc_sweep", paste0(dz, "_", st))
      else
        file.path(RESULTS_ROOT, d, "_osmc_sweep", paste0(dz, "_", st))
      run_osmc_step(filled_csv, mask_csv, out_dir, sweep_dir)
      res <- osmc_contract(dz, st, imputed_csv, sweep_dir = sweep_dir)
      if (is.null(res)) { cat("  no OSMC outputs, aborting cell\n"); return() }
    } else {
      HOLDOUT_MASK <<- mask
      res <- tryCatch(impute_R(method, x), error = function(e) {
        cat("  IMPUTE FAILED:", conditionMessage(e), "\n"); NULL })
      HOLDOUT_MASK <<- NULL
      if (is.null(res)) return()
    }

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

    P <- list(M = res$M, keys = pm$keys, cols = pm$cols)
    colnames(P$M) <- pm$cols
  }

  hier_finalize_cell(method, dz, st, densities)
}

# Weighted R2 (wr2) for a finished hier descent — cumulative per step. Each
# step's R2 is weighted by the columns it INTRODUCES (setdiff vs the running
# matrix P; the seed introduces all of its columns), summed and divided by the
# running matrix's total column count: wr2 at step d is the
# introduced-column-weighted mean R2 of the matrix factored at d. The R2
# itself is mode-dependent: held-out r2 for the real corpus, TRUTH-recovery
# gt_r2 (scored against the planted truth on the density's missing cells) for
# the synth corpus. Every input is re-derived from artifacts on disk
# (per-step imputed CSVs + rank sweeps + input tables + truth), so the numbers
# come out identically on --reimpute reuse and a cached re-run refreshes
# wr2/gt_r2 without re-imputing. Also hands the FINAL step's artifacts to the
# density-0 slice and writes that slice's imputation row with wr2 — the row
# (and value) the factoring gate reads at --density 0.
hier_finalize_cell <- function(method, dz, st, densities) {
  dataset <- paste0(dz, "_", st)
  tag     <- sprintf("%s/%s/%s", method, dz, st)
  dlast   <- densities[length(densities)]

  # Synth: the wr2 weight is TRUTH recovery, not held-out r2 — the planted
  # truth is what this corpus exists for. Per step, gt_r2 scores that step's
  # completed matrix against the planted truth on exactly the cells the
  # density's sparse input table is missing (gt_r2_score's column-balanced
  # definition); the seed step's number matches what the non-hier --synth
  # pass computes at the densest density.
  gt_for <- function(d, csv) {
    if (!SYNTH) return(NA_real_)
    truth_csv <- file.path(REPO, "data", "synth", d, "truth",
                           paste0(dataset, "_", SYNTH_CELL, "_complete.csv"))
    gt <- gt_r2_score(combos_path(dz, st, density = d), csv, truth_csv)
    cat(sprintf("  gt: %s @ %s gt_r2=%s\n", dataset, d,
                if (is.na(gt)) "NA" else sprintf("%.4f", gt)))
    gt
  }

  num <- 0
  prev_cols <- character(0)
  best <- NULL
  for (d in densities) {
    csv <- file.path(hier_imputed_dir(d, method, dz, st),
                     "imputed_model_benchmark_table.csv")
    sweep <- res_path(method, dz, st, "rank_sweep.csv", density = d)
    if (!file.exists(csv) || !file.exists(sweep)) {
      cat("  wr2: missing step artifacts at ", d, " — skipping hand-off\n")
      return(invisible())
    }
    cols <- setdiff(names(read.csv(csv, nrow = 0, check.names = FALSE)),
                    "collapse_key")
    sw <- read.csv(sweep)
    i <- which.min(replace(sw$rmse, is.na(sw$rmse), Inf))
    step_r2 <- if ("r2" %in% names(sw)) sw$r2[i] else NA_real_
    step_gt <- gt_for(d, csv)
    weight <- if (SYNTH) step_gt else step_r2
    num <- num + length(setdiff(cols, prev_cols)) * weight
    prev_cols <- cols
    wr2 <- if (length(cols)) num / length(cols) else NA_real_
    desc <- build_desc(sw$param, sw$param_name[1L], sw$param[i])
    db_insert_imputation(method, dataset, sw$rmse[i], step_r2, desc,
                         hier_db_at(d), overwrite = TRUE, wr2 = wr2,
                         table = if (SYNTH) "imputation_synth" else "imputation",
                         cell = SYNTH_CELL, mask = SYNTH_MASK)
    if (SYNTH)
      db_update_gt_r2(SYNTH_CELL, SYNTH_MASK, dataset, method, step_gt,
                      hier_db_at(d))
    best <- list(rmse = sw$rmse[i], r2 = step_r2, gt = step_gt, wr2 = wr2)
  }

  # Hand-off: the sparsest step's imputation + results, copied into the
  # density-0 slice (refreshed on every hier run).
  src_dir <- hier_imputed_dir(dlast, method, dz, st)
  dst_dir <- file.path(hier0_data_root(), "imputed", method, dz, st)
  dir.create(dst_dir, recursive = TRUE, showWarnings = FALSE)
  file.copy(file.path(src_dir, "imputed_model_benchmark_table.csv"), dst_dir,
            overwrite = TRUE)
  dst_res <- file.path(hier0_results_root(), method)
  dir.create(dst_res, recursive = TRUE, showWarnings = FALSE)
  for (suffix in c("rank_sweep.csv",
                   if (method %in% RAW_METHODS) "correlation.csv")) {
    src <- res_path(method, dz, st, suffix, density = dlast)
    if (file.exists(src)) file.copy(src, dst_res, overwrite = TRUE)
  }
  # The row the R2 gate reads at --density 0: wr2 carries the FINAL step's
  # weighted R2 (db_read_r2 selects COALESCE(wr2, r2)) — truth-weighted in
  # synth mode.
  db_insert_imputation(method, dataset, best$rmse, best$r2,
                       sprintf("hier %s", paste(densities, collapse = "->")),
                       file.path(hier0_results_root(), "database.db"),
                       overwrite = TRUE, wr2 = best$wr2,
                       table = if (SYNTH) "imputation_synth" else "imputation",
                       cell = SYNTH_CELL, mask = SYNTH_MASK)
  if (SYNTH)
    db_update_gt_r2(SYNTH_CELL, SYNTH_MASK, dataset, method, best$gt,
                    file.path(hier0_results_root(), "database.db"))
  cat(sprintf("  hier hand-off: %s @ %s -> density 0 (wr2=%.3f)\n",
              tag, dlast, best$wr2))
}

main_hier <- function() {
  densities <- hier_densities()
  if (!length(densities))
    stop("no density slices found under ",
         if (SYNTH) file.path(REPO, "data", "synth") else DATA_ROOT)
  if (SYNTH) {
    cat(sprintf("impute --hier --synth: methods=[%s]  densities=%s  results=results/synth/hierarchical  reimpute=%s\n",
                paste(METHODS, collapse = ","),
                paste(densities, collapse = " -> "), REIMPUTE))
    # One descent per (cell, mask); the seeds come from the standard --synth
    # runs at the densest density. Cells/masks are enumerated under the
    # densest slice; a step whose density lacks that root carries P forward
    # unchanged (run_hier_cell's missing-input branch).
    densest <- file.path(REPO, "data", "synth", densities[1])
    for (cell in sort(list.files(densest))) {
      cell_dir <- file.path(densest, cell)
      if (cell == "truth" || !dir.exists(cell_dir)) next
      for (mask in sort(list.files(cell_dir))) {
        if (!dir.exists(file.path(cell_dir, mask, "s0"))) next
        SYNTH_CELL <<- cell
        SYNTH_MASK <<- mask
        cat(sprintf("\n#### hier-synth %s/%s ####\n", cell, mask))
        for (method in METHODS) for (st in STRATEGIES) {
          for (dz in DENSIFIERS) {
            run_hier_cell(method, dz, st, densities)
          }
        }
      }
    }
    cat("\nDONE.\n  imputed CSVs -> data/synth/<density>/<cell>/<mask>/s0/hierarchical/imputed/<method>/<dz>/<st>/\n",
        "  results      -> results/synth/hierarchical/<cell>/<mask>/<density>/ (flat)\n",
        "  db           -> results/synth/hierarchical/<cell>/<mask>/<density>/database.db (imputation_synth.wr2 = weighted R2)\n",
        "  final step   -> data/synth/<cell>/<mask>/0/ + results/synth/hierarchical/<cell>/<mask>/0/ (density 0)\n",
        sep = "")
    return()
  }
  cat(sprintf("impute --hier: methods=[%s]  densities=%s  data_root=%s  results=%s  reimpute=%s\n",
              paste(METHODS, collapse = ","),
              paste(densities, collapse = " -> "),
              DATA_ROOT, RESULTS_ROOT, REIMPUTE))
  for (method in METHODS) for (st in STRATEGIES) {
    for (dz in DENSIFIERS) {
      run_hier_cell(method, dz, st, densities)
    }
  }
  cat("\nDONE.\n  imputed CSVs -> ", file.path(DATA_ROOT, "hierarchical"),
      "/<density>/imputed/<method>/<densifier>/<strategy>/\n  results       -> ",
      RESULTS_ROOT, "/<density>/ (flat)\n  db            -> ",
      RESULTS_ROOT, "/<density>/database.db (imputation.wr2 = weighted R2)\n",
      "  final step    -> ", hier0_data_root(),
      " + ", hier0_results_root(), "/ (density 0; factor there with --density 0)\n",
      sep = "")
}

main <- function() {
  if (HIER) return(main_hier())
  if (SYNTH) {
    for (root in .synth_roots) {
      SYNTH_CELL   <<- root$cell
      SYNTH_MASK   <<- root$mask
      DATA_ROOT    <<- root$root
      RESULTS_ROOT <<- file.path(REPO, "results", "synth", SYNTH_DENSITY,
                                 root$cell, root$mask)
      dir.create(RESULTS_ROOT, recursive = TRUE, showWarnings = FALSE)
      cat(sprintf("\n#### synth root %s/%s/%s ####\n", SYNTH_DENSITY,
                  root$cell, root$mask))
      if ("onesidedmc" %in% METHODS) run_osmc_subprocess()
      for (method in METHODS) for (st in STRATEGIES) {
        for (dz in DENSIFIERS) {
          run_cell(method, dz, st)
        }
      }
      score_gt_r2(root$cell)
    }
    cat("\nDONE.\n  imputed CSVs -> data/synth/<density>/<cell>/<mask>/s0/imputed/<method>/...\n",
        "  results      -> results/synth/<density>/<cell>/<mask>/ (flat)\n",
        "  db           -> results/synth/<density>/database.db (table imputation_synth)\n",
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
