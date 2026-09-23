# ─────────────────────────────────────────────────────────────────────────────
# Release-cohort factoring: the EFA half of scripts/release_date.py.
#
# Splits the rows of each completed matrix into release cohorts (<=2022, 2023,
# 2024, >=2025) and runs the pooled pipeline's own PA -> minres/promax ->
# Schmid-Leiman bifactor on each cohort. For every cohort it also factors
# --reps random subsets of the dated models of the same size, drawn regardless
# of year, as the null for "this cohort differs only because it is smaller".
#
# Unlike factor.R --timed (one cohort per calendar year, no null), cohorts here
# are binned so that every one holds enough models, and the null is built in.
#
# Called by scripts/release_date.py, which picks the methods (row-preserving
# imputers that pass factor.R's R² gate) and writes the year map. By hand:
#   Rscript src/run/release_cohorts.R --data-root <d> --results-root <r> \
#       --year-map <out>/year_map.csv --out <out> --methods softimpute,knn \
#       [--dz C,S] [--strategy all_standard] [--reps 50] [--cores 8]
#
# Writes <out>/cohort_fits.csv (one row per cohort x rep; rep 0 is the real
# cohort) and, for rep 0, <out>/cohorts/<method>/<method>_<dz>_<st>_bifactor_pa_y<bin>_*.
# ─────────────────────────────────────────────────────────────────────────────
suppressMessages(library(parallel))

.script_path <- sub("^--file=", "",
                    grep("^--file=", commandArgs(FALSE), value = TRUE))[1]
SRC_DIR <- if (length(.script_path) && nzchar(.script_path))
  dirname(normalizePath(.script_path)) else normalizePath("src/run")
SRC  <- dirname(SRC_DIR)
REPO <- dirname(SRC)

.renv_activate <- file.path(REPO, "renv", "activate.R")
if (file.exists(.renv_activate)) source(.renv_activate)
source(file.path(SRC, "factor", "factoring.R"))

parse_args <- function(args) {
  opt <- list(dz = "C,S", strategy = "all_standard", reps = "50",
              cores = as.character(max(1L, detectCores() - 1L)))
  i <- 1L
  while (i <= length(args)) {
    key <- sub("^--", "", args[[i]])
    key <- gsub("-", "_", key)
    opt[[key]] <- args[[i + 1L]]
    i <- i + 2L
  }
  for (k in c("data_root", "results_root", "year_map", "out", "methods"))
    if (is.null(opt[[k]])) stop("missing --", gsub("_", "-", k))
  opt
}
opt <- parse_args(commandArgs(trailingOnly = TRUE))
abs_path <- function(p) if (grepl("^/", p)) p else file.path(REPO, p)
DATA <- abs_path(opt$data_root); RES <- abs_path(opt$results_root)
OUT <- abs_path(opt$out)
METHODS <- strsplit(opt$methods, ",")[[1]]
DZS <- strsplit(opt$dz, ",")[[1]]
ST <- opt$strategy
B <- as.integer(opt$reps); CORES <- as.integer(opt$cores)

# Cohort bins. Keep in step with COHORTS in scripts/release_date.py.
bins <- function(y) ifelse(y <= 2022, 2022L, ifelse(y >= 2025, 2025L, as.integer(y)))
ym <- read.csv(abs_path(opt$year_map))
ym <- ym[!is.na(ym$year), ]
ymap <- setNames(bins(ym$year), ym$collapse_key)
tucker <- function(a, b) abs(sum(a * b)) / sqrt(sum(a^2) * sum(b^2))

fit <- function(M) {
  keep <- apply(M, 2, sd) > 1e-10; M <- M[, keep, drop = FALSE]
  fr <- tryCatch(suppressMessages(capture.output(r <- factor_matrix(M, pa_iter = 100L))),
                 error = function(e) NULL)
  if (is.null(fr)) return(NULL)
  capture.output(ho <- higher_order(M, r$nf))
  if (is.null(ho$bifactor_loadings)) return(NULL)
  list(nf = r$nf, var = extract_variance(r$efa), omega_h = ho$omega_h, ho = ho,
       g = setNames(ho$bifactor_loadings[, "g"], rownames(ho$bifactor_loadings)))
}

tasks <- list()
for (dz in DZS) for (m in METHODS) {
  f <- file.path(DATA, "imputed", m, dz, ST, "imputed_model_benchmark_table.csv")
  pooled_f <- file.path(RES, m, sprintf("%s_%s_%s_bifactor_pa_loadings.csv", m, dz, ST))
  if (!file.exists(f) || !file.exists(pooled_f)) {
    cat(sprintf("skipping %s %s: missing %s\n", m, dz,
                if (!file.exists(f)) f else pooled_f))
    next
  }
  df <- read.csv(f, check.names = FALSE)
  M <- as.matrix(df[, setdiff(names(df), "collapse_key")]); storage.mode(M) <- "double"
  coh <- unname(ymap[as.character(df$collapse_key)])
  pooled <- read.csv(pooled_f)
  gp <- setNames(pooled$g, pooled$benchmark)
  dated <- which(!is.na(coh))
  for (c in sort(unique(coh[dated]))) {
    idx <- which(coh == c)
    tasks[[length(tasks) + 1]] <- list(dz = dz, m = m, c = c, idx = idx, M = M, gp = gp,
                                       dated = dated, rep = 0L)
    for (b in seq_len(B)) tasks[[length(tasks) + 1]] <- list(dz = dz, m = m, c = c,
        idx = NULL, n = length(idx), M = M, gp = gp, dated = dated, rep = b)
  }
}
if (!length(tasks)) stop("no cohort fits to run")
cat(sprintf("release cohorts: %d fits on %d cores\n", length(tasks), CORES))

res <- mclapply(tasks, function(t) {
  # Seed per (rep, cohort): results do not depend on task order or core count.
  set.seed(1000L * t$rep + t$c)
  idx <- if (t$rep == 0L) t$idx else sample(t$dated, t$n)
  r <- fit(t$M[idx, , drop = FALSE])
  if (is.null(r)) return(data.frame(dz = t$dz, method = t$m, cohort = t$c, rep = t$rep,
                                    n = length(idx), p = ncol(t$M), nf = NA, var = NA,
                                    omega_h = NA, tucker_g = NA))
  j <- intersect(names(r$g), names(t$gp))
  if (t$rep == 0L) {
    d <- file.path(OUT, "cohorts", t$m); dir.create(d, recursive = TRUE, showWarnings = FALSE)
    base <- file.path(d, sprintf("%s_%s_%s_bifactor_pa_y%d", t$m, t$dz, ST, t$c))
    write_higher_order(r$ho, paste0(base, "_loadings.csv"), paste0(base, "_scalars.csv"),
                       paste0(base, "_omega_group.csv"), paste0(base, "_loadings.md"))
  }
  data.frame(dz = t$dz, method = t$m, cohort = t$c, rep = t$rep, n = length(idx),
             p = ncol(t$M), nf = r$nf, var = r$var, omega_h = r$omega_h,
             tucker_g = tucker(r$g[j], t$gp[j]))
}, mc.cores = CORES, mc.preschedule = FALSE)

errs <- vapply(res, inherits, logical(1), "try-error")
if (any(errs)) stop(sum(errs), " cohort fits crashed: ", as.character(res[[which(errs)[1]]]))
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
write.csv(do.call(rbind, res), file.path(OUT, "cohort_fits.csv"), row.names = FALSE)
cat("wrote", file.path(OUT, "cohort_fits.csv"), "\n")
