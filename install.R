#!/usr/bin/env Rscript
# Set up the project-scoped R environment via renv (parity with uv / Julia's
# Project.toml). Run via `make install-r` (or directly: Rscript install.R), from
# the repo root.
#
# First run: bootstraps renv, installs the packages into a project-local library,
# and writes renv.lock. Later runs (and collaborators): renv::restore() from the
# lockfile reproduces the exact same library.

REPO <- tryCatch(dirname(normalizePath(sub(
  "^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))),
  error = function(e) getwd())
setwd(REPO)

options(repos = c(CRAN = "https://cloud.r-project.org"))

# Parallelize compilation — renv forwards Ncpus to the underlying install step.
# (renv does NOT inherit the bare install.packages(Ncpus=) you may be used to,
# so set it explicitly here or compiles run ~serially.)
ncores <- max(1L, parallel::detectCores())
options(Ncpus = ncores)
Sys.setenv(MAKEFLAGS = paste0("-j", ncores))  # parallel make within each compile

pkgs <- c(
  "psych",       # factor analysis (PAF) + parallel analysis
  "softImpute",  # softimpute matrix completion
  "missMDA",     # iterative PCA imputation (deferred method)
  "VIM",         # KNN imputation
  "missForest",  # random-forest iterative imputation
  "mice",        # multiple imputation by chained equations
  "jsonlite",    # parallel-analysis cache (JSON)
  "magick",      # stacking result PNGs into combined figures
  "doParallel",  # parallel sensitivity seed-sweeps
  "foreach",     # parallel sensitivity seed-sweeps
  "boot"         # bootstrap CIs (reference code)
  # "parallel" is part of base R — no install needed.
)

# Bootstrap renv itself into the user library if absent.
if (!requireNamespace("renv", quietly = TRUE)) {
  cat("Installing renv...\n")
  install.packages("renv")
}

if (file.exists(file.path(REPO, "renv.lock"))) {
  # Reproduce the locked environment, then install + snapshot any pkgs added to
  # the list since the lock was written (e.g. new imputation methods).
  cat("renv.lock found -> renv::restore()\n")
  renv::restore(prompt = FALSE)
  missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing)) {
    cat("Installing newly-added packages:", paste(missing, collapse = ", "), "\n")
    renv::install(missing)
    renv::snapshot(prompt = FALSE)
  }
} else {
  # First-time setup: init a bare project env, install deps, snapshot the lock.
  cat("No renv.lock -> initializing project library + installing deps\n")
  renv::init(bare = TRUE, restart = FALSE)
  renv::install(pkgs)
  renv::snapshot(packages = pkgs, prompt = FALSE)
  cat("Wrote renv.lock.\n")
}
