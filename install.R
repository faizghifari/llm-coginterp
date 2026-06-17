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

pkgs <- c(
  "psych",       # factor analysis (PAF) + parallel analysis
  "softImpute",  # softimpute matrix completion
  "missMDA",     # iterative PCA imputation (deferred method)
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
  # Reproduce the locked environment.
  cat("renv.lock found -> renv::restore()\n")
  renv::restore(prompt = FALSE)
} else {
  # First-time setup: init a bare project env, install deps, snapshot the lock.
  cat("No renv.lock -> initializing project library + installing deps\n")
  renv::init(bare = TRUE, restart = FALSE)
  renv::install(pkgs)
  renv::snapshot(packages = pkgs, prompt = FALSE)
  cat("Wrote renv.lock.\n")
}
