suppressMessages({
  library(softImpute)
  library(psych)
})

source("impute.R")
source("factoring.R")
source("report.R")

run_pipeline <- function(path, max_rank = 15L, pa_iter = 30L,
                         prefix = "imputed_", seed = 1) {
  cat("\n======== ", path, " ========\n", sep = "")
  pm <- prep_matrix(path)
  x <- pm$x; keys <- pm$keys
  cat(sprintf("  matrix: %d x %d, %.1f%% observed\n",
              nrow(x), ncol(x), 100 * mean(!is.na(x))))

  xs <- biScale(x, row.center = FALSE, row.scale = FALSE,
                col.center = TRUE, col.scale = TRUE, maxit = 100)

  set.seed(seed)
  holdout <- make_holdout(xs, frac = 0.2, min_keep = 1L)

  ranks <- 1:min(max_rank, nrow(x) - 1L, ncol(x) - 1L)

  rmse_v   <- numeric(length(ranks))
  varexp_v <- numeric(length(ranks))
  pa_nf_v  <- integer(length(ranks))
  fits     <- vector("list", length(ranks))

  tag <- basename(dirname(path))

  for (k in seq_along(ranks)) {
    r <- ranks[k]
    fr <- fit_at_rank(xs, holdout, rank_cap = r)
    fits[[k]] <- fr
    rmse_v[k] <- fr$rmse

    M <- complete(x, fr$fit)

    efa <- run_efa(M, nf = r)
    varexp_v[k] <- extract_variance(efa)

    pa <- run_parallel_analysis(M, pa_iter)
    pa_nf_v[k] <- if (!is.null(pa)) max(1L, pa$nfact) else NA_integer_

    cat(sprintf("  rank %2d | RMSE %.4f | eff_rank %2d | cumVar %.3f | PA nf %s\n",
                r, fr$rmse, fr$eff_rank, varexp_v[k],
                ifelse(is.na(pa_nf_v[k]), "?", pa_nf_v[k])))
  }

  best_k <- which.min(rmse_v)
  best_r <- ranks[best_k]
  cat(sprintf("  >> CV-best imputation rank = %d (RMSE %.4f)\n",
              best_r, rmse_v[best_k]))

  best_fit <- fits[[best_k]]$fit
  M_best <- complete(x, best_fit)

  R <- cor(M_best)
  eig_best <- eigen(R, symmetric = TRUE, only.values = TRUE)$values
  pa_obj_best <- run_parallel_analysis(M_best, pa_iter)

  nf_best <- if (!is.null(pa_obj_best)) max(1L, pa_obj_best$nfact) else best_r
  efa_best <- fa(M_best, nfactors = nf_best, fm = "pa",
                 rotate = if (nf_best > 1) "promax" else "none")

  write_outputs(path, prefix, keys, M_best, efa_best, tag)

  cat("\n  --- loadings at CV-best rank (", nf_best, " factors) ---\n", sep = "")
  print(efa_best$loadings, cutoff = 0.30, sort = TRUE)

  marg_gain <- c(NA, -diff(rmse_v))
  sweep_df <- data.frame(rank = ranks, holdout_rmse = rmse_v,
                         marginal_gain = marg_gain,
                         cum_var = varexp_v, pa_nfactors = pa_nf_v)
  write.csv(sweep_df, paste0(tag, "_rank_sweep.csv"), row.names = FALSE)

  plot_dashboard(tag, x, ranks, rmse_v, marg_gain, varexp_v,
                 pa_nf_v, eig_best, pa_obj_best, efa_best, best_r, max_rank)
  plot_rmse(tag, ranks, rmse_v)
  plot_scree(tag, eig_best, pa_obj_best, max_rank)

  invisible(list(sweep = sweep_df, efa = efa_best, best_rank = best_r))
}

paths <- c(
  "combinations/all_standard/model_benchmark_table.csv",
  "combinations/whitelist_standard/model_benchmark_table.csv",
  "combinations/whitelist_none/model_benchmark_table.csv"
)

for (p in paths) {
  tryCatch(run_pipeline(p),
           error = function(e) cat("FAILED on", p, ":", conditionMessage(e), "\n"))
}
