# ─────────────────────────────────────────────────────────────────────────────
# Sensitivity plot for the orchestrator.
#
# The per-cell predictive-curve + scree + loadings panels all live in the single
# dashboard (main/dashboard.R). The only standalone plot is the seed-sweep
# SENSITIVITY analysis (run on demand with --sensitivity), kept separate because
# it's expensive and optional.
#
# ONE PNG per method x strategy, laid out as a grid:
#   rows    = densifier level (raw, C, S, R)
#   columns = [ RMSE/CV box | R^2 box | best-param stability barplot ]
# The R^2 column is blank for methods that don't produce r2_mat (e.g. the
# deferred iterativepca). Written flat to results/<method>_<strategy>_sensitivity.png.
# ─────────────────────────────────────────────────────────────────────────────

.blank_panel <- function(main = "") { plot.new(); title(main = main) }

# One grid row (3 panels) for a single densifier's sensitivity result.
# `sens` fields: rmse_mat|cv_mat (seeds x params), optional r2_mat, best_ranks,
# ranks, param.
.sens_row <- function(sens, dz_label) {
  if (is.null(sens)) {
    .blank_panel(paste0(dz_label, ": (failed)")); .blank_panel(); .blank_panel()
    return(invisible())
  }
  mat <- if (!is.null(sens$rmse_mat)) sens$rmse_mat else sens$cv_mat
  ranks <- sens$ranks; param <- sens$param
  ylab <- if (param == "ncp" && is.null(sens$rmse_mat)) "CV error" else "Held-out RMSE"

  # col 1: error distribution per param (auto-scaled; outline=FALSE hides fliers)
  boxplot(as.data.frame(mat), names = ranks, outline = FALSE,
          xlab = paste("imputation", param), ylab = ylab,
          main = paste0(dz_label, ": ", ylab, " spread"))
  med <- apply(mat, 2, median, na.rm = TRUE)
  lines(seq_along(ranks), med, col = "red", lwd = 2)
  points(seq_along(ranks), med, col = "red", pch = 19, cex = 0.7)

  # col 2: R^2 distribution per param (blank if the method has no r2_mat)
  if (!is.null(sens$r2_mat)) {
    boxplot(as.data.frame(sens$r2_mat), names = ranks, outline = FALSE,
            xlab = paste("imputation", param), ylab = "Held-out R2",
            main = paste0(dz_label, ": R2 spread"))
    medr <- apply(sens$r2_mat, 2, median, na.rm = TRUE)
    lines(seq_along(ranks), medr, col = "blue", lwd = 2)
    points(seq_along(ranks), medr, col = "blue", pch = 19, cex = 0.7)
    abline(h = 0, lty = 3, col = "grey50")  # R2=0 = no-skill baseline
  } else {
    .blank_panel(paste0(dz_label, ": R2 (n/a)"))
  }

  # col 3: best-param stability
  best <- sens$best_ranks[!is.na(sens$best_ranks)]
  tab <- table(factor(best, levels = ranks))
  barplot(tab, xlab = paste("CV-best", param), ylab = "count",
          main = paste0(dz_label, ": best-", param, " stability"))
}

# ── Combine / stack existing PNGs ────────────────────────────────────────────
# Read PNGs already written to results/ and stack them vertically into one
# aggregate image. Used at the end of a run so a separate `--raw` run and a
# default (C/S/R) run can be viewed together. Only files that exist are included;
# if fewer than 2 exist there's nothing to combine.

# Vertically stack a set of PNG paths into `out_path` (skips missing files).
stack_pngs <- function(paths, out_path) {
  suppressMessages(library(magick))
  paths <- paths[file.exists(paths)]
  if (length(paths) < 2) return(invisible(FALSE))  # nothing meaningful to stack
  imgs <- Reduce(c, lapply(paths, image_read))
  image_write(image_append(imgs, stack = TRUE), out_path)
  cat("  wrote", out_path, "\n")
  invisible(TRUE)
}

# Stack the raw + csr sensitivity grids for one method x strategy.
combine_sensitivity <- function(method, st, results_root) {
  paths <- file.path(results_root,
    sprintf("%s_%s_%s_sensitivity.png", method, st, c("raw", "csr")))
  out <- file.path(results_root,
    sprintf("%s_%s_sensitivity_combined.png", method, st))
  stack_pngs(paths, out)
}

# Stack all per-densifier dashboards (raw, C, S, R) for one method x strategy.
combine_dashboards <- function(method, st, results_root,
                               densifiers = c("raw", "C", "S", "R")) {
  paths <- file.path(results_root,
    sprintf("%s_%s_%s_dashboard.png", method, densifiers, st))
  out <- file.path(results_root,
    sprintf("%s_%s_dashboard_combined.png", method, st))
  stack_pngs(paths, out)
}

# `sens_by_dz` is a named list: densifier label -> sensitivity result (or NULL).
plot_sensitivity_grid <- function(sens_by_dz, path, title = "") {
  dzs <- names(sens_by_dz)
  nr <- length(dzs)
  png(path, width = 1500, height = 320 * nr, res = 110)
  op <- par(mfrow = c(nr, 3), mar = c(4, 4.2, 3, 1), oma = c(0, 0, 2.5, 0))
  for (dz in dzs) .sens_row(sens_by_dz[[dz]], dz)
  mtext(paste0("Seed-sweep sensitivity: ", title), outer = TRUE, cex = 1.1)
  par(op); dev.off()
  cat("  wrote", path, "\n")
}
