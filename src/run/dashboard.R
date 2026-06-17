# ─────────────────────────────────────────────────────────────────────────────
# Dashboard: a single 6-panel PNG per (method × densifier × strategy).
#
# The dashboard is a CROSS-STAGE artifact (imputation diagnostics × factoring
# results), so the ORCHESTRATOR owns it — not the imputers (which never factor)
# nor factor/ (which doesn't know about sweeps). This module drives the sweep
# via the imputer's complete_at() closure and calls the shared factor/ functions
# at each swept parameter value. Factoring stays in factor/.
#
# Panels (axis labels come from the method's param_name / metric_name, so panels
# 1-2 read "rank"/"ncp"/"r" and "Held-out RMSE"/"CV error"/... per method):
#   1. predictive metric vs sweep param      (+ rank-1 ref, CV-best marker)
#   2. marginal gain in the metric per param
#   3. cumulative variance explained vs param (needs EFA per param)
#   4. scree @ best param                     (observed eig vs cached PA cutoff)
#   5. SS loadings per factor @ best param
#   6. PA factor count vs param               (needs PA per param)
# Panels 3 & 6 are produced by factoring complete_at(v) at every v in params.
# ─────────────────────────────────────────────────────────────────────────────

# Walk the sweep: factor the completed matrix at each param value, collecting
# cumulative variance and the PA factor count. Returns two aligned vectors.
sweep_factor_curve <- function(res, pa_iter = 100L) {
  params <- res$params
  cumvar <- rep(NA_real_, length(params))
  pa_nf  <- rep(NA_integer_, length(params))
  # complete_at is NULL on a reused (--reimpute off) run — the per-param fits
  # aren't in memory, so panels 3 & 6 stay NA (rendered as "no data").
  if (is.null(res$complete_at)) return(list(cumvar = cumvar, pa_nf = pa_nf))
  for (k in seq_along(params)) {
    M <- tryCatch(res$complete_at(params[k]), error = function(e) NULL)
    if (is.null(M)) next
    nf_k <- tryCatch(choose_nfactors(M, n.iter = pa_iter, verbose = FALSE)$nf,
                     error = function(e) NA_integer_)
    pa_nf[k] <- nf_k
    efa <- run_efa(M, nf = if (is.na(nf_k)) params[k] else nf_k)
    cumvar[k] <- extract_variance(efa)
  }
  list(cumvar = cumvar, pa_nf = pa_nf)
}

# Plot x vs y, but if y is all-NA/empty draw a blank panel with the title rather
# than erroring on "need finite 'ylim' values".
.safe_plot <- function(x, y, main, xlab, ylab, ...) {
  if (!any(is.finite(y))) {
    plot.new(); title(main = main, xlab = xlab, ylab = ylab)
    text(0.5, 0.5, "no data", col = "grey50")
    return(invisible(FALSE))
  }
  plot(x, y, type = "b", pch = 19, main = main, xlab = xlab, ylab = ylab, ...)
  invisible(TRUE)
}

# Assemble the single dashboard PNG. `fr` is factor_matrix() output at best
# param; `sw` is sweep_factor_curve() output; `res` is the imputer contract;
# `ho` is higher_order() output (NULL -> higher-order panels show "no data").
plot_dashboard <- function(path, res, fr, sw, max_k = 10L, title = "",
                           dims = NULL, ho = NULL) {
  params <- res$params; curve <- res$curve; curve_r2 <- res$curve_r2
  pname <- res$param_name; mname <- res$metric_name
  best <- res$best_param

  png(path, width = 1400, height = 1350, res = 110)
  op <- par(mfrow = c(3, 3), mar = c(4, 4, 3, 4), oma = c(0, 0, 2, 0))

  # 1. predictive metric vs param: RMSE (left axis) + R^2 overlay (right axis).
  if (.safe_plot(params, curve, paste0("1. ", mname, " + R2 vs ", pname),
                 pname, mname)) {
    abline(v = best, lty = 3, col = "red")
    if (!is.null(curve_r2) && any(is.finite(curve_r2))) {
      par(new = TRUE)
      plot(params, curve_r2, type = "b", pch = 17, col = "blue", axes = FALSE,
           xlab = "", ylab = "", lty = 2)
      axis(4, col = "blue", col.axis = "blue")
      mtext("held-out R2", side = 4, line = 2.5, col = "blue", cex = .7)
      abline(h = 0, lty = 3, col = "blue")  # R2=0 = no-skill baseline
      legend("right", c(mname, "R2", "R2=0"), pch = c(19, 17, NA),
             lty = c(1, 2, 3), col = c("black", "blue", "blue"),
             bty = "n", cex = .7)
    } else {
      legend("topright", "CV-best", lty = 3, col = "red", bty = "n", cex = .8)
    }
  }

  # 2. marginal gain (reduction in metric per param step)
  marg <- c(NA, -diff(curve))
  barplot(ifelse(is.na(marg), 0, marg), names.arg = params,
          xlab = pname, ylab = paste(mname, "reduction vs prev"),
          main = "2. Marginal gain per step")

  # 3. cumulative variance vs param
  .safe_plot(params, sw$cumvar, "3. Variance explained", pname,
             "cumulative var explained")

  # 4. scree @ best param
  ne <- min(length(fr$eig), max_k + 5L)
  plot(seq_len(ne), fr$eig[seq_len(ne)], type = "b", pch = 19,
       xlab = "component", ylab = "eigenvalue",
       main = sprintf("4. Scree @best (PA nf: %d)", fr$nf))
  if (!is.null(fr$cutoffs))
    lines(seq_len(min(ne, length(fr$cutoffs))),
          fr$cutoffs[seq_len(min(ne, length(fr$cutoffs)))], col = "red", lty = 2)
  abline(h = 1, lty = 3, col = "grey60")

  # 5. SS loadings per factor @ best param
  ss <- colSums(unclass(fr$efa$loadings)^2)
  barplot(ss, names.arg = seq_along(ss), xlab = "factor",
          ylab = "SS loadings", main = "5. SSQ loadings @best")

  # 6. PA factor count vs param
  if (.safe_plot(params, sw$pa_nf, paste0("6. PA factor count vs ", pname),
                 pname, "PA factors"))
    abline(0, 1, lty = 2, col = "grey60")

  # ── higher-order panels (7-9): the two bifactor panels (8 g-loadings, 9 omega)
  #    are kept side by side; second-order (single hardcoded g — first-order
  #    factors are too noisy to justify a data-driven 2nd-order count) is panel 7.
  # 7. second-order loadings (first-order factors on the general factor)
  so <- if (!is.null(ho)) ho$second_loadings else NULL
  if (!is.null(so)) {
    v <- so[, 1]
    barplot(v, names.arg = rownames(so), las = 2, cex.names = .7,
            ylab = "2nd-order loading",
            main = "7. Second-order loadings (factors -> g)")
    abline(h = 0, col = "grey60")
  } else {
    plot.new(); title("7. Second-order loadings"); text(.5,.5,"no data",col="grey50")
  }

  # 8. bifactor general-factor (g) loadings across benchmarks
  g <- if (!is.null(ho) && !is.null(ho$bifactor_loadings) &&
           "g" %in% colnames(ho$bifactor_loadings))
    ho$bifactor_loadings[, "g"] else NULL
  if (!is.null(g) && any(is.finite(g))) {
    hist(g, breaks = 20, col = "grey80", xlab = "g loading",
         main = sprintf("8. Bifactor g loadings (omega_h=%.2f)",
                        if (is.finite(ho$omega_h)) ho$omega_h else NA))
    abline(v = mean(g, na.rm = TRUE), col = "red", lwd = 2)
  } else {
    plot.new(); title("8. Bifactor g loadings"); text(.5,.5,"no data",col="grey50")
  }

  # 9. omega coefficients: omega_total, omega_h (general), then per-group omega_hs
  #    (the `group` column of omega.group) — the g-vs-group reliability split.
  og <- if (!is.null(ho)) ho$omega_group else NULL
  if (!is.null(ho) && is.finite(ho$omega_total)) {
    hs <- if (!is.null(og) && "group" %in% colnames(og)) {
      gr <- og[rownames(og) != "g", "group", drop = TRUE]  # per-factor omega_hs
      gr
    } else numeric(0)
    vals <- c(omega_t = ho$omega_total, omega_h = ho$omega_h, hs)
    nm <- c("ωt", "ωh", if (length(hs)) paste0("ωhs.", seq_along(hs)) else character(0))
    cols <- c("grey40", "firebrick", rep("steelblue", length(hs)))
    barplot(vals, names.arg = nm, col = cols, ylim = c(0, 1), las = 2,
            cex.names = .8, ylab = "omega", main = "9. Omega coefficients")
    abline(h = c(0.5, 0.8), lty = 3, col = "grey70")
  } else {
    plot.new(); title("9. Omega coefficients")
    text(.5, .5, "no higher-order output", col = "grey50")
  }

  sub <- if (!is.null(dims))
    sprintf("%s  (%dx%d, %.1f%% obs)", title, dims[1], dims[2], dims[3]) else title
  mtext(sub, outer = TRUE, cex = 1.1)
  par(op); dev.off()
  cat("  wrote", path, "\n")
}
