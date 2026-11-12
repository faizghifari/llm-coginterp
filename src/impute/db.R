# ─────────────────────────────────────────────────────────────────────────────
# SQLite result persistence for the imputation step.
#
# Writes one row per (method × dataset) cell into results/database.db, table
# `imputation`, after the sweep completes.  No-op when the row already exists
# unless `overwrite = TRUE` (which mirrors the --reimpute flag).
#
# table = "imputation_synth" instead targets results/synth/database.db with
# the cell/mask in the key (impute.R --synth); see db_ensure_table.
# ─────────────────────────────────────────────────────────────────────────────

suppressMessages({ library(DBI); library(RSQLite) })

db_ensure_table <- function(db_file, table = "imputation") {
  con <- dbConnect(SQLite(), db_file)
  on.exit(dbDisconnect(con))
  if (identical(table, "imputation_synth")) {
    # Synthetic stress-corpus rows (impute.R --synth). Every data/synth root
    # shares the real corpus's `dataset` names, so cell/mask join the PK.
    # gt_r2 is reserved for scoring against the planted truth
    # (data/synth/truth/) and is left NA by the imputation stage.
    dbExecute(con, "
      CREATE TABLE IF NOT EXISTS imputation_synth (
        cell    TEXT NOT NULL,
        mask    TEXT NOT NULL,
        dataset TEXT NOT NULL,
        method  TEXT NOT NULL,
        rmse    REAL NOT NULL,
        r2      REAL,
        gt_r2   REAL,
        desc    TEXT,
        PRIMARY KEY (cell, mask, dataset, method)
      )
    ")
  } else if (identical(table, "imputation_hier")) {
    # Hierarchical descent rows (run/impute_hier.R). One database.db per
    # results root (results/<...>/hierarchical/), but `dataset` (<dz>_<st>)
    # repeats across density slices, so density joins the PK.
    dbExecute(con, "
      CREATE TABLE IF NOT EXISTS imputation_hier (
        density TEXT NOT NULL,
        dataset TEXT NOT NULL,
        method  TEXT NOT NULL,
        rmse    REAL NOT NULL,
        r2      REAL,
        desc    TEXT,
        PRIMARY KEY (density, dataset, method)
      )
    ")
  } else {
    dbExecute(con, "
      CREATE TABLE IF NOT EXISTS imputation (
        dataset TEXT NOT NULL,
        method  TEXT NOT NULL,
        rmse    REAL NOT NULL,
        r2      REAL,
        desc    TEXT,
        PRIMARY KEY (dataset, method)
      )
    ")
  }
}

db_insert_imputation <- function(method, dataset, rmse, r2, desc,
                                 db_file, overwrite = FALSE,
                                 table = "imputation",
                                 cell = NULL, mask = NULL, density = NULL) {
  synth <- identical(table, "imputation_synth")
  hier  <- identical(table, "imputation_hier")
  db_ensure_table(db_file, table)
  con <- dbConnect(SQLite(), db_file)
  on.exit(dbDisconnect(con))

  if (synth) {
    key_where <- "cell = ? AND mask = ? AND dataset = ? AND method = ?"
    key_vals  <- list(cell, mask, dataset, method)
  } else if (hier) {
    key_where <- "density = ? AND dataset = ? AND method = ?"
    key_vals  <- list(density, dataset, method)
  } else {
    key_where <- "dataset = ? AND method = ?"
    key_vals  <- list(dataset, method)
  }

  if (!overwrite) {
    existing <- dbGetQuery(con,
      paste("SELECT 1 FROM", table, "WHERE", key_where),
      params = key_vals)
    if (nrow(existing) > 0)
      return(invisible(FALSE))
  }

  if (synth) {
    res <- dbSendStatement(con, paste0(
      "INSERT OR REPLACE INTO imputation_synth ",
      "(cell, mask, dataset, method, rmse, r2, gt_r2, desc) ",
      "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"))
    vals <- c(key_vals, list(rmse,
                             if (is.na(r2)) NA_real_ else r2,
                             NA_real_, desc))
  } else if (hier) {
    res <- dbSendStatement(con, paste0(
      "INSERT OR REPLACE INTO imputation_hier ",
      "(density, dataset, method, rmse, r2, desc) VALUES (?, ?, ?, ?, ?, ?)"))
    vals <- c(key_vals, list(rmse,
                             if (is.na(r2)) NA_real_ else r2, desc))
  } else {
    res <- dbSendStatement(con, paste0(
      "INSERT OR REPLACE INTO imputation ",
      "(dataset, method, rmse, r2, desc) VALUES (?, ?, ?, ?, ?)"))
    vals <- list(dataset, method, rmse,
                 if (is.na(r2)) NA_real_ else r2, desc)
  }
  dbBind(res, vals)
  dbClearResult(res)
  if (synth)
    cat(sprintf("  db: [%s/%s] %s/%s rmse=%.4f r2=%.3f\n",
                cell, mask, method, dataset, rmse, r2))
  else if (hier)
    cat(sprintf("  db: [%s] %s/%s rmse=%.4f r2=%.3f\n",
                density, method, dataset, rmse, r2))
  else
    cat(sprintf("  db: %s/%s rmse=%.4f r2=%.3f\n",
                method, dataset, rmse, r2))
  invisible(TRUE)
}

# Refresh gt_r2 (truth-recovery R^2) for one imputation_synth row. Called by
# impute.R --synth after the imputation pass; recomputed on every invocation,
# independent of --reimpute. No-op when the row doesn't exist.
db_update_gt_r2 <- function(cell, mask, dataset, method, gt_r2, db_file) {
  db_ensure_table(db_file, "imputation_synth")
  con <- dbConnect(SQLite(), db_file)
  on.exit(dbDisconnect(con))
  res <- dbSendStatement(con, paste0(
    "UPDATE imputation_synth SET gt_r2 = ? ",
    "WHERE cell = ? AND mask = ? AND dataset = ? AND method = ?"))
  dbBind(res, list(gt_r2, cell, mask, dataset, method))
  dbClearResult(res)
  invisible(TRUE)
}

# Human-readable hyperparameter sweep description.
# Contiguous ranges like 1:10 become "rank=5 (swept 1..10)";
# sparse grids like c(50,100,200,400) become "ntree=100 (swept [50,100,200,400])".
build_desc <- function(params, param_name, best_param) {
  if (length(params) == 1L)
    return(sprintf("%s=%s", param_name, best_param))
  diffs <- diff(params)
  if (length(unique(diffs)) == 1L && unique(diffs)[1L] == 1L) {
    return(sprintf("%s=%s (swept %d..%d)",
                   param_name, best_param, min(params), max(params)))
  }
  sprintf("%s=%s (swept [%s])", param_name, best_param,
          paste(params, collapse = ","))
}