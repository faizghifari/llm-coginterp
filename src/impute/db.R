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
  } else {
    dbExecute(con, "
      CREATE TABLE IF NOT EXISTS imputation (
        dataset TEXT NOT NULL,
        method  TEXT NOT NULL,
        rmse    REAL NOT NULL,
        r2      REAL,
        desc    TEXT,
        wr2     REAL,
        PRIMARY KEY (dataset, method)
      )
    ")
    # wr2: hierarchical weighted R2 (impute.R --hier) — the held-out R2
    # weighted by per-step introduced columns. Old DBs predate the column;
    # ALTER is the in-place migration (errors away when it already exists).
    if (!"wr2" %in% dbListFields(con, "imputation"))
      dbExecute(con, "ALTER TABLE imputation ADD COLUMN wr2 REAL")
  }
}

db_insert_imputation <- function(method, dataset, rmse, r2, desc,
                                 db_file, overwrite = FALSE,
                                 table = "imputation",
                                 cell = NULL, mask = NULL,
                                 wr2 = NA_real_) {
  synth <- identical(table, "imputation_synth")
  db_ensure_table(db_file, table)
  con <- dbConnect(SQLite(), db_file)
  on.exit(dbDisconnect(con))

  if (synth) {
    key_where <- "cell = ? AND mask = ? AND dataset = ? AND method = ?"
    key_vals  <- list(cell, mask, dataset, method)
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
  } else {
    res <- dbSendStatement(con, paste0(
      "INSERT OR REPLACE INTO imputation ",
      "(dataset, method, rmse, r2, desc, wr2) VALUES (?, ?, ?, ?, ?, ?)"))
    vals <- list(dataset, method, rmse,
                 if (is.na(r2)) NA_real_ else r2, desc,
                 if (is.na(wr2)) NA_real_ else wr2)
  }
  dbBind(res, vals)
  dbClearResult(res)
  wr2_txt <- if (is.na(wr2)) "" else sprintf(" wr2=%.3f", wr2)
  if (synth)
    cat(sprintf("  db: [%s/%s] %s/%s rmse=%.4f r2=%.3f%s\n",
                cell, mask, method, dataset, rmse, r2, wr2_txt))
  else
    cat(sprintf("  db: %s/%s rmse=%.4f r2=%.3f%s\n",
                method, dataset, rmse, r2, wr2_txt))
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