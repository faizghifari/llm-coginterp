# ─────────────────────────────────────────────────────────────────────────────
# SQLite result persistence for the imputation step.
#
# Writes one row per (method × dataset) cell into results/database.db, table
# `imputation`, after the sweep completes.  No-op when the row already exists
# unless `overwrite = TRUE` (which mirrors the --reimpute flag).
# ─────────────────────────────────────────────────────────────────────────────

suppressMessages({ library(DBI); library(RSQLite) })

db_ensure_table <- function(db_file) {
  con <- dbConnect(SQLite(), db_file)
  on.exit(dbDisconnect(con))
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

db_insert_imputation <- function(method, dataset, rmse, r2, desc,
                                 db_file, overwrite = FALSE) {
  db_ensure_table(db_file)
  con <- dbConnect(SQLite(), db_file)
  on.exit(dbDisconnect(con))

  if (!overwrite) {
    existing <- dbGetQuery(con,
      "SELECT 1 FROM imputation WHERE dataset = ? AND method = ?",
      params = list(dataset, method))
    if (nrow(existing) > 0)
      return(invisible(FALSE))
  }

  res <- dbSendStatement(con,
    "INSERT OR REPLACE INTO imputation (dataset, method, rmse, r2, desc)
     VALUES (?, ?, ?, ?, ?)")
  dbBind(res, list(dataset, method, rmse,
                   if (is.na(r2)) NA_real_ else r2,
                   desc))
  dbClearResult(res)
  cat(sprintf("  db: %s/%s rmse=%.4f r2=%.3f\n",
              method, dataset, rmse, r2))
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