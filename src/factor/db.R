# ─────────────────────────────────────────────────────────────────────────────
# SQLite persistence for factor-analysis results.
#
# Writes factoring outcomes into results/<prefix>/database.db, table `factoring`.
# Reads imputation R² from the sibling table `imputation` (managed by
# src/impute/db.R) to gate factoring on imputation quality.
# ─────────────────────────────────────────────────────────────────────────────

suppressMessages({ library(DBI); library(RSQLite); library(jsonlite) })

db_ensure_factoring_table <- function(con) {
  dbExecute(con, "
    CREATE TABLE IF NOT EXISTS factoring (
      dataset       TEXT NOT NULL,
      method        TEXT NOT NULL,
      run           TEXT NOT NULL,
      nf            INTEGER NOT NULL,
      var_explained REAL,
      omega_t       REAL,
      omega_h       REAL,
      omega_hs      TEXT,
      PRIMARY KEY (dataset, method, run)
    )
  ")
}

db_read_r2 <- function(method, dataset, db_file) {
  con <- dbConnect(SQLite(), db_file)
  on.exit(dbDisconnect(con))
  row <- dbGetQuery(con,
    "SELECT r2 FROM imputation WHERE dataset = ? AND method = ?",
    params = list(dataset, method))
  if (nrow(row) == 0 || is.na(row$r2[1])) return(NA_real_)
  row$r2[1]
}

db_insert_factoring <- function(method, dataset, run, nf, var_explained,
                                omega_t, omega_h, omega_hs, db_file) {
  con <- dbConnect(SQLite(), db_file)
  on.exit(dbDisconnect(con))
  db_ensure_factoring_table(con)

  hs_json <- if (length(omega_hs) == 0) "[]" else toJSON(omega_hs, digits = 4, na = "null")

  dbExecute(con,
    "INSERT OR REPLACE INTO factoring
       (dataset, method, run, nf, var_explained, omega_t, omega_h, omega_hs)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
    params = list(dataset, method, run, nf, var_explained,
                  omega_t, omega_h, hs_json))
  cat(sprintf("  db: %s/%s/%s nf=%d var=%.3f ωt=%.3f ωh=%.3f ωhs=%s\n",
              method, dataset, run, nf, var_explained,
              omega_t, omega_h, hs_json))
  invisible(TRUE)
}