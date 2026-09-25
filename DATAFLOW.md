# DATAFLOW

> **WARNING: FORCED2F IS STALE AND SHOULD BE IGNORED.** The `forced2f` runs (and
> their `forced2f_y<year>` timed variants) recorded in `results/` and the
> `factoring` table of `database.db` are stale. No downstream consumer reads
> them: the release-date analysis (`scripts/release_date.py`) reads only
> `*_bifactor_pa_loadings.csv`, and the viewer
> (`viewer/compute_positions.py`) explicitly skips every tag except `pa`
> (plus legacy untagged files in `--legacy-names` mode). Do not use, cite, or
> extend the forced2f outputs.

How data moves through the MachineG2 pipeline: where every file is read from,
written to, and how the stages hand state to each other. This complements
`src/README.md` (which explains *why* each stage exists); this document is
about the *paths and the hand-offs*.

## Big picture

```
 source CSVs ──► densify ──► impute ──► factor ──► results (CSV/MD) + SQLite ──► latent scores
 (data/)   (scripts/)  (src/impute/ + Julia)  (src/factor/)     (results/)      (scripts/)
```

The pipeline is a **cross-product** of `{densifier C,S,R} × {strategy
all_standard, all_aggressive} × {imputation method}`. Every "cell" is one
`(method, densifier, strategy)` triple, and every cell follows the identical
path: read a sparse model×benchmark table → impute (or synthesize a surrogate)
→ factor → persist loadings + scalars + database rows.

Three separate orchestrators drive slices of this cross-product, all anchored to
the repo root so they run from any working directory:

| orchestrator | does | when |
|---|---|---|
| `src/run/main.R` | impute → factor → bifactor → DB, per cell | full pipeline |
| `src/run/impute.R` | impute + held-out sweep only | re-impute without re-factoring |
| `src/run/factor.R` | factor + bifactor only (reads completed CSVs) | re-factor without re-imputing |

All three resolve their own location via `commandArgs(--file=)`, compute
`REPO = dirname(dirname(SRC))`, and anchor every input/output path to `REPO`.

## Directory layout (the two trees)

Code lives under `src/`, helper scripts under `scripts/`, and all **data +
results at the repo root**.

```
data/                    multimodal-inclusive corpus (analysis-adjacent)
  results.csv            raw model × benchmark scores (the source of truth)
  models.csv benchmarks.csv
  combinations/<strategy>/model_benchmark_table.csv   aggregated, ~3–5% filled
  combinations_<C|R|S>/<strategy>/model_benchmark_table.csv   densified slices
  densify_summary.csv    shape/density rollup across all densifiers
  imputed/<method>/<dz>/<strategy>/imputed_model_benchmark_table.csv  completed/surrogate
data/text_only/          DEFAULT analysis corpus (subset of data/, mirror layout)
data/smoke/              tiny synthetic fixture (make_smoke.py), mirror layout

results/<method>/<method>_<dz>_<st>_<suffix>   all factoring outputs (flat names)
results/_osmc_sweep/<dz>_<st>/                OSMC per-r surrogates + curves
results/<...>/database.db                     SQLite: imputation, factoring, loco
results/smoke/...                             smoke variant of the above
```

> The default `data_root` differs by entry point: `main.R` defaults to `data`,
> while `impute.R` and `factor.R` (and `densify.py`) default to
> `data/text_only`. `--data-root` / `--results-root` overrides all of them.

## Stage 0 — densify (`scripts/densify.py`)

Python, independent of the R/Julia pipeline. Produces the input tables the
imputers read.

| | path |
|---|---|
| read | `<root>/combinations/<strategy>/model_benchmark_table.csv` |
| write | `<root>/combinations_<dz>/<strategy>/model_benchmark_table.csv` |
| write | `<root>/combinations_<dz>/<strategy>/summary.csv`, `<root>/combinations_<dz>/summary.csv`, `<root>/densify_summary.csv` |

Densifiers `C`/`R`/`S` greedily peel to `TARGET = 0.10` density with a
`MIN_OBS` floor on both axes (`MIN_OBS` is currently `3` in code — the docs
still say `2`; the floor is a constant at the top of the file). `raw` (the
undensified table) is *not* produced here — it is the source table read
directly.

## Stage 1 — imputation (`src/impute/`, OSMC in Julia)

**Input resolution** (shared by all three orchestrators):

```
combos_path(dz, st) =
  dz == "raw" ? <root>/combinations/<st>/model_benchmark_table.csv
              : <root>/combinations_<dz>/<st>/model_benchmark_table.csv
```

**Preprocessing** — `prep_matrix(path)` (`impute/common.R`): reads the CSV,
drops columns with `<2` observed cells **or** zero observed variance (breaks
`softImpute::biScale`), returns `list(x = matrix, keys = collapse_key)`. OSMC
mirrors this in Julia with `drop_degenerate_cols`.

**The uniform imputer contract.** Every `impute_<method>()` returns a list with
`M` (completed matrix, rows=models, cols=benchmarks), `best_param`, `params`,
`curve` (held-out RMSE per param), `curve_r2`, `param_name`, `metric_name`, and
`complete_at(v)` (used by the dashboard to factor at every swept value). The
fill-smooth methods additionally return `R`, the exact smoothed correlation
(see below).

Two families of imputers:

- **Cell-filling** (softimpute, knn, missforest, mice, iterativepca): mask
  ~20% of observed cells (`make_holdout`, column-stratified, `min_keep=2`),
  fit on the rest, predict the masked cells, score with `score_holdout`
  (column-balanced by default; `--no-balance` reverts to cell-weighted), return
  the completed matrix at the CV-best hyperparameter.
- **Correlation-matrix** (softimpute_corr, optspace, usvt, cvxr, ggm, and the
  fill-smooth pair `default`/`zeros`): build
  the observed pairwise correlation matrix (NA where never co-observed), impute
  those NA entries, `symmetrize_nearpd` to a valid PD correlation, predict
  held-out *cells* via the conditional-Gaussian best-linear predictor
  (`predict_cell_from_corr`), then emit a **covariance-matched surrogate** whose
  covariance equals the imputed correlation (`generate_surrogate`, original
  column scale). Shared driver `run_corr_single` (`corr_common.R`).
  **Fill-smooth** (`default`, `zeros`; driver `run_fill_smooth`, same held-out
  protocol) completes the correlation with a fixed recipe instead of a model:
  fill never-co-observed pairs with the mean finite off-diagonal (`default`) or
  0 (`zeros`), then `psych::cor.smooth`. No hyperparameter sweep (the reported
  param is the fill rule). Its final correlation is the *exact* recipe output —
  no nearPD on top — returned as `R` and persisted; that cached matrix is what
  its EFA and latent scoring run on.

**OSMC is the odd one out.** It never imputes cells. `main.R`/`impute.R` shell
out once up front to `julia impute/OneSidedMC/run.jl`, which recovers the
right-singular-vector product Θ̂ = V̂V̂ᵀ from *ragged* observations
(`RaggedObs = Vector{Tuple{cols, vals}}`), picks rank `r` by held-out
cell-level RMSE/R², and writes a surrogate whose covariance equals Θ̂. R then
reads those outputs back through `osmc_contract()` as if they were an imputer's
`M`.

**Outputs of imputation** (per cell):

| output | path | writer |
|---|---|---|
| completed matrix | `data/imputed/<method>/<dz>/<st>/imputed_model_benchmark_table.csv` | `write_completed` (or OSMC `write_surrogate`) |
| rank-sweep curve | `results/<method>/<method>_<dz>_<st>_rank_sweep.csv` | orchestrator |
| smoothed correlation (fill-smooth `default`/`zeros` only) | `results/<method>/<method>_<dz>_<st>_correlation.csv` | orchestrator (`write_correlation_csv`) |
| imputation row | `results/<...>/database.db` table `imputation` (PK `dataset,method`) | `db_insert_imputation` |

The completed CSV is the **hand-off artifact** to the factor stage: columns are
benchmarks, rows are models, first column `collapse_key`. For the fill-smooth
methods the persisted correlation cache is a second hand-off — it is the exact
matrix their EFA and their latent scoring run on (the surrogate cannot recover
it: it only matches the covariance up to sampling noise).

> `--reimpute` default OFF: if the completed CSV already exists, the
> orchestrator skips imputation and rebuilds a partial contract from disk
> (`read_matrix`), so higher-order/metric changes can be re-applied without
> re-running the slow imputation. Fill-smooth cells skip only when the
> correlation cache exists too (their imputation is cheap anyway).

## Stage 2 — factoring (`src/factor/`)

`factor.R` (or `main.R`) reads a completed matrix, then:

1. **Gate on imputation quality.** Reads `R²` for the cell from the SQLite
   `imputation` table (`db_read_r2`). If `R² < 0.3` (or absent), the cell is
   skipped — every method is gated, the fill-smooth pair included (they are
   imputers now and their held-out R² lives in the same table). Fill-smooth
   cells (`default`/`zeros`) then factor the **cached smoothed correlation**
   their imputation persisted (`factor_cached_R`) — the exact fill + PSD recipe
   output — instead of `cor(M)` of the surrogate.
2. **Factor count via Horn's parallel analysis** (`parallel_analysis.R`). The
   random-baseline eigenvalue cutoffs depend only on shape `(n, p, n.iter,
   quantile)`, so they are computed once and **cached as JSON in
   `factor/pa_cache/`**, keyed by shape. Observed eigenvalues come from
   `eigen(cor(M))` (`eigen(R_cached)` for fill-smooth, at `n_eff = nrow` —
   `choose_nfactors_cached_R`; the PA count is identical because `cor.smooth`
   only alters eigenvalues below 1e-6, which never clear the cutoffs).
   `nfactors = #(observed > cutoff)`, floor 2.
3. **Minres + promax EFA** at that count (`factor_matrix` → `fa_try`;
   `factor_cached_R` → `fa_try` for fill-smooth), degrading
   gracefully down to nf=2.
4. **Higher-order / bifactor** via `psych::omega` (`higher_order`) — Schmid-
   Leiman loadings plus ω_h, ω_total, per-group ω_hs. Two runs per cell: at the
   PA factor count (`pa`) and forced to 2 factors (`2f`).
5. **(optional `--loco`)** leave-one-covariate-out Δω_h per benchmark →
   SQLite table `loco`. LOCO never re-derives a correlation: fill-smooth cells
   load the imputation-persisted cache; the others use `cor(M)`.

**Outputs** (per cell, per run tag `pa` and `2f`):

| output | path |
|---|---|
| bifactor loadings | `results/<method>/<method>_<dz>_<st>_bifactor_<pa\|2f>_loadings.csv` + `.md` |
| omega scalars | `results/<method>/..._bifactor_<pa\|2f>_scalars.csv` |
| per-group ω_hs | `results/<method>/..._bifactor_<pa\|2f>_omega_group.csv` |
| factoring row | `results/<...>/database.db` table `factoring` (PK `dataset,method,run`) |

`write_higher_order` (CSV + MD) and `db_insert_factoring` (SQLite) are the only
writers here. `matrix_to_markdown` bolds |loading| ≥ 0.4 and sorts rows by
primary-factor assignment. The smoothed correlation used to be re-persisted
here for `default`/`zeros`; it is now written by the imputation stage (Stage 1)
so the recipe lives in exactly one place.

**Timed mode** (`make factor-timed` → `factor.R --timed`): year-separated EFA
instead of the pooled matrix. Reads
`<data_root>/combinations/<st>/collapse_mapping.csv` (written by
`scripts/collapse_results.py`), maps each `collapse_key` to a release year via
the first `19xx`/`20xx` in `release_date` (blank/junk dates excluded), then for
every `(method, dz, st)` cell partitions the completed matrix's rows into
release-year cohorts and reruns the standard imputed-path factoring per cohort
(R² gate → PA → bifactor at `pa`/`2f`). The fill-smooth methods (`default`,
`zeros`) are skipped — their cached correlation is global and cannot be cut
per cohort — and `--loco` is rejected in this mode. There is no minimum
cohort size: degenerate years log `FACTOR FAILED` and the loop continues.
Outputs insert the year into the run tag:
`<method>_<dz>_<st>_bifactor_pa_y<year>_loadings.csv` etc., DB table
`factoring` rows with `run = pa_y<year>` / `forced2f_y<year>` and `dataset`
unchanged. `latent_scores.py` skips these files (cohort loadings are not
scored against the pooled matrix).

**Release-date analysis** (`make release-date` → `scripts/release_date.py run`,
which calls `src/run/release_cohorts.R`) is a separate, downstream consumer.
It uses the same `collapse_mapping.csv` dates as timed mode, but bins them into
four cohorts (<=2022, 2023, 2024, >=2025) and factors each one against 50
random subsets of the same size. Only row-preserving imputers that pass
`R2_GATE` enter. It reads the completed matrices, the pooled `*_bifactor_pa_loadings.csv`
and the `imputation` table, and writes only to `<results_root>/release_date/`
(no DB rows). `make release-date-report` merges several runs into the paper's
tables and figure.

## Stage 3 — latent scores (`scripts/latent_scores.py`)

Python, reads factoring outputs and writes per-model factor scores next to the
loadings they came from. No SQLite involvement.

For every discovered `*_loadings.csv` (parsed into `(method, dz, st, tag)`,
tag ∈ {pa, 2f}), it locates the matching data, z-scores benchmark columns, and
applies regression (Thomson) scoring `F = Z R⁻¹Λ` — loadings turned into beta
weights via the correlation matrix, pseudoinverse fallback for low-rank R:

| cell type | data read | R used | writes |
|---|---|---|---|
| imputed methods (incl. onesidedmc) | `data/imputed/<method>/<dz>/<st>/imputed_model_benchmark_table.csv` | `cor(Z)` of the scored columns | `results/<method>/<method>_<dz>_<st>_bifactor_<tag>_scores.csv` |
| fill-smooth methods (default/zeros) | sparse `data/combinations[_<dz>]/<st>/model_benchmark_table.csv` + the imputation-persisted `..._correlation.csv` | the persisted smoothed fill+smooth R | `..._bifactor_<tag>_scores_conditional.csv` + `..._scores_prorated.csv` |

Fill-smooth cells get **two** estimators for triangulation (both use Λ; they differ
only in how they handle missing cells):

- **conditional** (`_scores_conditional.csv`): per model, the posterior-mean
  score given its observed benchmarks, `F̂ᵢ = Λ_Oᵢᵀ R_OᵢOᵢ⁻¹ zᵢ,Oᵢ` — i.e.
  implicit regression-imputation of the missing benchmarks, in closed form;
- **prorated** (`_scores_prorated.csv`): one global `W = pinv(R)Λ`, missing
  cells treated as z = 0, renormalized by observed weight mass.

Both raw outputs carry an `n_obs` column (observed benchmark count per model).
Every output has an `all_sum` column = rowwise sum of g + all group-factor
scores (the Schmid–Leiman group factors are orthogonal, so the sum is a valid
aggregate index in SD units). `h2`/`u2`/`uq2`/`p2`/`com` columns of the
loadings CSV are diagnostics and are never scored. Full methodology lives in
the comment block at the top of `scripts/latent_scores.py`.

## The SQLite database

One `database.db` per results root (`results/`, `results/smoke/`,
`results/text_only/`), written by both stages:

| table | PK | written by | purpose |
|---|---|---|---|
| `imputation` | `(dataset, method)` | `impute/db.R` | held-out RMSE/R² + param description; gates factoring |
| `factoring` | `(dataset, method, run)` | `factor/db.R` | nf, variance, ω_t/ω_h/ω_hs, φ (JSON-encoded vectors) |
| `loco` | `(dataset, method, run)` | `factor/db.R` | per-benchmark Δω_h from the LOCO sweep |

`dataset` = `"<dz>_<st>"` (e.g. `C_all_standard`).

## Hand-offs / contract summary

```
densify.py            ──CSV──►  impute_<method>            ──M──►  factor_<method>
(combinations_<dz>)                 │  │                          │
                                    │  └─ imputed CSV ────────────┘ (completed matrix)
                                    │       (data/imputed/...)
                                    └─ sweep ──► rank_sweep.csv
                                               └─► database.db: imputation
                                                  database.db: factoring
                                                  (read by factor for the R² gate)

factor.R              ──loadings + R──►  latent_scores.py
(fill-smooth R comes from                  (reads imputed CSVs / sparse tables +
 imputation's <...>_correlation.csv)        the persisted R, writes *_scores*.csv)
```

The one hard invariant: **imputers never factor; the orchestrator owns
factoring and all I/O.** A method just returns the contract above, and
`run/` (R) or `run.jl` (OSMC) handles paths, DB, and factoring.

## Gotchas

- **Path anchoring:** every entry point derives `REPO` from its own file
  location (`--file=` in R, `@__DIR__` in Julia, `__file__` in Python). Never
  assume the current working directory.
- **PA cache** is correctness-sensitive and keyed by shape only — never reuse a
  cutoff across shapes.
- **Column scale:** columns are z-scored, rows never (rows are models;
  row-scaling would erase the general factor). Held-out baseline is the *train*
  column mean (0 in z-space).
- **OSMC tests** (`impute/OneSidedMC/test/`) must not break — they cover the
  paper core; `realdata.jl` is additive and exported through the module.
- **Completed CSVs are the main artifact `factor.R` reads from imputation**;
  everything else (sweep, DB) is recomputed or stored alongside. The one
  exception: fill-smooth cells also read the `..._correlation.csv` cache —
  the exact matrix their EFA runs on.
- **Latent-score naming:** bare `..._scores.csv` is reserved for
  complete-data (imputed) cells; fill-smooth cells (default/zeros) always emit
  `..._scores_conditional.csv` / `..._scores_prorated.csv`. Their scoring also
  requires the `..._correlation.csv` persisted by imputation — re-run imputation
  for a fill-smooth cell if that file is missing (the scorer skips with a
  notice).
