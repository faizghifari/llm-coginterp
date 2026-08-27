# DATAFLOW

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
`complete_at(v)` (used by the dashboard to factor at every swept value).

Two families of imputers:

- **Cell-filling** (softimpute, knn, missforest, mice, iterativepca): mask
  ~20% of observed cells (`make_holdout`, column-stratified, `min_keep=2`),
  fit on the rest, predict the masked cells, score with `score_holdout`
  (column-balanced by default; `--no-balance` reverts to cell-weighted), return
  the completed matrix at the CV-best hyperparameter.
- **Correlation-matrix** (softimpute_corr, optspace, usvt, cvxr, ggm): build
  the observed pairwise correlation matrix (NA where never co-observed), impute
  those NA entries, `symmetrize_nearpd` to a valid PD correlation, predict
  held-out *cells* via the conditional-Gaussian best-linear predictor
  (`predict_cell_from_corr`), then emit a **covariance-matched surrogate** whose
  covariance equals the imputed correlation (`generate_surrogate`, original
  column scale). Shared driver `run_corr_single` (`corr_common.R`).

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
| imputation row | `results/<...>/database.db` table `imputation` (PK `dataset,method`) | `db_insert_imputation` |

The completed CSV is the **hand-off artifact** to the factor stage: columns are
benchmarks, rows are models, first column `collapse_key`. This is the only thing
`factor.R` consumes from imputation.

> `--reimpute` default OFF: if the completed CSV already exists, the
> orchestrator skips imputation and rebuilds a partial contract from disk
> (`read_matrix`), so higher-order/metric changes can be re-applied without
> re-running the slow imputation.

## Stage 2 — factoring (`src/factor/`)

`factor.R` (or `main.R`) reads a completed matrix, then:

1. **Gate on imputation quality.** Reads `R²` for the cell from the SQLite
   `imputation` table (`db_read_r2`). If `R² < 0.4` (or absent), the cell is
   skipped. `raw` / `default` / `zeros` methods bypass the gate (they factor the
   *sparse* table directly via pairwise-complete correlation + PSD smoothing,
   `factor_raw`).
2. **Factor count via Horn's parallel analysis** (`parallel_analysis.R`). The
   random-baseline eigenvalue cutoffs depend only on shape `(n, p, n.iter,
   quantile)`, so they are computed once and **cached as JSON in
   `factor/pa_cache/`**, keyed by shape. Observed eigenvalues come from
   `eigen(cor(M))`. `nfactors = #(observed > cutoff)`, floor 2.
3. **Minres + promax EFA** at that count (`factor_matrix` → `fa_try`), degrading
   gracefully down to nf=2.
4. **Higher-order / bifactor** via `psych::omega` (`higher_order`) — Schmid-
   Leiman loadings plus ω_h, ω_total, per-group ω_hs. Two runs per cell: at the
   PA factor count (`pa`) and forced to 2 factors (`2f`).
5. **(optional `--loco`)** leave-one-covariate-out Δω_h per benchmark →
   SQLite table `loco`.

**Outputs** (per cell, per run tag `pa` and `2f`):

| output | path |
|---|---|
| bifactor loadings | `results/<method>/<method>_<dz>_<st>_bifactor_<pa\|2f>_loadings.csv` + `.md` |
| smoothed correlation (raw methods only) | `results/<method>/<method>_<dz>_<st>_correlation.csv` |
| omega scalars | `results/<method>/..._bifactor_<pa\|2f>_scalars.csv` |
| per-group ω_hs | `results/<method>/..._bifactor_<pa\|2f>_omega_group.csv` |
| factoring row | `results/<...>/database.db` table `factoring` (PK `dataset,method,run`) |

`write_higher_order` (CSV + MD) and `db_insert_factoring` (SQLite) are the only
writers here. `matrix_to_markdown` bolds |loading| ≥ 0.4 and sorts rows by
primary-factor assignment. For raw methods (`default`/`zeros`),
`write_correlation_csv` additionally persists the PSD-smoothed pairwise-complete
correlation the factoring ran on, so downstream scoring does not have to
reimplement the fill/smoothing recipe.

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
| raw methods (default/zeros) | sparse `data/combinations[_<dz>]/<st>/model_benchmark_table.csv` + the persisted `..._correlation.csv` | the persisted smoothed pairwise-complete R | `..._bifactor_<tag>_scores_conditional.csv` + `..._scores_prorated.csv` |

Raw cells get **two** estimators for triangulation (both use Λ; they differ
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
(raw methods persist                     (reads imputed CSVs / sparse tables +
 <...>_correlation.csv)                   the persisted R, writes *_scores*.csv)
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
- **Completed CSVs are the only artifact `factor.R` reads from imputation**;
  everything else (sweep, DB) is recomputed or stored alongside.
- **Latent-score naming:** bare `..._scores.csv` is reserved for
  complete-data (imputed) cells; raw-method cells always emit
  `..._scores_conditional.csv` / `..._scores_prorated.csv`. Raw scoring also
  requires the `..._correlation.csv` persisted by factoring — re-run factoring
  for a raw cell if that file is missing (the scorer skips with a notice).