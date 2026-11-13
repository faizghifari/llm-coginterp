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
| `src/run/impute.R` | impute + held-out sweep only; `--hier` = experimental density-descent mode (below) | re-impute without re-factoring |
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
   <density>/combinations/            copy of the raw level (densify Stage 0 seeds
                                      every density dir with it)
   <density>/combinations_<C|R|S>/<strategy>/model_benchmark_table.csv  densified
                                      slices at that density (e.g. <density>=0.1)
   <density>/densify_summary.csv      shape/density rollup for that density
   <density>/imputed/<method>/<dz>/<strategy>/imputed_model_benchmark_table.csv
data/text_only/          DEFAULT analysis corpus (subset of data/, mirror layout;
                         density slices live one level down, data/text_only/<density>/;
                         plus hierarchical/ — the experimental --hier outputs, below;
                         and 0/ — the reserved hier final-step hand-off slice, below)
data/smoke/              tiny synthetic fixture (make_smoke.py), mirror layout,
                         density-free
data/synth/              synthetic MNAR stress corpus (make_mnar.py), density-nested:
                         <density>/truth/ + <density>/<cell>/<mask>/s0/ drop-in
                         --data-roots (side branch below)

results/<density>/<method>/<method>_<dz>_<st>_<suffix>   all factoring outputs
results/<density>/database.db                  SQLite: imputation, factoring, loco
results/smoke/...                              smoke variant of the above
```

> `--density` is REQUIRED at every entry point (e.g. `--density 0.1`); both roots
> gain the density segment: `<data-root>/<density>/` and `<results-root>/<density>/`.
> The raw `combinations/` level and the corpus root files (`results.csv`,
> `models.csv`, `benchmarks.csv`) are density-independent and stay at the plain
> data root (densify copies `combinations/` into each density dir). The default
> `data_root` differs by entry point: `main.R` defaults to `data`, while
> `impute.R` and `factor.R` (and `densify.py`) default to `data/text_only`.
> `--data-root` / `--results-root` overrides all of them.

## Stage 0 — densify (`scripts/densify.py`)

Python, independent of the R/Julia pipeline. Produces the input tables the
imputers read.

| | path |
|---|---|
| read | `<root>/combinations/<strategy>/model_benchmark_table.csv` |
| write | `<root>/<density>/combinations/` (verbatim copy of the raw level) |
| write | `<root>/<density>/combinations_<dz>/<strategy>/model_benchmark_table.csv` |
| write | `<root>/<density>/combinations_<dz>/<strategy>/summary.csv`, `<root>/<density>/combinations_<dz>/summary.csv`, `<root>/<density>/densify_summary.csv` |

`--density` is required (e.g. `--density 0.1`); the value is both the peel
target and the verbatim directory name. Densifiers `C`/`R`/`S` greedily peel to
that target with a `MIN_OBS` floor on both axes (`MIN_OBS` is currently `3` in
code — the docs still say `2`; the floor is a constant at the top of the file).
`raw` (the undensified table) is *not* produced here — it is the source table
read directly, and copied unchanged into each density dir so every density dir
is self-contained (downstream `raw`-level reads and `collapse_mapping.csv`
resolve inside `<root>/<density>/combinations/`).

## Stage 1 — imputation (`src/impute/`, OSMC in Julia)

**Input resolution** (shared by all three orchestrators):

```
combos_path(dz, st) =
  dz == "raw" ? <root>/<density>/combinations/<st>/model_benchmark_table.csv
              : <root>/<density>/combinations_<dz>/<st>/model_benchmark_table.csv
```

(`<root>` is `--data-root` minus the density segment; the orchestrators append
`--density` once, right after arg parsing, so `DATA_ROOT` already contains it.)

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
| completed matrix | `data/<density>/imputed/<method>/<dz>/<st>/imputed_model_benchmark_table.csv` | `write_completed` (or OSMC `write_surrogate`) |
| rank-sweep curve | `results/<density>/<method>/<method>_<dz>_<st>_rank_sweep.csv` | orchestrator |
| smoothed correlation (fill-smooth `default`/`zeros` only) | `results/<density>/<method>/<method>_<dz>_<st>_correlation.csv` | orchestrator (`write_correlation_csv`) |
| imputation row | `results/<density>/database.db` table `imputation` (PK `dataset,method`) | `db_insert_imputation` |

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

### Hierarchical mode (`impute.R --hier`, `make impute-hier` — experimental)

`--hier` descends the density slices under `<data-root>` **densest-first**
(`0.3 → 0.2 → 0.1`, discovered by globbing and sorted numerically; the
Makefile's `--density` flag is parsed but IGNORED), for one imputer method at
a time. `runmulti.sh` fans out one Rscript per method word exactly as for
`impute` (`--hier --reimpute` injected); bare `make impute-hier` defaults to
the full `IMPUTE_METHODS` set.

- **Seed.** The chain starts from the standard run's cached completed matrix
  at the densest density (`<data-root>/<d1>/imputed/<method>/<dz>/<st>/...`;
  fill-smooth seeds also require their `..._correlation.csv` cache). Missing →
  the cell logs and aborts; nothing is imputed. The seed is copied into the
  hierarchical output tree verbatim and becomes the running matrix `P`.
- **Descent.** Each sparser density `d` is imputed on the **union grid** of
  `P` and that density's sparse table `T`: rows matched by exact
  `collapse_key`, columns by exact benchmark name — never positional, since
  column sets need not nest (densify's greedy peel + `prep_matrix` drops).
  `T`'s observed cells win, `P`'s completed values fill the rest, NAs
  elsewhere; the filled input is written as `filled_model_benchmark_table.csv`
  for auditability. Each step's completed matrix becomes `P` for the next,
  until the densities are exhausted.
- **Honest holdouts.** Filled cells train but are NEVER held out or graded:
  `make_holdout` samples only mask-TRUE cells (`HOLDOUT_MASK` global in
  `common.R`, set per step — every R imputer inherits it; `prep_matrix` now
  returns its surviving columns so the mask can be realigned). OSMC gets the
  same treatment via the `holdout_mask.csv` sidecar: `run.jl`'s single-input
  mode (`OSMC_SINGLE_INPUT`) processes one filled table per descent step with
  `OSMC_HOLDOUT_MASK`, and `holdout_cells` (`pipeline.jl`) draws holdouts only
  from mask-TRUE cells. So each step's sweep/R² measures recovery of that
  density's REAL observations and is comparable to the standard run at the
  same density — training data differs (it has the seeded columns), which is
  exactly the effect under test, so some gain is expected by construction.
- **Reuse.** Same `--reimpute` semantics as the standard loop: an existing
  step CSV is loaded from disk and the chain continues from it (fill-smooth
  steps also require their per-step correlation cache).

| output | path |
|---|---|
| completed + filled-input + mask CSVs | `<data-root>/hierarchical/<density>/imputed/<method>/<dz>/<st>/imputed_model_benchmark_table.csv` (+ `filled_model_benchmark_table.csv`, `holdout_mask.csv`) |
| rank sweep (+ fill-smooth correlation) | `<results-root>/hierarchical/<density>/<method>/<method>_<dz>_<st>_rank_sweep.csv` |
| imputation row | `<results-root>/hierarchical/<density>/database.db` — the REGULAR per-density layout (table `imputation`, `dataset = "<dz>_<st>"`); `DB_FILE` is re-pointed per density step, the same pattern `--synth` uses per root. The seed step's standard-run sweep + best row are mirrored here too. |

These files sit outside every density slice, so nothing picks them up
accidentally — but they are no longer terminal:

**Weighted R² (`wr2`) and the density-0 hand-off.** After the descent,
`hier_finalize_cell` computes a cumulative weighted R² per step (re-derived
from the per-step imputed CSVs + rank sweeps, so it comes out identically on
`--reimpute` reuse): each step's held-out R² is weighted by the columns it
INTRODUCES (`setdiff` vs the running matrix `P`; the seed introduces all of
its columns), summed and divided by the running matrix's total column count —
so wr2 at step d is the introduced-column-weighted mean R² of the matrix as
of d. Every per-density `imputation` row gets its `wr2` (standard runs leave
the column NULL). The FINAL (sparsest) step's imputed CSV + rank sweep (+
correlation cache for fill-smooth) are then copied into the reserved
density-`0` slice (`<data-root>/0/`, `<results-root>/0/`) — refreshed on
every hier run — and that slice's `imputation` row is written with
`wr2` = the final weighted R². `factor.R --density 0` then factors the
hierarchical matrix through the completely standard path: `db_read_r2` reads
`COALESCE(wr2, r2)`, so `R2_GATE` gates on the weighted R² there.
`hier_densities()` excludes `"0"`, so the hand-off slice is never mistaken
for a descent step. The mode itself stays pure imputation — no factoring, no
`loco` — except through this hand-off.

## Stage 2 — factoring (`src/factor/`)

`factor.R` (or `main.R`) reads a completed matrix, then:

1. **Gate on imputation quality.** Reads the cell's gate value from the SQLite
   `imputation` table (`db_read_r2`): `COALESCE(wr2, r2)` — the hierarchical
   weighted R² when present, else the plain held-out R². If the value `< 0.3`
   (or absent), the cell is
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
| bifactor loadings | `results/<density>/<method>/<method>_<dz>_<st>_bifactor_<pa\|2f>_loadings.csv` + `.md` |
| omega scalars | `results/<density>/<method>/..._bifactor_<pa\|2f>_scalars.csv` |
| per-group ω_hs | `results/<density>/<method>/..._bifactor_<pa\|2f>_omega_group.csv` |
| factoring row | `results/<density>/database.db` table `factoring` (PK `dataset,method,run`) |

`write_higher_order` (CSV + MD) and `db_insert_factoring` (SQLite) are the only
writers here. `matrix_to_markdown` bolds |loading| ≥ 0.4 and sorts rows by
primary-factor assignment. The smoothed correlation used to be re-persisted
here for `default`/`zeros`; it is now written by the imputation stage (Stage 1)
so the recipe lives in exactly one place.

**Timed mode** (`make factor-timed` → `factor.R --timed`): year-separated EFA
instead of the pooled matrix. Reads
`<data_root>/<density>/combinations/<st>/collapse_mapping.csv` (the raw-level
copy densify.py seeds every density dir with; originally written by
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
| imputed methods (incl. onesidedmc) | `data/<density>/imputed/<method>/<dz>/<st>/imputed_model_benchmark_table.csv` | `cor(Z)` of the scored columns | `results/<density>/<method>/<method>_<dz>_<st>_bifactor_<tag>_scores.csv` |
| fill-smooth methods (default/zeros) | sparse `data/<density>/combinations[_<dz>]/<st>/model_benchmark_table.csv` + the imputation-persisted `..._correlation.csv` | the persisted smoothed fill+smooth R | `..._bifactor_<tag>_scores_conditional.csv` + `..._scores_prorated.csv` |

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

## Side branch — synthetic MNAR stress corpus (`scripts/make_mnar.py`)

Python, outside the main cross-product. Generates synthetic tables with
**known ground truth** (true correlation R, true factor count, true ω_h) so
imputers and factorers can be scored against the planted structure instead
of only held-out RMSE. Reads two real `text_only` tables, plants a truth,
and punches calibrated, cross-density-nested holes; each output root is a
drop-in `--data-root`.

| | path |
|---|---|
| read | `data/text_only/<density>/combinations_C/all_standard/model_benchmark_table.csv` |
| read | `data/text_only/<density>/combinations_S/all_standard/model_benchmark_table.csv` |
| write | `data/synth/<density>/truth/<tag>_<cell>_{complete.csv, correlation.csv, factors.npz, meta.json}` |
| write | `data/synth/<density>/<cell>/<mask>/s0/combinations[_<dz>]/<st>/model_benchmark_table.csv` |
| write | `data/synth/<density>/<cell>/<mask>/s0/truth/<tag>_<cell>_{mask.npz, meta.json}` |
| write | `data/synth/<density>/summary.csv` — one row per dataset (diagnostics + truth-cell columns) |

Grid: `{truth cell} × {mask} × {seed} × {table}` = 4 × 4 × 1 × 2 = **32
datasets**.

- **Truth cells** `f5_weak / f5_loud / f10_weak / f10_loud`: F ∈ {5, 10}
  group factors, group loadings ~ U(0.1, 0.5) or U(0.5, 0.9). One general
  factor per table with λ_g solved so the planted mean off-diagonal R
  equals that table's observed mean pairwise r. All loadings positive (R
  has only positive entries, matching the real tables); communalities
  capped at 0.95; the *achieved* mean r, loading ranges and ω_h are always
  recorded in the meta JSON — nothing silently distorted. The grid spans
  weak→detectable structure deliberately, so no single planted truth
  favors one imputer's inductive bias; rankings are per truth cell.
- **Masks**: `mcar` (calibrated control, s = 0), `cell` (s = γ·z_ij, good
  cells observed), `popularity` (s = κ·ρ̂_j·g_i — observability tied to a
  benchmark's expected correlation, interacting with model ability so the
  effect survives offset absorption), `real_mask` (the real mask verbatim;
  row/col MAE are 0 by construction — the sanity control).
- **Mask calibration**: logit P(obs_ij) = a_i + b_j + s_ij, with (a_i, b_j)
  fit by damped iterated bisection to the real table's row/column
  observation counts; margin targets clipped to [0.05, p−0.05]; an
  infeasible mechanism is halved until convergence (`mech_scale` < 1 in
  meta/summary records how much survived).
- **Mask nesting (default).** The real `text_only` slices nest exactly:
  densify's greedy peel only drops whole rows/columns, so the 0.5 grid and
  its observed cells are subsets of 0.3's, which are subsets of 0.1's. The
  synthetic masks reproduce that. Each invocation rebuilds a single chain
  from the densest slice down to the requested one: the densest slice gets
  the calibrated Bernoulli draw, and every sparser slice carries the
  denser slice's observed cells (`carry_mask`) and reveals additional ones
  in increasing `U / expit(a + b + s)` order until its own real row/column
  counts are met (`reveal_to_margins`; targets floored at the carried
  counts). Hence
  `mask(0.5) ⊆ mask(0.3) ⊆ mask(0.1)` by construction, and because the
  chain always restarts at the densest slice with fixed seeds, separate
  `make_mnar.py` invocations are deterministic and byte-identical for a
  given slice. `nested` / `density_chain` are recorded in the per-dataset
  meta JSON and `summary.csv`. `--no-nest` restores independent per-slice
  draws (ablation only; it deliberately breaks the property and the checker
  below fails on it).
- **Common random numbers**: one mask seed; the uniform behind the
  Bernoulli draw is identical across truth cells and masks for a fixed
  table, so all imputers see identical masked input and every
  cross-condition comparison is paired.

**Nesting check** (`scripts/ensure_synth_mnar_nests.py`): walks every
`(mask, tag)` group densest → sparsest and verifies both that the denser
grid (models × benchmarks) is a subset of the sparser one and that every
cell observed in the denser slice is observed in the sparser slice. It also
cross-checks each `_mask.npz` against the CSV's non-null pattern. Exit code
1 on any violation, so it must pass before any hier-over-synth run is
trustworthy:

```bash
uv run python scripts/ensure_synth_mnar_nests.py            # default root data/synth
```

**Hand-off:** each `data/synth/<density>/<cell>/<mask>/s0` directory is a
drop-in `--data-root` for `impute.R` / `factor.R`, but contains only the C/S
`all_standard` cells (no raw/R/aggressive). `impute.R --synth --density <d>`
automates the sweep over all 16 roots of that density in one invocation:
completed CSVs land inside each root as usual, flat results go to
`results/synth/<density>/<cell>/<mask>/`, and DB rows go to
`results/synth/<density>/database.db` table `imputation_synth` (PK
`cell, mask, dataset, method` — the extra columns are what keep the roots'
shared `dataset` names from colliding; `wr2` = hierarchical weighted R², NULL
outside `--hier --synth`). After the imputation pass, every completed matrix
is scored against the planted truth
(`data/synth/<density>/truth/<tag>_<cell>_complete.csv`) on its masked-in
cells — column-balanced, same aggregation shape as the held-out metric — and
the result lands in `imputation_synth.gt_r2`. That pass is refreshed on every
`--synth` invocation even when the imputation itself is skipped (no
`--reimpute`), since it only reads artifacts from disk. Scoring against the
truth is otherwise external: compare outputs to the truth files (truth R,
`true_nfactors = 1 + F`, `omega_hierarchical`). Nothing in `main.R` consumes
the corpus.

**Hier over the synth corpus** (`impute.R --hier --synth`): for each (cell,
mask), the standard hier descent runs over the synth density slices,
`data/synth/<d1>/<cell>/<mask>/s0 -> data/synth/<d2>/...` densest-first,
seeding from the standard `--synth` run's cached completed matrix at the
densest density. This mode is only meaningful on the nested corpus: the
descent assumes each denser slice's real observations reappear in the
sparser one (which `ensure_synth_mnar_nests.py` verifies). Per-step imputed
CSVs land in
`data/synth/<density>/<cell>/<mask>/s0/hierarchical/imputed/...`, flat
results + per-step DB in `results/synth/hierarchical/<cell>/<mask>/<density>/`
(table `imputation_synth`; each row's `gt_r2` scores that step against the
planted truth on the density's missing cells, and `wr2` is the
introduced-column-weighted mean of those gt_r2 values — truth-weighted, unlike
the real corpus's held-out weighting), and the final step is handed
off to the density-0 slice (`data/synth/<cell>/<mask>/0/` +
`results/synth/hierarchical/<cell>/<mask>/0/`).

## The SQLite database

One `database.db` per density slice (`results/<density>/database.db` — e.g.
`results/text_only/0.1/database.db`), written by both stages:

| table | PK | written by | purpose |
|---|---|---|---|
| `imputation` | `(dataset, method)` | `impute/db.R` | held-out RMSE/R² + param description; `wr2` = hierarchical weighted R² (NULL outside `--hier`); gates factoring |
| `factoring` | `(dataset, method, run)` | `factor/db.R` | nf, variance, ω_t/ω_h/ω_hs, φ (JSON-encoded vectors) |
| `loco` | `(dataset, method, run)` | `factor/db.R` | per-benchmark Δω_h from the LOCO sweep |

`dataset` = `"<dz>_<st>"` (e.g. `C_all_standard`).

## Hand-offs / contract summary

```
densify.py            ──CSV──►  impute_<method>            ──M──►  factor_<method>
(<density>/combinations_<dz>)       │  │                          │
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
- **Density slicing:** `--density` is required everywhere and names the
  directory verbatim (`--density 0.1` → `.../0.1/`). Data and results roots gain
  the segment (`<root>/<density>/`), except corpus root files (`results.csv`,
  `models.csv`, `benchmarks.csv`) and the `--synth`/`--smoke` fixtures, which
  stay density-free. Two densities share nothing: each has its own
  `combinations_<dz>`, `imputed/`, flat results, and `database.db`. The
  exception is the viewer: `viewer/compute_positions.py` takes no `--density` —
  it discovers every density slice under its results root and embeds all of
  them in one payload, with views keyed `<density>|<dz>|<tag>` and a density
  dropdown in the UI (defaulting to the densest slice). Density `0` (the
  hierarchical hand-off) shows up here too, displayed as "hier" — keys stay
  keyed on `0`, and its imputation line reports both the plain and the
  weighted R² (`wr2`).
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
- **Synth corpus is all-or-nothing:** `data/synth/` is a generated tree from
  one script version. When the generator's layout changes, wipe and
  regenerate the whole directory (`rm -rf data/synth`) rather than mixing
  layouts (the earlier 9-scenario layout is incompatible). Regenerate every
  density slice (`--density 0.5 0.4 0.3 0.2 0.1`, any order) and then run
  `ensure_synth_mnar_nests.py`: a corpus built with `--no-nest`, or one whose
  slices were generated by different script versions, fails that check.
- **Synth roots share `dataset` names with the real corpus** (`C_all_standard`,
  `S_all_standard` are the `<dz>_<st>` PK and the flat result filenames).
  `impute.R --synth` handles the segregation itself (per-root flat results +
  the `imputation_synth` table); when invoking `impute.R` / `factor.R`
  manually per root, use a dedicated `--results-root` per (cell, mask), or
  the real runs' DB rows and result files will be overwritten.
