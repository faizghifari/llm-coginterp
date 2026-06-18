# MachineG2 — LLM benchmark factor-structure pipeline

Recovering the latent factor structure of LLM capabilities from a **super-sparse,
MNAR** model × benchmark score matrix. Because the data is too sparse and too
biased to fix with one method, the pipeline runs a **cross-product of palliative
approaches** and compares them — the agreement (or disagreement) across cells is
the actual result.

```
data collection → aggregation → DENSIFY → IMPUTE → FACTOR
                                (this repo's three stages)
```

The cross-product is:

```
{densifier: raw, C, S, R} × {strategy: all_standard, all_aggressive}
  × {imputer: softimpute, knn, missforest, mice, onesidedmc, (iterativepca: deferred)}
```

Layout: the pipeline **code** lives under `src/` (`impute/`, `factor/`, `run/`),
the Python **scripts** under `scripts/` (`densify.py`, `make_smoke.py`,
`compare_loadings.py`), and all **data + results at the repo root**
(`data/`, `results/`). Paths are anchored to the repo root via each script's own
file location, so everything runs from any working directory. Imputed matrices
go to `data/imputed/<method>/<densifier>/<strategy>/`; all other outputs to
`results/<method>/`.

---

## Why each stage exists

The raw aggregated tables are ~3–5 % filled and **MNAR**: famous models get
scored on famous benchmarks, obscure benchmarks barely co-occur with anything.
No single fix works, so:

- **Densify** — drop rows/cols to reach a workable density (~30 %), producing
  several *bias profiles* (not one "best" table). Density is the only target
  (imputer-agnostic); we deliberately do **not** optimize pairwise overlap or
  positive-definiteness here, so the densifier stays neutral across imputers.
- **Impute** — three methods with different assumptions complete (or, for OSMC,
  side-step) the matrix.
- **Factor** — identical PAF + parallel analysis on every completed matrix, so
  the only thing that varies downstream is the imputed input.

---

## Stage 1 — Densify (`scripts/densify.py`)

Reads `data/combinations/<strategy>/model_benchmark_table.csv`, writes
`data/combinations_<C|R|S>/<strategy>/model_benchmark_table.csv` (+ `summary.csv`
per densifier and `data/densify_summary.csv` rollup).

Three densifiers, all greedy-peel to `TARGET` density (currently `0.10`), then
enforce a hardcoded `MIN_OBS = 2` floor on **both** axes (every kept model ≥2
scores AND every kept benchmark ≥2 scores — needed so all imputers, esp. OSMC's
pairwise loss, are well-posed):

| densifier | drop rule | bias profile |
|-----------|-----------|--------------|
| **C** | drop sparsest **benchmark**, then emptied models | famous benchmarks × wide model coverage |
| **R** | drop sparsest **model**, then emptied benchmarks | saturated models × wide benchmarks (incl. obscure) |
| **S** | drop whichever marginal has the lowest **fill-rate** | balanced, neither axis privileged |

Only `all_standard` and `all_aggressive` are processed (whitelist strategies are
ignored). C and S tend to converge on the dominant famous block; R keeps obscure
benchmarks but collapses to a handful of models (low data retention). See
`data/densify_summary.csv` for exact shapes.

```bash
python3 scripts/densify.py            # regenerate all 6 densified tables
python3 scripts/densify.py --peek     # preview shapes/density, write nothing
```

Knobs are constants at the top of `scripts/densify.py`: `TARGET`, `MIN_OBS`,
`STRATEGIES`, `DENSIFIERS`.

---

## Stage 2 — Impute (`impute/`)

Each method is **impute-only** — it emits a completed (or surrogate) matrix +
rank-selection diagnostics. **No factoring lives here** (that's Stage 3).

```
impute/
  common.R                 # prep_matrix(), make_holdout(), score_holdout(), imputed_dir() — shared
  softimpute/method.R      # impute_softimpute()  [R, softImpute] — low-rank, sweeps rank
  knn/method.R             # impute_knn()         [R, VIM]        — k-NN, sweeps k
  missforest/method.R      # impute_missforest()  [R, missForest] — random forest, sweeps ntree
  mice/method.R            # impute_mice()         [R, mice]       — chained equations, sweeps m
  iterativepca/method.R    # impute_iterativepca() [R, missMDA]   — EM-PCA, sweeps ncp (DEFERRED)
  OneSidedMC/              # [Julia] right-singular-vector recovery (Cao-Liang-Valiant 2023)
    src/                   #   data.jl loss.jl algorithm.jl baselines.jl metrics.jl (paper core, TESTED)
    src/realdata.jl        #   ragged loss/fit for variable-obs-per-row real data (in module, exported)
    src/pipeline.jl        #   CSV → Θ̂ → covariance-surrogate; cell metric (script-level, not in module)
    run.jl                 #   driver: loops densifier×strategy, writes surrogates
    test/                  #   DO NOT BREAK — hand-rolled tests for the paper core + ragged code
```

### The cell-filling R methods (softimpute, knn, missforest, mice, iterativepca)
Each sweeps one hyperparameter (`rank` / `k` / `ntree` / `m` / `ncp`), picks the
CV-best by held-out error, returns the completed matrix at that setting. Columns
are scaled, **rows are not** (rows are models — row-scaling would erase the
general-factor signal). Held-out RMSE + R² reported (see "Metric" below):
- **softimpute** — low-rank iterative soft-thresholded SVD (nuclear-norm; *not*
  EM). Primary validated method.
- **knn** — fill from the *k* most similar models. Assumption-light baseline.
- **missforest** — iterative random forest; captures nonlinearity the low-rank
  methods can't.
- **mice** — multiple imputation by chained equations; hands factoring the
  **mean** of *m* completions (ridge-regularized + `remove.collinear=FALSE`
  because this wide sparse matrix is highly collinear).
- **iterativepca** — EM regularized PCA. **Deferred/untested**: `estim_ncpPCA`
  CV is prohibitively slow here and its sensitivity isn't migrated to the
  held-out RMSE/R² mechanism. Treat as provisional.

#### Metric (shared, `score_holdout` in `common.R`)
A **column-stratified** holdout masks ~20 % of each column's observed cells
(`make_holdout`, keeping ≥2 in training). Held-out cells are scored in z-units
against the **train-cell column mean** baseline. **Column-balanced by default**
(`--no-balance` to disable): RMSE = mean of per-column RMSEs; R² = single pooled
ratio of column-balanced MSE / baseline-MSE (NOT a mean of per-column R², which
blows up on thin columns). Cell-weighting would let famous high-frequency columns
dominate and mask densification. Model selection uses R².

### onesidedmc (Julia) — the odd one out
OSMC does **not** impute cells. Its premise (the paper): when observations are
too few to complete the matrix, you can still recover the **right singular
vectors** (benchmark-space factors) from Θ̂ = (1/m)XᵀX. So it estimates
Θ̂ = V̂V̂ᵀ, then **synthesizes a surrogate data matrix whose covariance equals
Θ̂** (n = real model count, on original column scale). That surrogate is handed
to the same R factoring — psych never learns it is synthetic. This is a
covariance-surrogate, **not** an imputation of real cells.

OSMC has no built-in rank selector, so `r` is chosen by a held-out sweep
(r = 1..10). Its RMSE/R² is **cell-level** (for cross-method comparability):
each held-out cell is predicted from the recovered covariance via the
conditional-Gaussian best-linear predictor `ẑ_j = Vj' · pinv(Vs) · z_S` (solved
in the r-dim factor space, not by inverting the rank-deficient |S|×|S| covariance
— that blows up on richly-observed rows). The native pairwise-product metric is
kept as a dead branch (`OSMC_CELL_METRIC = false`). On this MNAR data the rank is
weakly identified (nearly flat curve); the seed-sweep sensitivity quantifies it.

> Real data has a **variable** number of observed benchmarks per model, so OSMC
> uses a *ragged* observation format (`RaggedObs = Vector{Tuple{cols, vals}}`),
> not the paper's fixed-k rectangular `idx`/`vals`. The method consumes only
> observed cells — never a full matrix + mask.

---

## Stage 3 — Factor (`factor/`)

Method-agnostic. Takes any completed matrix, returns factor loadings.

```
factor/
  factoring.R           # factor_matrix(), loadings CSV/MD writers, plot_scree()
  parallel_analysis.R   # cached Horn parallel analysis (PA)
  pa_cache/             # JSON cache of random-baseline eigenvalue cutoffs, keyed by shape
```

Factoring = principal-axis factoring (`fm="pa"`) + promax (when >1 factor). The
factor **count** comes from Horn's parallel analysis, split into:

1. **random baseline cutoffs** — depend only on shape `(n, p, n.iter, quantile)`,
   so computed once per shape and **cached as JSON** in `factor/pa_cache/`
   (PC-flavor: 95th percentile of random correlation-matrix eigenvalues, 100
   iters). Two datasets of the same shape reuse the same cutoffs.
2. **observed eigenvalues** — `eigen(cor(M))`, computed per dataset.

`nfactors = #(observed eigenvalue > random cutoff at same position)`.

> ⚠️ PA is **not** globally dataset-independent — the cutoffs depend on shape.
> The cache is keyed by shape for exactly this reason; do not assume one global
> cutoff vector.

**Higher-order** (`higher_order()` in `factoring.R`): since promax is oblique the
factors correlate, so each cell also gets (a) a **second-order FA** of the
factor-correlation matrix Φ → single g, and (b) a **bifactor / Schmid-Leiman**
solution via `psych::omega` → per-benchmark g + group loadings, plus **ω_h**,
**ω_total**, and the per-group **ω_hs** vector. This is EFA Schmid-Leiman, not a
constrained CFA bifactor (cross-loadings stay). No nf gate — runs at any nf.
Outputs per cell: `*_secondorder_loadings.{csv,md}`,
`*_bifactor_loadings.{csv,md}`, `*_bifactor_scalars.csv`,
`*_bifactor_omega_group.csv`.

---

## Stage 0 (orchestrator) — `run/`

```
run/
  main.R       # entry point: impute → factor → higher-order → dashboard, per cell
  dashboard.R  # the single 9-panel dashboard (orchestrator drives sweep + factoring)
  plots.R      # the seed-sweep sensitivity grid (rows=densifier, cols=RMSE/R²/best-param/ω_h)
```

`run/main.R` is how you run the whole thing. OSMC is Julia, so the orchestrator
shells out to `impute/OneSidedMC/run.jl` once up front to generate all
surrogates, then factors them in R like the other methods.

The 9 dashboard panels: 1 predictive curve + R² vs param, 2 marginal gain,
3 cumulative variance, 4 scree, 5 SS loadings, 6 PA-factor-count, 7 second-order
loadings, 8 bifactor g loadings, 9 omega coefficients (ω_t / ω_h / per-group
ω_hs).

**Modularity contract:** imputers never factor. Each `impute_<method>()` returns
a uniform contract — `M`, `best_param`, `params`, `curve`, `param_name`,
`metric_name`, and a `complete_at(v)` closure. The dashboard needs factoring at
every swept parameter value (panels 3 & 6), so the **orchestrator** drives that
sweep by calling `complete_at(v)` and the shared `factor/` functions — factoring
stays in `factor/`, owned by `run/`.

### Usage

```bash
# Everything (all methods × densifiers × strategies) on real data:
Rscript src/run/main.R

# One method:
Rscript src/run/main.R --method softimpute    # softimpute|knn|missforest|mice|onesidedmc

# Fast smoke run on the tiny synthetic fixture:
Rscript src/run/main.R --method softimpute --smoke

# Slow seed-sweep sensitivity:
Rscript src/run/main.R --method softimpute --sensitivity
```

Flags:
- `--method <name>` — `softimpute` | `knn` | `missforest` | `mice` |
  `onesidedmc` | `iterativepca`; omit to run all.
- `--raw` — run **only** the undensified `raw` level (slow); default runs C/S/R.
- `--smoke` — use `data/smoke/` instead of `data/`.
- `--reimpute` — force fresh imputation; **default reuses** an existing imputed
  CSV (re-factor + re-plot only), so higher-order/metric changes can be applied
  without re-running the slow imputation.
- `--sensitivity` — also run the seed-sweep sensitivity (off by default; slow).
- `--no-balance` — revert to the cell-weighted held-out metric (default is
  column-balanced).

Densifier levels are `raw`, `C`, `S`, `R`; strategies are `all_standard` and
`all_aggressive` (always both). `raw` is the undensified aggregated table, run
through the full pipeline so the effect of densifying is visible against
no-densifying — but because it's slow it's gated behind `--raw` and run on its
own. The sensitivity grid is named `..._csr_sensitivity.png` for the C/S/R run
and `..._raw_sensitivity.png` for the `--raw` run, so they don't clobber.

```bash
Rscript run/main.R --method softimpute               # C, S, R
Rscript run/main.R --method softimpute --raw         # raw only (slow)
```

### Outputs — two separate trees

**`data/imputed/<method>/<densifier>/<strategy>/`** (or `data/smoke/imputed/...`)
holds **only** the imputed data:
- `imputed_model_benchmark_table.csv` — completed (or, for OSMC, surrogate) matrix

**`results/<method>/`** (or `results/smoke/<method>/`) holds everything else,
named `<method>_<densifier>_<strategy>_<suffix>`:
- `..._dashboard.png` — single 9-panel dashboard per cell.
- `..._loadings.{csv,md}` — first-order factor loadings (PAF + promax).
- `..._secondorder_loadings.{csv,md}`, `..._bifactor_loadings.{csv,md}`,
  `..._bifactor_scalars.csv`, `..._bifactor_omega_group.csv` — higher-order.
- `..._rank_sweep.csv` — the param sweep (for `--reimpute`-off rebuilds).
- `<method>_<strategy>_<csr|raw>_sensitivity.png` — **one** grid per
  method×strategy with `--sensitivity`: rows = densifier, cols = [RMSE box | R²
  box | best-param stability | ω_h distribution]. Each seed = a fresh held-out
  split; the spread measures **holdout-split variance**, not MNAR robustness per
  se (every seed shares the same missingness). For OSMC the seed-sweep runs in
  Julia and writes `results/_osmc_sweep/<dz>_<st>/sensitivity.csv`, which R reads
  into the same grid.
- `..._dashboard_combined.png` / `..._sensitivity_combined.png` — raw+C/S/R
  stacked aggregates (auto-built at end of run via `magick`).
- `results/_osmc_sweep/<dz>_<st>/` — OSMC's per-r surrogate CSVs + curves
  (intermediate plotting inputs; kept out of `data/imputed`).

---

## Smoke fixture (`scripts/make_smoke.py`)

Generates a tiny synthetic dataset under `data/smoke/` (real low-rank structure +
MNAR-ish sparsity) so the full pipeline can be exercised in seconds. It runs the
synthetic tables through the **same densifier** as production (incl. the `raw`
level under `data/smoke/combinations/`).

```bash
python3 scripts/make_smoke.py
Rscript src/run/main.R --method softimpute --smoke
```

Note: the `C` densifier collapses small on the smoke set (famous-core collapse) —
faithful to its real behavior; the pipeline is robust to the resulting tiny/dense
matrices.

## Cross-method agreement (`scripts/compare_loadings.py`)

Brute-searches `results/` for loadings CSVs and computes pairwise factor
congruence (|cosine|, SSQ-sorted, sign-invariant) between methods, grouped by
dataset × loadings-kind (first-order / second-order / bifactor) × shape — only
same-kind, same-shape solutions are compared. Writes
`results/loadings_congruence.md`. Safe to run before all methods finish (only
compares what exists).

---

## Adding a new imputation method

A method plugs into the orchestrator by satisfying ONE contract: given a sparse
input matrix, return a completed matrix + a rank-selection sweep, scored with the
**held-out cell-level RMSE + R²** that every other method uses. The orchestrator
owns factoring, dashboards, sensitivity, and I/O — your method must not factor.

### 1. The imputer contract

Write `impute/<yourmethod>/method.R` (R) or a Julia driver that emits the same
CSVs (see OSMC for the Julia pattern). The core function returns a list:

| field | type | meaning |
|---|---|---|
| `M` | matrix | completed matrix at the best sweep value (rows = models, cols = benchmarks) |
| `best_param` | scalar | the chosen rank/ncp/r |
| `params` | vector | the swept parameter grid |
| `curve` | vector | **held-out RMSE** per param (the selection metric) |
| `curve_r2` | vector | **held-out R²** per param (same residuals, reported alongside) |
| `param_name` | string | axis label: `"rank"`, `"ncp"`, `"r"`, … |
| `metric_name` | string | always `"Held-out RMSE"` (keep methods comparable) |
| `complete_at(v)` | function | returns the completed matrix at param value `v` — the orchestrator calls this to factor at every swept value for dashboard panels 3 & 6. Make it cheap (reuse cached fits); return `NULL` if unavailable. |

### 2. The required metric — held-out cell RMSE + R²

This is the **non-negotiable common currency**. Hold out ~20% of *observed*
cells, fit on the rest, predict the held-out cells, then:

```
RMSE = sqrt(mean((ẑ - z)^2))                 over held-out cells, standardized
R²   = 1 - SS_resid / SS_baseline            SS_baseline = TRAIN-mean baseline
```

Key rules (so your numbers are comparable):
- **Standardize columns** (z-scores), never rows. RMSE is in column-SD units;
  RMSE = 1 / R² = 0 is the no-skill baseline (predicting the train column mean).
- **Baseline uses the TRAIN mean**, not the holdout mean — the baseline must see
  only what the model sees.
- If your method's native error isn't cell-level (e.g. OSMC scores pairwise
  products), you must still produce a cell-level RMSE/R² for the comparison —
  derive cell predictions from your method's output (OSMC does this via the
  conditional-Gaussian predictor `ẑ_j = Θ̂[j,S]·pinv(Θ̂[S,S])·z_S`). Keep your
  native metric as a secondary/dead branch if you like, but the headline must be
  cell-level.

### 3. Sensitivity (optional, `--sensitivity`)

Return a list with `rmse_mat` (seeds × params), `r2_mat` (seeds × params),
`best_ranks` (per seed), `ranks`, `param`. Each seed = a fresh held-out split
(MNAR ⇒ different splits give different answers; that spread is the point).
Parallelize it (R: `doParallel`/`foreach`; Julia: `Threads.@threads`) — it
refits at every param × every seed and is the slowest part. The grid plotter
draws RMSE-box | R²-box | best-param-bar; the R²-box is blank if `r2_mat` is
absent.

### 4. Wire it into `run/main.R`

- add the name to `ALL_METHODS`;
- dispatch it in `impute_R()` / `sensitivity_R()` (or, for a non-R method, the
  subprocess pattern like `run_osmc_subprocess()` + a `*_contract()` reader);
- the orchestrator handles output paths (`data/imputed/<method>/…` for the CSV,
  flat `results/` for plots), factoring, dashboard, and the combined aggregates.

### 5. Don't

- don't factor inside the method (factoring lives in `factor/`, called by `run/`);
- don't row-scale; don't invent a new metric name; don't write plots/loadings
  (the orchestrator does). Just impute and report the contract.

---

## Conventions & gotchas for future agents

- **Runs from any CWD.** Scripts anchor to the repo root via their own file
  location (Python `__file__`, R `commandArgs(--file=)`, Julia `@__DIR__`) — they
  do **not** assume the working directory. Data/results are at the repo root;
  code is under `src/`; Python scripts under `scripts/`.
- **Install** all three environments with `make install` (uv / renv / Julia).
- **Don't break OSMC tests** (`impute/OneSidedMC/test/`). The paper core
  (`data.jl`/`loss.jl`/`algorithm.jl`/`baselines.jl`/`metrics.jl`) is tested;
  `realdata.jl` is additive and exported through `OneSidedMC.jl`.
- **Scaling:** columns scaled, rows never. Held-out metric is column-balanced by
  default; model selection uses R².
- **PA cache** is correctness-sensitive — keyed by shape, never global.
- **No invented gates.** Higher-order has no nf floor; the only data drops are
  the `MIN_OBS = 2` floor (densifier) + zero-variance/<2-obs column drops
  (`prep_matrix` / OSMC `drop_degenerate_cols`). `safe_nf` only caps nf at the
  matrix's numeric rank to stop `fa()` erroring.
- **iterativepca** is deferred/untested (slow). Validated methods: softimpute,
  knn, missforest, mice, onesidedmc.
```
