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
{densifier: C, R, S} × {strategy: all_standard, all_aggressive} × {imputer: softimpute, iterativepca, onesidedmc}
```

Every leaf produces imputed/surrogate data + factor loadings, written under
`data/imputed/<method>/<densifier>/<strategy>/`.

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

## Stage 1 — Densify (`densify.py`)

Reads `data/combinations/<strategy>/model_benchmark_table.csv`, writes
`data/combinations_<C|R|S>/<strategy>/model_benchmark_table.csv` (+ `summary.csv`
per densifier and `data/densify_summary.csv` rollup).

Three densifiers, all greedy-peel to `TARGET = 0.30` density, then enforce a
hardcoded `MIN_OBS = 2` floor on **both** axes (every kept model ≥2 scores AND
every kept benchmark ≥2 scores — needed so all imputers, esp. OSMC's pairwise
loss, are well-posed):

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
python3 densify.py     # regenerate all 6 densified tables
```

Knobs are constants at the top of `densify.py`: `TARGET`, `MIN_OBS`,
`STRATEGIES`, `DENSIFIERS`.

---

## Stage 2 — Impute (`impute/`)

Each method is **impute-only** — it emits a completed (or surrogate) matrix +
rank-selection diagnostics. **No factoring lives here** (that's Stage 3).

```
impute/
  common.R                 # prep_matrix(), imputed_dir(), write_completed() — shared by the R methods
  softimpute/method.R      # impute_softimpute(), sensitivity_softimpute()  [R, softImpute]
  iterativepca/method.R    # impute_iterativepca(), sensitivity_iterativepca() [R, missMDA]
  OneSidedMC/              # [Julia] right-singular-vector recovery (Cao-Liang-Valiant 2023)
    src/                   #   data.jl loss.jl algorithm.jl baselines.jl metrics.jl (paper core, TESTED)
    src/realdata.jl        #   ragged loss/fit for variable-obs-per-row real data (in module, exported)
    src/pipeline.jl        #   CSV → Θ̂ → covariance-surrogate (script-level, not in module)
    run.jl                 #   driver: loops densifier×strategy, writes surrogates
    test/                  #   DO NOT BREAK — hand-rolled tests for the paper core + ragged code
```

### softimpute / iterativepca (R)
Standard matrix completion. Each sweeps its rank parameter (`rank` / `ncp`,
capped at `MAX_RANK = 10`), picks the CV-best by held-out error, returns the
completed matrix at that setting. Columns are scaled, **rows are not** (rows are
models — row-scaling would erase the general-factor signal). RMSE is reported in
standardized (column-SD) units, with held-out R² alongside.

> ⚠️ **iterativepca is slow and currently not exercised.** `estim_ncpPCA`'s
> cross-validation is very expensive at this matrix size, so it's effectively
> deferred — its sensitivity has **not** been migrated to the held-out RMSE+R²
> mechanism the other methods use (it still reports `estim_ncpPCA`'s CV criterion
> and has no R² distribution). Treat its results as provisional / untested. The
> primary, validated methods are **softimpute** and **onesidedmc**.

### onesidedmc (Julia) — the odd one out
OSMC does **not** impute cells. Its premise (the paper): when observations are
too few to complete the matrix, you can still recover the **right singular
vectors** (benchmark-space factors) from Θ̂ = (1/m)XᵀX. So it estimates
Θ̂ = V̂V̂ᵀ, then **synthesizes a surrogate data matrix whose covariance equals
Θ̂** (n = real model count, on original column scale). That surrogate is handed
to the same R factoring — psych never learns it is synthetic. This is a
covariance-surrogate, **not** an imputation of real cells.

OSMC has no built-in rank selector, so `r` is chosen by a held-out
**pairwise-product RMSE** sweep (r = 1..10) — the analog of softimpute's RMSE
sweep. On this MNAR data the rank is weakly identified (nearly flat RMSE curve);
that's expected, and the seed-sweep sensitivity quantifies it.

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

---

## Stage 0 (orchestrator) — `run/`

```
run/
  main.R        # entry point: impute → factor → dashboard, per cell
  dashboard.R  # the single 6-panel dashboard (orchestrator drives sweep + factoring)
  plots.R      # the 4x2 seed-sweep sensitivity grid
```

`run/main.R` is how you run the whole thing. OSMC is Julia, so the orchestrator
shells out to `impute/OneSidedMC/run.jl` once up front to generate all
surrogates, then factors them in R like the other methods.

**Modularity contract:** imputers never factor. Each `impute_<method>()` returns
a uniform contract — `M`, `best_param`, `params`, `curve`, `param_name`,
`metric_name`, and a `complete_at(v)` closure. The dashboard needs factoring at
every swept parameter value (panels 3 & 6), so the **orchestrator** drives that
sweep by calling `complete_at(v)` and the shared `factor/` functions — factoring
stays in `factor/`, owned by `run/`.

### Usage

```bash
# Everything (all methods × densifiers × strategies) on real data:
Rscript run/main.R

# One method:
Rscript run/main.R --method softimpute        # or iterativepca | onesidedmc

# Fast smoke run on the tiny synthetic fixture (data/smoke):
Rscript run/main.R --method softimpute --smoke

# Include the slow seed-sweep sensitivity analysis:
Rscript run/main.R --method iterativepca --sensitivity
```

Flags:
- `--method <name>` — `softimpute` | `iterativepca` | `onesidedmc`; omit to run all.
- `--raw` — run **only** the undensified `raw` level (slow); without it, only the
  densified levels `C`, `S`, `R` run. So raw is run as a separate invocation.
- `--smoke` — use `data/smoke/` instead of `data/`.
- `--sensitivity` — also run the seed-sweep sensitivity (off by default; slow).

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

**`results/`** (or `results/smoke/`) holds everything else, **flat**, named
`<method>_<densifier>_<strategy>_<suffix>`:
- `..._dashboard.png` — single 6-panel dashboard per cell (predictive curve,
  marginal gain, cumulative variance, scree@best, SSQ loadings@best, PA-nf vs param)
- `..._loadings.csv`, `..._loadings.md` — factor loadings (PAF + promax)
- `<method>_<strategy>_<csr|raw>_sensitivity.png` — **one** grid per
  method×strategy with `--sensitivity`: rows = densifier, cols = [RMSE/CV box |
  best-param stability]. (No aggregation — just co-located rows.) For OSMC the
  seed-sweep runs in Julia (50 seeds, each a fresh held-out split → MNAR
  robustness), writes `results/_osmc_sweep/<dz>_<st>/sensitivity.csv`, and R
  reads it into the same grid.
- `results/_osmc_sweep/<dz>_<st>/` — OSMC's per-r surrogate CSVs + rank curve
  (intermediate plotting inputs; kept out of `data/imputed`).

---

## Smoke fixture (`make_smoke.py`)

Generates a tiny synthetic dataset under `data/smoke/` (real low-rank structure +
MNAR-ish sparsity) so the full pipeline can be exercised in seconds. It runs the
synthetic tables through the **same densifier** as production.

```bash
python3 make_smoke.py                       # writes data/smoke/combinations_<C|S|R>/... (+ raw under data/smoke/combinations/)
Rscript run/main.R --method softimpute --smoke
```

Note: the `C` densifier collapses small on the smoke set (famous-core collapse) —
faithful to its real behavior; the pipeline is robust to the resulting tiny/dense
matrices.

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

- **Run everything from the repo root.** Paths in the R/Julia code are
  repo-root-relative (`source("factor/factoring.R")`, etc.).
- **`tag = basename(dirname(path))`** is the strategy name; densifier outputs are
  laid out so this keying stays intact.
- **Don't break OSMC tests** (`impute/OneSidedMC/test/`). The paper core
  (`data.jl`/`loss.jl`/`algorithm.jl`/`baselines.jl`/`metrics.jl`) is tested;
  `realdata.jl` is additive and exported through `OneSidedMC.jl`.
- **Scaling:** columns scaled, rows never. RMSE is standardized-unit only.
- **PA cache** is correctness-sensitive — keyed by shape, never global.
- **Dead files pending deletion** (superseded by the refactor; nothing sources
  them): `impute/softimpute/{run,impute,factoring,report,sensitivity}.R`,
  `impute/iterativepca/{run,impute,factoring,report,sensitivity,reference}.R`,
  and the old per-method `combinations/` input copies + loose `*.png`/`*.csv`
  artifacts. Keep `method.R`, `common.R`, the `factor/` and `run/` modules, and
  all of `impute/OneSidedMC/` except its empty `combinations/`.
```
