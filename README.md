# Large-scale factor analysis shows machine intelligence is only partially interpretable

This project borrows an idea from human intelligence research, where people who do well on
one kind of mental test also tend to do well on very different ones, pointing to one broad
general ability layered with narrower specific ones. It asks whether that same structure
shows up in large language models when a large number of them are compared across many
different kinds of tasks — from math and coding to safety behavior, multiple languages, and
medical knowledge — or whether it does not hold up the way it does for people.

The repo is two related but distinct pieces of work:

1. **A curated dataset** (`data/benchmarks.csv`, `data/models.csv`, `data/results.csv`) of
   LLM benchmark evaluations across cognitive domains, assembled and cleaned via `scripts/`
   — documented below.
2. **MachineG2** — an R/Julia pipeline (`src/`) that recovers the latent factor structure of
   LLM capabilities from that dataset, treated as a super-sparse, MNAR (missing-not-at-random)
   model × benchmark score matrix — see [MachineG2 Pipeline](#machineg2-pipeline) below.

`scripts/` (Python) builds and maintains the dataset; `src/` (R + Julia) consumes it. They are
independently runnable.

## Running

Use the docker container to have a replicable environment.

```bash
# Example
sudo docker build -t machineg .
```

The container does not copy data/ or results/, so build them first in host:

``` bash
make preproc

# note: default is "fill-mean" in the paper, and zeros is "fill-zeros"
# ggm and cvxr are omitted: they do not complete on these matrices
sudo docker build -t machineg .
sudo docker run --rm -it -v $PWD/data:/app/data -v $PWD/results:/app/results machineg make impute softimpute softimpute_corr missforest onesidedmc knn usvt default zeros
sudo docker run --rm -it -v $PWD/data:/app/data -v $PWD/results:/app/results machineg make factor softimpute softimpute_corr missforest onesidedmc knn usvt default zeros
sudo docker run --rm -it -v $PWD/data:/app/data -v $PWD/results:/app/results machineg make factor-timed softimpute softimpute_corr missforest onesidedmc knn usvt default zeros
sudo docker run --rm -it -v $PWD/data:/app/data -v $PWD/results:/app/results machineg make loco softimpute softimpute_corr missforest onesidedmc knn usvt default zeros

# the container writes its outputs as root; take ownership back on the host
sudo chown -R $USER:$USER data
sudo chown -R $USER:$USER results

mkdir -p results/pyout

for c in \
  viewer/compute_positions.py \
  scripts/miss_corr.py \
  scripts/correlations.py \
  scripts/plot_missing.py \
  scripts/label_cohesion.py \
  scripts/top_g_ci.py \
  scripts/plot_cohesion_summary.py \
  scripts/impute_summary.py \
  scripts/factor_summary.py; do
  log="results/pyout/$(basename ${c%.py}).log"
  uv run $c 2>&1 | tee "$log"
done

make release-date DATA_ROOT=data/text_only RESULTS_ROOT=results/text_only
make release-date-report RUNS="10%=results/text_only/release_date"
```

### Scripts

```bash
# Dataset maintenance (Python, from repo root or anywhere — scripts anchor to repo root)
python3 scripts/verify_data.py                 # integrity checks; run after every data edit
python3 scripts/manage_data.py --help           # dupes, dedup, find-aliases, apply-aliases,
                                                 #   standardize-ids, categorize-models, recompute-stats

# Derive the analysis view from the archive (after ANY change to data/*.csv
# or to the knowledge bases in scripts/lib/config.py)
python3 scripts/make_text_only_copy.py          # regenerate data/text_only/
python3 scripts/make_text_only_copy.py --check  # assert it is reproducible; changes nothing

# MachineG2 pipeline. Everything below defaults to the TEXT-ONLY analysis view.
python3 scripts/collapse_results.py             # preproc stage 0a
python3 scripts/densify.py                      # preproc stage 1 (or: make preproc runs both)
Rscript src/run/impute.R --method softimpute    # impute + held-out sweep
Rscript src/run/factor.R --method softimpute    # factor the completed matrix
python3 scripts/compare_loadings.py             # cross-method factor congruence
make release-date                               # release-cohort EFA + benchmark release year (after factor)
make runall                                     # canned sequence across all methods

# The multimodal-inclusive corpus is retained; reach it by naming both roots:
make runall DATA_ROOT=data RESULTS_ROOT=results
Rscript src/run/impute.R --method softimpute --data-root data --results-root results

# methods: softimpute | softimpute_corr | knn | missforest | mice | onesidedmc
#          | optspace | usvt | cvxr | ggm | default | zeros (fill-smooth)
# optionally, per run:
# --reimpute      force fresh imputation (default reuses an existing imputed CSV)
# --raw           run the slow undensified level instead of the C/S/R densifiers
# --smoke         fast synthetic-fixture smoke run
# --loco          (factor.R) leave-one-benchmark-out delta omega_h
```

## Dataset Overview

The dataset exists twice: **`data/*.csv` is the archive**, recording what sources
published, and **`data/text_only/` is the analysis view**, generated from it and the
corpus the MachineG2 pipeline actually runs on.

| Table | Archive (`data/`) | Analysis view (`data/text_only/`) | Description |
|-------|------:|------:|-------------|
| `benchmarks.csv` | 624 | 456 | Benchmark metadata: name, venue, category, source URLs |
| `models.csv` | 2,014 | 1,618 | Model metadata: family, developer, size, type |
| `results.csv` | 19,030 | 13,251 | Evaluation results: scores, metrics, setup parameters |

The view drops image- and audio-based benchmarks, score-redundant duplicate columns,
translations of an in-corpus original, and all but one metric per benchmark. Every one of
those decisions is encoded in `scripts/lib/config.py` and applied by
`scripts/make_text_only_copy.py` — the view is regenerated, never hand-edited, and
`--check` asserts it reproduces byte-for-byte.

Even after cleanup, the table of models × benchmarks is extremely sparse — under 2% of all
possible (model, benchmark) pairs have a recorded score, since well-known models get tested
repeatedly while lesser-known benchmarks barely get touched. This is *why* the MachineG2
pipeline exists: it's the reason recovering a factor structure needs densifying + imputing
before it can be run at all.

## Data Schema

> **The one thing to know before writing any query:** `results.csv`'s
> foreign key to `models.csv` is the **`model_name`** column, not
> `model_id` (results.csv also has its own `model_id`, which is a
> denormalized convenience field — usually the model's HuggingFace repo
> slug or a similar source-specific identifier — and is *not* what joins
> to `models.csv`). This is enforced by `scripts/verify_data.py`.

### benchmarks.csv (41 columns)
Primary key `benchmark_id` (lowercase). Core fields you'll actually use:
`benchmark_id`, `benchmark_name`, `year`, `venue`, `category`,
`subcategory`, `source_url`, `organization`, `task_types`, `metrics`.
The rest (`paper_url`, `github_url`, `hf_url`, `other_url`, `title`,
`acronym`, `domain`, ...) are legacy/overlapping fields accumulated
across different extraction batches — mostly redundant with the core
fields above, kept for provenance rather than as a clean schema.

### models.csv (25 columns)
Primary key `model_id`. Core fields: `model_id`, `model_name`,
`model_family`, `developer`, `model_size`, `model_type` (`open`/`closed`),
`provider`, `parameters_billion`. `benchmark_count`, `total_results`, and
`avg_score` are denormalized aggregates computed from results.csv —
**recompute them with `scripts/manage_data.py recompute-stats --write`
after editing results.csv**, they don't update automatically. (`avg_score`
is a plain mean across every row for that model regardless of metric
scale — most scores are 0-100, but a few, like Chatbot Arena's Elo
ratings, are on a ~1000-1500 scale, so for models evaluated on mixed
scales this average isn't a single meaningful number.)

### results.csv (38 columns)
One row per (model, benchmark, evaluation-setup) data point. Core
fields: `benchmark_id`, `model_name` (the real FK, see above), `score`,
`metric_name`, `setup`, `language` (sub-task/sub-language label when a
benchmark reports more than one metric per model), `reasoning_enabled`,
`generation_temperature`, `source_url`. A model can legitimately have
many rows for the same benchmark — different `setup`/`source_url`/
`language` values mean different real evaluations, not duplicates.

## Usage Examples

```python
import pandas as pd

benchmarks = pd.read_csv("data/benchmarks.csv")
models = pd.read_csv("data/models.csv")
results = pd.read_csv("data/results.csv")

# Join results to model + benchmark metadata. Note the FK: results.model_name -> models.model_id.
joined = results.merge(models, left_on="model_name", right_on="model_id", suffixes=("", "_model")) \
                 .merge(benchmarks, on="benchmark_id", suffixes=("", "_benchmark"))

# All scores for one model across every benchmark it's been evaluated on.
gpt4 = results[results["model_name"] == "GPT-4"][["benchmark_id", "score", "metric_name", "setup"]]

# Compare two models head-to-head on benchmarks they both have results for.
a, b = "GPT-4", "Claude 3 Opus"
pivot = results[results["model_name"].isin([a, b])].pivot_table(
    index="benchmark_id", columns="model_name", values="score", aggfunc="mean"
).dropna()

# Every benchmark in one category, with how many models cover it.
safety = benchmarks[benchmarks["category"].str.contains("Safety", case=False, na=False)]
coverage = results[results["benchmark_id"].isin(safety["benchmark_id"])] \
    .groupby("benchmark_id")["model_name"].nunique().sort_values(ascending=False)
```

For data maintenance (checking integrity, finding/fixing duplicate
evaluations, deduping model aliases) use the CLI in the next section
instead of writing one-off scripts against the CSVs directly.

## Categories Covered

Benchmarks span a broad set of cognitive domains, including multilingual/crosslingual/cultural
understanding, alignment & safety, general reasoning, coding, math, general knowledge, medical
knowledge, machine translation, multimodal (vision/audio) tasks, and more — `benchmarks.csv`'s
`category`/`subcategory` columns hold the fine-grained (and sometimes messy, multi-batch)
labels. The multimodal (vision/audio) benchmarks are retained in the archive but set aside in
the analysis view, since such a benchmark measures a model's perceptual front-end at least as
much as its language ability; the pipeline can still be pointed at the full corpus to check
what that exclusion changed (see below).

## Verification

Run `scripts/verify_data.py` to check data integrity:
```bash
python3 scripts/verify_data.py
```

Checks include:
- Foreign key validity (all results reference valid benchmarks + models)
- No duplicate primary keys
- No orphaned benchmarks/models with zero results
- Score values are valid floats

## Utility Scripts

| Script | Purpose |
|--------|---------|
| `scripts/verify_data.py` | Data integrity checks (FK, orphans, exhaustion) — run after every change |
| `scripts/manage_data.py` | Dataset maintenance CLI — duplicate detection/resolution, alias fixes, model categorization. Run `python3 scripts/manage_data.py --help` for the full command list |
| `scripts/standardise.py` | Normalize/standardise models, results, and benchmarks from a JSON rules file (remove/rename/remap/setup-extract/merge-benchmark) plus dedup cleanups. Dry-run by default; `--write` to apply |
| `scripts/export_eee_jsonl.py` | Export to EEE JSONL schema |
| `scripts/export_xlsx.py` | Export to Excel workbook |

All of these scripts are thin entry points over the shared, reusable toolkit in
`scripts/lib/` (config/trust-tier data, CSV I/O, integrity checks, dedup
resolution, alias/standardization helpers, model categorization, exports). Each
script also works if run from anywhere, not just the repo root.

## MachineG2 Pipeline

```
data collection → aggregation → DENSIFY → IMPUTE → FACTOR
                                (this pipeline's three stages)
```

The pipeline runs the cross-product `{densifier: raw, C, S, R} × {strategy: all_standard,
all_aggressive} × {imputer: softimpute, softimpute_corr, knn, missforest, mice, onesidedmc, optspace,
usvt, cvxr, ggm}`, plus the no-imputation `default` and `zeros` variants that factor a smoothed
pairwise-complete correlation directly (`iterativepca` is implemented but deferred/untested):

- **Densify** (`scripts/densify.py`) — three greedy-peel strategies that each produce a
  different bias profile, not one "best" table: **C** drops the sparsest benchmarks (favors
  keeping famous benchmarks and most models), **R** drops the sparsest models (favors keeping
  famous models and most benchmarks), **S** balances both.
- **Impute** (`src/impute/`) — each method fills in the missing cells of the sparse matrix and
  reports a held-out cell-level RMSE/R² sweep. `onesidedmc` is the odd one out — Julia,
  recovering benchmark-space singular vectors and synthesizing a covariance-equivalent
  surrogate rather than imputing individual cells.
- **Factor** (`src/factor/`) — method-agnostic: minimum-residual factoring + promax rotation,
  factor count from Horn's parallel analysis (capped at 20), and a bifactor/Schmid-Leiman
  (general-ability-vs-group-factor) decomposition run twice per cell — once at the
  parallel-analysis count, once forced to two factors.
- **Compare** (`scripts/compare_loadings.py`) — cross-method factor congruence, i.e. whether
  different imputation methods agree on the factor structure they recover.
- **Release date** (`scripts/release_date.py`, `make release-date`) — whether the structure moves
  with model release date, and whether a benchmark's release year relates to its place in it.
  Written up in the paper's release-date appendix; see [Release-date analysis](#release-date-analysis).

Outputs, under whichever root is active (`data/text_only` + `results/text_only` by default):
`<data-root>/imputed/<method>/<densifier>/<strategy>/` holds the imputed CSV;
`<results-root>/<method>/` holds the bifactor loadings and scalars; numeric results also land in
a SQLite store at `<results-root>/database.db`, which `scripts/impute_summary.py`,
`factor_summary.py` and `correlations.py` read.

### Release-date analysis

Run after `make factor` on the same roots. One run covers one target density:

```bash
make release-date DATA_ROOT=<data-root> RESULTS_ROOT=<results-root>
# = uv run python scripts/release_date.py run --data-root ... --results-root ...
#   [--dz C,S] [--strategy all_standard] [--reps 50] [--cores N] [--skip-efa]
```

- Models are binned into release cohorts (<=2022, 2023, 2024, >=2025) from
  `combinations/<strategy>/collapse_mapping.csv`, the same dates `factor.R --timed` uses.
- Each cohort of each completed matrix is refactored with the pooled pipeline
  (`src/run/release_cohorts.R`), next to 50 random subsets of the same size as the null.
  Only row-preserving imputers enter (`softimpute`, `missforest`, `knn`, `mice`,
  `iterativepca`), and only where they pass `factor.R`'s `R2_GATE`.
- Also written: coverage per cohort, mean g score per cohort, an imputation-free check on the
  benchmarks every cohort took, and benchmark release year against the pooled loadings
  (every gated solution except `default`/`zeros`).
- Output lands in `<results-root>/release_date/`, and `summary.md` there has every table.
  The cohort EFA is the slow step (one to two minutes per density on 36 cores). `--skip-efa`
  reuses its `cohort_fits.csv`.
- Every step is seeded per cell, so a rerun on the same inputs gives the same numbers.

The paper reports two densities side by side. After a `release-date` run on each:

```bash
make release-date-report RUNS="10%=<results-10>/release_date 20%=<results-20>/release_date"
```

This writes `release_date_tables.md` (the appendix tables, plus the counts quoted in its text)
and `release-cohort-omega.png` (the appendix figure) to `$(RESULTS_ROOT)/release_date_report/`.

The analysis imputes once over all models and then splits by cohort. It is valid when release year
shifts the level of performance without changing how benchmarks correlate. The paper's appendix
states this assumption and its caveat.

## License

MIT — see [LICENSE](LICENSE).

## Authors / Contact

Author information has been removed for anonymous review.

## Citation

If you use this dataset or pipeline in your research, please cite:
```
@misc{anonymous2026largescale,
  title={Large-scale factor analysis shows machine intelligence is only partially interpretable},
  author={Anonymous Authors},
  year={2026}
}
```
