# Understanding the Structure of Language Model Abilities

*Working title of the research; the repo itself is organized around two related pieces of work — see below.*

This project borrows an idea from human intelligence research, where people who do well on
one kind of mental test also tend to do well on very different ones, pointing to one broad
general ability layered with narrower specific ones. It asks whether that same structure
shows up in large language models when a large number of them are compared across many
different kinds of tasks — from math and coding to safety behavior, multiple languages, and
medical knowledge — or whether it does not hold up the way it does for people.

For the current findings, written for a general audience, see
**[docs/RESEARCH_OVERVIEW.md](docs/RESEARCH_OVERVIEW.md)**.

The repo is two related but distinct pieces of work:

1. **A curated dataset** (`data/benchmarks.csv`, `data/models.csv`, `data/results.csv`) of
   LLM benchmark evaluations across cognitive domains, assembled and cleaned via `scripts/`
   — documented below.
2. **MachineG2** — an R/Julia pipeline (`src/`) that recovers the latent factor structure of
   LLM capabilities from that dataset, treated as a super-sparse, MNAR (missing-not-at-random)
   model × benchmark score matrix — see [MachineG2 Pipeline](#machineg2-pipeline) below and
   [src/README.md](src/README.md) for the full statistical detail.

`scripts/` (Python) builds and maintains the dataset; `src/` (R + Julia) consumes it. They are
independently runnable.

## Running

```bash
# One-time environment setup (Python via uv, R via renv, Julia via Project.toml)
make deps    # apt install r-base, install julia + uv
make env     # env-py (uv sync) + env-r (Rscript install.R) + env-jl (Pkg.instantiate)

# Dataset maintenance (Python, from repo root or anywhere — scripts anchor to repo root)
python3 scripts/verify_data.py                 # integrity checks; run after every data edit
python3 scripts/manage_data.py --help           # dupes, dedup, find-aliases, apply-aliases,
                                                 #   standardize-ids, categorize-models, recompute-stats

# MachineG2 pipeline (R, from repo root or anywhere)
python3 scripts/collapse_results.py             # preproc stage 0a
python3 scripts/densify.py                      # preproc stage 1 (or: make preproc runs both)
Rscript src/run/main.R                          # run everything (all methods x C/S/R x both strategies)
Rscript src/run/main.R --method softimpute      # one method: softimpute|knn|missforest|mice|onesidedmc
python3 scripts/compare_loadings.py             # cross-method factor congruence -> results/loadings_congruence.md
make runall                                     # canned sequence of the above across all methods

# optionally, per run:
# --reimpute      force fresh imputation (default reuses an existing imputed CSV)
# --raw           run the slow undensified level instead of the C/S/R densifiers
# --smoke         fast synthetic-fixture smoke run
# --sensitivity   add the slow seed-sweep sensitivity grid (very slow)
```

See `src/README.md` for the full command reference, output paths, and pipeline architecture.

## Dataset Overview

| Table | Rows | Description |
|-------|------|-------------|
| `benchmarks.csv` | 637 | Benchmark metadata: name, venue, category, source URLs |
| `models.csv` | 2,028 | Model metadata: family, developer, size, type |
| `results.csv` | 19,078 | Evaluation results: scores, metrics, setup parameters |

Even after cleanup, the table of models × benchmarks is extremely sparse — under 2% of all
possible (model, benchmark) pairs have a recorded score, since well-known models get tested
repeatedly while lesser-known benchmarks barely get touched. This is *why* the MachineG2
pipeline exists: it's the reason recovering a factor structure needs densifying + imputing
before it can be run at all. See [docs/RESEARCH_OVERVIEW.md](docs/RESEARCH_OVERVIEW.md) §1 for
the current density numbers per trimming strategy.

## Data Schema

> **The one thing to know before writing any query:** `results.csv`'s
> foreign key to `models.csv` is the **`model_name`** column, not
> `model_id` (results.csv also has its own `model_id`, which is a
> denormalized convenience field — usually the model's HuggingFace repo
> slug or a similar source-specific identifier — and is *not* what joins
> to `models.csv`). This is enforced by `scripts/verify_data.py`; see
> [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for the full normalization rules.

### benchmarks.csv (37 columns)
Primary key `benchmark_id` (lowercase). Core fields you'll actually use:
`benchmark_id`, `benchmark_name`, `year`, `venue`, `category`,
`subcategory`, `source_url`, `organization`, `task_types`, `metrics`.
The rest (`paper_url`, `github_url`, `hf_url`, `other_url`, `title`,
`acronym`, `domain`, ...) are legacy/overlapping fields accumulated
across different extraction batches — mostly redundant with the core
fields above, kept for provenance rather than as a clean schema.

### models.csv (24 columns)
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
benchmark reports more than one metric per model — see "Multiple Scores
per Model-Benchmark Pair" in METHODOLOGY.md), `reasoning_enabled`,
`generation_temperature`, `source_url`. A model can legitimately have
many rows for the same benchmark — different `setup`/`source_url`/
`language` values mean different real evaluations, not duplicates.

Full schema documentation in [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

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
labels. A `data/text_only/` copy of the dataset, with image- and audio-based benchmarks set
aside, is also maintained for the MachineG2 pipeline (see below).

See `notes/` for per-category research notes.

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
resolution, alias/standardization helpers, model categorization, exports) — new
cleanup needs should extend that library rather than adding another
one-off script. Past one-off cleanup scripts are kept for audit-trail
purposes in `scripts/archive/` (see `scripts/archive/README.md`). Each
script also works if run from anywhere, not just the repo root.

## MachineG2 Pipeline

```
data collection → aggregation → DENSIFY → IMPUTE → FACTOR
                                (this pipeline's three stages)
```

The pipeline runs the cross-product `{densifier: raw, C, S, R} × {strategy: all_standard,
all_aggressive} × {imputer: softimpute, knn, missforest, mice, onesidedmc}` (`iterativepca` is
implemented but deferred/untested):

- **Densify** (`scripts/densify.py`) — three greedy-peel strategies that each produce a
  different bias profile, not one "best" table: **C** drops the sparsest benchmarks (favors
  keeping famous benchmarks and most models), **R** drops the sparsest models (favors keeping
  famous models and most benchmarks), **S** balances both.
- **Impute** (`src/impute/`) — each method fills in the missing cells of the sparse matrix and
  reports a held-out cell-level RMSE/R² sweep. `onesidedmc` is the odd one out — Julia,
  recovering benchmark-space singular vectors and synthesizing a covariance-equivalent
  surrogate rather than imputing individual cells.
- **Factor** (`src/factor/`) — method-agnostic: principal-axis factoring + promax rotation,
  factor count from Horn's parallel analysis, plus higher-order (second-order) and
  bifactor/Schmid-Leiman (general-ability-vs-group-factor) decompositions.
- **Compare** (`scripts/compare_loadings.py`) — cross-method factor congruence, i.e. whether
  different imputation methods agree on the factor structure they recover.

Outputs: `data/imputed/<method>/<densifier>/<strategy>/` holds the imputed CSV;
`results/<method>/` holds everything else (dashboards, loadings, sensitivity grids); a parallel
`results/text_only/` tree holds the same pipeline run against the image/audio-stripped dataset
copy. `src/README.md` is the authoritative reference for pipeline internals, output formats,
and how to add a new imputation method — this section only summarizes it.

## Current Findings

See [docs/RESEARCH_OVERVIEW.md](docs/RESEARCH_OVERVIEW.md) for the up-to-date, plain-language
writeup of what the pipeline has found so far — how well each method predicts held-out scores,
how many distinct ability groupings the data supports (and why the methods disagree on that
number), and the open questions the project is currently working through.

## Methodology

See [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for:
- Strict source verification principles
- Model inclusion/exclusion criteria
- Data normalization rules
- Inference environment collection methodology
- Generation parameter extraction approach
- The required checklist for adding new data

## Changelog

See [docs/CHANGELOG.md](docs/CHANGELOG.md) for a history of all data additions and changes.

## Notes

Research notes per category are in `notes/`. The backlog is tracked in `notes/TODO.md`.

## License

MIT — see [LICENSE](LICENSE).

## Authors / Contact

If you are interested in the work, anything from a simple feedback to any forms of
collaboration, feel free to reach us:
- Faiz Ghifari Haznitrama ([haznitrama@kaist.ac.kr](mailto:haznitrama@kaist.ac.kr))
- Afrizal Hasbi Azizy ([letter.afrizal@gmail.com](mailto:letter.afrizal@gmail.com))
- Faeyza Rishad Ardi ([faeyza.rishad@gmail.com](mailto:faeyza.rishad@gmail.com))

## Citation

If you use this dataset or pipeline in your research, please cite:
```
@misc{llm-coginterp-2026,
  title={Understanding the Structure of Language Model Abilities},
  author={Haznitrama, Faiz Ghifari and Azizy, Afrizal Hasbi and Ardi, Faeyza Rishad},
  year={2026},
  url={https://github.com/faizghifari/llm-coginterp}
}
```
