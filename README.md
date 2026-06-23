# LLM Cognitive Interpretability Benchmarks

A curated, verified dataset of LLM benchmark evaluations across diverse cognitive domains — built for research into how language models interpret, reason, and generalize across tasks.

## Running

```bash
# install python, R, julia
make deps
# install language-specific deps
make install
# preprocess dataset
make preproc
# run everything
make runall

# optionally run individual runs
# use --reimpute to do actual imputation even if first time
# use --raw to run on the supersparse data (not actually the raw dataset)
# use --sensitivity to run random-seed sensitivity (slow)
#   Rscript src/run/main.R --method softimpute --reimpute --sensitivity
#   Rscript src/run/main.R --method softimpute --reimpute --sensitivity --raw
```

## Dataset Overview

| Table | Rows | Description |
|-------|------|-------------|
| `benchmarks.csv` | 842 | Benchmark metadata: name, venue, category, source URLs |
| `models.csv` | 4,262 | Model metadata: family, developer, size, type |
| `results.csv` | 27,070 | Evaluation results: scores, metrics, setup parameters |

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

The dataset spans 12 benchmark categories:
- Multilingual, Crosslingual, Cultural
- Alignment & Safety, Cognitive Science
- Coding, Math, Reasoning
- General Knowledge, Music
- Multimodal, Medical, Audio/Speech
- Machine Translation, Morality/Ethics
- Humor/Creativity, History/Culture/Time

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
| `scripts/export_eee_jsonl.py` | Export to EEE JSONL schema |
| `scripts/export_xlsx.py` | Export to Excel workbook |

All four scripts are thin entry points over the shared, reusable toolkit in
`scripts/lib/` (config/trust-tier data, CSV I/O, integrity checks, dedup
resolution, alias/standardization helpers, model categorization, exports) — new
cleanup needs should extend that library rather than adding another
one-off script. Past one-off cleanup scripts are kept for audit-trail
purposes in `scripts/archive/` (see `scripts/archive/README.md`). Each
script also works if run from anywhere, not just the repo root.

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

## Citation

If you use this dataset in your research, please cite:
```
@misc{llm-coginterp-2026,
  title={LLM Cognitive Interpretability Benchmarks Dataset},
  author={Haznitrama, Faiz Ghifari},
  year={2026},
  url={https://github.com/faizghifari/llm-coginterp}
}
```
