# LLM Cognitive Interpretability Benchmarks

A curated, verified dataset of LLM benchmark evaluations across diverse cognitive domains — built for research into how language models interpret, reason, and generalize across tasks.

## Dataset Overview

| Table | Rows | Description |
|-------|------|-------------|
| `benchmarks.csv` | 200 | Benchmark metadata: name, venue, category, source URLs |
| `models.csv` | 1,125 | Model metadata: family, developer, size, type |
| `results.csv` | 8,126 | Evaluation results: scores, metrics, setup parameters |

## Data Schema

### benchmarks.csv
Core fields: `benchmark_id`, `benchmark_name`, `year`, `venue`, `category`, `subcategory`, `source_url`, `organization`, `task_types`, `metrics`

### models.csv
Core fields: `model_id`, `model_name`, `model_family`, `developer`, `model_size`, `model_type`, `provider`, `parameters_billion`

### results.csv
Core fields: `benchmark_id`, `model_id`, `score`, `metric`, `setup`, `reasoning_enabled`, `generation_temperature`, `source_url`

Full schema documentation in [METHODOLOGY.md](METHODOLOGY.md).

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

Run `verify_data.py` to check data integrity:
```bash
python3 verify_data.py
```

Checks include:
- Foreign key validity (all results reference valid benchmarks + models)
- No duplicate primary keys
- No orphaned benchmarks/models with zero results
- Score values are valid floats

## Utility Scripts

| Script | Purpose |
|--------|---------|
| `verify_data.py` | Data integrity checks (FK, duplicates, orphans) |
| `merge_models.py` | Merge duplicate model entries |
| `standardize_model_ids.py` | Normalize model naming conventions |
| `export_eee_jsonl.py` | Export to EEE JSONL schema |
| `export_xlsx.py` | Export to Excel workbook |

Additional cleanup scripts (alias detection, dupe analysis, FK repair) are in development and will be added after refactoring.

## Methodology

See [METHODOLOGY.md](METHODOLOGY.md) for:
- Strict source verification principles
- Model inclusion/exclusion criteria
- Data normalization rules
- Inference environment collection methodology
- Generation parameter extraction approach

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a history of all data additions and changes.

## Notes

Research notes per category are in `notes/`. The backlog is tracked in `notes/TODO.md`.

## License

[Add your license here]

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
