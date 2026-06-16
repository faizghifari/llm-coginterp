# TODO

## Data cleanup
- [x] Deduplicate models with multiple model_id/model_name entries (same model, different aliases) — 2026-06-16 pass using `scripts/manage_data.py`: merged 11 genuine duplicate models.csv entries (e.g. `Gemini-2.0-Flash`/`Gemini 2.0 Flash`, `Qwen3 Max`/`qwen3_max`), fixed all 26 orphan `model_name` FK violations in results.csv (curated alias map, cross-checked against actual model context — not blind fuzzy-matching), and registered 1 genuinely new model (`Llama-4-Large`, EngiBench's own naming, doesn't map cleanly to Scout/Maverick so it wasn't guessed). Result: 0 invalid FKs, 0 orphans either direction (`scripts/verify_data.py` clean). See docs/CHANGELOG.md "Data Cleanup" entry for the full list.
- [x] Remove experimental/inproper model entries (e.g., names describing training steps rather than model identity) — removed 21 zero-result stub entries that had zero matching results.csv rows under *any* casing/format (5 were malformed HF-repo-prefixed duplicates with a parsing bug putting the model size into the `developer` field, e.g. `tiiuae/falcon-7b` dev=`7B`; 16 were leftover model registrations from the lost extraction sweep that never got result rows committed).
- [x] Resolved the 393 HF Open LLM Leaderboard duplicate-evaluation groups (24 pure redundancy + 369 conflicting — 256 from the deprecated v1/"old" space, 113 from the current v2 space) by re-verifying each against the live source rather than guessing: v2 via the `open-llm-leaderboard/contents` parquet dataset, v1 via the per-model timestamped JSON result files in `open-llm-leaderboard-old/results` (merged per-benchmark-task across all of a model's resubmissions, since some tasks were only present in earlier/later submissions, not the single latest file). 254/256 v1 and 113/113 v2 conflicts had one of the two stored scores match the live value exactly — kept that one, discarded the other. 2 rows (1 model, 2 benchmarks: `Open-Orca/OpenOrcaxOpenChat-Preview2-13B` gsm8k/winogrande) had neither stored value match — both stored scores were from older resubmissions, superseded by an even later one; overwrote with the verified current value (15.0872 / 77.8216) rather than picking either stale one. results.csv: 8240 → 7829 rows. `scripts/verify_data.py` still clean.
- [x] Resolved 71 of the remaining 126 non-HF conflicting groups. Two structural fixes to the dedup tool itself (`scripts/lib/config.RESULT_IDENTITY_KEY` now also includes `model_id` and `language`, not just `model_name`/`metric_name`) eliminated 90 false-positive groups that were never real duplicates — see docs/CHANGELOG.md "Remaining Dupe Cleanup" for the full breakdown (model-checkpoint disambiguation via `model_id`; sub-task/sub-language disambiguation via `language`). The other 36 were verified against live sources and fixed: mt-rag (24 groups — 3 real metrics mislabeled as one), vectara hallucination-leaderboard (7 — 2 complementary metrics mislabeled as one), crux-eval (5 — CoT vs Direct setup wasn't recorded), p_mmeval (2 of 3 — stale value discarded after matching the paper's stated number).
- [ ] **19 conflicting groups remain** (down from 519). Run `scripts/manage_data.py dupes --verbose` to see them.
  - `opencompass` (3 groups, 21 rows): almost certainly the same "multiple real metrics under one metric_name" pattern as mt-rag/vectara (ERNIE 5.0, Qwen3 235B A22B, Qwen3-Next-80B-A3B each have 7 wildly-varying rows) — but `rank.opencompass.org.cn` is a React SPA backed by an API (`/api/v1/rank/listNewModelsV2`, `/api/v1/rank/listArenaRankings`) that returned the SPA shell instead of data on a plain GET, needs proper request reverse-engineering to pull the real per-category breakdown.
  - ~16 single-paper conflicts (`xstest`, `hagendorff_biases_2023`, `complexbench`, `followbench`, `eifbench`, `benchmax`, `mmar`, `neuro_eval`, `xcr_bench`, and Claude-3.5-Sonnet's `p_mmeval` row) each need their specific paper's table checked by hand — automated WebFetch-based table extraction proved unreliable here (gave two different, mutually inconsistent answers from the same table on separate fetches for arXiv 2411.09116), so these were left flagged rather than risk enshrining a misread number as "verified."
  - `chat.lmsys.org` (11 groups) and `arena-hard-auto`/`llm-stats.com arena-hard-v2` (7 groups) were confirmed as NOT duplicates (genuinely distinct model checkpoints sharing a display name) — no action needed, already correctly handled by the `model_id` key fix.
- [ ] `scripts/manage_data.py categorize-models` flags 46 models as `REMOVE`-candidates (fine-tuned name, no `model_family`/`base_model` set) and 38 as `FLAG` (unclear origin) — **none have zero results**, so none were deleted. These need metadata enrichment (fill in `model_family`/`base_model`), not removal; run `scripts/manage_data.py categorize-models --output data/models_categorized.csv` to get the full list.
- [x] Run scripts/verify_data.py after any data changes to ensure FK integrity — now also the documented step in docs/METHODOLOGY.md's "Adding New Data" checklist.

## Data expansion — large extraction tasks

### Stanford HELM
- [ ] Extract Stanford HELM data per individual benchmark across all HELM sub-projects (Safety, Audio, Image2Struct, Reasoning, Long-Context, MMLU-Winogrande-Afr, MedHELM, ThaiExam, TORR, EWoK, Finance, Arabic Enterprise, SEA-HELM, etc.)
- [ ] Extract per-benchmark scores, NOT mean/aggregate scores averaged across benchmark groups
- [ ] Follow existing methodology (strict source verification, model inclusion criteria, normalization rules)

### Kaggle Benchmarks
- [ ] Extract all benchmarks from Kaggle Research category (104 benchmarks as of 2026-06-16)
- Source: https://www.kaggle.com/benchmarks/?browse=true&type=research
- [ ] Extract per-benchmark model scores individually
- [ ] Follow existing methodology

### Papers With Code
- [ ] Extract benchmarks from Papers With Code tasks page
- Source: https://paperswithcode.co/tasks — iterate through every topic/subtask that has associated benchmarks
- [ ] Extract per-benchmark model scores individually
- [ ] Follow existing methodology

### Previous HELM sweep (lost)
- [x] ~~HELM sweep data (312 benchmarks, 1155 models, 11208 results) extracted locally but not yet committed~~ — Data was lost (never committed to git) when a repo refactor reset the data files to the last commit. Re-extract via tasks above.
- [x] Partially recovered 2026-06-16 from a local `data/*.csv.bak5` snapshot that survived the reset: the PwC scan (10 benchmarks) and HELM FACTS family (5 benchmarks), 138 result rows total — committed this time. See docs/CHANGELOG.md "Data Recovery" entry.
- [ ] Still missing and must be re-extracted from scratch (no surviving backup): the rest of the HELM "other groups" sweep (safety, audio, image2struct, reasoning, air-bench-2024, long-context, mmlu-winogrande-afr, mmlu standalone, medhelm, thaiexam, torr, ewok, finance, arabic-enterprise, seahelm — ~973+ rows) and the entire Kaggle sweep below.
- [ ] **Lesson learned:** commit data/*.csv after each extraction batch instead of relying on local .bak snapshots — uncommitted working-tree state is not safe from external tooling/refactors.

## Methodology updates
- [x] Add to docs/METHODOLOGY.md: If there is more than 1 score for the same model in a benchmark (due to different setup, different provider/evaluator running the benchmark, etc.), keep each as a **separate row** in results.csv rather than averaging. Distinguish rows via the `setup` and `source_url` fields. — Done (see "Multiple Scores per Model-Benchmark Pair" section); the same rule is now also encoded as `config.RESULT_IDENTITY_KEY` in `scripts/lib/config.py`, so the dedup tooling can't accidentally violate it.

## Scripts
- [x] Refactor scripts/ directory — consolidated all duplicate-checking logic (check_dupes.py, check_dupes2.py, analyze_dupes.py, deduplicate_results.py, analyze_source_trust.py) and model-categorization logic (analyze_models.py, categorize_models.py) into one reusable library: `scripts/lib/` (config, io, integrity, dedup, aliases, categorize), exposed via `scripts/manage_data.py`. One-off historical scripts moved to `scripts/archive/` (not part of the active toolkit — see its README). `scripts/` is no longer gitignored.
- [x] Refactor the remaining root-level scripts (`export_eee_jsonl.py`, `export_xlsx.py`) onto the same shared library — added `scripts/lib/export.py`; both scripts are now thin wrappers. Also fixed a latent bug found in the process: the EEE JSONL exporter read a `developer` column that doesn't exist in results.csv (the real column is `model_developer`), so every exported record's `model_info.developer` silently fell back to `"unknown"`.
- [x] Add docstrings and CLI help to all utility scripts — `scripts/manage_data.py --help` / `scripts/manage_data.py <command> --help` covers the new toolkit; `scripts/verify_data.py` and the lib modules have module docstrings.
- [x] Move `manage_data.py`, `verify_data.py`, `export_eee_jsonl.py`, `export_xlsx.py` from the repo root into `scripts/` — the repo root now has no loose Python files, only `data/`, `docs/`, `notes/`, `scripts/`, `README.md`, `LICENSE`. Each script bootstraps `sys.path` to the repo root itself, so they work when run as `python3 scripts/<name>.py` from any working directory.
- [ ] benchmark_analysis.md — refactor analysis output into proper report format

## Documentation
- [ ] Expand README with usage examples and data schema reference
- [ ] Add CONTRIBUTING.md if opening to collaborators
