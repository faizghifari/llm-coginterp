# TODO

## Data cleanup
- [x] Deduplicate models with multiple model_id/model_name entries (same model, different aliases) — 2026-06-16 pass using `scripts/manage_data.py`: merged 11 genuine duplicate models.csv entries (e.g. `Gemini-2.0-Flash`/`Gemini 2.0 Flash`, `Qwen3 Max`/`qwen3_max`), fixed all 26 orphan `model_name` FK violations in results.csv (curated alias map, cross-checked against actual model context — not blind fuzzy-matching), and registered 1 genuinely new model (`Llama-4-Large`, EngiBench's own naming, doesn't map cleanly to Scout/Maverick so it wasn't guessed). Result: 0 invalid FKs, 0 orphans either direction (`scripts/verify_data.py` clean). See docs/CHANGELOG.md "Data Cleanup" entry for the full list.
- [x] Remove experimental/inproper model entries (e.g., names describing training steps rather than model identity) — removed 21 zero-result stub entries that had zero matching results.csv rows under *any* casing/format (5 were malformed HF-repo-prefixed duplicates with a parsing bug putting the model size into the `developer` field, e.g. `tiiuae/falcon-7b` dev=`7B`; 16 were leftover model registrations from the lost extraction sweep that never got result rows committed).
- [x] Resolved the 393 HF Open LLM Leaderboard duplicate-evaluation groups (24 pure redundancy + 369 conflicting — 256 from the deprecated v1/"old" space, 113 from the current v2 space) by re-verifying each against the live source rather than guessing: v2 via the `open-llm-leaderboard/contents` parquet dataset, v1 via the per-model timestamped JSON result files in `open-llm-leaderboard-old/results` (merged per-benchmark-task across all of a model's resubmissions, since some tasks were only present in earlier/later submissions, not the single latest file). 254/256 v1 and 113/113 v2 conflicts had one of the two stored scores match the live value exactly — kept that one, discarded the other. 2 rows (1 model, 2 benchmarks: `Open-Orca/OpenOrcaxOpenChat-Preview2-13B` gsm8k/winogrande) had neither stored value match — both stored scores were from older resubmissions, superseded by an even later one; overwrote with the verified current value (15.0872 / 77.8216) rather than picking either stale one. results.csv: 8240 → 7829 rows. `scripts/verify_data.py` still clean.
- [x] Resolved 71 of the remaining 126 non-HF conflicting groups. Two structural fixes to the dedup tool itself (`scripts/lib/config.RESULT_IDENTITY_KEY` now also includes `model_id` and `language`, not just `model_name`/`metric_name`) eliminated 90 false-positive groups that were never real duplicates — see docs/CHANGELOG.md "Remaining Dupe Cleanup" for the full breakdown (model-checkpoint disambiguation via `model_id`; sub-task/sub-language disambiguation via `language`). The other 36 were verified against live sources and fixed: mt-rag (24 groups — 3 real metrics mislabeled as one), vectara hallucination-leaderboard (7 — 2 complementary metrics mislabeled as one), crux-eval (5 — CoT vs Direct setup wasn't recorded), p_mmeval (2 of 3 — stale value discarded after matching the paper's stated number).
- [x] Worked through the 19 remaining conflicts individually (see docs/CHANGELOG.md "Final Dupe Cleanup Pass"): 3 small-gap groups averaged (genuine same-metric noise); xstest/hagendorff_biases_2023/xcr_bench/neuro_eval/complexbench/followbench all verified against their live paper sources and fixed (real distinct sub-tests/conditions mislabeled identically, except complexbench's 51.2 which traced to an unrelated baseline figure and was discarded as a mis-extraction). `chat.lmsys.org` (11 groups) and `arena-hard-auto`/`llm-stats.com arena-hard-v2` (7 groups) were confirmed as NOT duplicates (genuinely distinct model checkpoints sharing a display name) — no action needed, already correctly handled by the `model_id` key fix.
- [x] **Cross-source benchmark deduplication (2026-06-23)** — the PwC/Kaggle/HELM imports created duplicate benchmarks under source-prefixed (`pwc_*`) or underscore-variant IDs that the string-keyed dedup tooling never caught. `scripts/merge_duplicate_benchmarks.py` normalized benchmark IDs+names, found 14 genuine collision pairs, and cascade-merged each into a canonical key (dropped `pwc_` prefix into native IDs; collapsed HELM underscore-variants like `med_qa`/`truthful_qa` into the native concatenated form). Metadata (source_url, description, etc.) ported into survivors; 0 result rows lost (multi-source rows differ by source_url, kept separate per methodology). A fuzzy substring sweep confirmed remaining near-matches are legitimately distinct (language splits, dataset versions, different benchmarks sharing a stem). Net: 856 → 842 benchmarks.
- [x] **All 6 remaining conflicting groups resolved (2026-06-23)** (down from 519 originally) via `scripts/resolve_conflict_benchmarks.py`. Re-verification against each cited source showed the conflicts were symptoms of import mis-extraction, not genuine score disputes; resolution policy "correct + prune to source":
  - `mmar` (1 group): MMAR is an **audio-reasoning** benchmark (arXiv 2505.13032), not multilingual/MT. "Gemini 1.5 Pro" does not appear in the paper at all → its row was dropped. The other 3 stored scores didn't match paper Table 2 either and were corrected to the Avg column (GPT-4o Audio 54.3→63.5, Qwen2-Audio 52.1→30.4, SALMONN-13B 38.2→33.2). Benchmark recategorized multilingual→audio/speech, subcategory/task_type→audio-reasoning.
  - `opencompass` (3 groups): the CompassAcademic leaderboard's 7 capability dimensions were imported all mislabeled `accuracy`; pass 5 had kept only the first (the overall/average — confirmed ≈ mean of the other 6 per model) and dropped the rest. Relabeled metric_name accuracy→overall and recorded in `notes` that the 6 per-dimension scores are unrecoverable (live API still returns only the SPA shell). No row change.
  - `eifbench` (2 groups): EIFBench (arXiv 2506.08375, EMNLP 2025) reports ILA/CLA sub-scores (0–1) per scenario; our single `accuracy` values can't be verified against any paper table, so the 15 result rows were left untouched. Benchmark recategorized alignment/safety→instruction-following, paper_url + year added.
- [x] **Release-date enrichment, models AND benchmarks (done 2026-09-01).** Both tables are
  now effectively fully dated: **models 2007/2014 (100%), of which 1765 (88%) reach month
  precision**; **benchmarks 623/624 (100%), of which 457 (73%) reach month precision**. The
  remainder carry year precision, which the corroboration ladder emits on purpose when two
  passes agree on the year but not the month. Provenance is per row in `release_date_source`
  (`hf_createdat` 691, `sourced_evidence` 378, `corroborated_year` 223, `single_haiku` 218,
  `web_verified` 199, `corroborated_month` 135, `single_hermes` 86, `web_cited` 45,
  `existing` 27, `verified_arxiv` 5). See docs/CHANGELOG.md for method and measured error.

  Both defects listed when this item was opened are fixed: the `128.0` context-window leak
  and the bare `1` in `benchmarks.release_date` are gone, and granularity is settled — month
  is the floor, existing `YYYY-MM-DD` values keep their precision (19 benchmark rows).

  **Do not read the column as uniformly reliable.** The tiers differ a lot, and the known
  bias is documented in docs/CHANGELOG.md: LLM-guessed dates run systematically EARLY for
  models released after the guessing model's training cutoff. The 2024+ region was repaired
  and is largely evidence-backed; the pre-2024 region is the weaker half.

- [ ] **Residual release-date work.** Much smaller after the pre-2024 repair
  (2026-09-01, see docs/CHANGELOG.md "Pre-2024 repair completed"):
  - 7 models and 1 benchmark are still undated.
  - **47 models** at year precision, down from 242. Benchmarks still have 165 —
    that side has had no equivalent pass and is now the weaker table.
  - The eval/method-variant question is **settled**: a variant takes its base
    model's date, tagged `base_model_date`, but only when the base is
    better-evidenced than the variant. Direct evidence outranks inheritance —
    `LLaVA-1.5-13B (+CSR)` had inherited 2023-10 from LLaVA-1.5 while its own
    paper is 2024-05, and the paper wins.
  - 4 rows are knowingly unresolved and should stay that way unless a real source
    turns up: `Claude 1.3`, `Palmyra X5`, `RuGPT3Large`, `Neo-6B`.

- [ ] ~~`scripts/manage_data.py categorize-models` flags 46 models as `REMOVE`-candidates~~ **(superseded by the holistic pass above; counts also stale — the current split is 0 REMOVE / 492 FLAG)** (fine-tuned name, no `model_family`/`base_model` set) and 38 as `FLAG` (unclear origin) — **none have zero results**, so none were deleted. These need metadata enrichment (fill in `model_family`/`base_model`), not removal; run `scripts/manage_data.py categorize-models --output data/models_categorized.csv` to get the full list.
- [x] Fixed `models.csv`'s stale aggregate columns (`benchmark_count`, `total_results`, `avg_score`) — they hadn't been recomputed in a long time (e.g. GPT-4's row said 55/75/42.91, actual was 76/122/88.29). Added `scripts/lib/stats.py` + `scripts/manage_data.py recompute-stats` and ran it for all 1096 models. Re-run this after any batch of changes to results.csv (it's not automatic). Caveat documented in the module docstring: `avg_score` is a plain mean across every row regardless of metric scale (most are 0-100, but Elo ratings like Chatbot Arena's are ~1000-1500), so for models evaluated on mixed scales it isn't a meaningful "typical score" — that's how the column was already defined, just now correctly computed.
- [x] Verified the "~4,985 HF Open LLM Leaderboard results pending Official-Providers-only filter" item from old session memory — checked all 448 unique v2-sourced model_ids in our results.csv against the live `open-llm-leaderboard/contents` dataset's `Official Providers` column: **100% are already official-provider models.** v1/"old" leaderboard rows (1865 of them) predate the Official-Providers concept entirely, so it doesn't apply there. This was a non-issue, already correctly filtered at extraction time — no fix needed.
- [x] Run scripts/verify_data.py after any data changes to ensure FK integrity — now also the documented step in docs/METHODOLOGY.md's "Adding New Data" checklist.
- [x] **Non-LLM removal + multilingual benchmark de-duplication (2026-07-16)** — see docs/CHANGELOG.md "Non-LLM Removal & Multilingual Benchmark De-duplication" and notes/multilingual_duplication_audit.md for full detail/sourcing. Added a second, orthogonal `classify_scope` axis to `scripts/lib/categorize.py` (modality/inclusion-scope vs. the existing fine-tune-provenance axis) surfaced via `scripts/manage_data.py audit-model-scope`; removed 10 non-LLM models (CLIP/PMC-CLIP/ST5-XXL/monoT5-3B/Knowledge Review/NLLB/SeamlessM4T/m2ugen/mullama/musilingo) + the 1 benchmark (`pwc_vsr`) it orphaned. Added `remove_benchmark` and a `merge_benchmark` language-backfill mode to `scripts/lib/standardise.py`, plus a `find-language-clusters` discovery command (`scripts/lib/benchmark_clusters.py`); researched each candidate cluster against its source paper and removed 28 literal-translation benchmark_ids (MGSM/Global-MMLU-Lite non-English Kaggle variants, MBZUAI's translated Arabic MMLU) while consolidating 8 more (XCOPA/XNLI/XQuAD HELM per-language sub-pages, no data lost) into their parent id with `language` backfilled. MultiLoKo, FLORES direction pairs, LINDSEA, Thai sub-exams, ArabicMMLU, and PwC WMT/CoNLL-2002 pairs were investigated and confirmed as genuinely distinct content — left untouched. `verify_data.py` clean throughout. benchmarks.csv: 679 → 642; models.csv: 2053 → 2043; results.csv: 20234 → 19226.
- [x] **Residual cleanup after validation sweep (2026-07-16)** — see docs/CHANGELOG.md "Residual Cleanup" entry. Removed 5 non-suffixed translated benchmarks (`mmmlu`, `global_mmlu`, `kaggle_nanliao7_global_mmlu_lite_{ca,cs}`, `indicxnli` — 126 rows; originals `mmlu`/`xnli` kept) and 12 narrow-task/encoder-only models found by sweeping for models confined to MT/NER leaderboards (BioMegatron, LUKE（Large）, Pooled Flair, Straková et al. 2019, LS-unLLaMA, HeadMask, PartialFormer, Variational Attention, Caglayan, BigBird, BigBird-etc, YiSi-1 — 22 rows), plus the 10 pure-NER/MT PwC benchmarks and 3 mmmlu-only models they orphaned. Fixed `distilgpt2` metadata (`model_name`/`developer` had the HF repo path/namespace) — `audit-model-scope` is now 100% KEEP. Totals: 642 → 627 benchmarks, 2043 → 2028 models, 19226 → 19078 results.
- [x] **`GPT-2-Medium 355M` is dated wrong, and so is its variant (done 2026-09-01).**
  Confirmed against `openai/gpt-2`'s own commits — "updates for 345M model"
  2019-05-03, "push 774M model" 2019-08-20, the 1.5B model card 2019-11-05 — and
  the other staged sizes did have the same problem: `GPT-2 Medium (355M)` and
  `GPT-2 Large (774M)` both sat at 2019-11, `GPT-2 (0.7B)` and `GPT-2 XL 1.5B
  (pre-trained)` both at 2019-02. Six rows corrected via `set_model_field`. The
  same sweep caught `SGPT-2.7B-msmarco` at 2019-02 — it is SGPT (2022-02), not a
  GPT-2 model, mis-dated on a name collision.

- [ ] **`mexa` has three candidate identities.** The row is recorded as
  `multilingual` / `mexican-languages` (year 2024), but its `paper_url` was a 2025
  *multimodal reasoning* MEXA (arXiv 2506.17113) and its `github_url` is
  `cisnlp/MEXA`, a *cross-lingual alignment* benchmark. Three different benchmarks
  share the name. The bad `hf_url` was cleared; the paper/github links were left
  as-is rather than guessed. Needs one sourcing pass against the 7 result rows to
  decide which MEXA was actually evaluated.

- [ ] **Docs findings from the 2026-09-01 consistency pass** (see
  docs/CHANGELOG.md "Archive metadata corrections"):
  - `src/README.md` is stale in the same way `docs/METHODOLOGY.md` was: it still
    documents `main.R`, `dashboard.R`, the 9-panel dashboard and second-order FA,
    all of which are gone. METHODOLOGY has been corrected; src/README has not, and
    CLAUDE.md points at it as authoritative for anything statistical.
  - **`src/run/main.R`'s status is unresolved.** CLAUDE.md says it is dead because
    it calls `prepare_raw_cor()`, which exists nowhere — but `prepare_raw_cor`
    appears nowhere in `main.R` either, so that reason is wrong. main.R parses,
    every file it sources exists, and the functions it calls resolve (transitively,
    via `factoring.R` → `parallel_analysis.R`). It is simply not wired into the
    Makefile. Settle it with an actual `--smoke` run before either reviving or
    deleting it.
  - **CLAUDE.md states the factoring R² gate as 0.4; the code uses 0.3** in all
    four places in `src/run/factor.R`. METHODOLOGY now says 0.3.
  - `notes/cleanup_passes/` is empty (git does not track empty directories, so it
    exists only locally) — remove it or put something in it.

- [ ] **Case-by-case calls left open from the residual sweep**: `mgsm` (llm-stats "avg langs" aggregate over translated GSM8K; `gsm8k` has 460 rows), `belebele` ("avg 122 langs", parallel translated content, no per-language rows), `humaneval_xl` ("avg 23 NLs" rows), `BigBird-Pegasus` (summarization-only seq2seq — fails the "arbitrary instructions" test T5 passes?), and the 10 PwC `truthfulqa` rows with negative log-prob-style scores mixed into a 0-100 column.

## Data expansion — large extraction tasks

> **CLOSED 2026-08-25.** Benchmark collection is finished; the corpus is frozen
> for the analysis. Items below are kept as a record of what was done and what
> was left, not as planned work. `[~]` marks deliberately-not-planned items.

### Stanford HELM
- [x] **Staging extraction complete (2026-06-18).** `scripts/extract_helm_staging.py` fetches all 13 HELM sub-projects from their public GCS APIs and writes properly schema-aligned staging CSVs. Results: **188 benchmarks, 302 models, 6,158 result rows** across Classic, Lite, Safety, MedHELM, ThaiExam, TORR, EWoK, Finance, SEA-HELM, Arabic, Audio, Image2Struct, Reasoning. Validated: correct schema, 0-100 score scale (BPB kept absolute), no special-char model names, 0 null scores.
- [x] **Merge staging data into main files (2026-06-18).** `scripts/merge_helm_staging.py --write` applied cleanly: +182 benchmarks (6 skipped as already in main), +237 models (58-entry alias map collapsed HELM-style names to existing model_ids; 8 exact collisions skipped), +6158 results (1266 rows remapped via alias). `verify_data.py` clean (0 FK violations), aggregate stats recomputed for all 1333 models. Totals: **397 benchmarks / 1333 models / 13981 results**.
- [~] **NOT PLANNED — benchmark collection closed 2026-08-25.** Recorded because
  the blocker that made this a "revisit later" item is gone, so if collection ever
  reopens the groundwork is here. Verified 2026-08-25: all three previously-inaccessible sub-projects are now live at the standard `crfm-helm-public/{project}/benchmark_output/runs/` path, with data present:
  - `long-context` — v1.0.0
  - `mmlu-winogrande-afr` — v1.0.0, v1.1.0
  - `air-bench` — 20 versions, latest v1.9.0 (the TODO's "air-bench-2024" is this project)

  Extraction is now a normal data-collection job rather than a blocker. Note `scripts/extract_helm_staging.py` was archived after the 2026-06-18 sweep and would need restoring from `scripts/archive/` (or git history) first.
- [~] **NOT PLANNED (collection closed).** Six further HELM projects exist in the bucket that the corpus does not cover (found by the same 2026-08-25 sweep): `capabilities`, `cleva`, `efficient_helm`, `instruct`, `robo-reward-bench`, `arabic-enterprise`. All are text-relevant and would be in scope. (`heim` and `vhelm` are image/vision suites — out of scope for the text-only analysis.)

### Kaggle Benchmarks
- [x] **Extraction complete (2026-06-19).** Research filter = `type IN (INDIVIDUAL, SUITE)` → exactly 104 benchmarks. `scripts/extract_kaggle_staging.py` used the gRPC-gateway 3-step flow (ListBenchmarks → GetBenchmark → GetBenchmarkLeaderboard). Staging: 104 benchmarks, 79 models, 4,082 results. Key fix: task version ID ≠ benchmark version ID; versionIdSelector must use the inner benchmark version ID from GetBenchmark.
- [x] **Merge complete (2026-06-19).** `scripts/merge_kaggle_staging.py --write`: +94 benchmarks (10 aliased to existing IDs), +30 models (49 aliased to existing IDs), +4,082 results. 0 FK violations.

### Papers With Code
- [x] **Extraction complete (2026-06-19).** `scripts/extract_pwc_staging.py` read pwc-archive/evaluation-tables HF parquet (4 shards). Staging: 386 benchmarks, 4,036 models, 10,654 results. Removed 4 junk models (Model name, Anonymous, Baseline Model, tes).
- [x] **Merge complete (2026-06-19).** `scripts/merge_pwc_staging.py --write`: +367 benchmarks (19 aliased to existing IDs), +3,931 models (105 aliased to existing IDs), +10,654 results. 0 FK violations.
- [x] **Setup-in-name / compound model cleanup** — seven standardisation passes total (`fix_setup_in_names.py`, `standardise_models.py`, `standardise_pass3.py`–`pass7.py`). Pass 5: 86 compound/pipeline models removed. Pass 6: 430 task-specific/discriminative models removed (scope rule: general-purpose generative LMs only), 54 inference-time techniques extracted to `results.setup`. Pass 7: 9 remaining stragglers removed (GRAFT-Net, GCN-Align, AlignEA, ChatDev, InstructABSA, InstructDS, etc. — all discriminative/task-specific models that lacked the `+` pattern). Fine-tuned LLM identity names intentionally kept. Net across all passes: 4,873 → 4,265 models; 28,242 → 27,074 results.

### Previous HELM sweep (lost)
- [x] ~~HELM sweep data (312 benchmarks, 1155 models, 11208 results) extracted locally but not yet committed~~ — Data was lost (never committed to git) when a repo refactor reset the data files to the last commit. Re-extract via tasks above.
- [x] Partially recovered 2026-06-16 from a local `data/*.csv.bak5` snapshot that survived the reset: the PwC scan (10 benchmarks) and HELM FACTS family (5 benchmarks), 138 result rows total — committed this time. See docs/CHANGELOG.md "Data Recovery" entry.
- [x] All previously-lost HELM sub-projects re-extracted (2026-06-18): safety, audio, image2struct, reasoning, medhelm, thaiexam, torr, ewok, finance, arabic, seahelm, mmlu standalone — all covered by the new `extract_helm_staging.py` sweep. Only air-bench-2024, long-context, mmlu-winogrande-afr remain inaccessible (tracked above).
- [x] **Lesson learned applied:** all extraction output committed immediately via staging CSVs + merge script.

## Methodology updates
- [x] Add to docs/METHODOLOGY.md: If there is more than 1 score for the same model in a benchmark (due to different setup, different provider/evaluator running the benchmark, etc.), keep each as a **separate row** in results.csv rather than averaging. Distinguish rows via the `setup` and `source_url` fields. — Done (see "Multiple Scores per Model-Benchmark Pair" section); the same rule is now also encoded as `config.RESULT_IDENTITY_KEY` in `scripts/lib/config.py`, so the dedup tooling can't accidentally violate it.

## Scripts
- [x] Refactor scripts/ directory — consolidated all duplicate-checking logic (check_dupes.py, check_dupes2.py, analyze_dupes.py, deduplicate_results.py, analyze_source_trust.py) and model-categorization logic (analyze_models.py, categorize_models.py) into one reusable library: `scripts/lib/` (config, io, integrity, dedup, aliases, categorize), exposed via `scripts/manage_data.py`. One-off historical scripts moved to `scripts/archive/` (not part of the active toolkit — see its README). `scripts/` is no longer gitignored.
- [x] Refactor the remaining root-level scripts (`export_eee_jsonl.py`, `export_xlsx.py`) onto the same shared library — added `scripts/lib/export.py`; both scripts are now thin wrappers. Also fixed a latent bug found in the process: the EEE JSONL exporter read a `developer` column that doesn't exist in results.csv (the real column is `model_developer`), so every exported record's `model_info.developer` silently fell back to `"unknown"`.
- [x] Add docstrings and CLI help to all utility scripts — `scripts/manage_data.py --help` / `scripts/manage_data.py <command> --help` covers the new toolkit; `scripts/verify_data.py` and the lib modules have module docstrings.
- [x] Move `manage_data.py`, `verify_data.py`, `export_eee_jsonl.py`, `export_xlsx.py` from the repo root into `scripts/` — the repo root now has no loose Python files, only `data/`, `docs/`, `notes/`, `scripts/`, `README.md`, `LICENSE`. Each script bootstraps `sys.path` to the repo root itself, so they work when run as `python3 scripts/<name>.py` from any working directory.
- [x] benchmark_analysis.md — refactored into one coherent status table (was an append-only per-session log with duplicate/contradictory entries, e.g. `akata_games_2023` listed twice, `GaslightingBench` both "saturated" and "✅ improved to 7" with a strikethrough hack). Moved to `docs/benchmark_analysis.md` (no longer gitignored) and regenerated against current row counts. Found in the process: `global_mmlu` and `GaslightingBench` were both previously reported as expanded (26 and 7 rows) but are currently back down to 4 and 3 — almost certainly the same lost-work pattern as the HELM/Kaggle sweep in docs/CHANGELOG.md "Data Recovery", flagged in the new report rather than silently re-trusting the old claim.

## Documentation
- [ ] Expand README with usage examples and data schema reference
