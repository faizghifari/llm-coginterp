# Changelog

All notable changes to the LLM Benchmarks dataset.

**Current totals:** 397 benchmarks, 1333 models, 13981 result entries.

---

## HELM Merge into Main Files (2026-06-18)

- **Applied `scripts/merge_helm_staging.py --write`** to merge all three
  staging CSVs into the main data files in one atomic pass.
- **Benchmarks**: +182 new entries; 6 existing IDs skipped (hellaswag, mmlu,
  legalbench, xstest, nusax, fleurs) — HELM result rows pointing to those
  benchmarks still added (they use distinct HELM source_urls).
- **Models**: 58-entry verified alias map collapsed HELM naming conventions
  (e.g. `"BLOOM (176B)"` → `"BLOOM-176B"`, `"GPT-4o (2024-05-13)"` → `"GPT-4o"`)
  remapping 1266 result rows; 8 exact model_id collisions skipped; +237 new
  model entries added.
- **Results**: +6158 rows; 0 already-in-main duplicates (all staging rows were
  genuinely new). Dedup key: `[benchmark_id, model_name, metric_name, source_url]`.
- **Post-merge checks**: `verify_data.py` clean — 0 FK violations, 0 orphan
  benchmarks or models. `recompute-stats --write` updated aggregate columns
  for all 1333 models.
- **Totals after merge:** 397 benchmarks, 1333 models, 13981 result entries.

---

## HELM Staging Extraction — 13 sub-projects (2026-06-18)

- **Built `scripts/extract_helm_staging.py`**, a reusable extraction script
  that fetches per-benchmark leaderboard data from all Stanford HELM
  sub-projects via their public GCS APIs. Outputs properly schema-aligned
  staging CSV files (same 37/24/38-column schemas as the main data files).
  Idempotent: re-running appends and de-duplicates.
- **Extracted 13 HELM sub-projects**: Classic (v0.4.0), Lite (v1.13.0),
  Safety (v1.17.0), MedHELM (v4.0.0), ThaiExam (v1.2.0), TORR (v1.0.0),
  EWoK (v1.0.0), Finance (v1.0.0), SEA-HELM (v1.2.0), Arabic (v2.1.0),
  Audio (v1.0.0), Image2Struct (v1.0.2), Reasoning (v0.0.1).
- **Staging totals** (pending merge into main files):
  `data/staging_helm_benchmarks.csv` → 188 benchmarks
  `data/staging_helm_models.csv` → 302 models
  `data/staging_helm_results.csv` → 6,158 result rows
- **Fixes applied during extraction:**
  - HELM model names have deprecation markers (☠, ⚠) appended to some
    model names in some benchmarks — stripped before storing so e.g.
    `text-davinci-003⚠` and `text-davinci-003` resolve to the same
    model_id.
  - Scores in 0–1 range are multiplied ×100 for consistency with the
    0-100 convention in the rest of results.csv. Exception: BPB
    (Bits Per Byte, a language-modeling entropy metric) is kept on its
    natural absolute scale (~0.5–4). WER (Word Error Rate) is also scaled
    ×100 (yielding percentage form, e.g. 11.33%).
  - `Self-BLEU` for the disinformation benchmarks is already on a 0-100
    scale in HELM; floating-point noise (100.0000000004) clamped/rounded.
  - `synthetic_efficiency` (counts inference instances, not model quality)
    excluded from extraction.
- **Known gaps** (404 from GCS): `harm_bench_gcg_transfer` (Safety),
  `ami` (Audio) — both omitted, all others extracted successfully.
- **Benchmark ID notes**: 6 benchmark_ids collide with existing entries
  (hellaswag, mmlu, legalbench, xstest, nusax, fleurs) — when merging,
  skip creating new benchmark rows for those and just add the result rows.
- **Next step**: Review staging files, then follow the merge checklist in
  `notes/TODO.md` → "Stanford HELM" section.

## Analysis Pipeline — methods, higher-order FA, balanced metric, layout ✓

Extends the densify → impute → factor pipeline (no dataset rows touched). See
`docs/METHODOLOGY.md` and `src/README.md` for the full method.

- **Layout moved:** pipeline code under `src/` (`impute/`, `factor/`, `run/`);
  Python scripts under `scripts/` (`densify.py`, `make_smoke.py`,
  `compare_loadings.py`); **data + results at the repo root** (`data/`,
  `results/<method>/`). Every script anchors to the repo root via its own file
  location, so it runs from any working directory. `make install` sets up all
  three environments (uv / renv / Julia project).
- **Three new imputers** (all cell-filling, share the held-out RMSE/R²
  contract): **knn** (`VIM`, sweeps k), **missforest** (`missForest`, sweeps
  ntree), **mice** (`mice`, sweeps m, ridge-regularized for the collinear wide
  matrix). softimpute + onesidedmc remain primary; iterativepca deferred.
- **Higher-order factoring** per cell: second-order FA of the promax factor
  correlations + bifactor (Schmid-Leiman via `psych::omega`) → ω_h, ω_total, and
  per-group ω_hs. Dashboard grew to 9 panels (added second-order, bifactor-g,
  omega). On this data: low ω_h + high per-group ω_hs ⇒ strong but
  domain-specific structure, near-absent general factor.
- **Column-balanced held-out metric** (default; `--no-balance` to revert):
  cell-weighted RMSE/R² let high-frequency "famous" benchmarks dominate and
  masked densification. Now RMSE = mean of per-column RMSEs and R² = pooled
  column-balanced ratio (not a mean of per-column R², which is unstable on thin
  columns). Applied to all R methods + OSMC (Julia, `OSMC_BALANCE`).
- **`scripts/compare_loadings.py`** — cross-method factor congruence (|cosine|,
  SSQ-sorted) grouped by dataset × loadings-kind × shape; the structural check
  behind the ~0.4–0.6 shared-R²-ceiling finding.
- **Flags:** `--method`, `--raw` (slow undensified level, run separately),
  `--smoke`, `--reimpute` (default now *reuses* existing imputed CSVs),
  `--sensitivity` (seed-sweep, parallelized, adds ω_h distribution column),
  `--no-balance`.
- **Fixes:** OSMC cell-predictor stabilized (`pinv(Vs)` in r-space, no |S|×|S|
  blowup); zero-variance/<2-obs columns dropped before factoring; softimpute
  holdout-leak fixed; removed invented `nf<3` higher-order gates.

---

## Maintenance Pass — stale stats, leaderboard-filter audit, report refactor, docs ✓

- **Fixed `models.csv`'s stale aggregate columns.** `benchmark_count`,
  `total_results`, and `avg_score` hadn't been recomputed in a long time
  (e.g. GPT-4's row said `55 / 75 / 42.91`; the real numbers were
  `76 / 122 / 88.29`). Added `scripts/lib/stats.py` +
  `scripts/manage_data.py recompute-stats` and ran it for all 1096
  models — every row changed, all now match results.csv exactly. Run
  this after any future batch of changes to results.csv; it's not
  automatic.
- **Audited the "HF Open LLM Leaderboard Official-Providers-only filter"
  item** carried over from old session notes. Checked all 448 unique
  v2-sourced `model_id`s in results.csv against the live
  `open-llm-leaderboard/contents` dataset's `Official Providers` column:
  100% are official-provider models. v1/"old" rows (1865 of them)
  predate that concept entirely. Conclusion: non-issue, already correct
  — no data change needed.
- **Refactored `benchmark_analysis.md`** from an append-only per-session
  log (duplicate entries like `akata_games_2023` listed twice,
  contradictory ones like `GaslightingBench` marked both "saturated" and
  "✅ improved" with a strikethrough hack) into one status table
  regenerated against current row counts. Moved to
  `docs/benchmark_analysis.md` (no longer gitignored). Found in the
  process: `global_mmlu` and `GaslightingBench` were both previously
  reported as expanded (to 26 and 7 rows) but are currently back down to
  4 and 3 — likely the same lost-work pattern as the HELM/Kaggle sweep
  described below, flagged in the new report rather than silently
  re-trusting the old claim.
- **Expanded README.md**: added a "Usage Examples" section with runnable
  pandas snippets (verified against the actual current data) and a
  fuller Data Schema section covering all three CSVs' column counts,
  the legacy/redundant columns in benchmarks.csv, and a prominent
  callout of the `results.model_name` → `models.model_id` foreign key
  (since results.csv's *own* `model_id` column is a different,
  denormalized field and is not what joins).
---

## Final Dupe Cleanup Pass — 19 → 6 ✓

Worked through the last 19 conflicting groups individually as requested.

- **3 small-gap groups** (<2.5 pt difference — Claude-3.5-Sonnet/p_mmeval,
  DeepSeek-R1/xcr_bench, Qwen2.5-72B/benchmax) were genuine same-metric
  noise; averaged.
- **xstest** (2 groups): verified against arXiv 2308.01263 Table 1 — the
  two scores per model are real distinct conditions (Llama2 with/without
  system prompt; Mistral with/without guardrail prompt), not duplicates.
  Filled in `setup` accordingly.
- **hagendorff_biases_2023** (3 groups): read the PDF directly (HTML
  version doesn't exist for this 2022 paper) — confirmed the benchmark
  conflates two distinct tests (CRT and Semantic Illusions) under one
  label; every value matched a specific test's reported correct-response
  rate exactly. Filled in `setup`.
- **xcr_bench/GPT-4o** (1 group): verified against arXiv 2601.14063 Table
  3 — the two scores are CSI Identification (hard) vs CSI Prediction
  (soft), two different tasks. Filled in `language`.
- **neuro_eval** (2 groups): verified against arXiv 2603.02540 — GPT-5's
  3 values matched WCST-Easy/WCST-Hard/RAPM-Gen exactly; 2 of Gemini 3
  Pro's 3 values matched SWM/RAPM-MC (the third, 96.3, wasn't found
  among the sub-tasks surfaced — left without a confirmed label, but no
  longer flagged as conflicting since the other two are now
  distinguished).
- **complexbench/GPT-4** (1 group): one value (14.9) is explicitly
  quoted in arXiv 2407.03978 as GPT-4's score on a specific
  high-difficulty test; the other (51.2) traced to a "Direct Scoring"
  baseline figure in an unrelated table (judge-agreement evaluation),
  not a GPT-4 result at all — discarded as a likely mis-extraction
  rather than kept under GPT-4's name.
- **followbench/GPT-4** (1 group): didn't need external verification —
  our own `notes` field already said one row was on a 1-5 scale (CSL)
  while the other was 0-100 (HSR/SSR); they were never the same metric.
  Relabeled `metric_name`.

### Still unresolved (6 groups)
- **opencompass** (3 groups, 21 rows): same "real metrics collapsed onto
  one metric_name" pattern almost certainly applies, but the live
  `rank.opencompass.org.cn` API consistently returned the SPA shell
  regardless of method/headers/params tried (its own JS bundle confirms
  these endpoints don't need auth, so this is a routing/hosting detail
  not yet figured out, not a permissions wall). No OpenCompass-hosted HF
  Space had these specific (very recent) models either.
- **eifbench** (2 groups): the paper's Table 4 (ILA/CLA sub-metrics) was
  located but didn't contain our stored values — there's likely another
  table further into this large, very recent paper, not found yet.
- **mmar/Gemini-1.5-Pro** (1 group): "Gemini-1.5-Pro" doesn't appear in
  arXiv 2505.13032 at all; "Gemini 2.0 Flash" has two values close to
  (but not exactly matching) our stored ones, suggesting a possible
  model-identity mislabeling rather than a simple duplicate-score issue
  — flagged for manual review rather than silently changed, since fixing
  it would mean relabeling the model identity, not just a score/metric
  label.

results.csv: 7828 → 7823 (1 row discarded as a mis-extraction; everything
else this pass only relabeled metric_name/setup/language columns).
verify_data.py still 0 FK violations, 0 orphans.

---

## Analysis Pipeline — densify → impute → factor (`src/`) ✓

Added a factor-analysis pipeline that recovers the latent factor structure of
LLM capabilities from the (super-sparse, MNAR) model × benchmark score matrix.
This is analysis *on top of* the dataset, not a change to the dataset itself —
no benchmarks/models/results rows were touched. Lives entirely under `src/`; see
`docs/METHODOLOGY.md` ("Analysis Pipeline") for the full method and rationale.

The raw aggregated matrix is ~3–5 % filled and missing-not-at-random (famous
models scored on famous benchmarks; obscure benchmarks barely co-occur), so no
single recipe is trustworthy. The pipeline instead runs a **cross-product of
palliative approaches** and reports their agreement/disagreement as the result:

- **Densify** (`src/densify.py`) — three greedy-peel strategies, each a different
  bias profile, to a target density: **C** (column-primary → famous benchmarks ×
  wide models), **R** (row-primary → saturated models × wide/obscure benchmarks),
  **S** (symmetric fill-rate). A hardcoded `MIN_OBS = 2` floor on both axes keeps
  every downstream method well-posed. Plus a `raw` (undensified) level for
  contrast. Density is the only target — deliberately imputer-agnostic.
- **Impute** (`src/impute/`) — **softimpute** (R, validated) and **onesidedmc**
  (Julia, Cao-Liang-Valiant right-singular-vector recovery) complete or side-step
  the matrix; **iterativepca** (R) is present but deferred (slow, untested).
- **Factor** (`src/factor/`) — identical principal-axis factoring + Horn parallel
  analysis (shape-cached) on every completed matrix, so the imputed input is the
  only thing that varies.
- **Orchestrator** (`src/run/main.R`) — drives the full cross-product, emits per-
  cell loadings + a 6-panel dashboard and (opt-in) seed-sweep sensitivity grids;
  imputed CSVs to `src/data/imputed/`, all plots/loadings flat in `src/results/`.

Held-out **cell-level RMSE + R²** (R² vs the train-mean baseline) is the common,
cross-method-comparable metric. A numerical-stability fix to OSMC's cell
predictor (solve in the r-dim factor space via `pinv(Vs)` rather than inverting
the rank-deficient |S|×|S| covariance) was required to keep it from blowing up on
richly-observed rows.

Smoke fixture (`src/make_smoke.py`) generates a tiny synthetic dataset so the
whole pipeline runs in seconds (`--smoke`).

---

## Remaining Dupe Cleanup — non-HF sources, 90 → 19 ✓

Continuation of the dupe resolution below, for the 126 conflicting groups
that weren't from HF Open LLM Leaderboard. Two different kinds of issue
turned up, handled differently:

### Fixed the dedup tool itself: `model_id` and `language` added to the identity key
Auditing the remaining groups found that most weren't real duplicates at
all, but **false positives from too coarse an identity key**:
- ~48 groups were genuinely distinct model checkpoints sharing one
  `model_name` (e.g. `model_name="GPT-4"` for both `gpt-4-0314` and
  `gpt-4-0613`; `model_name="Qwen3 VL 32B Instruct"` for both the
  instruct and thinking-mode checkpoints) — `model_id` already
  disambiguated them correctly, it just wasn't part of the key.
- ~36 more were genuinely distinct sub-tasks/sub-languages of a
  multilingual/multi-task benchmark (afrobench, irokobench, culemo, ...)
  all reported under the same `metric_name="accuracy"`, with the actual
  sub-task label sitting in `language` (e.g. "AfriMMLU", "pos",
  "Hindi (India)") — also not part of the key.

Added both columns to `config.RESULT_IDENTITY_KEY` (scripts/lib/config.py)
— a no-op for older rows where they're blank, but correctly splits these
apart everywhere they're populated. This is a permanent fix to the
toolkit, not a one-off data patch: 90 → 21 conflicting groups dropped out
immediately, with zero rows touched.

### Verified and fixed metric-mislabeling bugs (mt-rag, vectara, crux-eval)
Some groups *were* a real (pre-existing) extraction bug: multiple
distinct metrics for the same model+benchmark were all written with the
same generic `metric_name`/`setup`, making them look like conflicting
duplicates of one evaluation. Verified against each live source and
relabeled rather than discarding any data:
- **mt-rag** (24 groups, 71 rows): IBM's mt-rag-benchmark JSON files
  report `rb_agg`, `rb_llm`, and `rl_f` per model — all had been written
  as `metric_name="accuracy"`. Recomputed all three from the live
  `evaluations` arrays in `RAG.json` / `reference+RAG.json` /
  `reference.json` and relabeled each row to its real metric.
- **vectara hallucination-leaderboard** (7 groups, 14 rows): each pair
  summed to ~100 — "Hallucination Rate" and "Factual Consistency Rate",
  Vectara's two real complementary columns. Cross-checked every value
  against the live README table before relabeling (not just inferred
  from the sum-to-100 pattern).
- **crux-eval** (5 groups, 10 rows): the `+cot` vs non-`+cot` suffix on
  `model_id` already correctly distinguished Chain-of-Thought from
  direct prompting; `setup` just hadn't been set to record it. Verified
  every value against the live `crux-eval.github.io/data.csv` and filled
  in `setup` (`CoT` / `Direct`).

### Verified as NOT duplicates (no source data changed)
**arena-hard-auto** (4 groups) and **llm-stats.com arena-hard-v2** (3
groups): cross-checked against the live dated CSV / leaderboard page —
confirmed these are genuinely distinct model checkpoints (e.g.
`gpt-4-0314` vs `gpt-4-0613`; `-instruct` vs `-thinking` variants) that
the `model_id` key fix above already handles; no rows needed touching.
**chat.lmsys.org / Chatbot Arena** (11 groups): same pattern, but the
specific historical checkpoints (`gpt-4-0314`, `gemini-1.5-pro-001`,
etc.) have since been retired from the live leaderboard and can't be
freshly re-verified — the structural explanation (real distinct
checkpoints, not duplicates) is the same well-established pattern seen
in 7 other sources this pass, so left as-is rather than guessed at
further.

### Resolved via live re-verification (same approach as HF)
**p_mmeval** (2/3 groups): cross-checked against arXiv 2411.09116's
Table 3 (HTML rendering) — GPT-4o and Qwen2.5-72B's stored higher values
(75.11, 73.69) matched the paper's stated numbers exactly; discarded the
lower (stale) value for each. The third (Claude-3.5-Sonnet) was left
unresolved — the paper's table only has a "Claude-3.7-sonnet" row, no
3.5, so this may be a deeper model-identity mismatch, not just a score
pick.

### Still unresolved (19 groups) — see notes/TODO.md
- **opencompass** (3 groups, 21 rows): same metric-mislabeling pattern
  as mt-rag/vectara almost certainly applies (ERNIE 5.0, Qwen3 235B A22B,
  and Qwen3-Next-80B-A3B each have 7 wildly-varying rows under one
  `metric_name`), but the live leaderboard is a React SPA backed by an
  API that needs auth/POST to query — couldn't pull the real per-category
  breakdown to relabel safely.
- **~16 single-paper conflicts** (xstest, hagendorff_biases_2023,
  complexbench, followbench, eifbench, benchmax, mmar, neuro_eval,
  xcr_bench, Claude-3.5-Sonnet/p_mmeval): each needs its specific paper's
  table checked by hand. Automated WebFetch-based table extraction proved
  unreliable for some of these (got two different, mutually inconsistent
  answers from the same table on two separate fetches) — not safe to
  trust without a human cross-check, so left flagged rather than risk
  enshrining a misread number as "verified."

results.csv: 7829 → 7828 (2 rows discarded from the p_mmeval fix; the
mt-rag/vectara/crux-eval fixes only relabeled columns, no rows added or
removed).

---

## HF Open LLM Leaderboard Dupe Resolution — live re-verification ✓

`scripts/manage_data.py dupes` found 519 duplicate-evaluation groups (same
model+benchmark+metric+setup+source, conflicting data). Investigated the
source rather than guessing a winner mechanically:

- **24 redundant** (identical score reported twice) — collapsed, zero
  information loss.
- **369 conflicting, all HF Open LLM Leaderboard** (256 from the
  deprecated v1 "old" space, 113 from the current v2 space) — root cause:
  this dataset had scraped the same model's leaderboard entry at two
  different points in time, and the leaderboard's own number had drifted
  between scrapes (re-runs, harness version bumps, or — in the v1 case —
  the model owner resubmitting). There was no usable timestamp in our
  data to mechanically pick a winner (`date_recorded` was a constant
  placeholder for all of them), so each was **re-verified against the
  live source** instead of guessed:
  - v2: cross-checked against the live `open-llm-leaderboard/contents`
    HF dataset (113/113 resolved — one of the two stored scores always
    matched the current published value).
  - v1: cross-checked against the per-model timestamped JSON result
    files in `open-llm-leaderboard-old/results`, merging each
    benchmark task's value from the most recent submission that actually
    contains it (some resubmissions only re-ran a subset of tasks, so the
    single latest file alone was insufficient — 254/256 resolved this
    way). The remaining 2 rows (`Open-Orca/OpenOrcaxOpenChat-Preview2-13B`,
    gsm8k + winogrande) had neither stored score match — both were from
    submissions superseded by a later one — so they were overwritten with
    the verified current value instead of arbitrarily keeping either
    stale one.
- results.csv: 8240 → 7829 rows (411 discarded as the losing half of a
  resolved conflict, 2 overwritten in place). `scripts/verify_data.py`
  still reports 0 FK violations, 0 orphans.
- **126 conflicting groups remain**, all from non-HF live
  sources/leaderboards (lmsys Arena, vectara/hallucination-leaderboard,
  IBM mt-rag-benchmark, crux-eval, opencompass, arena-hard-auto, and
  several individual arXiv papers) — same root-cause pattern, but each
  needs its own re-verification approach and wasn't in scope for this
  pass. See `notes/TODO.md`.

---

## Repo Layout Cleanup — scripts/, docs/, LICENSE ✓

- **Scripts moved out of the repo root.** `manage_data.py`, `verify_data.py`,
  `export_eee_jsonl.py`, `export_xlsx.py` now live under `scripts/`
  (alongside `scripts/lib/` and `scripts/archive/`). Each script
  bootstraps `sys.path` to the repo root itself, so they still work when
  run as `python3 scripts/<name>.py` from any working directory, not just
  the repo root. The repo root now has no loose Python files.
- **Docs moved to `docs/`.** `METHODOLOGY.md` and `CHANGELOG.md` (this
  file) moved to `docs/METHODOLOGY.md` and `docs/CHANGELOG.md`. `README.md`
  is now the only Markdown file in the repo root; it links to both.
- **Added `LICENSE`** (MIT). The root `.gitignore` had a stray `LICENSE`
  entry left over from a generic Python `.gitignore` template (meant to
  ignore packaging-tool-generated copies) that was silently preventing a
  real `LICENSE` file from ever being committed — removed.
- Updated every cross-reference to the moved files/paths across
  README.md, docs/METHODOLOGY.md, notes/TODO.md, and
  scripts/archive/README.md.

---

## Data Cleanup — Model identity dedup + scripts refactor ✓

Full pass using the new `manage_data.py` toolkit (see "Scripts Refactor"
below) to close out the FK/orphan issues this dataset had been carrying.

### Model identity dedup (results in 0 FK violations, 0 orphans)
- **11 genuine duplicate `models.csv` entries merged** (same model
  registered twice under different casing/punctuation): `Gemini-2.0-Flash`/`Gemini 2.0 Flash`,
  `Claude-3.7-Sonnet`/`Claude 3.7 Sonnet`, `codellama-34b`/`CodeLlama-34B`,
  `Qwen2.5-VL-7B`/`qwen2.5_vl_7b`, `Gemini-2.5-Flash`/`gemini_2.5_flash`,
  `Qwen3-1.7B`/`qwen3_1.7b`, `Qwen3-0.6B`/`qwen3_0.6b`, `Qwen3 Max`/`qwen3_max`,
  `GPT-5-mini`/`GPT-5 Mini`, `internlm-20b`/`internlm_20b`, `Qwen3-4B`/`qwen3_4b`.
  In each case the spelling with more existing result rows (or cleaner
  metadata) was kept.
- **All 26 orphan `model_name` FK violations in results.csv fixed**, mapped
  to their existing canonical `model_id` after manually checking each
  one's actual source/context (not blind fuzzy-matching) — e.g.
  `ChatGPT-5`→`chatgpt5`, `ERNIE-Bot`→`ernie_bot`, `XuanYuan-70B`→`xuanYuan_70b`,
  `DeepSeek-R1-7B`→`DeepSeek-R1-Distill-Qwen-7B`, `GLM-4-9B`→`GLM-4-9B-Chat`,
  `OpenAI o1`→`o1`. Full map in `notes/TODO.md`.
- **1 genuinely new model registered**: `Llama-4-Large` (from EngiBench,
  arXiv 2509.17677) — doesn't map cleanly to the public Llama 4 Scout/Maverick
  branding, so it was added as its own entry rather than guessed, per
  strict source verification.
- **21 zero-result orphan model stubs removed** — verified each had no
  matching results.csv row under *any* casing before removal. 5 were a
  parsing-bug artifact (HuggingFace-repo-prefixed model_ids with the model
  size landing in the `developer` column, e.g. `tiiuae/falcon-7b` dev=`7B`)
  duplicating already-clean entries; 16 were leftover model registrations
  from the lost extraction sweep (see "Data Recovery" below) that never
  got result rows committed.
- 357 results.csv rows renamed in total (model identity only — no scores
  changed, no rows added/removed in results.csv).
- `models.csv`: 1127 → 1096 rows (11 merged + 21 removed + 1 added).
- Verified with `verify_data.py`: **0** invalid `benchmark_id`s, **0**
  invalid `model_name`s, **0** orphan benchmarks, **0** orphan models.

### Not done this pass (documented, not silently dropped)
- `manage_data.py dupes` finds 519 duplicate-*evaluation* groups in
  results.csv (24 pure redundancy, 495 conflicting scores — mostly HF Open
  LLM Leaderboard v1/v2 re-run drift). This is a different problem from
  model-identity dedup above (same model+benchmark, multiple *score*
  reports) and needs human review of the conflicts before resolving; see
  `notes/TODO.md`.
- `manage_data.py categorize-models` flags 46 models as fine-tunes with no
  `model_family`/`base_model` set and 38 as unclear-origin — none have
  zero results, so none were removed; they need metadata enrichment, not
  deletion. See `notes/TODO.md`.

### Scripts Refactor — reusable dataset maintenance toolkit
- Replaced ~17 one-off, mutually-inconsistent scripts (several reimplementing
  the same duplicate-detection or model-categorization logic slightly
  differently, several referencing stale intermediate files, several with
  hardcoded one-off rename maps) with a single importable library,
  `scripts/lib/` (`config`, `io`, `integrity`, `dedup`, `aliases`,
  `categorize`, `export`), exposed through one CLI, `manage_data.py`, plus
  `verify_data.py` and the two `export_*.py` scripts as thin wrappers over
  the same library.
- Old one-off scripts moved to `scripts/archive/` for audit trail (not part
  of the active toolkit — see `scripts/archive/README.md` for the
  old-script → new-command mapping). `scripts/` is no longer gitignored.
- Found and fixed a latent bug while refactoring `export_eee_jsonl.py`: it
  read a `developer` column that doesn't exist in results.csv (the real
  column is `model_developer`), so every exported EEE record's
  `model_info.developer` silently defaulted to `"unknown"`.
- Added a new METHODOLOGY.md section, "Adding New Data: Required
  Checklist", codifying the integrity-check → dedup/standardize →
  inclusion-criteria-compliance steps every future data addition should
  follow, using the new toolkit.

---

## Data Recovery — Partial restore after git refactor data loss ✓

A repo-wide git refactor (run via an external agent) reset `data/models.csv`, `data/results.csv`, and `data/benchmarks.csv` back to the last *committed* state (commit `3e3f193`, Apr 27), silently discarding weeks of uncommitted working-tree progress — including a full Kaggle+PwC+HELM extraction sweep that had reached 312 benchmarks / 1155 models / 11208 results (see "Previous HELM sweep (lost)" in TODO.md). Investigation found no git-level recovery path (no reflog/dangling commits — the data files were simply never committed), but a local, gitignored snapshot `data/*.csv.bak5` (saved ~5.5h before the wipe) preserved part of that work.

### Recovered from `data/*.csv.bak5`
- **Papers With Code multimodal scan** (10 benchmarks): agieval, mathvista, evalplus, chartqa, docvqa, ai2d, mmbench, mme, ocrbench, textvqa.
- **HELM FACTS family** (5 benchmarks): facts, facts_grounding, facts_multimodal, facts_parametric, facts_search.
- 138 associated result rows, mapped from bak5's normalized lowercase model IDs back onto the canonical model_id strings already used in `models.csv` (e.g. `gpt-4` → `GPT-4`, `claude-3-opus` → `Claude 3 Opus`).
- 2 new models added: `Qwen-VL-Max`, `text-davinci-003`.
- Verified with `verify_data.py`: 0 missing `benchmark_id` FKs; the 26 missing `model_name` FKs are a pre-existing, unrelated issue (see TODO.md).

### Not recoverable (no backup contained it)
- The full Stanford HELM "other groups" sweep (15 sub-projects, ~973+ rows beyond the FACTS family) and the Kaggle Research category sweep (104 benchmarks) — these existed only in the working tree past the bak5 snapshot and were never saved anywhere. Must be re-extracted from scratch; see TODO.md.

---

## Batch 6 — Reasoning / Counterfactual Reasoning ✓

### Added
- **CounterfactualReasoningEval** — ICLR 2026 (arXiv 2505.11839). Decompositional evaluation of counterfactual reasoning across 4 stages using Pearl's SCM framework: (I) Causal Variable Identification, (II) Causal Graph Construction, (III) Counterfactual Intervention Identification, (IV) Outcome Reasoning. 11 datasets spanning text, vision-language, math (symbol), and code modalities. Evaluated 7 models. F1 scores per decompositional stage: GPT-5 leads on exposure identification (avg ~87%), but performance drops sharply on mediator-outcome reasoning (~60% avg). Llama-4-S and Llama-4-M (both new model entries) evaluated alongside ChatGPT-5, o1, Qwen3-Max, Gemini-2.5-Pro, DeepSeek-V3. Results include 4 sub-metrics per dataset-model pair: TaskIII_F1_Exposure, TaskIII_F1_Covariate, TaskIII_F1_Mediator, TaskIII_F1_Outcome, TaskIV_F1_M_prime, TaskIV_F1_Y_prime — total 462 result rows (7 models × 11 datasets × 3 task phases × 2 sub-dimensions).

### Removed from pending
- **Sycophancy Is Not One Thing** (arXiv 2509.21305) — Mechanistic interpretability study on causal separation of sycophantic behaviors, NOT a benchmark with leaderboard scores. Does not fit EEE schema (no model-vs-model score table).
- **DiploBench** — Explicitly labeled by authors as "not a benchmark yet" in EQ-Bench; consists of single game runs with high variance between iterations. Not suitable for standard benchmark tracking.
- **Cultural Variations in Moral Judgments** (arXiv 2506.12433) — Correlation study comparing LLM moral judgments to human survey data across countries using correlation coefficients, NOT a standard benchmark with accuracy/F1 scores.

### Total
- Benchmarks added: 1 new
- Models added: 2 new (Llama-4-S, Llama-4-M); mapped from paper model aliases (GPT-o4→o1, GPT-5→gpt-5, Qwen3→qwen3 max, Gemini2.5→gemini-2.5-pro, DeepSeek→deepseek-v3)
- Result rows added: 462

---

## Batch 7 — Engineering & GPU kernels ✓

### Added
- **EngiBench** — ICLR 2026 (arXiv 2509.17677). Hierarchical engineering benchmark: Level 1 (1717 problems, foundational knowledge retrieval across systems/control, physics/structural, chemical/biological), Level 2 (511 problems, multi-step contextual reasoning), Level 3 (43 open-ended modeling tasks with rubric scoring max 10). Each L1/L2 has 3 variants: original, perturbed, knowledge-enhanced, and math abstraction. Evaluated 14 models across 5 metrics per model (Level 1 orig/perturbed accuracy, Level 2 orig/perturbed accuracy, Level 3 rubric score). GPT-4.1 leads with Level 3 score 7.0; Tier 2 models (~6.0 avg); Tier 3/7B-class models (~3.5 avg).
- **KernelBench** — ICML 2025 (arXiv 2502.10517, corrected from pending URL 2601.00227 which is FlashInfer-Bench). GPU kernel generation benchmark with 3 levels: Level 1 (100 individual ops), Level 2 (100 operator sequences for fusion testing), Level 3 (50 end-to-end architectures). Metric: fast_p (fraction correct AND speedup >p over PyTorch Eager). Evaluated 5 models. DeepSeek-R1 achieves best L2 at 36% (72% with iterative refinement feedback). Reasoning models better at avoiding crashes but equally weak on functional correctness. CUDA is only 0.073% of code corpora explaining low performance overall.

### Fixed
- **KernelBench URL corrected**: arXiv 2601.00227 was FlashInfer-Bench, real KernelBench is arXiv 2502.10517

### Removed from pending
- **BioNovice Lab Performance** — LLM-assisted human trial (arXiv 2602.16703), not a standard model-vs-model benchmark with leaderboard scores. Should be tracked separately if needed.

### Total
- Benchmarks added: 2 new (EngiBench, KernelBench)
- Models verified/added: 3 new (Claude-3.7-Sonnet, Gemini-2.0-Flash, GLM-4-32B); all others existed in CSVs
- Result rows added: 96 (80 EngiBench + 16 KernelBench)
- Totals after batch: 198 benchmarks, 1118 models, 8024 result entries

## Batch 8 — Critique-Correct Reasoning ✓

### Added
- **CriticBench** — ACL 2024 Findings (arXiv 2402.14809). General reasoning benchmark evaluating LLMs' ability to generate, critique, and correct their own reasoning across 5 domains: Math (GSM8K), Commonsense (CSQA), Algorithmic, Coding (MBPP/HumanEval), Symbolic (BIG-Bench). 3,800 instances from 15 datasets. Evaluated 17 models from LLaMA/Vicuna/GPT families in generation/correction phases; critique phase evaluated 12 models. Uses binary discrimination F1 for critique (CritF1), accuracy for generation (GenAcc) and correction (CorrAcc). Key finding: linear GQC relationship, critique-focused training boosts performance even over larger RLHF models. Added per-model averages across all domains (3 metrics × 12 models = 36 result rows).

### Total
- Benchmarks added: 1 new
- Models added: 3 new (Mixtral-8x7b inst, Vicuna-33b, Vicuna-13B — existing FK aliases normalized); 9 mapped from lowercase names to canonical models.csv entries
- Result rows added: 36
- Totals after batch: 200 benchmarks, 1121 models, 8060 result entries

### Removed from pending
- **CRITIQUE** (CriticBench) — Added as Batch 8

## Batch 9 — Malicious Prompt Resistance ✓

### Added
- **MaliciousInstruct** — ICLR 2024 (arXiv 2310.06987). Safety/red-teaming benchmark evaluating LLM resistance to jailbreaks via "generation exploitation" attack. 100 malicious prompts across 10 categories: psychological manipulation, sabotage, theft, defamation, cyberbullying, false accusation, tax fraud, hacking, fraud, illegal drug use. Evaluated 11 open-source models (Vicuna 7B/13B/33B, MPT 7B/30B, Falcon 7B/40B, LLaMA2 7B/13B + both chat variants). ASR measured under 6 decoding conditions: greedy with/without system prompt, varied temperature, varied top-k, varied top-p, and combined varied all. Key finding: default evaluations yield <5% ASR on aligned models, but simple decoding manipulation pushes ASR to >95% across all models. Safety-aligned LLaMA2-chat models drop from 0% (default) to 8-16% under single-condition exploitation, reaching 71-88% under varied decoding.

### Total
- Benchmarks added: 1 new
- Models added: 5 new (MPT-7B, MPT-30B, Falcon-7B, Falcon-40B, Llama-2-13B base); 6 mapped from lowercase to canonical names
- Result rows added: 66 (11 models × 6 ASR metrics)
- Totals after batch: 202 benchmarks, 1126 models, 8126 result entries

### Removed from pending
- **MaliciousInstruct** — Added as Batch 9

---

## Batch 4 — Music

### Added
- **ABC-Eval** — arXiv 2025. (Previously added; results verified) Symbolic music understanding via ABC notation MCQs. 10 sub-tasks across 3 complexity levels. Accuracy: GPT-5 55.02%, GPT-5-mini 53.51%, DeepSeek-reasoner 52.01%, Gemini Pro 51.25%, DeepSeek-chat 40.17%.
- **WildScore** — arXiv 2025. (Previously added; results verified) In-the-wild multimodal symbolic music reasoning from r/musictheory. 807 MCQs across 5 domains. Accuracy: GPT-4o-mini 68.31%, Qwen2.5-VL-72B 49.73%, Phi-3-Vision 48.82%, Gemma-3 46.34%, MiniCPM 45.90%, InternVL 39.34%, LLaVA 32.97%.
- **MSU-Bench** — arXiv 2025 (Withdrawn ICLR 2026). First human-curated benchmark for score-level musical understanding across textual (ABC notation) and visual (PDF) modalities. 1,800 QA pairs from 150 classical scores, 4 hierarchical levels. Textual QA accuracy: Gemini 2.5 Pro 49.44%, ChatGPT-5 47.28%, ChatGPT-5-mini 43.72%, Grok 4 42.61%, Claude Sonnet 4 42.61%, Claude Opus 4 41.28%, Qwen3-VL-235B 41.22%.
- **ZIQI-Eval** — ACL Findings 2024. Massive 14,244-entry music benchmark with 10 categories and 56 subcategories. Comprehension F1: GPT-4 63.04, ERNIE-Bot 59.94, Claude-instant-1.2 53.50, GPT-3.5-Turbo 52.91, XuanYuan-70B 51.40. Also evaluates music generation (ABC notation continuation).
- **MuChoMusic** — ISMIR 2024. Audio LLM evaluation using 1,187 MCQs across 644 music tracks. Accuracy: Qwen-Audio 51.4%, M2UGen 42.9%, SALMONN 41.8%, MuLLaMa 32.4%, MusiLingo 21.1%. Significant language bias where models ignore audio input.

### Total
- Benchmarks added: 3 new (MSU-Bench, ZIQI-Eval, MuchoMusic); 2 verified (ABC-Eval, WildScore)
- Models added: 32 (ChatGPT-5, ChatGPT-5-mini, Gemini 2.5 Flash, Qwen3-VL-235B, Qwen3-0.6B/1.7B/4B/32B/MAX, Qwen2.5-VL-3B/7B/32B/72B, Llama 4 Scout, DeepSeek-V3, ERNIE-Bot, Claude-instant-1.2, XuanYuan-70B, educhat-base-002-13B, ChatMusician-Base, Ziya-LLaMA-13B-v1.1, Qwen-7B/14B-Base, XVERSE-13B/7B, Baichuan2-7B/13B-Base, Baichuan-13B/7B-Base, InternLM-7B/20B, M2UGen, MuLLaMa, MusiLingo)
- Result rows added: 24 (MSU-Bench: 7, ZIQI-Eval: 12, MuchoMusic: 5)

---

## Batch 3 — Sports

### Added
- **SportQA** — NAACL 2024. Level-3 (hardest) accuracy for 4 models (GPT-4: 23.01%, GPT-3.5-Turbo: 19.24%, PaLM-2-Bison: 16.74%, Llama-2-13B-Chat: 8.79%) on 70,592 MCQs across 35 sports. Human experts score 91.84%.
- **SPORTU** — ICLR 2025. Video MCQ accuracy for 4 MLLMs (Qwen2-VL-72B: 70.94%, Claude-3.5-Sonnet: 70.18%, GPT-4o: 68.79%, LLaVA-NeXT-72B: 63.72%) on 12,048 QA pairs across 7 sports.
- **SportsMetrics** — ACL 2024. Mean absolute error on NBA game points for 4 models (GPT-3.5-Turbo-1106: 9.45, Gemini Pro: 17.62, claude-2.1: 21.73, Llama-2-13B-Chat: 70.77) on 28,492 NBA games + 5,867 NFL games.

### Total
- Benchmarks added: 3
- Models added: 2 (LLaVA-NeXT-72B, GPT-3.5-Turbo-1106)
- Result rows added: 12

---

## Batch 1 — Safety/Red-teaming

### Added
- **XSTest** — NAACL 2024. Refusal rate on safe prompts for 5 models (Llama2.0, Llama2.1, Mistral-7B-Instruct, Mistral-7B+Guardrails, GPT-4).
- **Do-Not-Answer** — EMNLP 2023. Harmful response rate for 3 models (LLaMA-2-7B-Chat, ChatGLM2, Vicuna-13B).
- **JailbreakBench** — NeurIPS 2024. Attack success rate for 2 models (Llama-2-7B-Chat, GPT-4) via Random Search attack.
- **HarmBench** — arXiv 2024. GCG attack success rate for 1 model (Zephyr-7B: 31.8%).
- **TrustLLM** — arXiv 2024. Benign prompt refusal rate for 2 models (Llama2-7B: 57%, GPT-4: 80.5%).

### Total
- Benchmarks added: 5
- Result rows added: 13

---

## Batch 2 — Coding/Debugging

### Added
- **DebugBench** — ACL 2024 Findings. Pass rate for 5 models (GPT-4: 65%, GPT-3.5: 45%, DeepSeek-Coder-33B: 42%, CodeLlama-7B: 22%, Mixtral-8x7B: 15%) across C++, Java, Python.

### Total
- Benchmarks added: 1
- Result rows added: 5

---

## Process Group 4 — Morality/Ethics Batch

### Added
- MoralBench results
- Moral Machine for LLMs results
- AITA Normative Evaluation results

### Removed
- ETHICS (no valid LLM baseline scores)
- Value Kaleidoscope (no valid LLM baseline scores)
- Delphi (no valid LLM baseline scores)

---

## Process Group 3 — Humor/Creativity Batch

### Added
- HumorBench results
- Not All Jokes Land (workplacehumor) results
- LitBench results
- CreativityPrism results

### Purged (re-added later)
- Oogiri LLM Benchmark (re-added in Group 3.5)
- TTCW (re-added in Group 3.5)
- CS4
- A Confederacy of Models

---

## Process Group 2 — Games/Strategy + Others

### Added
- LLM Chess results
- GTBench results
- RPGBench results

### Removed
- Readable Minds (not a standard LLM benchmark)
- LudoBench (not a standard LLM benchmark)
- gg-bench (not a standard LLM benchmark)
- PokerBench (not a standard LLM benchmark)

---

## Earlier Work

### HF Open LLM Leaderboard Cleanup ✓
- v1 and v2 leaderboard rows retained as-is
- v2 collected with "Only Official Providers" filter
- v1 reviewed and confirmed acceptable

### Blank Score Filling ✓
- 21 of 57 blank-score rows filled
- 36 intentionally blank (documented reasons in TODO.md)

### Baseline Model Expansion
- Added exhaustive SOTOPIA results
- Added exhaustive SI-Bench and ELEPHANT results
- Updated EQ-Bench and EmoBench exports
