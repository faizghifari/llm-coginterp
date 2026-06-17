# LLM Benchmark Dataset: Methodology

## General Data Collection Methodology

The core philosophy of this data collection effort is **"Strict Source Verification"**. To prevent the hallucination or fabrication of evaluation parameters, no data was inferred using generalized heuristics (e.g., assuming all open models use HuggingFace Transformers). If a detail was not explicitly documented by the benchmark's authors or the evaluator, the field was intentionally left blank.

### Platforms & Tools Used for Verification
- **Web Search & Indexing:** Direct web searches to locate missing benchmark repositories and original papers.
- **Paper Parsing:** Automated conversion of ArXiv PDFs/Abstracts to HTML formats to extract methodology sections.
- **Repository Scanning:** Direct scanning of `raw.githubusercontent.com` (Main and Master branches) for README files and evaluation scripts.
- **HuggingFace:** Scanning Dataset cards via the HF API.

---

## Adding New Data: Required Checklist

Every time benchmarks/models/results are added or edited, run through this
checklist before committing. All of these tools are described in more
detail in their own sections below; this is the order to run them in.

1. **Verify data integrity.**
   ```bash
   python3 scripts/verify_data.py
   ```
   Confirms 0 FK violations (`benchmark_id` in results must exist in
   benchmarks.csv; `model_name` in results must exist as a `model_id` in
   models.csv), 0 orphan benchmarks/models (zero result rows), and flags
   any benchmark with fewer than 5 result rows as a possible incomplete
   extraction (see "Multiple Scores per Model-Benchmark Pair" below for
   what *not* to do about that — don't average to hit a row-count target).

2. **Dedup and standardize model identity.**
   - New model names almost always collide with an existing entry under
     different casing/spacing (`GPT-4o` vs `gpt-4o`, `Qwen3 Max` vs
     `qwen3_max`). Before adding a model row, check whether it already
     exists: `python3 scripts/manage_data.py find-aliases` lists every
     `model_name` in results.csv that has no matching `model_id` in
     models.csv, with fuzzy-match suggestions.
   - If the new data really does introduce a duplicate (same model,
     different spelling), fix it with a rename map rather than leaving
     both spellings in the dataset:
     ```bash
     echo '{"Old-Spelling": "canonical-model-id"}' > /tmp/renames.json
     python3 scripts/manage_data.py apply-aliases --map-file /tmp/renames.json --write
     ```
   - Check for duplicate *evaluations* (same model + benchmark + metric +
     setup + source reported more than once) with
     `python3 scripts/manage_data.py dupes --verbose`. Pure redundancy (identical
     score reported twice) and genuine conflicts (different scores for
     what should be the same evaluation) are reported separately —
     resolve conflicts by hand or with `scripts/manage_data.py dedup --write`
     (trust-tier + recency), but always read the `--verbose` output
     first; don't run `dedup --write` blind.
   - Do **not** run a blanket `standardize-ids` pass to relabel every
     model_id to one casing convention — this dataset's existing
     convention is mixed Title-Case (`GPT-4`, `Claude 3 Opus`), and a
     wholesale relabel would touch hundreds of unrelated, already-correct
     rows for no integrity benefit. Only rename entries that are
     genuinely the same model under two different spellings.

3. **Ensure models comply with the inclusion/exclusion criteria below.**
   Run `python3 scripts/manage_data.py categorize-models` to classify every model
   as `KEEP` / `FLAG` / `REMOVE` per the Model Inclusion Criteria. `FLAG`
   and `REMOVE` are *prompts for manual review*, not auto-delete signals:
   cross-check against `model_type`/results before removing anything, and
   never delete a model that still has result rows (that breaks FK
   integrity — `scripts/verify_data.py` would catch it, but don't get there).
   Only delete a model row if it both (a) fails the inclusion criteria
   and (b) has zero result rows after step 1's orphan check.

4. **Re-run `scripts/verify_data.py` one more time** after any fixes from steps 2–3,
   to confirm the change didn't introduce a new FK violation or orphan.

---

## Model Inclusion Criteria

We enforce strict criteria for which models are tracked in this dataset. The core requirement is that the model must be a **generative language model capable of handling arbitrary prompts**.

**Included:**
- General-purpose LLMs (e.g., GPT-4, Llama 3, Claude 3).
- Models trained or fine-tuned for a specific task or domain (e.g., coding, medical, legal), provided they retain the ability to process arbitrary text prompts.
- Multimodal models (vision-language, audio-language) that add an encoder or other modality processor on top of an LLM, as long as the base model can still handle arbitrary prompts.

**Excluded:**
- Classification-only or encoder-only models (e.g., BERT, RoBERTa) that cannot generate arbitrary text.
- Task-specific models restricted to narrow inputs/outputs (e.g., dedicated TTS, ASR, or specialized translation models that cannot be prompted generally).
- Community-uploaded models (e.g., random HuggingFace uploads) that are experimental, undocumented, or lack reliable provenance and evaluation data.
- Models that represent a different *setup* of the same model (e.g., different context length, CoT prompting, effort levels). These are captured via the `setup` and `reasoning_enabled` columns instead of as separate model entries.

---

## Data Normalization

### Benchmark Normalization
- `benchmark_id` is always lowercase, used as the primary key.
- 22 zero-result benchmark stubs were removed from `benchmarks.csv`. Two (`CRUX`, `VerifyQA`) were confirmed duplicates of existing entries (`cruxeval`, `simpleqa`); the remaining 20 were genuine empty stubs with no associated results. All 20 were added to `notes/pending_benchmarks.md` for future data collection.
- Benchmarks with zero result rows must not exist in benchmarks.csv.

### Model Normalization
- **HuggingFace org prefix stripping:** All model IDs had organization prefixes removed (e.g., `meta-llama/Llama-3-8B` → `Llama-3-8B`). One collision was pre-resolved before stripping (`mistral-community/Mixtral-8x22B-v0.1` merged into `mistralai/Mixtral-8x22B-v0.1`).
- **Thinking/reasoning tag removal:** Model names with thinking-mode or effort-level tags (e.g., `claude-3-7-sonnet-thinking`, `o3 high`) were merged into their canonical base names; affected rows received `reasoning_enabled = True`.
- **Context-length variants merged:** `gpt-4-32k`, `gpt-4-128k`, etc. were merged into `GPT-4` since context length is an evaluation setup, not a model identity.
- **Llama family disambiguation:** Llama variants are kept distinct using `model_size` + `year_evaluated` heuristics (e.g., 7B/13B/70B → Llama 2; 405B → Llama 3.1). Llama 4 variants (Scout, Maverick) are tracked as separate entries.
- **Non-LLM removal:** Models that are not generative LLMs (e.g., SeamlessM4T, encoder-only models) were removed from both `models.csv` and `results.csv`.

### Data Integrity
- FK violations (results rows referencing unknown benchmarks or models): **0** (benchmarks), **0** (models — last alias cleanup pass: 2026-06-16, see CHANGELOG.md "Data Cleanup").
- Models with zero result rows: **0**.
- Benchmarks with zero result rows: **0**.
- Always run `scripts/verify_data.py` after making changes to the 3 main data files: `benchmarks.csv`, `models.csv`, `results.csv`.

### Multiple Scores per Model-Benchmark Pair
- If a model has more than 1 score for the same benchmark (due to different evaluation setup, different provider/evaluator running the benchmark, different prompting strategy, etc.), each score is kept as a **separate row** in `results.csv` — never averaged or collapsed.
- Rows are distinguished by the `setup` field (describing the evaluation configuration) and `source_url` (which evaluator/provider published the result).
- Example: GPT-4 evaluated on MMLU by HELM (one row) and by the original MMLU authors (another row) → two separate rows with different `setup` and `source_url`.

---

## Inference Environment Collection

We mapped the `inference_platform` and `inference_engine` used for each evaluation row.

**Methodology:**
1. Traced each evaluation to its `source_url`.
2. If the source was an aggregator (like `llm-stats.com` or `pricepertoken.com`) that did not publish their engineering stack, the inference data was marked as unverified and left blank.
3. If the source was an official benchmark, we scanned the GitHub README and ArXiv paper for explicit mentions of frameworks (e.g., `vLLM`, `lm-eval-harness`, `SGLang`, `Lighteval`).

**Assumptions Applied:**
- **Closed Models via API:** Models with `model_type == 'closed'` (e.g., GPT-4, Claude 3.5, Gemini 1.5) cannot be run locally. Their `inference_platform` was universally set to `api` and engine to `api (unspecified)`.

---

## Generation Configuration

We collected explicit generation parameters (`generation_temperature`, `generation_max_tokens`, `generation_top_p`).

**Methodology:**
1. Scanned all benchmarks' original GitHub READMEs and ArXiv HTML papers, or the benchmark sources.
2. Filtered for keywords: `temperature`, `max_tokens`, `top_p`, `greedy decoding`.
3. Extracted only explicit, stated values (e.g., "We suggest using greedy decoding", "temperature was fixed at 0").
4. Benchmarks that relied on "bring your own predictions" (e.g., PubMedQA) or whose papers omitted decoding parameters were strictly left blank.

**Assumptions Applied:**
- **LM-Eval-Harness Default:** The EleutherAI `lm-eval-harness` calculates log-likelihoods for multiple-choice tasks without temperature, and defaults to `greedy=True` (`temperature=0.0`) for generative tasks unless overridden. For benchmarks strictly verified to use `lm-eval-harness` (e.g., AfroBench), we applied `generation_temperature = 0.0`.

---

## Quality Assurance (URL Validation)

Before finalizing the dataset, a multi-threaded URL validator was run across both CSVs:
- Validated HTTP status codes (ignoring false-positive 403s from anti-bot protections on ArXiv/Cloudflare).
- Hunted down and replaced dead links (404s), such as resolving moved GitHub repositories (e.g., `HiTZ/BasqueGLUE` → `orai-nlp/BasqueGLUE`) or finding mirrors for dead DOI links.
- Filled in 53 entirely blank benchmark source links by cross-referencing benchmark names with active ArXiv/GitHub links.

---

## Export Outputs

### EEE JSONL (`scripts/export_eee_jsonl.py`)
- `data/eee_output/by_benchmark/{benchmark_id}.jsonl` — one file per benchmark (EEE schema v0.2.1)
- `data/eee_output/all_evaluations.jsonl` — single consolidated file with all 7,823 records
- `evaluation_id` is an MD5 hash of `model_name + benchmark_id + source_url`

### Excel Workbook (`scripts/export_xlsx.py`)
- `data/llm_benchmarks_export.xlsx` — three-sheet workbook (benchmarks / models / results)

---

## Analysis Pipeline (`src/`)

Beyond curating the dataset, `src/` holds an analysis pipeline that recovers the
**latent factor structure** of LLM capabilities from the model × benchmark score
matrix. It does not modify the dataset; it consumes the aggregated matrix and
produces factor loadings + diagnostics.

### The core problem: super-sparse and MNAR

The aggregated matrix (≈970–1096 models × ≈200–215 benchmarks) is only **~3–5 %
filled**, and the missingness is **not at random**: famous models are scored on
famous benchmarks, while obscure benchmarks barely co-occur with anything. There
is no single imputation or densification that "fixes" this honestly. The guiding
philosophy mirrors the dataset's "strict source verification": rather than pick
one recipe and present its output as the answer, the pipeline runs a
**cross-product of palliative approaches** and treats their agreement (or
disagreement) as the actual finding.

```
aggregation → DENSIFY → IMPUTE → FACTOR
```

### Stage 1 — Densify (`src/densify.py`)

Drops rows/columns to reach a workable density, producing several *bias
profiles* (not one "best" table). Density is the only target — pairwise-overlap
and positive-definiteness are deliberately **not** optimized, so the densifier
stays neutral across downstream imputers.

- **C** (column-primary peel): drop the sparsest benchmark, then any emptied
  model → famous benchmarks × wide model coverage.
- **R** (row-primary peel): drop the sparsest model, then any emptied benchmark →
  saturated models × wide benchmarks (retains obscure benchmarks, at the cost of
  collapsing to few models).
- **S** (symmetric peel): drop whichever marginal has the lowest fill-rate →
  balanced, neither axis privileged.
- **raw**: the undensified matrix, run through the pipeline as a contrast level.

After peeling, a hardcoded `MIN_OBS = 2` floor is enforced on **both** axes
(every kept model and benchmark must have ≥2 scores) so all downstream methods
are well-posed. Knobs (`TARGET`, `MIN_OBS`) are constants at the top of the file;
`--peek` previews shapes/density without writing.

### Stage 2 — Impute (`src/impute/`)

Each method is impute-only: it completes (or side-steps) the matrix and reports
rank-selection diagnostics, but does **not** factor.

- **softimpute** (R, `softImpute`) — low-rank matrix completion; rank chosen by
  held-out cell RMSE. The primary validated method.
- **onesidedmc** (Julia, Cao-Liang-Valiant 2023) — does not impute cells; it
  recovers the benchmark covariance Θ̂ from pairwise products of observed scores,
  then synthesizes a covariance-matched surrogate matrix for factoring. Faithful
  to the paper's "one-sided" recovery (right singular vectors are recoverable
  when cells are not). Real data has variable observations per row, handled via a
  ragged observation format.
- **iterativepca** (R, `missMDA`) — present but **deferred**: `estim_ncpPCA`'s
  cross-validation is prohibitively slow at this matrix size, and its sensitivity
  has not been migrated to the held-out RMSE/R² metric. Treat as provisional.

**Scaling:** columns are standardized (z-scores), rows are never scaled — rows
are models, and row-scaling would erase the general-capability (g) signal the
factor analysis is meant to find.

**Metric:** held-out **cell-level RMSE and R²**, identical across methods so they
are comparable. ~20 % of observed cells are held out; the model predicts them;
R² = 1 − SS_resid / SS_baseline where the baseline predicts the **train-cell**
column mean (honest out-of-sample). R² = 0 is the no-skill baseline, 1 is
perfect, negative means worse than the mean. For onesidedmc, cells are predicted
from the recovered covariance via the conditional-Gaussian (best-linear)
predictor — an off-label use of the method, chosen so its number sits on the same
scale as softimpute.

### Stage 3 — Factor (`src/factor/`)

Method-agnostic: identical **principal-axis factoring** (`fm = "pa"`) + promax
rotation on every completed matrix, so the imputed input is the only thing that
varies. The number of factors comes from **Horn's parallel analysis**, split for
efficiency into (a) random-baseline eigenvalue cutoffs that depend only on matrix
shape `(n, p)` and are cached as JSON, and (b) the observed eigenvalues computed
per dataset. The cutoffs are **shape-keyed, never global** — parallel analysis is
not dataset-independent.

### Orchestration (`src/run/main.R`)

Runs the cross-product `{densifier} × {strategy} × {imputer}`. For each cell:
impute → factor → a single 6-panel dashboard (predictive curve, marginal gain,
cumulative variance, scree, SS loadings, PA-factor-count). Imputed matrices are
written under `src/data/imputed/<method>/<densifier>/<strategy>/` (CSV only);
everything else (loadings, dashboards, sensitivity grids) lands flat in
`src/results/`. Flags: `--method`, `--raw` (the slow undensified level, run
separately), `--smoke` (tiny synthetic fixture from `src/make_smoke.py`),
`--reimpute` (force fresh imputation instead of reusing existing CSVs),
`--sensitivity` (opt-in seed-sweep, parallelized; each seed is a fresh held-out
split, so the spread quantifies MNAR fragility).

### Interpreting the results (caveat)

Good held-out RMSE/R² means good per-cell prediction, not necessarily a
trustworthy factor structure — the loadings and their **stability across
densifiers and seeds** are the real output. On this MNAR data the factor count is
typically weakly identified (flat RMSE curves, scattered best-rank across seeds);
that instability is itself a reportable finding, not a bug to tune away.