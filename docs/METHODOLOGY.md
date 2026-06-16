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
- `data/eee_output/all_evaluations.jsonl` — single consolidated file with all 7,828 records
- `evaluation_id` is an MD5 hash of `model_name + benchmark_id + source_url`

### Excel Workbook (`scripts/export_xlsx.py`)
- `data/llm_benchmarks_export.xlsx` — three-sheet workbook (benchmarks / models / results)