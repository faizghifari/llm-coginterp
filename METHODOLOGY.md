# LLM Benchmark Dataset: Methodology & Descriptive Statistics

## 1. Descriptive Statistics

**Benchmarks (`benchmarks.csv`)**
- **Total Benchmarks:** 162
- **Source URL Coverage:** 100% (Every benchmark has at least one verified working link to its ArXiv Paper, GitHub Repo, or HuggingFace Dataset).

**Evaluation Results (`results.csv`)**
- **Total Evaluation Rows:** 8444
- **Model Types:** 6446 Open Weights, 1719 Closed/API Models, 279 Unknown.
- **Inference Info Coverage:** 6857 / 8444 rows (81.2%)
- **Generation Config Coverage:** 5316 / 8444 rows (63.0%)

---

## 2. General Data Collection Methodology

The core philosophy of this data collection effort is **"Strict Source Verification"**. To prevent the hallucination or fabrication of evaluation parameters, no data was inferred using generalized heuristics (e.g., assuming all open models use HuggingFace Transformers). If a detail was not explicitly documented by the benchmark's authors or the evaluator, the field was intentionally left blank.

### Platforms & Tools Used for Verification
- **Web Search & Indexing:** Direct web searches to locate missing benchmark repositories and original papers.
- **Paper Parsing:** Automated conversion of ArXiv PDFs/Abstracts to HTML formats to extract methodology sections.
- **Repository Scanning:** Direct scanning of `raw.githubusercontent.com` (Main and Master branches) for README files and evaluation scripts.
- **HuggingFace:** Scanning Dataset cards via the HF API.

---

## 3. Model Inclusion Criteria

We enforce strict criteria for which models are tracked in this dataset. The core requirement is that the model must be a **generative language model capable of handling arbitrary prompts**.

**Included:**
- General-purpose LLMs (e.g., GPT-4, Llama 3, Claude 3).
- Models trained or fine-tuned for a specific task or domain (e.g., coding, medical, legal), provided they retain the ability to process arbitrary text prompts.
- Multimodal models (vision-language, audio-language) that add an encoder or other modality processor on top of an LLM, as long as the base model can still handle arbitrary prompts.

**Excluded:**
- Classification-only or encoder-only models (e.g., BERT, RoBERTa) that cannot generate arbitrary text.
- Task-specific models restricted to narrow inputs/outputs (e.g., dedicated TTS, ASR, or specialized translation models that cannot be prompted generally).
- Community-uploaded models (e.g., random HuggingFace uploads) that are experimental, undocumented, or lack reliable provenance and evaluation data.

---

## 4. Data Normalization (Tasks 1-3)
- **Model Normalization:** Model families were normalized without erasing crucial variant details. For instance, Llama models were carefully disambiguated using heuristics like `model_size` and `year_evaluated` (e.g., 7B/13B/70B -> Llama 2, 405B -> Llama 3.1) instead of grouping them into a monolithic "Llama" category.
- **Data Integrity:** Ensured that structural updates to the CSVs preserved all historical metrics.

---

## 4. Inference Environment Collection (Task 4)
We mapped the `inference_platform` and `inference_engine` used for each evaluation row.

**Methodology:**
1. Traced each evaluation to its `source_url`.
2. If the source was an aggregator (like `llm-stats.com` or `pricepertoken.com`) that did not publish their engineering stack, the inference data was marked as unverified and left blank.
3. If the source was an official benchmark, we scanned the GitHub README and ArXiv paper for explicit mentions of frameworks (e.g., `vLLM`, `lm-eval-harness`, `SGLang`, `Lighteval`).

**Assumptions Applied:**
- **Closed Models via API:** We applied a safe deduction that models with `model_type == 'closed'` (e.g., GPT-4, Claude 3.5, Gemini 1.5) cannot be run locally. Therefore, their `inference_platform` was universally set to `api` and engine to `api (unspecified)`.

---

## 5. Generation Configuration (Task 5)
We collected explicit generation parameters (`generation_temperature`, `generation_max_tokens`, `generation_top_p`).

**Methodology:**
1. Scanned all 147 benchmarks' original GitHub READMEs and ArXiv HTML papers.
2. Filtered for keywords: `temperature`, `max_tokens`, `top_p`, `greedy decoding`.
3. Extracted only explicit, stated values (e.g., "We suggest using greedy decoding", "temperature was fixed at 0"). 
4. Benchmarks that relied on "bring your own predictions" (e.g., PubMedQA) or whose papers omitted decoding parameters were strictly left blank.

**Assumptions Applied:**
- **LM-Eval-Harness Default:** The EleutherAI `lm-eval-harness` calculates log-likelihoods for multiple-choice tasks without temperature, and defaults to `greedy=True` (`temperature=0.0`) for generative tasks unless overridden. For benchmarks strictly verified to use `lm-eval-harness` (e.g., AfroBench), we applied `generation_temperature = 0.0`.

---

## 6. Quality Assurance (URL Validation)
Before finalizing the dataset, a multi-threaded URL validator was run across both CSVs:
- Validated HTTP status codes (ignoring false-positive 403s from anti-bot protections on ArXiv/Cloudflare).
- Hunted down and replaced dead links (404s), such as resolving moved GitHub repositories (e.g., `HiTZ/BasqueGLUE` -> `orai-nlp/BasqueGLUE`) or finding mirrors for dead DOI links.
- Filled in 53 entirely blank benchmark source links by cross-referencing benchmark names with active ArXiv/GitHub links.
