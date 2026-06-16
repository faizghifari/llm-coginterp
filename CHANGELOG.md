# Changelog

All notable changes to the LLM Benchmarks dataset.

**Current totals:** 215 benchmarks, 1096 models, 8264 result entries.

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
