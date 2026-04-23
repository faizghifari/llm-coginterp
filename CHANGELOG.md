# Changelog

All notable changes to the LLM Benchmarks dataset.

---

## [Unreleased]

### Added
- EmoBench results
- EQ-Bench results

### Fixed
- Blank `score` column: All previously blank rows now have scores filled in

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
