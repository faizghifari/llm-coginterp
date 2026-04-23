# Music

This document categorizes benchmarks related to music knowledge, understanding, and reasoning.

## Symbolic Music Understanding

### ABC-Eval (`abceval`) [Link](https://arxiv.org/abs/2509.23350)
**Subcategory:** symbolic-music
**Level:** arXiv 2025

First open-source benchmark for symbolic music understanding using text-based ABC notation. 1,086 test samples across 10 sub-tasks spanning 3 complexity levels: basic syntax, segment-level reasoning, and sequence-level reasoning. Accuracy: GPT-5 55.02%, GPT-5-mini 53.51%, DeepSeek-reasoner 52.01%, Gemini Pro 51.25%, DeepSeek-chat 40.17%. Even the best models only reach ~55%; high-level tasks approach random-guess levels.

### WildScore (`wildscore`) [Link](https://arxiv.org/abs/2509.04744)
**Subcategory:** symbolic-music
**Level:** arXiv 2025

First in-the-wild multimodal symbolic music reasoning benchmark. 807 MCQs from authentic r/musictheory Reddit discussions (2012-2022), categorized into 5 domains: Harmony & Tonality, Rhythm & Meter, Texture, Expression & Performance, and Form. Accuracy (with image): GPT-4o-mini 68.31%, Qwen2.5-VL-72B 49.73%, Phi-3-Vision 48.82%, Gemma-3 46.34%, MiniCPM 45.90%, InternVL 39.34%, LLaVA 32.97%. Several models show image penalty (worse performance with score images than without).

### MSU-Bench (`msu_bench`) [Link](https://arxiv.org/abs/2511.20697)
**Subcategory:** symbolic-music
**Level:** arXiv 2025 (Withdrawn ICLR 2026)

First large-scale human-curated benchmark for score-level musical understanding across both textual (ABC notation) and visual (PDF) modalities. 1,800 QA pairs from 150 classical scores (Bach, Beethoven, Chopin, Debussy, Mussorgsky), organized into 4 hierarchical comprehension levels: L1-Onset Information, L2-Notation & Note, L3-Chord & Harmony, L4-Texture & Form. Textual QA accuracy: Gemini 2.5 Pro 49.44%, ChatGPT-5 47.28%, ChatGPT-5-mini 43.72%, Grok 4 42.61%, Claude Sonnet 4 42.61%, Claude Opus 4 41.28%, Qwen3-VL-235B 41.22%. Visual QA remains much harder (Claude Opus 4: 24.22%). Fine-tuning with LoRA yields substantial gains while preserving general knowledge.

## Music Knowledge & Comprehension

### ZIQI-Eval (`ziqi_eval`) [Link](https://arxiv.org/abs/2406.15885)
**Subcategory:** music-knowledge
**Level:** ACL Findings 2024

Comprehensive 14,244-entry music benchmark evaluating 16 LLMs across 10 major categories (Music Theory, Composition, Genres, Instruments, History, Aesthetics, Education, World Ethnic Music, Female Composers, Popular Music) and 56 subcategories. Also includes 200 MCQs for music continuation (generation). Comprehension F1: GPT-4 63.04, ERNIE-Bot 59.94, Claude-instant-1.2 53.50, GPT-3.5-Turbo 52.91, XuanYuan-70B 51.40 (top open-source). GPT-4 performs at average Music Ph.D. student level in breadth; human experts still outperform in specialized domains. 43.75% of models show Eurocentric bias. Generation F1: GPT-4 54.31, GPT-3.5-Turbo 31.12, ERNIE-Bot 29.52, Claude-instant-1.2 25.06, XuanYuan-70B 22.22.

## Audio-Music Understanding

### MuChoMusic (`muchomusic`) [Link](https://arxiv.org/abs/2408.01337)
**Subcategory:** audio-music-understanding
**Level:** ISMIR 2024

Multiple-choice music QA benchmark for evaluating Audio LLMs. 1,187 human-validated 4-option MCQs across 644 music tracks from MusicCaps and Song Describer Dataset. Questions span Knowledge (melody, harmony, rhythm, instrumentation, sound texture, performance, structure - 56%) and Reasoning (mood/expression, temporal relations, lyric interpretation, genre/style, cultural context, functional context - 44%). Accuracy: Qwen-Audio 51.4%, M2UGen 42.9%, SALMONN 41.8%, MuLLaMa 32.4%, MusiLingo 21.1%. Significant language bias: models ignore audio and rely on text priors; all models fail the audio attention test (performance doesn't drop when audio replaced with white noise).
