# Benchmark Completeness Report

Tracks which benchmarks in `data/benchmarks.csv` have too few result rows
(< 5, matching `verify_data.py`'s exhaustion check) to be confident the
extraction is complete, and what's known about whether more data exists.

This is a living report, not a session log — it always reflects the
*current* row counts (regenerate the counts column with
`python3 scripts/manage_data.py verify` any time this looks stale) and
each benchmark appears exactly once. For a history of what changed and
when, see `docs/CHANGELOG.md`; this file only tracks open
research/extraction leads.

**Last regenerated:** 2026-06-17, against committed state (215
benchmarks / 1096 models / 7823 results).

## Status legend
- **Has leads** — research links below point to a likely source for more data; not yet extracted.
- **Stuck** — a specific, identified blocker (no comparable models, no aggregate score, broken leaderboard, etc.).
- **Saturated** — investigated and confirmed exhausted; the benchmark's own source paper/leaderboard genuinely only has this many models.
- **Needs research** — below threshold, no investigation done yet.

## Extraction strategies for blocked benchmarks
Many low-count benchmarks have results locked in **image-only tables**
in arXiv HTML renders, where programmatic extraction fails. In rough
order of likely success:
1. Check GitHub repos for `results.csv` / `leaderboard.json` files.
2. Check HuggingFace Spaces/datasets for an interactive leaderboard or backing dataset.
3. Check the arXiv HTML rendering (`arxiv.org/html/<id>`) — usually has real tables even when the abstract page doesn't; older pre-~2023 papers may not have one, fall back to the PDF.
4. Check Papers With Code for a text-based results table.
5. If all fail, the data may only exist in an image/figure — mark **Stuck**, don't keep re-attempting the same dead end every session.

---

## Open Items (62 benchmarks below 5 rows)

| Benchmark | Rows | Status | Notes / leads |
|---|---|---|---|
| exams_v | 4 | Has leads | https://github.com/mbzuai-oryx/EXAMS-V · https://huggingface.co/datasets/MBZUAI/EXAMS-V |
| mmau | 4 | Has leads | https://github.com/MMAU-Bench/MMAU · arXiv:2410.19168 Table 3 has salmonn-13b and others; only audio-specific models (gemini-1.5-pro, gpt-4o, qwen2-audio, SALMONN variants) are in models.csv, limiting expansion. |
| air_bench | 4 | Stuck | arXiv:2402.07729 evaluates audio-specific models (SALMONN, Qwen-Audio, BLSP, PandaGPT, NExT-GPT, SpeechGPT, Macaw-LLM) not in models.csv. Adding more rows requires adding those models first. |
| xifbench | 4 | Has leads | https://arxiv.org/abs/2405.09505 · https://huggingface.co/datasets/xifbench/XIFBench |
| complexbench | 4 | Has leads | https://github.com/thu-coai/ComplexBench — verified GPT-4's two prior rows in the 2026-06-16 dupe cleanup; one was a mis-extraction from an unrelated table and was removed. Needs other models from Table 5. |
| egoschema | 4 | Has leads | https://egoschema.github.io/ (official) · https://huggingface.co/datasets/lmms-lab/egoschema · https://llm-stats.com/benchmarks/egoschema |
| global_mmlu | 4 | Has leads | arXiv:2412.04261 (Aya Expanse, 12 models, 5-shot, 23 langs) + arXiv:2509.14233 (Apertus, 10 models, 0-shot, 15 langs) — **a prior session reported expanding this to 26 rows but it's currently only 4**; likely lost the same way as the HELM/Kaggle sweep in docs/CHANGELOG.md "Data Recovery". Re-extract from these two papers rather than assuming it's done. |
| mega | 4 | Has leads | https://github.com/microsoft/Multilingual-Evaluation-of-Generative-AI-MEGA · https://huggingface.co/datasets/microsoft/MEGA |
| include | 4 | Has leads | https://github.com/nightingal3/include · https://huggingface.co/datasets/nightingal3/include |
| akata_games_2023 | 4 | Saturated | Single paper studying game-theoretic coordination (Battle of the Sexes etc.); only GPT-3/3.5/4 + Claude-2, Llama-2-70B evaluated. Not a maintained benchmark — 4 is genuinely all there is. |
| bharatbench | 4 | Has leads | arXiv:2407.10917 · https://huggingface.co/datasets/ai4bharat/BharatBench |
| cac | 4 | Has leads | arXiv:2311.09580 · https://huggingface.co/datasets/CulturaX/CAC |
| mena_bench | 4 | Has leads | arXiv:2407.10897 · https://huggingface.co/datasets/MBZUAI/MenaBench |
| xtreme | 4 | Needs research | Not yet investigated. |
| parsiNLU | 4 | Saturated | Persian NLU benchmark, low LLM-community adoption, largely superseded by broader multilingual benchmarks. Unlikely to grow. |
| StrongREJECT | 4 | Saturated | arXiv:2402.10260 evaluates exactly 4 "victim models" (GPT-4, GPT-3.5, Llama-2, Dolphin) in the human-eval dataset; Figure 3's separate 3-model eval has no aggregate table. Genuinely exhausted. |
| LegalEval-Q | 4 | Has leads | https://github.com/Legal-NLP-EkStep/legal-eval-NLP4Justice · https://huggingface.co/datasets/shivkarthik/LegalEval |
| StatEval | 4 | Has leads | https://stateval.github.io/ · arXiv:2510.09517 · https://huggingface.co/datasets/stateval/StatEval |
| RealMath | 4 | Has leads | arXiv:2505.12575 · https://github.com/eth-sri/realmath |
| FlashInfer-Bench | 4 | Saturated | Leaderboard at bench.flashinfer.ai has exactly 4 entries (gemini-2.5-pro, gpt-5, claude-opus-4.1, o3) — these are competition submissions, not a fixed eval set. Can't expand without new entries appearing. |
| ReasonBENCH | 4 | Has leads | arXiv:2512.07795 · https://huggingface.co/datasets/ReasonBENCH/ReasonBENCH |
| DarkBench | 4 | Has leads | arXiv:2503.10728 · https://openreview.net/forum?id=odjMSBSWRt · https://github.com/Jnnes/DarkBench |
| ShoppingMMLU | 4 | Has leads | arXiv:2410.20745 · https://huggingface.co/datasets/withmartian/ShoppingMMLU |
| SpiralBench | 4 | Needs research | Not yet investigated. |
| ProphetArena | 4 | Has leads | arXiv:2310.20421 · https://huggingface.co/datasets/prophets/ProphetArena |
| sportqa | 4 | Saturated | arXiv:2402.15862 evaluates exactly 4 models (Llama2-13b, PaLM2, GPT-3.5, GPT-4); no external leaderboard, GitHub repo inaccessible. |
| sportu | 4 | Has leads | https://github.com/njunlp/SportU · https://huggingface.co/datasets/njunlp/SportU |
| sportsmetrics | 4 | Has leads | arXiv:2401.02862 · https://huggingface.co/datasets/sportsmetrics/SportsMetrics |
| p_mmeval | 3 | Has leads | https://github.com/open-compass/MMEval — partially re-verified during the 2026-06-16 dupe cleanup (GPT-4o/Qwen2.5-72B fixed against arXiv:2411.09116 Table 3); Claude-3.5-Sonnet's row may be a model-identity mismatch (the paper's table only has "Claude-3.7-sonnet") — check before adding more. |
| mvbench | 3 | Has leads | https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2 · https://huggingface.co/datasets/OpenGVLab/MVBench |
| blend | 3 | Needs research | Not yet investigated. |
| xnli | 3 | Has leads | https://huggingface.co/datasets/xnli · https://llm-stats.com/benchmarks/xnli |
| mclm | 3 | Saturated | Minimal community uptake; 3 rows likely exhausts available public results. |
| ullman_tom_2023 | 3 | Saturated | Small-scale Theory of Mind probe (Ullman 2023) — a psychology probing paper, not a maintained benchmark. 3 rows is likely full realistic coverage. |
| culturevlm | 3 | Has leads | arXiv:2501.01056 · https://huggingface.co/datasets/MichaelFan/CultureVLM-Bench |
| dialogbench | 3 | Has leads | https://github.com/LLM-Evaluation-s-Always-Fatiguing/DiagGMBench |
| medal | 3 | Needs research | Not yet investigated. |
| temporalbench | 3 | Has leads | https://github.com/eigenein/temporalbench · https://huggingface.co/datasets/microsoft/TemporalBench |
| v_star | 3 | Saturated | Small-scope visual chain-of-thought benchmark; no broad leaderboard coverage found. |
| pinocchio | 3 | Has leads | arXiv:2310.05177 · https://huggingface.co/datasets/Zhengbao/Pinocchio · https://openreview.net/forum?id=9OevMUdods — moderate citation trajectory; if a re-extraction attempt fails, downgrade to Saturated. |
| nusamt | 3 | Needs research | Not yet investigated. |
| ttcw | 3 | Saturated | Limited evidence of community adoption beyond the original paper. |
| oogiri | 3 | Saturated | Japanese humor/wordplay benchmark — extremely niche and culturally specific; likely only evaluated in the original paper and one or two Japanese-language LLM papers. |
| moralbench | 3 | Has leads | https://github.com/agiresearch/MoralBench · arXiv:2406.04428 |
| HistRev | 3 | Saturated | Niche historical reasoning/revision benchmark; likely a single paper with minimal follow-up. |
| MoralMachine | 3 | Stuck | arXiv:2411.06790 evaluates 52 LLMs but **all scores are in image-only tables/graphs** — confirmed not extractable via screenshot or programmatic means. Don't remove; this is a real result, just unextractable with current tooling. |
| GaslightingBench | 3 | Has leads | Adversarial benchmark testing gaslighting behaviors. **A prior session reported expanding this to 7 rows but it's currently only 3** — likely lost the same way as global_mmlu above; check docs/CHANGELOG.md "Data Recovery" before re-doing the extraction work from scratch. |
| DeceptionBench | 3 | Needs research | Not yet investigated. |
| ChipBench | 3 | Needs research | Not yet investigated. |
| LemmaBench | 3 | Has leads | arXiv:2602.24173 · https://huggingface.co/datasets/LemmaBench/LemmaBench |
| Vericoding | 3 | Saturated | Single preprint, minimal follow-up community evaluation. |
| do_not_answer | 3 | Has leads | https://github.com/Libr-AI/do-not-answer · https://huggingface.co/datasets/LibrAI/do-not-answer |
| naturalquestions | 2 | Has leads | https://ai.google.com/research/NaturalQuestions · https://huggingface.co/datasets/google-research-datasets/natural_questions |
| xquad | 2 | Has leads | https://huggingface.co/datasets/google/xquad · aclanthology.org/2024.acl-long.595 (IndicGenBench uses XQuAD across many models) |
| americasnli | 2 | Saturated | NLI for indigenous American languages — extremely low-resource and niche; almost no systematic LLM coverage beyond a handful of multilingual fairness papers. |
| fleurs | 2 | Has leads | https://huggingface.co/datasets/google/fleurs · arXiv:2205.12446 — many speech/audio LLMs are evaluated on this but results are scattered across individual model papers, not one leaderboard. |
| MenatQA | 2 | Has leads | arXiv:2312.15920 · https://huggingface.co/datasets/iSicLab/MenatQA |
| SocialStigmaQA | 2 | Needs research | Not yet investigated. |
| jailbreakbench | 2 | Has leads | https://jailbreakbench.github.io/ (official attack/defense leaderboard) · https://github.com/JailbreakBench/jailbreakbench |
| trustllm | 2 | Needs research | Not yet investigated. |
| wmt | 1 | Stuck | WMT23/24 findings are PDF-only (aclanthology.org/2024.wmt-1.1); table data hasn't been extractable. |
| harmbench | 1 | Needs research | Not yet investigated. |

## Resolved / no longer below threshold
These were previously flagged here and have since been expanded past the
5-row threshold — kept as a one-line pointer in case the count drops
again (e.g. from a future dedup pass discarding a mis-extracted row):
`followbench` (5), `audiobench` (5), `cvqa` (5), `benchmax` (5),
`alm_bench` (5), `marco_bench_mif` (5), `flores_200` (5), `xstest` (5),
`indicgenbench` (5), `culemo` (11 — disambiguated by the 2026-06-16 dupe
cleanup's `language` fix, was wrongly read as a handful of conflicting
duplicates before that), `dialectbench` (12), `opencompass` (21 — see
the unresolved opencompass entry in notes/TODO.md; the row *count* is
fine, but the per-row metric labeling has a known unresolved issue).

## Known framework/aggregator caveat
**opencompass**: OpenCompass is a benchmark *framework/aggregator*, not
a single benchmark. Its 21 rows are very likely several distinct,
mislabeled sub-metrics per model rather than 21 independent evaluations
— same pattern already confirmed and fixed for mt-rag/vectara/crux-eval
in the 2026-06-16 dupe cleanup, but opencompass's live ranking API
wouldn't return data over plain HTTP requests (see notes/TODO.md). Don't
add more opencompass rows without first resolving that.
