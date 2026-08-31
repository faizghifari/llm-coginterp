# Text-only derived copy

Generated 2026-09-01 by `scripts/make_text_only_copy.py` from `data/*.csv`.

**This directory is tracked in git, but it is NOT hand-maintained.** It is fully reproducible from the canonical tables plus the knowledge bases in `scripts/lib/config.py`; `python3 scripts/make_text_only_copy.py --check` asserts exactly that. Never hand-edit these CSVs -- encode the decision in `config.py` and regenerate, or the edit is silently lost on the next run.

Three orthogonal removals are applied, then cascades (benchmark -> its `results.csv` rows -> any model left with zero results):

1. **Modality** -- every benchmark classified NON_TEXT by `scripts.lib.modality.classify_benchmark_modality` (requires the model to consume/produce image, audio, or video content).
2. **Score redundancy** -- every benchmark id in `config.SCORE_REDUNDANT_BENCHMARKS`: near-duplicate columns whose variance is already carried by a kept benchmark, decided from pairwise-correlation audits rather than from names.
3. **Canonical metric** -- one metric per benchmark (`scripts.lib.metrics`), so a matrix cell measures one thing. Without this the aggregation averages incommensurable metrics, and because which metric a model received is largely decided by which leaderboard scored it, part of each column's variance would be a function of its source rather than of capability.

Totals: benchmarks 624 -> 456, models 2014 -> 1618, results 19030 -> 13251.

## Removed benchmarks by category

- Visual Question Answering (VQA): 27
- Multimodal: 13
- vision/multimodal: 12
- multimodal: 7
- Audio/Speech: 6
- audio/speech: 6
- multilingual: 6
- general_knowledge: 4
- Alignment & Safety: 4
- General Knowledge: 3
- Visual Question Answering: 3
- Factual Inconsistency Detection in Chart Captioning: 3
- : 3
- Question Answering: 2
- cross-cultural: 2
- alignment/safety: 2
- Visual Reasoning: 2
- Chart Question Answering: 2
- Multimodal Reasoning: 2
- Cognitive Science: 1
- Language Modelling: 1
- Machine Translation: 1
- Multilingual, Crosslingual, Cultural: 1
- Humor/Creativity: 1
- Natural Language Visual Grounding: 1
- academic: 1
- Reasoning: 1
- domain-specific: 1
- reasoning: 1

## Allow-list exceptions (config.TEXT_MODALITY_ALLOWLIST)

Benchmarks whose category/description matched a non-text pattern but were confirmed text-only on inspection:

- `abceval`: Evaluates text-based ABC notation only, no audio/image input despite category=audio/speech
- `ziqi_eval`: Pure text-QA music-knowledge benchmark, no audio/image input despite category=audio/speech
- `aci_bench`: Stanford MedHELM task is text-transcript-to-clinical-note summarization; model never receives audio despite description mentioning 'spoken medical dialogue'

## Non-text overrides (config.NON_TEXT_MODALITY_OVERRIDES)

Benchmarks confirmed non-text on inspection despite keyword-free or mislabeled metadata (found by the 2026-07-17 audit):

- `alm_bench`: ALM-Bench (arXiv:2411.16508): image-based cultural VQA across 100 languages; rows are VLMs (GLM-4V, InternVL2); metadata only says 'alignment'
- `exams_v`: EXAMS-V (arXiv:2403.10378): multimodal multilingual exam benchmark with images; rows include GPT-4V/Gemini Pro Vision; metadata says only 'exam questions'
- `mmau`: MMAU (arXiv:2410.19168): Massive Multi-Task AUDIO Understanding; rows are audio LMs (Qwen2-Audio, SALMONN); subcategory mislabeled 'australian-languages'
- `mmt_bench`: MMT-Bench (arXiv:2404.16006, OpenGVLab): massive multitask MULTIMODAL benchmark; rows are VLMs (GPT-4V, LLaVA-NeXT); category mislabeled 'machine-translation'
- `cmmMU`: Actually CMMMU (arXiv:2401.11944), Chinese multi-discipline MULTIMODAL understanding; rows are VLMs (Qwen-VL, Yi-VL, GPT-4V); benchmark_name mislabeled 'Chinese Multilingual MMLU'
- `temporalbench`: TemporalBench (arXiv:2410.10818): fine-grained temporal understanding for multimodal VIDEO models; rows include Qwen2-VL
- `voice_jailbreak_attacks`: Stanford HELM Audio leaderboard: voice-mode (audio-input) jailbreak attacks; rows include Qwen2-Audio; 'voice' not in pattern set
- `pwc_next_qa_open_ended_videoqa`: NExT-QA open-ended VideoQA; rows are video LMs (Video-ChatGPT, MovieChat); 'VideoQA' is one token so no word-boundary pattern matches
- `pwc_salmon`: SALMon: acoustic/speech language-model suite (metrics: Speaker/Room/Background Consistency); PwC task mislabeled plain 'Language Modelling'
- `kaggle_aminmohamedmohami_video_qa`: Kaggle VideoQA leaderboard; name 'VideoQA' is one token so no word-boundary pattern matches
- `kaggle_sjmikler_mathvista_testmini`: MathVista testmini (arXiv:2310.02255): image-based math VQA; the curated 'mathvista' twin is already pattern-removed
- `kaggle_andrewmingwang_parsebench`: ParseBench (arXiv:2604.08538): document-image parsing/OCR for agents (tables, charts, visual grounding); metadata otherwise blank
- `longshot`: LongShOTBench (arXiv:2512.16978): omni-modal reasoning in long VIDEOS (vision + speech + ambient audio); rows are VLMs (LLaVA, InternVL, Qwen3-VL); subcategory misleadingly says 'long document reasoning'

## Benchmarks removed (49) — redundant, translated, or defective

From `config.SCORE_REDUNDANT_BENCHMARKS` (near-duplicate columns, decided from full pairwise Pearson correlation over the models evaluated on both), `config.TRANSLATION_DUPLICATE_BENCHMARKS` (translations of an in-corpus original, decided on construct), and `config.DEFECTIVE_BENCHMARKS`:

- `elephant`: metric_name holds MODEL CONFIGURATIONS ('DPO-All-Llama-8B', 'iti-llama-70b', 'perspective-gpt-4o'), not metrics -- an extraction defect. Choosing a canonical metric keeps one arbitrary config, which is not a measurement of anything. 9 models; drop until re-extracted.
- `humaneval_xl`: HumanEval-X/XL is HumanEval's problem set re-prompted in 23 natural languages; the original `humaneval` is in the corpus (21 models). Corroborated but NOT decided by correlation: the 3 models scored on both correlate at r=0.993 (humaneval_xl running 13.1 points lower), which is far too few shared models to carry the decision on its own -- the construct is the basis, the correlation only agrees with it.
- `kaggle_aminmohamedmohami_mmlu`: Cross-source re-import of MMLU (44 rows) duplicating the canonical, far better-populated `mmlu` (468 rows)
- `kaggle_andrewmingwang_gpqa_few_shot_diamond_set`: GPQA prompting/subset variant; mean r=0.944 (worst 0.915) over 46-47 shared models; canonical gpqa/gpqa_diamond kept from better-populated sources
- `kaggle_andrewmingwang_gpqa_few_shot_main_set`: GPQA prompting/subset variant; mean r=0.944 (worst 0.915) over 46-47 shared models; canonical gpqa/gpqa_diamond kept from better-populated sources
- `kaggle_andrewmingwang_gpqa_zero_shot_diamond_set`: GPQA prompting/subset variant; mean r=0.944 (worst 0.915) over 46-47 shared models; canonical gpqa/gpqa_diamond kept from better-populated sources
- `kaggle_andrewmingwang_gpqa_zero_shot_main_set`: GPQA prompting/subset variant; mean r=0.944 (worst 0.915) over 46-47 shared models; canonical gpqa/gpqa_diamond kept from better-populated sources
- `kaggle_andrewmingwang_multiloko_arabic`: MultiLoKo per-language split; mean pairwise r=0.82 (0.98+ among well-resourced languages); kept the paper-sourced across-language aggregate `multiloko` (arXiv:2504.10356) instead of electing a representative
- `kaggle_andrewmingwang_multiloko_bengali`: MultiLoKo per-language split; mean pairwise r=0.82 (0.98+ among well-resourced languages); kept the paper-sourced across-language aggregate `multiloko` (arXiv:2504.10356) instead of electing a representative
- `kaggle_andrewmingwang_multiloko_cantonese`: MultiLoKo per-language split; mean pairwise r=0.82 (0.98+ among well-resourced languages); kept the paper-sourced across-language aggregate `multiloko` (arXiv:2504.10356) instead of electing a representative
- `kaggle_andrewmingwang_multiloko_czech`: MultiLoKo per-language split; mean pairwise r=0.82 (0.98+ among well-resourced languages); kept the paper-sourced across-language aggregate `multiloko` (arXiv:2504.10356) instead of electing a representative
- `kaggle_andrewmingwang_multiloko_dutch`: MultiLoKo per-language split; mean pairwise r=0.82 (0.98+ among well-resourced languages); kept the paper-sourced across-language aggregate `multiloko` (arXiv:2504.10356) instead of electing a representative
- `kaggle_andrewmingwang_multiloko_english`: MultiLoKo per-language split; mean pairwise r=0.82 (0.98+ among well-resourced languages); kept the paper-sourced across-language aggregate `multiloko` (arXiv:2504.10356) instead of electing a representative
- `kaggle_andrewmingwang_multiloko_farsi`: MultiLoKo per-language split; mean pairwise r=0.82 (0.98+ among well-resourced languages); kept the paper-sourced across-language aggregate `multiloko` (arXiv:2504.10356) instead of electing a representative
- `kaggle_andrewmingwang_multiloko_french`: MultiLoKo per-language split; mean pairwise r=0.82 (0.98+ among well-resourced languages); kept the paper-sourced across-language aggregate `multiloko` (arXiv:2504.10356) instead of electing a representative
- `kaggle_andrewmingwang_multiloko_german`: MultiLoKo per-language split; mean pairwise r=0.82 (0.98+ among well-resourced languages); kept the paper-sourced across-language aggregate `multiloko` (arXiv:2504.10356) instead of electing a representative
- `kaggle_andrewmingwang_multiloko_hebrew`: MultiLoKo per-language split; mean pairwise r=0.82 (0.98+ among well-resourced languages); kept the paper-sourced across-language aggregate `multiloko` (arXiv:2504.10356) instead of electing a representative
- `kaggle_andrewmingwang_multiloko_hindi`: MultiLoKo per-language split; mean pairwise r=0.82 (0.98+ among well-resourced languages); kept the paper-sourced across-language aggregate `multiloko` (arXiv:2504.10356) instead of electing a representative
- `kaggle_andrewmingwang_multiloko_indonesian`: MultiLoKo per-language split; mean pairwise r=0.82 (0.98+ among well-resourced languages); kept the paper-sourced across-language aggregate `multiloko` (arXiv:2504.10356) instead of electing a representative
- `kaggle_andrewmingwang_multiloko_italian`: MultiLoKo per-language split; mean pairwise r=0.82 (0.98+ among well-resourced languages); kept the paper-sourced across-language aggregate `multiloko` (arXiv:2504.10356) instead of electing a representative
- `kaggle_andrewmingwang_multiloko_japanese`: MultiLoKo per-language split; mean pairwise r=0.82 (0.98+ among well-resourced languages); kept the paper-sourced across-language aggregate `multiloko` (arXiv:2504.10356) instead of electing a representative
- `kaggle_andrewmingwang_multiloko_khmer`: MultiLoKo per-language split; mean pairwise r=0.82 (0.98+ among well-resourced languages); kept the paper-sourced across-language aggregate `multiloko` (arXiv:2504.10356) instead of electing a representative
- `kaggle_andrewmingwang_multiloko_korean`: MultiLoKo per-language split; mean pairwise r=0.82 (0.98+ among well-resourced languages); kept the paper-sourced across-language aggregate `multiloko` (arXiv:2504.10356) instead of electing a representative
- `kaggle_andrewmingwang_multiloko_malay`: MultiLoKo per-language split; mean pairwise r=0.82 (0.98+ among well-resourced languages); kept the paper-sourced across-language aggregate `multiloko` (arXiv:2504.10356) instead of electing a representative
- `kaggle_andrewmingwang_multiloko_marathi`: MultiLoKo per-language split; mean pairwise r=0.82 (0.98+ among well-resourced languages); kept the paper-sourced across-language aggregate `multiloko` (arXiv:2504.10356) instead of electing a representative
- `kaggle_andrewmingwang_multiloko_polish`: MultiLoKo per-language split; mean pairwise r=0.82 (0.98+ among well-resourced languages); kept the paper-sourced across-language aggregate `multiloko` (arXiv:2504.10356) instead of electing a representative
- `kaggle_andrewmingwang_multiloko_portuguese`: MultiLoKo per-language split; mean pairwise r=0.82 (0.98+ among well-resourced languages); kept the paper-sourced across-language aggregate `multiloko` (arXiv:2504.10356) instead of electing a representative
- `kaggle_andrewmingwang_multiloko_romanian`: MultiLoKo per-language split; mean pairwise r=0.82 (0.98+ among well-resourced languages); kept the paper-sourced across-language aggregate `multiloko` (arXiv:2504.10356) instead of electing a representative
- `kaggle_andrewmingwang_multiloko_russian`: MultiLoKo per-language split; mean pairwise r=0.82 (0.98+ among well-resourced languages); kept the paper-sourced across-language aggregate `multiloko` (arXiv:2504.10356) instead of electing a representative
- `kaggle_andrewmingwang_multiloko_simplified_mandarin`: MultiLoKo per-language split; mean pairwise r=0.82 (0.98+ among well-resourced languages); kept the paper-sourced across-language aggregate `multiloko` (arXiv:2504.10356) instead of electing a representative
- `kaggle_andrewmingwang_multiloko_spanish`: MultiLoKo per-language split; mean pairwise r=0.82 (0.98+ among well-resourced languages); kept the paper-sourced across-language aggregate `multiloko` (arXiv:2504.10356) instead of electing a representative
- `kaggle_andrewmingwang_multiloko_swedish`: MultiLoKo per-language split; mean pairwise r=0.82 (0.98+ among well-resourced languages); kept the paper-sourced across-language aggregate `multiloko` (arXiv:2504.10356) instead of electing a representative
- `kaggle_andrewmingwang_multiloko_tagalog`: MultiLoKo per-language split; mean pairwise r=0.82 (0.98+ among well-resourced languages); kept the paper-sourced across-language aggregate `multiloko` (arXiv:2504.10356) instead of electing a representative
- `kaggle_andrewmingwang_multiloko_thai`: MultiLoKo per-language split; mean pairwise r=0.82 (0.98+ among well-resourced languages); kept the paper-sourced across-language aggregate `multiloko` (arXiv:2504.10356) instead of electing a representative
- `kaggle_andrewmingwang_multiloko_traditional_mandarin`: MultiLoKo per-language split; mean pairwise r=0.82 (0.98+ among well-resourced languages); kept the paper-sourced across-language aggregate `multiloko` (arXiv:2504.10356) instead of electing a representative
- `kaggle_andrewmingwang_multiloko_turkish`: MultiLoKo per-language split; mean pairwise r=0.82 (0.98+ among well-resourced languages); kept the paper-sourced across-language aggregate `multiloko` (arXiv:2504.10356) instead of electing a representative
- `kaggle_andrewmingwang_multiloko_urdu`: MultiLoKo per-language split; mean pairwise r=0.82 (0.98+ among well-resourced languages); kept the paper-sourced across-language aggregate `multiloko` (arXiv:2504.10356) instead of electing a representative
- `kaggle_andrewmingwang_multiloko_vietnamese`: MultiLoKo per-language split; mean pairwise r=0.82 (0.98+ among well-resourced languages); kept the paper-sourced across-language aggregate `multiloko` (arXiv:2504.10356) instead of electing a representative
- `kaggle_andrewmingwang_scicode`: Corrupted composite id: splices scicode_main_standard values for 30/46 models and scicode_subproblem_standard for the other 13 (scraping artifact)
- `kaggle_sjmikler_livecodebench_release_v1`: LiveCodeBench release-window split; mean pairwise r=0.995 (worst 0.987) over 45 shared models with the kept aggregate kaggle_sjmikler_livecodebench
- `kaggle_sjmikler_livecodebench_release_v2`: LiveCodeBench release-window split; mean pairwise r=0.995 (worst 0.987) over 45 shared models with the kept aggregate kaggle_sjmikler_livecodebench
- `kaggle_sjmikler_livecodebench_release_v3`: LiveCodeBench release-window split; mean pairwise r=0.995 (worst 0.987) over 45 shared models with the kept aggregate kaggle_sjmikler_livecodebench
- `kaggle_sjmikler_livecodebench_release_v4`: LiveCodeBench release-window split; mean pairwise r=0.995 (worst 0.987) over 45 shared models with the kept aggregate kaggle_sjmikler_livecodebench
- `kaggle_sjmikler_livecodebench_release_v5`: LiveCodeBench release-window split; mean pairwise r=0.995 (worst 0.987) over 45 shared models with the kept aggregate kaggle_sjmikler_livecodebench
- `kaggle_sjmikler_livecodebench_release_v6`: LiveCodeBench release-window split; mean pairwise r=0.995 (worst 0.987) over 45 shared models with the kept aggregate kaggle_sjmikler_livecodebench
- `thai_exam_ic`: ThaiExam knowledge-cluster duplicate; r=0.92-0.95 with the kept thai_exam_a_level
- `thai_exam_onet`: ThaiExam knowledge-cluster duplicate; r=0.92-0.95 with the kept thai_exam_a_level
- `twitter_aae_aa`: TwitterAAE dialect split; r=0.993-0.999 with the kept parent twitter_aae over 32 shared models
- `twitter_aae_white`: TwitterAAE dialect split; r=0.993-0.999 with the kept parent twitter_aae over 32 shared models

## Audited but deliberately KEPT (config.KEPT_DESPITE_CORRELATION)

Families whose columns correlate highly but *not uniformly* -- the spread is real capability variance, not duplication. Recorded so these decisions are not silently re-litigated:

- `scicode_{main,subproblem}_{standard,with_background}`: The 4 explicit SciCode split variants correlate r=0.71-0.96 -- real, but not uniform enough to treat as duplicates the way GPQA's were (floor 0.915). Only the corrupted bare `kaggle_andrewmingwang_scicode` id was removed.
- `thai_exam_{tgat,tpat1}`: Aptitude/reasoning-style Thai exams; correlate only r=0.70-0.88 with the {ONET, IC, A-Level} knowledge cluster and with each other. Verified the TGAT vs A-Level gap is systematic, not noise: several general-purpose multilingual models (Command R, SeaLLM, GPT-3.5-Turbo) score 35-45 points higher on TGAT (general reasoning) than on A-Level (Thai-curriculum subject knowledge). Distinct variance -- keep.
- `multiloko`: The canonical paper-sourced aggregate is KEPT; only the 31 Kaggle per-language splits were removed.
- `flores_*, lindsea, arabicmmlu, pwc wmt/conll language pairs`: Natively multilingual or genuinely distinct per-language content, not parallel translations -- see notes/multilingual_duplication_audit.md.

## Canonical metric selection (91 contested benchmarks)

1408 result rows dropped, 705 model-cells lost. Selection order: normalise (case/whitespace) -> alias (`config.METRIC_NAME_ALIASES`) -> override (`config.CANONICAL_METRIC_OVERRIDES`) -> most models covered.

| benchmark | chosen | why | models kept | models lost | candidates |
|---|---|---|---:|---:|---|
| `mmlu` | accuracy | coverage | 311 | 144 | accuracy(311), em(144) |
| `truthfulqa` | accuracy | coverage | 325 | 83 | accuracy(325), em(67), mc1(18), % info(11), % true(11), mc2(5), % true (gpt-judge)(4), bleu(4), bleurt(4), rouge(4) |
| `math_regular` | accuracy | coverage | 103 | 66 | accuracy(103), equivalent(69) |
| `bbq` | bbq accuracy | coverage | 85 | 42 | bbq accuracy(85), em(42) |
| `boolq` | em | coverage | 67 | 42 | em(67), accuracy(44) |
| `medqa` | em | coverage | 99 | 41 | em(99), accuracy(42) |
| `hellaswag` | accuracy | coverage | 308 | 32 | accuracy(308), em(32) |
| `openbookqa` | em | coverage | 120 | 21 | em(120), accuracy(22) |
| `triviaqa` | em | coverage | 21 | 18 | em(21), accuracy(17), f1(1) |
| `simpleqa` | score | coverage | 31 | 17 | score(31), accuracy(17) |
| `ifeval` | accuracy | coverage | 447 | 14 | accuracy(447), prompt-level accuracy(10), inst-level loose-accuracy(4), inst-level strict-accuracy(4), prompt-level loose-accuracy(4), prompt-level strict-accuracy(4) |
| `mbpp` | accuracy | coverage | 62 | 14 | accuracy(62), pass@1(14) |
| `facts_grounding` | score | coverage | 41 | 13 | score(41), accuracy(15) |
| `facts_search` | score | coverage | 31 | 13 | score(31), accuracy(15) |
| `pubmedqa` | accuracy | coverage | 35 | 13 | accuracy(35), em(13) |
| `aime25` | score | coverage | 44 | 11 | score(44), accuracy(11) |
| `math500` | score | coverage | 39 | 11 | score(39), accuracy(11) |
| `nusax` | macro f1 score | coverage | 21 | 10 | macro f1 score(21), accuracy(10) |
| `pwc_record` | em | coverage | 13 | 10 | em(13), f1(13) |
| `thaiexam` | em | coverage | 42 | 10 | em(42), accuracy(10) |
| `mmlu_prox` | score | coverage | 47 | 9 | score(47), accuracy(9) |
| `multiloko` | score | coverage | 50 | 8 | score(50), accuracy(8) |
| `arabicmmlu` | em | coverage | 38 | 7 | em(38), accuracy(7) |
| `medmcqa` | em | coverage | 13 | 6 | em(13), accuracy(6) |
| `the_pile` | bpb | coverage | 55 | 6 | bpb(55), test perplexity(6) |
| `imdb` | em | coverage | 67 | 5 | em(67), accuracy(5) |
| `legalbench` | em | coverage | 90 | 5 | em(90), accuracy(5) |
| `xcopa` | em | coverage | 21 | 5 | em(21), accuracy(5) |
| `mtrag` | rb_agg | coverage | 10 | 4 | rb_agg(10), rb_llm(9), rl_f(9), accuracy(8) |
| `raft` | em | coverage | 67 | 3 | em(67), ade(3), avg(3), b77(3), nis(3), ose(3), over(3), sot(3), sri(3), tai(3), tc(3), teh(3), tos(3) |
| `xnli` | em | coverage | 21 | 3 | em(21), accuracy(3) |
| `flores_200` | spbleu; chrf | coverage | 4 | 2 | spbleu; chrf(4), bleu(2) |
| `naturalquestions` | em | coverage | 19 | 2 | em(19), accuracy(2) |
| `pwc_race` | accuracy (high) | coverage | 12 | 2 | accuracy (high)(12), accuracy (middle)(12), accuracy(2) |
| `xquad` | squad macro-averaged f1 score | coverage | 21 | 2 | squad macro-averaged f1 score(21), f1; em(2) |
| `DarkBench` | brand bias score | coverage | 1 | 1 | brand bias score(1), user retention score(1) |
| `harmbench` | lm evaluated safety score | coverage | 85 | 1 | lm evaluated safety score(85), attack_success_rate(1) |
| `pwc_apps` | introductory pass@1 | coverage | 5 | 1 | introductory pass@1(5), competition pass@1(4), interview pass@1(4), competition pass@any(1) |
| `pwc_gem_xsum` | rouge-2 | coverage | 4 | 1 | rouge-2(4), bleu score(1) |
| `pwc_multinli` | matched | coverage | 7 | 1 | matched(7), mismatched(6) |
| `pwc_multirc` | f1 | coverage | 18 | 1 | f1(18), em(4) |
| `pwc_svamp` | execution accuracy | coverage | 13 | 1 | execution accuracy(13), accuracy(1) |
| `summarization_cnndm` | rouge-2 | coverage | 48 | 1 | rouge-2(48), rouge-1(8), rouge-l(8) |
| `swiss_legal_bench` | accuracy | coverage | 2 | 1 | accuracy(2), correct rate(1) |
| `tab_fact` | accuracy | coverage | 15 | 1 | accuracy(15), test(1), val(1) |
| `xstest` | lm evaluated safety score | coverage | 85 | 1 | lm evaluated safety score(85), refusal_rate_on_safe_prompts(1) |
| `criticbench` | corracc_alldomains_avg | coverage | 12 | 0 | corracc_alldomains_avg(12), critf1_alldomains_avg(12), genacc_alldomains_avg(12) |
| `emobench` | emotional application | coverage | 12 | 0 | emotional application(12), emotional understanding(12) |
| `engibench` | level1_original_accuracy | coverage | 1 | 0 | level1_original_accuracy(1), level1_perturbed_accuracy(1), level2_original_accuracy(1), level2_perturbed_accuracy(1), level3_rubric_score(1) |
| `followbench` | accuracy | coverage | 4 | 0 | accuracy(4), csl_1_to_5(1) |
| `kernelbench` | fast_1_l1 | coverage | 1 | 0 | fast_1_l1(1), fast_1_l2(1), fast_1_l3(1), fast_1_l2_feedback(1) |
| `maliciousinstruct` | asr_greedy_nosys | coverage | 11 | 0 | asr_greedy_nosys(11), asr_greedy_wsys(11), asr_variatedall(11), asr_variatedtemp(11), asr_variatedtopk(11), asr_variatedtopp(11) |
| `pwc_abstractive_text_summarization_from_fanpage` | bertscore | coverage | 1 | 0 | bertscore(1), rouge-1(1), rouge-2(1), rouge-l(1) |
| `pwc_abstractive_text_summarization_from_il_post` | rouge-1 | coverage | 2 | 0 | rouge-1(2), bertscore(1), rouge-2(1), rouge-l(1) |
| `pwc_anli_test` | a2 | coverage | 18 | 0 | a2(18), a3(18), a1(14) |
| `pwc_arxiv_hep_th_citation_graph` | rouge-1 | coverage | 3 | 0 | rouge-1(3), rouge-2(2), rouge-l(2) |
| `pwc_arxiv_summarization_dataset` | rouge-1 | coverage | 1 | 0 | rouge-1(1), rouge-l(1) |
| `pwc_asqp` | f1 (r15) | coverage | 3 | 0 | f1 (r15)(3), f1 (r16)(1) |
| `pwc_cc3m_tagmask` | accuracy | coverage | 3 | 0 | accuracy(3), f1(3), precision(3), recall(3) |
| `pwc_citesum` | rouge-1 | coverage | 1 | 0 | rouge-1(1), rouge-2(1), rouge-l(1) |
| `pwc_cnn_daily_mail_anonymized` | rouge-1 | coverage | 1 | 0 | rouge-1(1), rouge-2(1), rouge-l(1) |
| `pwc_codecontests` | test set pass@1 | coverage | 4 | 0 | test set pass@1(4), val set pass@1(3), test set pass@5(1), val set pass@5(1) |
| `pwc_commitmentbank` | accuracy | coverage | 15 | 0 | accuracy(15), f1(5) |
| `pwc_conala` | bleu | coverage | 3 | 0 | bleu(3), exact match accuracy(2) |
| `pwc_django` | accuracy | coverage | 2 | 0 | accuracy(2), bleu score(2) |
| `pwc_gigaword` | rouge-1 | coverage | 4 | 0 | rouge-1(4), rouge-2(4), rouge-l(4) |
| `pwc_kilt_eli5` | f1 | coverage | 1 | 0 | f1(1), kilt-f1(1), kilt-rl(1), r-prec(1), recall@5(1), rouge-l(1) |
| `pwc_kvret` | entity f1 | coverage | 2 | 0 | entity f1(2), bleu(1) |
| `pwc_lambada` | accuracy | coverage | 26 | 0 | accuracy(26), perplexity(9) |
| `pwc_mteb` | accuracy | coverage | 7 | 0 | accuracy(7), spearman correlation(4) |
| `pwc_muserc` | average f1 | coverage | 6 | 0 | average f1(6), em(6) |
| `pwc_newsqa` | em | coverage | 8 | 0 | em(8), f1(8) |
| `pwc_openapi_completion_refined` | correctness, avg., % | coverage | 3 | 0 | correctness, avg., %(3), correctness, max., %(3), validness, avg., %(3), validness, max., %(3) |
| `pwc_openwebtext` | eval_perplexity | coverage | 6 | 0 | eval_perplexity(6), eval_loss(4) |
| `pwc_peerqa` | alignscore | coverage | 6 | 0 | alignscore(6), prometheus-2 answer correctness(6), rouge-l(6) |
| `pwc_penn_treebank_word_level` | test perplexity | coverage | 3 | 0 | test perplexity(3), validation perplexity(1) |
| `pwc_pubmed` | rouge-1 | coverage | 2 | 0 | rouge-1(2), rouge-2(1), rouge-l(1) |
| `pwc_rcb` | accuracy | coverage | 6 | 0 | accuracy(6), average f1(6) |
| `pwc_reddit_tifu` | rouge-1 | coverage | 1 | 0 | rouge-1(1), rouge-2(1), rouge-l(1) |
| `pwc_rucos` | average f1 | coverage | 6 | 0 | average f1(6), em(6) |
| `pwc_safim` | algorithmic | coverage | 15 | 0 | algorithmic(15), api(15), average(15), control(15) |
| `pwc_samsum` | rouge-l | coverage | 3 | 0 | rouge-l(3), rouge-1(2), rouge-2(1) |
| `pwc_squad1_1_dev` | em | coverage | 2 | 0 | em(2), f1(2) |
| `pwc_tasd` | f1 (r16) | coverage | 3 | 0 | f1 (r16)(3), f1 (r15)(1) |
| `pwc_vietnews` | rouge-1 | coverage | 1 | 0 | rouge-1(1), rouge-2(1), rouge-l(1) |
| `pwc_wikitext_2` | test perplexity | coverage | 9 | 0 | test perplexity(9), validation perplexity(1) |
| `pwc_wits` | bertscore | coverage | 1 | 0 | bertscore(1), rouge-1(1), rouge-2(1), rouge-l(1) |
| `sibench` | cause | coverage | 6 | 0 | cause(6), communicative strategy(6), emotional attitude(6), motivation(6), reply cot(6), reply direct(6), social intention(6), win rate(6) |
| `sotopia` | believability | coverage | 4 | 0 | believability(4), financial benefits(4), goal completion(4), knowledge(4), relationship(4), secret(4), social rules(4) |
| `summarization_xsum` | rouge-2 | coverage | 46 | 0 | rouge-2(46), rouge-1(1) |
| `vectara` | factual_consistency_rate | coverage | 7 | 0 | factual_consistency_rate(7), hallucination_rate(7) |

## Anomalous rows dropped (config.ANOMALOUS_RESULT_ROWS)

Individual values inconsistent with their own column by a margin no plausible model difference explains, and not repairable from source. Canonical data keeps them; only this derived copy drops them:

- `DeepSeek V3.2` on `kaggle_jonlipovetz_game_arena`: score=3114.0 while every other model on this benchmark falls in 2.97-363.72 -- 8.5x the next-highest value. Almost certainly a unit error at source. Not repairable: the Kaggle benchmark page serves no leaderboard data without authentication, and 3114 is equally consistent with a mis-scaled 311.4 or 31.14, so the intended value cannot be inferred from the column either. Verified unrecoverable 2026-08-25.

## Source-level scale conflicts (config.SOURCE_SCALE_CONFLICTS)

One metric *name* covering two incompatible scoring conventions, separable only by source. Correlations are unchanged by a linear rescaling of a whole column, so a normalised column is fine -- provided every row shares the convention, which is what these removals enforce:

- `gpqa`: kept 447 rows from **Open LLM Leaderboard v2**, dropped 54 on an incompatible scale

## Benchmarks removed as structurally defective (config.DEFECTIVE_BENCHMARKS)

Columns that no metric choice can make interpretable -- the defect is in the data. Re-add if the source is re-extracted:

- `elephant`: metric_name holds MODEL CONFIGURATIONS ('DPO-All-Llama-8B', 'iti-llama-70b', 'perspective-gpt-4o'), not metrics -- an extraction defect. Choosing a canonical metric keeps one arbitrary config, which is not a measurement of anything. 9 models; drop until re-extracted.

### Metric-column defects on record (config.KNOWN_METRIC_COLUMN_DEFECTS)

Choosing a canonical metric does NOT fix these -- they need a source-level re-extraction. The choice made for them is arbitrary:

- `pwc_lambada`: RESOLVED. 'accuracy' and 'perplexity' in one column -- different scales AND opposite directions. The filter now keeps accuracy only.
- `vectara`: RESOLVED. 'factual_consistency_rate' and 'hallucination_rate' are COMPLEMENTS (differ by +86.5 across 7 shared models, i.e. ~100-x), so averaging them averaged x with 100-x. The filter now keeps factual_consistency_rate only.
