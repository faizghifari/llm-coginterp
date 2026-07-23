# Text-only derived copy

Generated 2026-07-17 by `scripts/make_text_only_copy.py` from `data/*.csv`. Regenerate any time by rerunning that script -- this directory is gitignored, not hand-maintained.

Every benchmark classified NON_TEXT by `scripts.lib.modality.classify_benchmark_modality` (requires the model to consume/produce image, audio, or video content) was cascade-removed, along with its `results.csv` rows and any model left with zero remaining results.

Totals: benchmarks 627 -> 506, models 2028 -> 1682, results 19078 -> 17054.

## Removed benchmarks by category

- Visual Question Answering (VQA): 29
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
