"""Shared configuration for the dataset maintenance toolkit.

Canonical paths, the duplicate-row identity key, the source-trust
hierarchy, and the model-categorization knowledge base all live here.
When one of these needs to grow (a new trusted developer, a new trusted
source domain, a new from-scratch model family) edit ONLY this file —
no other script in this toolkit should hardcode this data again.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
NOTES_DIR = REPO_ROOT / "notes"

BENCHMARKS_CSV = DATA_DIR / "benchmarks.csv"
MODELS_CSV = DATA_DIR / "models.csv"
RESULTS_CSV = DATA_DIR / "results.csv"
DUPLICATES_CSV = DATA_DIR / "results_duplicates.csv"
PENDING_BENCHMARKS_MD = NOTES_DIR / "pending_benchmarks.md"

# The columns that together define "the same evaluation" in results.csv.
# Two rows that match on every one of these are reporting on the literal
# same eval — true duplicates (either redundant or conflicting), never a
# legitimate "multiple scores per model-benchmark pair" case.
#
# Per METHODOLOGY.md "Multiple Scores per Model-Benchmark Pair": rows
# distinguished by `setup` and/or `source_url` are kept separate on
# purpose and must NOT be collapsed — hence both are part of the key.
#
# `model_id` is included alongside `model_name` because `model_name` is
# sometimes a coarse display label shared by several genuinely distinct
# model checkpoints (e.g. model_name="GPT-4" for both `gpt-4-0314` and
# `gpt-4-0613`, or "-instruct" vs "-thinking" variants under one display
# name) while `model_id` correctly disambiguates them.
#
# `language` is included because several multilingual/multi-task
# benchmarks (e.g. afrobench, irokobench, culemo) report one row per
# sub-task/sub-language using the *same* metric_name ("accuracy") for
# all of them, with `language` holding the actual sub-task label (e.g.
# "AfriMMLU", "pos", "Hindi (India)") -- without it those rows look like
# conflicting duplicates of "the same" evaluation when they're actually
# unrelated scores for different sub-tasks.
#
# Found via a 2026-06-16 audit: of ~519 "duplicate" groups flagged before
# these two columns were added to the key, ~48 were really distinct
# models sharing one model_name, and a further ~36 were really distinct
# sub-tasks/languages sharing one metric_name -- neither was a real
# duplicate. When a column is blank (common for older/simpler
# extractions) it's a no-op for this key.
RESULT_IDENTITY_KEY = [
    "model_name", "model_id", "benchmark_id", "metric_name",
    "setup", "reasoning_enabled", "num_shot_sample", "source_url", "language",
]

# Source trust hierarchy used to resolve conflicting duplicate rows (same
# evaluation, different reported score). Lower tier number wins; anything
# matching no pattern below falls into UNKNOWN_TRUST_TIER.
SOURCE_TRUST_TIERS = {
    1: [  # Official: paper authors, the benchmark's own site, model developer
        "arxiv.org", "aclanthology.org", "openreview.net",
        "anthropic.com", "openai.com", "ai.google",
        "swebench.com", "crux-eval.github.io",
        "chat.lmsys.org", "lmsys.org", "lmarena.ai",
        "raw.githubusercontent.com/lmarena",
        "raw.githubusercontent.com/ibm/mt-rag",
    ],
    2: [  # Reputable aggregators with documented methodology
        "huggingface.co/spaces/open-llm-leaderboard",
    ],
    3: [  # Third-party aggregators, methodology not independently verified
        "llm-stats.com", "vellum.ai", "emergentmind.com",
        "gorilla.cs.berkeley.edu", "artificialanalysis.ai",
        "pricepertoken.com",
    ],
}
UNKNOWN_TRUST_TIER = 4

# Developers trusted to publish proprietary or genuinely from-scratch
# models (used by `categorize-models` to flag KEEP vs needs-research).
TRUSTED_DEVELOPERS = {
    "Meta", "OpenAI", "Anthropic", "Google", "Mistral AI", "DeepSeek",
    "Alibaba Cloud", "Microsoft", "NVIDIA", "01.AI", "Zhipu AI",
    "TII UAE", "StabilityAI", "IBM Granite", "Cohere", "xAI",
    "AI21 Labs", "Baichuan", "Baidu", "Tencent", "ByteDance",
    "Moonshot AI", "Qwen Team", "Core42", "Sakana AI", "Upstage",
    "Facebook",
}

# Model-family / org name fragments indicating a genuinely from-scratch
# open-weights model, as opposed to a community fine-tune of someone
# else's base model.
FROM_SCRATCH_PATTERNS = {
    "llama", "mistral", "mixtral", "gemma", "palm", "gemini",
    "qwen", "phi", "deepseek", "yi", "pythia", "gpt-j", "gpt-neox",
    "gpt-neo", "bloom", "falcon", "dbrx", "olmo", "redpajama", "gpt-jt",
    "codegen", "smollm", "mpt", "opt", "xglm", "dolly", "internlm",
    "exaone", "jais", "granite", "glm", "chatglm", "hyperclova",
    "command-r", "grok", "aya", "minicpm", "baichuan", "starcoder",
    "santacoder", "tinyllama", "vicuna", "internvl",
}

# Keyword fragments suggesting a model name describes a fine-tune /
# alignment pass rather than identifying a from-scratch base model.
FINE_TUNE_KEYWORDS = {
    "instruct", "chat", "fine-tuned", "ft-", "-ft", "-chat", "-instruct",
    "align", "dpo", "rlhf", "orpo", "sft", "alpaca", "hermes",
    "openhermes", "zephyr", "openchat", "open-orca", "orca", "kto",
    "simpo", "ppo", "wizard", "magpie", "solar",
}

# --- Model-inclusion SCOPE axis (used by `classify_scope` in categorize.py) ---
#
# Orthogonal to everything above: FROM_SCRATCH_PATTERNS/FINE_TUNE_KEYWORDS
# answer "is this a traceable base model or an untraceable fine-tune?".
# The three sets below answer a different question -- "is this a
# generative LLM (or an LLM-backed multimodal model) at all?", per
# METHODOLOGY.md's "Model Inclusion Criteria". A model can KEEP on one
# axis and REMOVE on the other (e.g. mT5-XXL: KEEP under
# categorize_model() since it has a clear model_family, but REMOVE under
# classify_scope() since T5 is encoder-decoder, not a decoder-only
# generative LLM).

# Architecture-name fragments identifying non-generative models: encoder-
# only classifiers, embedding-only models, bare pure-vision/CV models, or
# encoder-decoder (T5-family) models -- none of which are decoder-only
# generative LLMs capable of handling arbitrary prompts. Checked ONLY
# after VLM_ALM_ALLOWLIST -- several legitimate VLMs (e.g.
# BLIP2-FLAN-T5-XXL) have architecture names that contain these fragments
# as their LLM backbone's name, not as the model's own identity.
#
# Note: plain T5/mT5/FLAN-T5 (encoder-decoder but still a general-purpose
# generative LM that can be prompted with arbitrary instructions) are
# deliberately NOT in this set -- only T5-derived architectures fine-tuned
# into a narrow, non-generative head (ST5 = sentence-embedding, monoT5 =
# binary relevance reranker) are.
NON_GENERATIVE_PATTERNS = {
    "bert", "roberta", "albert", "electra", "deberta",
    "clip", "resnet", "vit", "convnext", "knowledge review",
    "monot5", "st5",
    "sentence-transformer", "sbert", "simcse",
}

# Narrow single-modality task families (dedicated TTS/ASR/MT-only, music
# generation) -- cannot be prompted with arbitrary text even though some
# internally use an LLM-like text component.
NARROW_TASK_PATTERNS = {
    "seamlessm4t", "whisper", "wav2vec", "musicgen",
    "m2ugen", "mullama", "musilingo", "nllb", "madlad", "vall-e",
}

# Explicit allow-list, checked BEFORE NON_GENERATIVE_PATTERNS /
# NARROW_TASK_PATTERNS: models that add a modality encoder ON TOP OF an
# LLM backbone (policy-compliant per METHODOLOGY.md) but whose family/name
# happens to contain an exclude-pattern fragment naming their backbone
# (e.g. "BLIP2-FLAN-T5-XXL" contains "t5"). Extend THIS set, never the
# exclude patterns, when a new VLM/ALM family is added to the dataset.
VLM_ALM_ALLOWLIST = {
    "llava", "blip2", "blip-2", "instructblip", "qwen2-audio", "qwen-audio",
    "phi-3-vision", "phi-3.5-vision", "llama-3.2-vision",
    "gemini", "glm4v", "glm-4v", "palmyra-vision",
    "internlm2+vit", "taco (llama3-8b",
}

# --- Benchmark-modality axis (used by `classify_benchmark_modality` in
# scripts/lib/modality.py) ---
#
# A different axis again: not "is this model in scope" but "does THIS
# BENCHMARK require the model to consume or produce a non-text modality
# (vision, audio, video) to do the task at all". benchmarks.csv has no
# dedicated modality column -- this is inferred from free-text category/
# subcategory/task_type/task_types/domain/benchmark_name/description
# fields left over from several merged source-CSV schemas.
#
# Word-fragment set, matched on WHOLE-WORD boundaries only (never raw
# substring -- "vision" as a substring would false-positive on e.g.
# "Historical Revisionism Detection"). Deliberately a little
# over-inclusive (some fragments currently match zero rows) -- cheap
# future-proofing, matching this repo's existing pattern-list style.
NON_TEXT_MODALITY_PATTERNS = {
    "vision", "visual", "image", "video", "multimodal", "vqa",
    "audio", "speech", "spoken", "acoustic", "ocr", "asr", "tts",
    "photo", "diagram", "chart", "screenshot", "music", "song", "sound",
}

# Explicit allow-list, checked BEFORE NON_TEXT_MODALITY_PATTERNS:
# benchmark_ids where the source data's own category/subcategory tag is
# misleading -- confirmed text-only on inspection of the benchmark's own
# description despite matching an exclude pattern. Extend THIS set,
# never the exclude patterns, when a new false positive is found.
TEXT_MODALITY_ALLOWLIST = {
    "abceval": "Evaluates text-based ABC notation only, no audio/image input despite category=audio/speech",
    "ziqi_eval": "Pure text-QA music-knowledge benchmark, no audio/image input despite category=audio/speech",
    "aci_bench": "Stanford MedHELM task is text-transcript-to-clinical-note summarization; model never receives audio despite description mentioning 'spoken medical dialogue'",
}

# The mirror-image override: benchmark_ids confirmed NON-text on inspection
# whose metadata carries no matchable keyword at all (blank/sparse rows) or
# is outright mislabeled (e.g. cmmMU's benchmark_name says "Chinese
# Multilingual MMLU" but its rows cite arXiv:2401.11944 -- CMMMU, the
# *multimodal* benchmark, scored on Qwen-VL/Yi-VL/GPT-4V). Found by a
# 2026-07-17 audit of the text_only copy: cross-referencing each surviving
# benchmark's result-row models and source papers. Checked BEFORE the
# pattern match, after TEXT_MODALITY_ALLOWLIST (the two sets must stay
# disjoint).
NON_TEXT_MODALITY_OVERRIDES = {
    "alm_bench": "ALM-Bench (arXiv:2411.16508): image-based cultural VQA across 100 languages; rows are VLMs (GLM-4V, InternVL2); metadata only says 'alignment'",
    "exams_v": "EXAMS-V (arXiv:2403.10378): multimodal multilingual exam benchmark with images; rows include GPT-4V/Gemini Pro Vision; metadata says only 'exam questions'",
    "mmau": "MMAU (arXiv:2410.19168): Massive Multi-Task AUDIO Understanding; rows are audio LMs (Qwen2-Audio, SALMONN); subcategory mislabeled 'australian-languages'",
    "mmt_bench": "MMT-Bench (arXiv:2404.16006, OpenGVLab): massive multitask MULTIMODAL benchmark; rows are VLMs (GPT-4V, LLaVA-NeXT); category mislabeled 'machine-translation'",
    "cmmMU": "Actually CMMMU (arXiv:2401.11944), Chinese multi-discipline MULTIMODAL understanding; rows are VLMs (Qwen-VL, Yi-VL, GPT-4V); benchmark_name mislabeled 'Chinese Multilingual MMLU'",
    "temporalbench": "TemporalBench (arXiv:2410.10818): fine-grained temporal understanding for multimodal VIDEO models; rows include Qwen2-VL",
    "voice_jailbreak_attacks": "Stanford HELM Audio leaderboard: voice-mode (audio-input) jailbreak attacks; rows include Qwen2-Audio; 'voice' not in pattern set",
    "pwc_next_qa_open_ended_videoqa": "NExT-QA open-ended VideoQA; rows are video LMs (Video-ChatGPT, MovieChat); 'VideoQA' is one token so no word-boundary pattern matches",
    "pwc_salmon": "SALMon: acoustic/speech language-model suite (metrics: Speaker/Room/Background Consistency); PwC task mislabeled plain 'Language Modelling'",
    "kaggle_aminmohamedmohami_video_qa": "Kaggle VideoQA leaderboard; name 'VideoQA' is one token so no word-boundary pattern matches",
    "kaggle_sjmikler_mathvista_testmini": "MathVista testmini (arXiv:2310.02255): image-based math VQA; the curated 'mathvista' twin is already pattern-removed",
    "kaggle_andrewmingwang_parsebench": "ParseBench (arXiv:2604.08538): document-image parsing/OCR for agents (tables, charts, visual grounding); metadata otherwise blank",
    "longshot": "LongShOTBench (arXiv:2512.16978): omni-modal reasoning in long VIDEOS (vision + speech + ambient audio); rows are VLMs (LLaVA, InternVL, Qwen3-VL); subcategory misleadingly says 'long document reasoning'",
}

# --- Multilingual benchmark-cluster discovery (used by
# scripts/lib/benchmark_clusters.find_language_clusters) ---
#
# Language-name / ISO-code fragments that, when found as a trailing
# `_<suffix>` on a benchmark_id, HINT that it's a per-language sibling of
# a shared-stem benchmark family (e.g. "kaggle_vijitsingh1_mgsm_chinese"
# hints at stem "kaggle_vijitsingh1_mgsm" + language "chinese"). This is
# heuristic-only -- a hint for human review, not a translation-vs-distinct
# classifier. Multi-word entries (e.g. "simplified_mandarin") are matched
# against the LAST TWO underscore-separated tokens before falling back to
# the last one, so both single- and two-word suffixes are caught.
LANGUAGE_SUFFIX_HINTS = {
    "english", "en", "chinese", "zh", "simplified_mandarin", "traditional_mandarin",
    "cantonese", "mandarin", "japanese", "ja", "korean", "ko",
    "thai", "th", "vietnamese", "vi", "indonesian", "id", "malay", "ms",
    "tagalog", "khmer", "km", "burmese", "my", "hindi", "hi", "bengali", "bn",
    "urdu", "ur", "marathi", "mr", "tamil", "ta", "telugu", "te", "swahili", "sw",
    "yoruba", "arabic", "ar", "hebrew", "he", "farsi", "persian", "fa",
    "turkish", "tr", "russian", "ru", "german", "de", "french", "fr",
    "spanish", "es", "italian", "it", "portuguese", "pt", "dutch", "nl",
    "polish", "pl", "czech", "cs", "romanian", "ro", "swedish", "sv",
    "greek", "el", "ukrainian", "uk", "finnish", "fi", "hungarian", "hu",
}
