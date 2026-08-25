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
    # Pegasus is pretrained with gap-sentence generation specifically FOR
    # summarization and cannot be given an arbitrary instruction -- unlike
    # T5/FLAN-T5, which are deliberately kept (see the note above
    # NON_GENERATIVE_PATTERNS). Added 2026-08-25 after BigBird-Pegasus reached
    # the corpus: plain "bigbird" catches the encoder-only variants, but the
    # seq2seq one carries the Pegasus name and slipped through. Its only two
    # benchmarks were both summarization tasks, which is the tell.
    "pegasus",
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

# --- Score-redundant benchmark splits (used by
# scripts/make_text_only_copy.py, applied alongside the modality filter) ---
#
# A third axis, orthogonal to both the model-scope and the benchmark-modality
# classifiers: not "is this in scope" but "does this benchmark_id carry
# score variance that some OTHER kept benchmark_id doesn't already carry".
#
# Public leaderboards routinely publish one benchmark as several near-identical
# columns -- version-dated re-releases, difficulty/subset variants, per-language
# splits of a translated test set, or the same benchmark re-imported from a
# second source. Each enters the model x benchmark matrix as a nominally
# distinct benchmark, so a factor analysis recovers a "factor" that is one
# benchmark's identity replicated k times. Factor COUNT and general-factor
# saturation are exactly the quantities that inflates, so these are pruned
# before the matrix is built.
#
# NOT heuristic. Every entry below was decided from the full pairwise Pearson
# correlation among the family's columns over the models evaluated on both,
# audited family-by-family (2026-08-10). Families whose correlations showed
# real structure rather than redundancy were deliberately KEPT -- see
# KEPT_DESPITE_CORRELATION below, which exists so those decisions are not
# silently re-litigated by whoever extends this map next.
#
# Applied ONLY to the derived text-only copy, never to the canonical tables:
# the canonical dataset records what sources published, and dropping a
# published benchmark there would lose data. Deduplication for analysis is a
# modelling decision and belongs in the derived copy.
SCORE_REDUNDANT_BENCHMARKS = {
    # LiveCodeBench release windows: same benchmark administered at different
    # problem-release cutoffs, not distinct capability measurements.
    # mean pairwise r=0.995, worst pair r=0.987, across 45 shared models.
    # Kept: kaggle_sjmikler_livecodebench (the aggregate).
    **{f"kaggle_sjmikler_livecodebench_release_v{v}":
       "LiveCodeBench release-window split; mean pairwise r=0.995 (worst 0.987) "
       "over 45 shared models with the kept aggregate kaggle_sjmikler_livecodebench"
       for v in range(1, 7)},

    # TwitterAAE dialect splits: the AA/white split does not differentiate
    # model capability -- it is the same LM-perplexity task twice.
    # r=0.993-0.999 with each other and with the parent, across 32 shared models.
    # Kept: twitter_aae (the parent).
    "twitter_aae_aa": "TwitterAAE dialect split; r=0.993-0.999 with the kept parent twitter_aae over 32 shared models",
    "twitter_aae_white": "TwitterAAE dialect split; r=0.993-0.999 with the kept parent twitter_aae over 32 shared models",

    # GPQA few-shot/zero-shot x diamond/main-set variants: prompting regime and
    # subset do not meaningfully change model ranking. mean r=0.944, worst pair
    # r=0.915, across 46-47 shared models. The dataset already carries much
    # better-populated canonical gpqa (501 rows) and gpqa_diamond (53 rows) from
    # other sources, so these 4 Kaggle variants are pure redundancy.
    **{f"kaggle_andrewmingwang_gpqa_{shot}_{subset}_set":
       "GPQA prompting/subset variant; mean r=0.944 (worst 0.915) over 46-47 "
       "shared models; canonical gpqa/gpqa_diamond kept from better-populated sources"
       for shot in ("few_shot", "zero_shot") for subset in ("diamond", "main")},

    # MultiLoKo per-language splits: highly correlated for well-resourced
    # languages (Simplified/Traditional Mandarin r=0.989, Italian/Swedish
    # r=0.983); mean pairwise r=0.82 overall, pulled down by low-resource pairs
    # with small model overlap and correspondingly noisy correlation (e.g.
    # Khmer/Urdu n=19). Rather than elect one language as representative, keep
    # the dataset's own canonical `multiloko` -- already an avg-across-languages
    # aggregate sourced directly from the paper (arXiv:2504.10356).
    **{f"kaggle_andrewmingwang_multiloko_{lang}":
       "MultiLoKo per-language split; mean pairwise r=0.82 (0.98+ among "
       "well-resourced languages); kept the paper-sourced across-language "
       "aggregate `multiloko` (arXiv:2504.10356) instead of electing a representative"
       for lang in (
           "arabic", "bengali", "cantonese", "czech", "dutch", "english", "farsi",
           "french", "german", "hebrew", "hindi", "indonesian", "italian",
           "japanese", "khmer", "korean", "malay", "marathi", "polish",
           "portuguese", "romanian", "russian", "simplified_mandarin", "spanish",
           "swedish", "tagalog", "thai", "traditional_mandarin", "turkish",
           "urdu", "vietnamese")},

    # Cross-source duplicate rather than an artificial split -- but the effect
    # on the matrix is the same (dilutes one capability with a near-identical
    # extra column).
    "kaggle_aminmohamedmohami_mmlu":
        "Cross-source re-import of MMLU (44 rows) duplicating the canonical, far "
        "better-populated `mmlu` (468 rows)",

    # Data-integrity fix, NOT a redundancy dedup: this id is not a coherent
    # third metric. Exact per-model value matching shows it silently splices
    # scicode_main_standard scores (whole-problem solve rate, ~5% mean) for 30
    # of its 46 models and scicode_subproblem_standard scores (partial sub-step
    # credit, ~25% mean) for the other 13 -- almost certainly an artifact of
    # scraping the Kaggle overview page's "headline" number at different times
    # as the leaderboard split evolved.
    "kaggle_andrewmingwang_scicode":
        "Corrupted composite id: splices scicode_main_standard values for 30/46 "
        "models and scicode_subproblem_standard for the other 13 (scraping artifact)",

    # Stanford HELM ThaiExam: two clusters, not uniform redundancy.
    # {ONET, IC, A-Level} intercorrelate at r=0.92-0.95 (general-secondary,
    # finance-license and subject-academic exams all reward the same underlying
    # knowledge). Kept A-Level as the more discriminative representative.
    "thai_exam_onet": "ThaiExam knowledge-cluster duplicate; r=0.92-0.95 with the kept thai_exam_a_level",
    "thai_exam_ic": "ThaiExam knowledge-cluster duplicate; r=0.92-0.95 with the kept thai_exam_a_level",
}

# Families audited on the same pass and deliberately RETAINED. Documented so a
# future extension of SCORE_REDUNDANT_BENCHMARKS does not "helpfully" remove
# them: in each case the correlations are high but NOT uniform, and the spread
# is real capability variance rather than duplication.
KEPT_DESPITE_CORRELATION = {
    "scicode_{main,subproblem}_{standard,with_background}":
        "The 4 explicit SciCode split variants correlate r=0.71-0.96 -- real, but "
        "not uniform enough to treat as duplicates the way GPQA's were (floor 0.915). "
        "Only the corrupted bare `kaggle_andrewmingwang_scicode` id was removed.",
    "thai_exam_{tgat,tpat1}":
        "Aptitude/reasoning-style Thai exams; correlate only r=0.70-0.88 with the "
        "{ONET, IC, A-Level} knowledge cluster and with each other. Verified the "
        "TGAT vs A-Level gap is systematic, not noise: several general-purpose "
        "multilingual models (Command R, SeaLLM, GPT-3.5-Turbo) score 35-45 points "
        "higher on TGAT (general reasoning) than on A-Level (Thai-curriculum "
        "subject knowledge). Distinct variance -- keep.",
    "multiloko":
        "The canonical paper-sourced aggregate is KEPT; only the 31 Kaggle "
        "per-language splits were removed.",
    "flores_*, lindsea, arabicmmlu, pwc wmt/conll language pairs":
        "Natively multilingual or genuinely distinct per-language content, not "
        "parallel translations -- see notes/multilingual_duplication_audit.md.",
}

# --- Canonical metric selection (used by scripts/lib/metrics.py) ---
#
# Spelling variants of ONE metric, mapped onto a single name. Keys are already
# normalized (stripped, casefolded, internal whitespace collapsed), so plain
# case differences -- "Accuracy" vs "accuracy" -- need NO entry here; the
# normalizer handles those.
#
# Curated, never inferred. Detecting aliases from small within-model score
# differences was tried and rejected: it mislabels task FACETS as aliases
# (sibench's "cause"/"motivation"/"social intention", sotopia's "secret"/
# "social rules" all sit within a point of each other on a 0-10 scale) and it
# mislabels misfiled CONFIG names as aliases (elephant's metric_name column
# holds model configurations like "dpo-all-llama-8b", not metrics at all --
# see KNOWN_METRIC_COLUMN_DEFECTS). Add an entry only when the two names are
# the same measurement under different spelling.
METRIC_NAME_ALIASES = {
    # Verified on math_chain_of_thought: 13 models carry both spellings, mean
    # within-model difference +0.16 (sd 1.82) -- the same measurement typed two
    # ways by two importers. Merging them keeps all 146 models; treating them as
    # rivals would have discarded 56.
    "equivalent (chain of thought)": "equivalent (cot)",
    # the_pile: "bits per byte" spelled out vs abbreviated. Merging recovers 23
    # models that the coverage rule would otherwise have discarded.
    "bits per byte": "bpb",
    # Plain abbreviation. Harmless where it is the only spelling present.
    "acc": "accuracy",
    # pwc_race: the RACE paper's own shorthand for its two subsets.
    "race-h": "accuracy (high)",
    "race-m": "accuracy (middle)",
}

# Benchmarks where "metric covering the most models" picks badly, pinned by
# hand. Value is the metric name (matched after normalisation + aliasing).
CANONICAL_METRIC_OVERRIDES = {
}

# ---------------------------------------------------------------------------
# SETTLED (2026-08-25): 'accuracy' vs 'em' are NOT aliases. Do not merge them.
# ---------------------------------------------------------------------------
# 16 benchmarks carry both. It is tempting to alias them -- on multiple-choice
# tasks they look like the same construct, and merging would recover ~450
# model-cells (mmlu alone keeps 144 more models). Do not. Two findings settle it:
#
# 1. The two names are effectively SOURCE LABELS, not measurement labels.
#    'em' is Stanford HELM's metric name (HELM is the top 'em' source on 13 of
#    the 16); 'accuracy' is Open LLM Leaderboard / Papers With Code / arXiv.
#    Merging them does not merge two metrics, it merges two evaluation regimes.
#
# 2. Those regimes have almost DISJOINT model populations. Measured over the
#    derived copy, by the source that owns each column:
#        HELM    x HELM    : median  4 shared models, 33% of pairs uncomputable
#        OpenLLM x OpenLLM : median 160 shared models,  0% uncomputable
#        HELM    x OpenLLM : median  0 shared models, 87% uncomputable
#    So a merged column would be bimodal by source, with an offset that cannot
#    be estimated -- not one model in the corpus is scored BOTH ways on ANY of
#    the 16, so there is no overlap to calibrate on. And because HELM owns 138
#    columns, the same bias would repeat corpus-wide as a spurious "source
#    factor" -- exactly the artifact the analysis is designed to avoid. Worse,
#    it would LOOK like an improvement: the matrix would appear better
#    connected while the new bridges rest on an unverifiable assumption.
#
# Corollary, equally settled: do NOT pin 'accuracy' globally via
# CANONICAL_METRIC_OVERRIDES either. The coverage rule already selects
# 'accuracy' where it genuinely dominates (mmlu, truthfulqa, hellaswag,
# pubmedqa) and 'em' on the other 12. Forcing 'accuracy' everywhere would cost
# 449 further model-cells (openbookqa 120 -> 22, legalbench 90 -> 5, imdb
# 67 -> 6, medqa 99 -> 42) and would systematically evict HELM -- a
# standardised harness with documented setup and one evaluator across many
# models -- in favour of community-submitted leaderboard runs. 'accuracy' is
# the more conventional NAME; in this corpus it is not a quality signal.
#
# What this does leave in place: because different columns are owned by
# different sources, the matrix retains a near-block structure. That is a
# property of which models each leaderboard chooses to evaluate, not of this
# decision -- the metric choice only determines which block those 16 columns
# join. A BETWEEN-column source offset is benign (column standardisation
# absorbs a mean shift); a WITHIN-column bimodal split is not.

# Benchmarks whose metric_name column is defective in a way that choosing a
# canonical metric CANNOT fix -- recorded so the choice made for them is
# understood to be arbitrary, and so the underlying defect stays visible.
# Fixing these means re-extracting the source, not picking a different metric.
KNOWN_METRIC_COLUMN_DEFECTS = {
    # RESOLVED by the canonical-metric filter -- kept here as a record of what
    # the filter is actually protecting against, not as outstanding problems.
    "vectara": "RESOLVED. 'factual_consistency_rate' and 'hallucination_rate' are "
               "COMPLEMENTS (differ by +86.5 across 7 shared models, i.e. ~100-x), "
               "so averaging them averaged x with 100-x. The filter now keeps "
               "factual_consistency_rate only.",
    "pwc_lambada": "RESOLVED. 'accuracy' and 'perplexity' in one column -- "
                   "different scales AND opposite directions. The filter now keeps "
                   "accuracy only.",
}

# Benchmarks removed from the derived copy because they are a TRANSLATION of an
# original that is itself in the corpus. Kept separate from
# SCORE_REDUNDANT_BENCHMARKS because the basis of the decision differs: those
# entries rest on pairwise-correlation audits over dozens of shared models,
# these rest on what the benchmark IS. Conflating the two would let a
# construct argument borrow the correlational evidence's authority.
#
# Same rule as the earlier corpus-level translation pass (see
# docs/CHANGELOG.md): remove a translation only when the original is present
# to fall back on; where no original exists in our import, the translated
# benchmark is KEPT and consolidated instead. Cross-language *aggregates* with
# no in-corpus original (multiloko, belebele, mgsm) are therefore kept -- they
# measure multilingual transfer, which is not what the monolingual original
# measures, and they duplicate no column we hold.
TRANSLATION_DUPLICATE_BENCHMARKS = {
    "humaneval_xl": "HumanEval-X/XL is HumanEval's problem set re-prompted in 23 "
                    "natural languages; the original `humaneval` is in the corpus "
                    "(21 models). Corroborated but NOT decided by correlation: the "
                    "3 models scored on both correlate at r=0.993 (humaneval_xl "
                    "running 13.1 points lower), which is far too few shared models "
                    "to carry the decision on its own -- the construct is the basis, "
                    "the correlation only agrees with it.",
}

# Benchmarks removed from the derived copy because the column cannot be made
# interpretable by choosing a metric -- the defect is in the data itself.
# Separate from SCORE_REDUNDANT_BENCHMARKS: those columns are fine but
# duplicated; these are broken. Re-add if the source is ever re-extracted.
DEFECTIVE_BENCHMARKS = {
    "elephant": "metric_name holds MODEL CONFIGURATIONS ('DPO-All-Llama-8B', "
                "'iti-llama-70b', 'perspective-gpt-4o'), not metrics -- an "
                "extraction defect. Choosing a canonical metric keeps one "
                "arbitrary config, which is not a measurement of anything. "
                "9 models; drop until re-extracted.",
}

# --- Source-level scale conflicts (used by scripts/lib/metrics.py) ---
#
# The residue that canonical-metric selection cannot reach: one metric NAME
# covering two incompatible scoring conventions, told apart only by source.
# Value is the source_name to KEEP; every other source's rows for that
# benchmark are dropped.
#
# Detected by looking for benchmarks where two sources' score ranges do not
# overlap at all. Only act where there is a MECHANISM, not merely a gap --
# frontier-model trackers (llm-stats, Artificial Analysis) legitimately show
# higher ranges than broad leaderboards because they evaluate better models,
# and three benchmarks (wildbench, sea_exam, multipl_e) show gaps of that
# benign kind on small n. gpqa is the one case with a known mechanism.
SOURCE_SCALE_CONFLICTS = {
    # Open LLM Leaderboard v2 publishes NORMALIZED accuracy: the random-chance
    # baseline is mapped to 0 and negatives are clamped there. Its 447 rows span
    # 0.00-24.94 (median 4.36); the other 7 rows are raw accuracy from papers and
    # llm-stats and span 39.0-94.1. The ranges do not overlap, and the metric
    # name is 'accuracy' for both.
    #
    # Keep the 447. A linear rescaling of a whole column leaves correlations
    # unchanged, so normalized accuracy is a perfectly good column PROVIDED every
    # row shares the convention -- which is exactly what dropping the 7 achieves.
    # Back-transforming instead (raw = 0.75*norm + 25) would assume the formula
    # and would not undo the clamp, for the sake of 7 models out of 454.
    #
    # Documented caveat: 56 of the 447 (13%) sit exactly at 0.00, i.e. tied at
    # the clamp, so this column under-discriminates among weak models.
    "gpqa": "Open LLM Leaderboard v2",
}

# --- Anomalous individual result rows (used by scripts/lib/metrics.py) ---
#
# Single rows dropped from the DERIVED copy because the value is almost
# certainly wrong at source and cannot be repaired. Canonical data keeps them:
# it is the archive of what sources published, and we cannot prove whether the
# error was theirs or ours in transcription. Dropping only in the derived copy
# keeps the decision reversible and visible.
#
# Keyed by (benchmark_id, model_id) -- the pair that identifies the row.
# Use sparingly: a value being surprising is not grounds for removal. The bar
# is that the value is inconsistent with its own column by a margin no
# plausible model difference explains, AND that the source cannot be
# re-checked to confirm it.
ANOMALOUS_RESULT_ROWS = {
    ("kaggle_jonlipovetz_game_arena", "DeepSeek V3.2"):
        "score=3114.0 while every other model on this benchmark falls in "
        "2.97-363.72 -- 8.5x the next-highest value. Almost certainly a unit "
        "error at source. Not repairable: the Kaggle benchmark page serves no "
        "leaderboard data without authentication, and 3114 is equally "
        "consistent with a mis-scaled 311.4 or 31.14, so the intended value "
        "cannot be inferred from the column either. Verified unrecoverable "
        "2026-08-25.",
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
