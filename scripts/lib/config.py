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
