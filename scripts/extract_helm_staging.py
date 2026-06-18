#!/usr/bin/env python3
"""
HELM Data Extraction — Staging File Generator
==============================================
Fetches per-benchmark leaderboard data from all HELM sub-projects via their
public GCS APIs and writes properly-schema-aligned staging CSV files:

  data/staging_helm_benchmarks.csv  — same 37-column schema as benchmarks.csv
  data/staging_helm_models.csv      — same 24-column schema as models.csv
  data/staging_helm_results.csv     — same 38-column schema as results.csv

These staging files are *appended to* (not overwritten) on re-runs, so they
accumulate data across runs.  Duplicate rows (same benchmark_id + model_name +
metric_name + source_url) are de-duplicated at write time.

Usage:
  python3 scripts/extract_helm_staging.py              # all sub-projects
  python3 scripts/extract_helm_staging.py --project classic
  python3 scripts/extract_helm_staging.py --project lite safety medhelm
  python3 scripts/extract_helm_staging.py --dry-run    # fetch & print, no write

After generating staging files, review them, then run:
  python3 scripts/manage_data.py verify
to check the main data files' integrity (the staging files are separate and
must be merged manually after review).
"""
import argparse
import csv
import gzip
import json
import re
import sys
import time
import urllib.request
from io import StringIO
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap sys.path so this script runs from any cwd
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

# ---------------------------------------------------------------------------
# Sub-project registry
# ---------------------------------------------------------------------------
# Each entry: (project_id, display_name, base_url, version, path_prefix, site_url, category_default)
# path_prefix: "releases" for RELEASE-type, "runs" for SUITE-type (audio/image2struct/reasoning)
HELM_PROJECTS = {
    "classic": {
        "name": "HELM Classic",
        "base_url": "https://storage.googleapis.com/crfm-helm-public/classic/benchmark_output/",
        "version": "v0.4.0",
        "path_type": "releases",
        "site_url": "https://crfm.stanford.edu/helm/classic/latest/",
        "year": "2023",
        "venue": "NeurIPS 2023",
        "source_name": "Stanford HELM Classic",
        "organization": "Stanford CRFM",
    },
    "lite": {
        "name": "HELM Lite",
        "base_url": "https://storage.googleapis.com/crfm-helm-public/gzip/lite/benchmark_output/",
        "version": "v1.13.0",
        "path_type": "releases",
        "site_url": "https://crfm.stanford.edu/helm/lite/latest/",
        "year": "2024",
        "venue": "arXiv 2024",
        "source_name": "Stanford HELM Lite",
        "organization": "Stanford CRFM",
    },
    "safety": {
        "name": "HELM Safety",
        "base_url": "https://storage.googleapis.com/crfm-helm-public/gzip/safety/benchmark_output/",
        "version": "v1.17.0",
        "path_type": "releases",
        "site_url": "https://crfm.stanford.edu/helm/safety/latest/",
        "year": "2024",
        "venue": "arXiv 2024",
        "source_name": "Stanford HELM Safety",
        "organization": "Stanford CRFM",
    },
    "medhelm": {
        "name": "MedHELM",
        "base_url": "https://storage.googleapis.com/crfm-helm-public/gzip/medhelm/benchmark_output/",
        "version": "v4.0.0",
        "path_type": "releases",
        "site_url": "https://crfm.stanford.edu/helm/medhelm/latest/",
        "year": "2024",
        "venue": "arXiv 2024",
        "source_name": "Stanford MedHELM",
        "organization": "Stanford CRFM",
    },
    "thaiexam": {
        "name": "HELM ThaiExam",
        "base_url": "https://storage.googleapis.com/crfm-helm-public/gzip/thaiexam/benchmark_output/",
        "version": "v1.2.0",
        "path_type": "releases",
        "site_url": "https://crfm.stanford.edu/helm/thaiexam/latest/",
        "year": "2024",
        "venue": "arXiv 2024",
        "source_name": "Stanford HELM ThaiExam",
        "organization": "Stanford CRFM",
    },
    "torr": {
        "name": "HELM TORR",
        "base_url": "https://storage.googleapis.com/crfm-helm-public/gzip/torr/benchmark_output/",
        "version": "v1.0.0",
        "path_type": "releases",
        "site_url": "https://crfm.stanford.edu/helm/torr/latest/",
        "year": "2024",
        "venue": "arXiv 2024",
        "source_name": "Stanford HELM TORR",
        "organization": "Stanford CRFM",
    },
    "ewok": {
        "name": "HELM EWoK",
        "base_url": "https://storage.googleapis.com/crfm-helm-public/ewok/benchmark_output/",
        "version": "v1.0.0",
        "path_type": "releases",
        "site_url": "https://crfm.stanford.edu/helm/ewok/latest/",
        "year": "2024",
        "venue": "arXiv 2024",
        "source_name": "Stanford HELM EWoK",
        "organization": "Stanford CRFM",
    },
    "finance": {
        "name": "HELM Finance",
        "base_url": "https://storage.googleapis.com/crfm-helm-public/finance/benchmark_output/",
        "version": "v1.0.0",
        "path_type": "releases",
        "site_url": "https://crfm.stanford.edu/helm/finance/latest/",
        "year": "2024",
        "venue": "arXiv 2024",
        "source_name": "Stanford HELM Finance",
        "organization": "Stanford CRFM",
    },
    "seahelm": {
        "name": "SEA-HELM",
        "base_url": "https://storage.googleapis.com/crfm-helm-public/seahelm/benchmark_output/",
        "version": "v1.2.0",
        "path_type": "releases",
        "site_url": "https://crfm.stanford.edu/helm/seahelm/latest/",
        "year": "2024",
        "venue": "arXiv 2024",
        "source_name": "Stanford SEA-HELM",
        "organization": "Stanford CRFM / SEACrowd",
    },
    "arabic": {
        "name": "HELM Arabic",
        "base_url": "https://nlp.stanford.edu/helm/arabic/benchmark_output/",
        "version": "v2.1.0",
        "path_type": "releases",
        "site_url": "https://crfm.stanford.edu/helm/arabic/latest/",
        "year": "2024",
        "venue": "arXiv 2024",
        "source_name": "Stanford HELM Arabic",
        "organization": "Stanford CRFM / MBZUAI",
    },
    "audio": {
        "name": "HELM Audio",
        "base_url": "https://storage.googleapis.com/crfm-helm-public/gzip/audio/benchmark_output/",
        "version": "v1.0.0",
        "path_type": "runs",
        "site_url": "https://crfm.stanford.edu/helm/audio/latest/",
        "year": "2025",
        "venue": "arXiv 2025",
        "source_name": "Stanford HELM Audio",
        "organization": "Stanford CRFM",
    },
    "image2struct": {
        "name": "HELM Image2Struct",
        "base_url": "https://storage.googleapis.com/crfm-helm-public/gzip/image2struct/benchmark_output/",
        "version": "v1.0.2",
        "path_type": "runs",
        "site_url": "https://crfm.stanford.edu/helm/image2struct/latest/",
        "year": "2024",
        "venue": "arXiv 2024",
        "source_name": "Stanford HELM Image2Struct",
        "organization": "Stanford CRFM",
    },
    "reasoning": {
        "name": "HELM Reasoning",
        "base_url": "https://storage.googleapis.com/crfm-helm-public/gzip/reasoning/benchmark_output/",
        "version": "v0.0.1",
        "path_type": "runs",
        "site_url": "https://crfm.stanford.edu/helm/reasoning/latest/",
        "year": "2025",
        "venue": "arXiv 2025",
        "source_name": "Stanford HELM Reasoning",
        "organization": "Stanford CRFM",
    },
}

# ---------------------------------------------------------------------------
# Category mapping (HELM group_id → our benchmark category)
# ---------------------------------------------------------------------------
CATEGORY_MAP = {
    # Classic — QA
    "boolq": "General Knowledge", "narrative_qa": "General Knowledge",
    "natural_qa_closedbook": "General Knowledge", "natural_qa_openbook_longans": "General Knowledge",
    "quac": "General Knowledge", "openbookqa": "General Knowledge",
    "truthful_qa": "Alignment & Safety",
    "mmlu": "General Knowledge", "wikifact": "General Knowledge",
    "msmarco_regular": "General Knowledge", "msmarco_trec": "General Knowledge",
    # Classic — Reasoning
    "hellaswag": "Reasoning", "babi_qa": "Reasoning", "dyck_language": "Reasoning",
    "synthetic_reasoning": "Reasoning", "synthetic_reasoning_natural": "Reasoning",
    "lsat_qa": "Reasoning",
    # Classic — Math/Code
    "gsm": "Math", "math_regular": "Math", "math_chain_of_thought": "Math",
    "code_humaneval": "Coding",
    # Classic — Summarization/Language Modeling
    "summarization_cnndm": "Reasoning", "summarization_xsum": "Reasoning",
    "ice": "General Knowledge", "the_pile": "General Knowledge",
    "twitter_aae": "General Knowledge", "twitter_aae_aa": "General Knowledge",
    "twitter_aae_white": "General Knowledge",
    # Classic — Classification/Detection
    "imdb": "General Knowledge", "raft": "General Knowledge",
    "civil_comments": "Alignment & Safety", "blimp": "Cognitive Science",
    # Classic — Safety/Bias
    "bbq": "Alignment & Safety", "bold": "Alignment & Safety",
    "real_toxicity_prompts": "Alignment & Safety",
    "disinformation_reiteration": "Alignment & Safety",
    "disinformation_wedging": "Alignment & Safety",
    "copyright_text": "Alignment & Safety", "copyright_code": "Alignment & Safety",
    # Classic — Legal/Data
    "legal_support": "Reasoning", "entity_data_imputation": "General Knowledge",
    "entity_matching": "General Knowledge",
    # Lite extras
    "legalbench": "Reasoning", "med_qa": "Medical",
    "wmt_14": "Machine Translation",
    # Safety
    "harm_bench": "Alignment & Safety", "harm_bench_gcg_transfer": "Alignment & Safety",
    "simple_safety_tests": "Alignment & Safety", "xstest": "Alignment & Safety",
    "anthropic_red_team": "Alignment & Safety",
    # MedHELM
    "medcalc_bench": "Medical", "clear": "Medical", "mtsamples_replicate": "Medical",
    "medec": "Medical", "ehrshot": "Medical", "head_qa": "Medical",
    "medbullets": "Medical", "med_mcqa": "Medical", "medalign": "Medical",
    "dischargeme": "Medical", "aci_bench": "Medical", "mtsamples_procedures": "Medical",
    "mimic_rrs": "Medical", "mimic_bhc": "Medical", "chw_care_plan": "Medical",
    "medication_qa": "Medical", "starr_patient_instructions": "Medical",
    "med_dialog": "Medical", "medi_qa": "Medical", "mental_health": "Medical",
    "pubmed_qa": "Medical", "ehr_sql": "Medical", "medhallu": "Medical",
    "n2c2_ct_matching": "Medical", "race_based_med": "Medical",
    "mimiciv_billing_code": "Medical",
    # ThaiExam
    "thai_exam": "General Knowledge", "thai_exam_onet": "General Knowledge",
    "thai_exam_ic": "General Knowledge", "thai_exam_tgat": "General Knowledge",
    "thai_exam_tpat1": "General Knowledge", "thai_exam_a_level": "General Knowledge",
    # TORR (table/structured data reasoning)
    "fin_qa": "Reasoning", "numeric_nlg": "Reasoning", "qtsumm": "Reasoning",
    "scigen": "Reasoning", "tab_fact": "Reasoning",
    "tablebench_data_analysis": "Reasoning", "tablebench_fact_checking": "Reasoning",
    "tablebench_numerical_reasoning": "Reasoning",
    "turl_col_type": "General Knowledge", "wikitq": "Reasoning",
    # EWoK (world knowledge)
    "ewok": "Cognitive Science", "ewok_agent_properties": "Cognitive Science",
    "ewok_material_dynamics": "Cognitive Science", "ewok_material_properties": "Cognitive Science",
    "ewok_physical_dynamics": "Cognitive Science", "ewok_physical_interactions": "Cognitive Science",
    "ewok_physical_relations": "Cognitive Science", "ewok_quantitative_properties": "Cognitive Science",
    "ewok_social_interactions": "Cognitive Science", "ewok_social_properties": "Cognitive Science",
    "ewok_social_relations": "Cognitive Science", "ewok_spatial_relations": "Cognitive Science",
    # Finance
    "financebench": "Reasoning", "banking77": "General Knowledge",
    # SEA-HELM (multilingual SE Asian)
    "tydiqa": "Multilingual, Crosslingual, Cultural",
    "xquad_vi": "Multilingual, Crosslingual, Cultural",
    "xquad_th": "Multilingual, Crosslingual, Cultural",
    "indicqa": "Multilingual, Crosslingual, Cultural",
    "nusax": "Multilingual, Crosslingual, Cultural",
    "uitvsfc": "Multilingual, Crosslingual, Cultural",
    "wisesight": "Multilingual, Crosslingual, Cultural",
    "indicsentiment": "Multilingual, Crosslingual, Cultural",
    "mlhsd": "Alignment & Safety", "vihsd": "Alignment & Safety",
    "thaitoxicitytweets": "Alignment & Safety",
    "flores_en_id": "Machine Translation", "flores_en_vi": "Machine Translation",
    "flores_en_th": "Machine Translation", "flores_en_ta": "Machine Translation",
    "flores_id_en": "Machine Translation", "flores_vi_en": "Machine Translation",
    "flores_th_en": "Machine Translation", "flores_ta_en": "Machine Translation",
    "indonli": "Multilingual, Crosslingual, Cultural",
    "xnli_vi": "Multilingual, Crosslingual, Cultural",
    "xnli_th": "Multilingual, Crosslingual, Cultural",
    "indicxnli": "Multilingual, Crosslingual, Cultural",
    "xcopa_id": "Multilingual, Crosslingual, Cultural",
    "xcopa_vi": "Multilingual, Crosslingual, Cultural",
    "xcopa_th": "Multilingual, Crosslingual, Cultural",
    "xcopa_ta": "Multilingual, Crosslingual, Cultural",
    "lindsea_syntax_minimal_pairs_id": "Multilingual, Crosslingual, Cultural",
    "lindsea_pragmatics_presuppositions_id": "Multilingual, Crosslingual, Cultural",
    "lindsea_pragmatics_scalar_implicatures_id": "Multilingual, Crosslingual, Cultural",
    # Arabic
    "alghafa": "Multilingual, Crosslingual, Cultural",
    "arabic_mmlu": "Multilingual, Crosslingual, Cultural",
    "arabic_exams": "Multilingual, Crosslingual, Cultural",
    "madinah_qa": "Multilingual, Crosslingual, Cultural",
    "aratrust": "Alignment & Safety",
    "alrage": "Multilingual, Crosslingual, Cultural",
    "mbzuai_human_translated_arabic_mmlu": "Multilingual, Crosslingual, Cultural",
    # Audio
    "covost2": "Machine Translation", "vocal_sound": "Audio/Speech",
    "multilingual_librispeech": "Audio/Speech", "fleurs": "Multilingual, Crosslingual, Cultural",
    "fleurs_fairness": "Alignment & Safety", "audiocaps": "Audio/Speech",
    "voxceleb2": "Audio/Speech", "speech_robust_bench": "Audio/Speech",
    "meld_audio": "Cognitive Science", "air_bench_chat_knowledge": "General Knowledge",
    "air_bench_chat_reasoning": "Reasoning", "air_bench_foundation": "General Knowledge",
    "mutox": "Alignment & Safety", "mustard": "Humor/Creativity",
    "voice_jailbreak_attacks": "Alignment & Safety",
    "ami": "Audio/Speech", "librispeech": "Audio/Speech",
    "librispeech_fairness": "Alignment & Safety",
    "parade": "Multilingual, Crosslingual, Cultural", "corebench": "General Knowledge",
    # Image2Struct
    "image2latex": "Multimodal", "image2latex_equation": "Multimodal",
    "image2latex_table": "Multimodal", "image2latex_algorithm": "Multimodal",
    "image2latex_plot": "Multimodal", "image2latex_wild": "Multimodal",
    "image2webpage": "Multimodal", "image2webpage_css": "Multimodal",
    "image2webpage_html": "Multimodal", "image2webpage_javascript": "Multimodal",
    "image2webpage_wild": "Multimodal", "image2musicsheet": "Multimodal",
    "image2struct_wild": "Multimodal",
    # Reasoning
    "math500": "Math", "aime": "Math", "aime25": "Math",
}

# Known aggregate/category group IDs to skip (they link to sub-groups, not models)
AGGREGATE_GROUP_IDS = {
    "core_scenarios", "all_scenarios", "targeted_evaluations", "question_answering",
    "information_retrieval", "summarization", "sentiment_analysis", "toxicity_detection",
    "miscellaneous_text_classification", "language", "knowledge", "reasoning",
    "harms", "efficiency", "calibration", "ablation_in_context",
    "ablation_multiple_choice", "ablation_prompts", "robustness_contrast_sets",
    # Operational/meta benchmarks — not model quality measures
    "synthetic_efficiency",
    # Sub-project aggregate groups
    "safety_scenarios", "medhelm_scenarios", "thai_scenarios", "table_scenarios",
    "world_knowledge_scenarios", "seahelm_nlu", "seahelm_nlg", "seahelm_nlr",
    "seahelm_lindsea", "arabic_scenarios", "audio_scenarios",
    "auditory_perception", "emotion_detection", "robustness", "multilinguality",
    "safety", "fairness", "bias", "clinical_decision_support",
    "clinical_note_generation", "patient_communication", "medical_research",
    "administration_and_workflow",
    # Image2struct aggregate
    "core_scenarios",
    # Reasoning aggregate
    "reasoning_scenarios",
}

# Known metrics that should NOT be scaled ×100 (they are absolute, not 0-1 percentages)
NO_SCALE_METRICS = {
    "Bits/byte", "bits/byte",
    "BPB",          # Bits Per Byte — absolute language-modeling entropy (0.8–4 range)
    "BLEURT", "BERTScore", "Roberta-large",
    "Perplexity", "perplexity",
    "# eval",       # Synthetic efficiency — count of instances, not a quality score
    # Elo ratings
    "Elo", "elo",
}

# ---------------------------------------------------------------------------
# HELM Classic model → developer/family metadata
# (populated by hand for models known at HELM Classic time; all others default
#  to empty strings — fill in later via categorize-models or manual edit)
# ---------------------------------------------------------------------------
MODEL_METADATA = {
    # OpenAI
    "ada (350M)":           ("OpenAI", "GPT-3",   "350M",  "closed"),
    "babbage (1.3B)":       ("OpenAI", "GPT-3",   "1.3B",  "closed"),
    "curie (6.7B)":         ("OpenAI", "GPT-3",   "6.7B",  "closed"),
    "davinci (175B)":       ("OpenAI", "GPT-3",   "175B",  "closed"),
    "code-cushman-001 (12B)": ("OpenAI", "Codex",  "12B",  "closed"),
    "code-davinci-002":     ("OpenAI", "Codex",   "",      "closed"),
    "text-ada-001":         ("OpenAI", "GPT-3",   "350M",  "closed"),
    "text-babbage-001":     ("OpenAI", "GPT-3",   "1.3B",  "closed"),
    "text-curie-001":       ("OpenAI", "GPT-3",   "6.7B",  "closed"),
    "text-davinci-002":     ("OpenAI", "GPT-3",   "175B",  "closed"),
    "text-davinci-003":     ("OpenAI", "GPT-3",   "175B",  "closed"),
    "gpt-3.5-turbo-0301":   ("OpenAI", "GPT-3.5", "",      "closed"),
    "gpt-3.5-turbo-0613":   ("OpenAI", "GPT-3.5", "",      "closed"),
    # Anthropic
    "Anthropic-LM v4-s3 (52B)": ("Anthropic", "Claude", "52B", "closed"),
    # Meta / LLaMA
    "LLaMA (7B)":   ("Meta", "LLaMA",   "7B",  "open"),
    "LLaMA (13B)":  ("Meta", "LLaMA",   "13B", "open"),
    "LLaMA (30B)":  ("Meta", "LLaMA",   "30B", "open"),
    "LLaMA (65B)":  ("Meta", "LLaMA",   "65B", "open"),
    "Llama 2 (7B)": ("Meta", "Llama 2", "7B",  "open"),
    "Llama 2 (13B)": ("Meta", "Llama 2", "13B", "open"),
    "Llama 2 (70B)": ("Meta", "Llama 2", "70B", "open"),
    # Stanford / Alpaca
    "Alpaca (7B)":  ("Stanford", "Alpaca", "7B", "open"),
    # Vicuna
    "Vicuna v1.3 (7B)":  ("LMSYS", "Vicuna", "7B",  "open"),
    "Vicuna v1.3 (13B)": ("LMSYS", "Vicuna", "13B", "open"),
    # BigScience
    "BLOOM (176B)":  ("BigScience", "BLOOM", "176B", "open"),
    # EleutherAI
    "GPT-J (6B)":     ("EleutherAI", "GPT-J",   "6B",  "open"),
    "GPT-NeoX (20B)": ("EleutherAI", "GPT-NeoX", "20B", "open"),
    "Pythia (6.9B)":  ("EleutherAI", "Pythia",  "6.9B", "open"),
    "Pythia (12B)":   ("EleutherAI", "Pythia",  "12B",  "open"),
    # Tsinghua
    "GLM (130B)": ("Tsinghua KEG Lab", "GLM", "130B", "open"),
    # Cohere
    "Cohere Command beta (6.1B)":     ("Cohere", "Command", "6.1B",  "closed"),
    "Cohere Command beta (52.4B)":    ("Cohere", "Command", "52.4B", "closed"),
    "Cohere large v20220720 (13.1B)": ("Cohere", "Cohere",  "13.1B", "closed"),
    "Cohere medium v20220720 (6.1B)": ("Cohere", "Cohere",  "6.1B",  "closed"),
    "Cohere medium v20221108 (6.1B)": ("Cohere", "Cohere",  "6.1B",  "closed"),
    "Cohere small v20220720 (410M)":  ("Cohere", "Cohere",  "410M",  "closed"),
    "Cohere xlarge v20220609 (52.4B)": ("Cohere", "Cohere", "52.4B", "closed"),
    "Cohere xlarge v20221108 (52.4B)": ("Cohere", "Cohere", "52.4B", "closed"),
    # AI21 Labs
    "J1-Large v1 (7.5B)":      ("AI21 Labs", "Jurassic-1", "7.5B",  "closed"),
    "J1-Grande v1 (17B)":      ("AI21 Labs", "Jurassic-1", "17B",   "closed"),
    "J1-Grande v2 beta (17B)": ("AI21 Labs", "Jurassic-1", "17B",   "closed"),
    "J1-Jumbo v1 (178B)":      ("AI21 Labs", "Jurassic-1", "178B",  "closed"),
    "Jurassic-2 Large (7.5B)": ("AI21 Labs", "Jurassic-2", "7.5B",  "closed"),
    "Jurassic-2 Grande (17B)": ("AI21 Labs", "Jurassic-2", "17B",   "closed"),
    "Jurassic-2 Jumbo (178B)": ("AI21 Labs", "Jurassic-2", "178B",  "closed"),
    # TII (Falcon)
    "Falcon (7B)":           ("Technology Innovation Institute", "Falcon", "7B",  "open"),
    "Falcon (40B)":          ("Technology Innovation Institute", "Falcon", "40B", "open"),
    "Falcon-Instruct (7B)":  ("Technology Innovation Institute", "Falcon", "7B",  "open"),
    "Falcon-Instruct (40B)": ("Technology Innovation Institute", "Falcon", "40B", "open"),
    # Mistral
    "Mistral v0.1 (7B)": ("Mistral AI", "Mistral", "7B", "open"),
    # MosaicML / MPT
    "MPT (30B)":          ("MosaicML", "MPT", "30B", "open"),
    "MPT-Instruct (30B)": ("MosaicML", "MPT", "30B", "open"),
    # Aleph Alpha
    "Luminous Base (13B)":    ("Aleph Alpha", "Luminous", "13B", "closed"),
    "Luminous Extended (30B)": ("Aleph Alpha", "Luminous", "30B", "closed"),
    "Luminous Supreme (70B)":  ("Aleph Alpha", "Luminous", "70B", "closed"),
    # Microsoft / TNLG
    "TNLG v2 (530B)": ("Microsoft / NVIDIA", "TNLG", "530B", "closed"),
    "TNLG v2 (6.7B)": ("Microsoft / NVIDIA", "TNLG", "6.7B",  "closed"),
    # Google
    "T5 (11B)":  ("Google", "T5",  "11B", "open"),
    "UL2 (20B)": ("Google", "UL2", "20B", "open"),
    "T0pp (11B)": ("HuggingFace / BigScience", "T0pp", "11B", "open"),
    # Writer
    "Palmyra X (43B)":       ("Writer", "Palmyra", "43B", "closed"),
    "InstructPalmyra (30B)": ("Writer", "Palmyra", "30B", "closed"),
    # Yandex
    "YaLM (100B)": ("Yandex", "YaLM", "100B", "open"),
    # RedPajama
    "RedPajama-INCITE-Base (7B)":        ("Together AI", "RedPajama", "7B", "open"),
    "RedPajama-INCITE-Base-v1 (3B)":     ("Together AI", "RedPajama", "3B", "open"),
    "RedPajama-INCITE-Instruct (7B)":    ("Together AI", "RedPajama", "7B", "open"),
    "RedPajama-INCITE-Instruct-v1 (3B)": ("Together AI", "RedPajama", "3B", "open"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_json(url: str, retries: int = 3, delay: float = 1.0):
    """Fetch a URL and return parsed JSON; handles gzip transparently."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url)
            resp = urllib.request.urlopen(req, timeout=15)
            raw = resp.read()
            try:
                return json.loads(gzip.decompress(raw))
            except Exception:
                return json.loads(raw)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt < retries - 1:
                time.sleep(delay)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)
    return None


def group_json_url(proj: dict, group_id: str) -> str:
    path_type = proj["path_type"]
    version = proj["version"]
    base = proj["base_url"]
    return f"{base}{path_type}/{version}/groups/{group_id}.json"


def groups_index_url(proj: dict) -> str:
    path_type = proj["path_type"]
    version = proj["version"]
    base = proj["base_url"]
    return f"{base}{path_type}/{version}/groups.json"


def is_leaderboard_table(table: dict) -> bool:
    """Return True if this table has per-model scores (not a category listing)."""
    header = table.get("header", [])
    if not header:
        return False
    first_col = header[0].get("value", "")
    return first_col in ("Model/adapter", "Model")


def extract_primary_metric(table: dict) -> tuple[str, bool]:
    """Return (metric_name, lower_is_better) for the primary metric column."""
    header = table.get("header", [])
    if len(header) < 2:
        return ("score", False)
    col = header[1]
    metric_val = col.get("value", "score")
    lower = col.get("lower_is_better", False)
    # Extract just the short metric name (strip description)
    metric_name = metric_val.strip()
    return metric_name, lower


def scale_score(raw_score, metric_name: str, lower_is_better: bool) -> float:
    """Convert raw HELM score to our 0-100 scale (or keep absolute).

    Rules:
    - Metrics in NO_SCALE_METRICS: keep as-is (absolute scale, e.g. BPB, Elo)
    - All other 0.0–1.0 values: multiply by 100 (they are 0-1 percentages in HELM)
    - Values outside 0–1: keep as-is (already on some other absolute scale)
    Note: lower_is_better does NOT affect scaling — e.g. "Toxic fraction" is 0-1
    and gets ×100 just like accuracy; the metric_lower_is_better column records directionality.
    """
    if raw_score is None:
        return None
    try:
        val = float(raw_score)
    except (TypeError, ValueError):
        return None
    # Keep absolute-scale metrics as-is
    if metric_name in NO_SCALE_METRICS:
        return round(val, 6)
    # HELM reports percentages as 0-1; multiply by 100
    if 0.0 <= val <= 1.0:
        return min(round(val * 100, 4), 100.0)
    # Already on 0-100 scale or some other absolute scale
    # Round to avoid floating-point noise (e.g. 100.0000000000004 → 100.0)
    return round(val, 4)


# Regex for HELM model status markers (deprecated ☠, warning ⚠, etc.)
_HELM_STATUS_MARKERS = re.compile(r'[☠⚠†★✗✓⓪①②③④⑤]')


def clean_model_name(name: str) -> str:
    """Strip HELM-specific status markers from model display names.

    HELM appends symbols like ☠ (deprecated) or ⚠ (warning) to some model
    names, causing the same model to appear under two different strings across
    benchmarks.  Strip them so the model_id is stable.
    """
    return _HELM_STATUS_MARKERS.sub("", name).strip()


def clean_description(desc: str) -> str:
    """Strip markdown citation links from description text."""
    # e.g. [(Clark et al., 2019)](https://...) -> (Clark et al., 2019)
    desc = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', desc)
    # Remove trailing metric explanation after double newline
    parts = desc.split('\n\n', 1)
    return parts[0].strip()


def parse_benchmark_description(table: dict) -> str:
    """Pull the benchmark description from the primary metric's header column."""
    header = table.get("header", [])
    if len(header) >= 2:
        raw = header[1].get("description", "")
        return clean_description(raw)
    return ""


def get_model_info(model_name: str) -> dict:
    """Return developer/family/size/type metadata for a HELM model name."""
    meta = MODEL_METADATA.get(model_name, ("", "", "", ""))
    developer, family, size, mtype = meta
    return {
        "model_id": model_name,
        "model_name": model_name,
        "model_family": family,
        "developer": developer,
        "model_size": size,
        "model_type": mtype,
    }


# ---------------------------------------------------------------------------
# Staging file columns (must match main files exactly)
# ---------------------------------------------------------------------------
BENCHMARKS_COLS = [
    "benchmark_id", "benchmark_name", "abbreviation", "year", "venue", "category",
    "subcategory", "paper_url", "github_url", "hf_url", "languages_count", "languages",
    "cultures_regions", "task_types", "metrics", "dataset_size", "annotation_method",
    "key_finding", "notes", "other_url", "description", "task_type", "domain",
    "source_url", "title", "acronym", "huggingface_url", "language_count", "geography",
    "metric_type", "data_source", "organization", "name", "release_date", "paper_title",
    "source_type", "source_name",
]

MODELS_COLS = [
    "model_id", "model_name", "model_family", "developer", "sizes", "types",
    "benchmark_count", "total_results", "avg_score", "model_type", "model_size",
    "provider", "type", "base_model", "year_evaluated", "inference_platform",
    "inference_engine", "organization", "release_date", "architecture",
    "context_window", "license", "url", "parameters_billion",
]

RESULTS_COLS = [
    "benchmark_id", "model_name", "model_family", "model_size", "active_size",
    "model_type", "score", "metric_name", "setup", "num_shot_sample", "language",
    "source_url", "year_evaluated", "notes", "evaluation_id", "source_name",
    "source_type", "source_organization", "model_developer", "model_id", "hf_repo",
    "hf_split", "samples_number", "metric_lower_is_better", "metric_score_type",
    "metric_min_score", "metric_max_score", "reasoning_enabled", "inference_platform",
    "inference_engine", "generation_temperature", "generation_max_tokens",
    "generation_top_p", "benchmark_name", "metric_score", "metric", "score_type",
    "date_recorded",
]


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_project(proj_id: str, proj: dict, dry_run: bool = False):
    """Fetch all per-model leaderboards for one HELM sub-project."""
    print(f"\n{'='*60}")
    print(f"  {proj['name']}  ({proj['version']})")
    print(f"{'='*60}")

    # Fetch the groups index
    idx_url = groups_index_url(proj)
    index = fetch_json(idx_url)
    if index is None:
        print(f"  ERROR: could not fetch groups index: {idx_url}")
        return [], [], []

    # Collect all group IDs from the index
    group_ids = []
    for table in index:
        for row in table.get("rows", []):
            if not row:
                continue
            cell = row[0]
            if isinstance(cell, dict) and cell.get("href", "").startswith("?group="):
                gid = cell["href"].split("=")[1]
                if gid not in AGGREGATE_GROUP_IDS:
                    group_ids.append((gid, cell.get("value", gid)))

    print(f"  Found {len(group_ids)} non-aggregate group IDs")

    benchmarks_rows = []
    models_seen: dict[str, dict] = {}
    results_rows = []

    for gid, gname in group_ids:
        url = group_json_url(proj, gid)
        data = fetch_json(url)
        if data is None:
            print(f"  SKIP {gid}: 404 or error")
            continue
        if not isinstance(data, list) or not data:
            print(f"  SKIP {gid}: unexpected format")
            continue

        # Find the first table that has per-model scores
        leaderboard = None
        for table in data:
            if is_leaderboard_table(table):
                leaderboard = table
                break

        if leaderboard is None:
            print(f"  SKIP {gid}: no model/adapter leaderboard table found")
            continue

        metric_name, lower_is_better = extract_primary_metric(leaderboard)
        description = parse_benchmark_description(leaderboard)
        source_url = f"{proj['site_url']}#/leaderboard/{gid}"
        category = CATEGORY_MAP.get(gid, "General Knowledge")

        # Build benchmark row
        bench_row = {col: "" for col in BENCHMARKS_COLS}
        bench_row.update({
            "benchmark_id": gid,
            "benchmark_name": gname,
            "year": proj["year"],
            "venue": proj["venue"],
            "category": category,
            "metrics": metric_name,
            "description": description,
            "source_url": source_url,
            "source_type": "leaderboard",
            "source_name": proj["source_name"],
            "organization": proj["organization"],
        })
        benchmarks_rows.append(bench_row)

        # Extract model rows
        n_models = 0
        for row in leaderboard.get("rows", []):
            if not row:
                continue
            model_cell = row[0]
            if not isinstance(model_cell, dict):
                continue
            raw_model_name = model_cell.get("value", "")
            if not raw_model_name:
                continue
            model_name = clean_model_name(raw_model_name)

            # Score: second column (index 1)
            score_raw = None
            if len(row) > 1:
                sc = row[1]
                if isinstance(sc, dict):
                    score_raw = sc.get("value")

            score = scale_score(score_raw, metric_name, lower_is_better)
            if score is None:
                continue

            # Model metadata
            if model_name not in models_seen:
                models_seen[model_name] = get_model_info(model_name)

            minfo = models_seen[model_name]

            result_row = {col: "" for col in RESULTS_COLS}
            result_row.update({
                "benchmark_id": gid,
                "model_name": model_name,
                "model_family": minfo["model_family"],
                "model_size": minfo["model_size"],
                "model_type": minfo["model_type"],
                "score": str(score),
                "metric_name": metric_name,
                "setup": "",
                "language": "",
                "source_url": source_url,
                "year_evaluated": proj["year"],
                "source_name": proj["source_name"],
                "source_type": "leaderboard",
                "source_organization": proj["organization"],
                "model_developer": minfo["developer"],
                "model_id": model_name,
                "metric_lower_is_better": "True" if lower_is_better else "False",
                "benchmark_name": gname,
            })
            results_rows.append(result_row)
            n_models += 1

        status = "DRY-RUN" if dry_run else "OK"
        print(f"  [{status}] {gid:45s} metric={metric_name:25s} models={n_models}")
        time.sleep(0.05)  # be polite to GCS

    # Build models rows
    models_rows = []
    for mname, minfo in models_seen.items():
        model_row = {col: "" for col in MODELS_COLS}
        model_row.update({
            "model_id": mname,
            "model_name": mname,
            "model_family": minfo["model_family"],
            "developer": minfo["developer"],
            "model_size": minfo["model_size"],
            "model_type": minfo["model_type"],
        })
        models_rows.append(model_row)

    print(f"\n  Summary: {len(benchmarks_rows)} benchmarks, {len(models_seen)} models, {len(results_rows)} result rows")
    return benchmarks_rows, models_rows, results_rows


# ---------------------------------------------------------------------------
# Staging file I/O
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGING_BENCHMARKS = REPO_ROOT / "data" / "staging_helm_benchmarks.csv"
STAGING_MODELS = REPO_ROOT / "data" / "staging_helm_models.csv"
STAGING_RESULTS = REPO_ROOT / "data" / "staging_helm_results.csv"


def load_existing_staging() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load existing staging files (or empty DataFrames if missing/wrong schema)."""
    def _safe_load(path, cols):
        if path.exists():
            try:
                df = pd.read_csv(path, dtype=str, keep_default_na=False)
                # If schema doesn't match (e.g. old agent's files), start fresh
                if set(cols).issubset(set(df.columns)):
                    return df[cols]
                else:
                    print(f"  WARNING: {path.name} has wrong schema — starting fresh")
                    return pd.DataFrame(columns=cols)
            except Exception:
                return pd.DataFrame(columns=cols)
        return pd.DataFrame(columns=cols)

    b = _safe_load(STAGING_BENCHMARKS, BENCHMARKS_COLS)
    m = _safe_load(STAGING_MODELS, MODELS_COLS)
    r = _safe_load(STAGING_RESULTS, RESULTS_COLS)
    return b, m, r


def save_staging(benchmarks: pd.DataFrame, models: pd.DataFrame, results: pd.DataFrame):
    """Write staging files with CRLF line endings (matches main file convention)."""
    for df, path in [(benchmarks, STAGING_BENCHMARKS),
                     (models, STAGING_MODELS),
                     (results, STAGING_RESULTS)]:
        df.to_csv(path, index=False, lineterminator="\r\n")
        print(f"  Wrote {len(df):5d} rows → {path.name}")


def merge_new_rows(existing: pd.DataFrame, new_rows: list[dict],
                   dedup_keys: list[str]) -> pd.DataFrame:
    """Append new rows to existing DataFrame, de-duplicating on dedup_keys."""
    if not new_rows:
        return existing
    new_df = pd.DataFrame(new_rows)
    # Ensure all columns present
    cols = existing.columns.tolist() if len(existing.columns) > 0 else list(new_df.columns)
    for col in cols:
        if col not in new_df.columns:
            new_df[col] = ""
    new_df = new_df[cols]
    if len(existing) == 0:
        combined = new_df
    else:
        combined = pd.concat([existing, new_df], ignore_index=True)
    # Drop duplicates on the key columns that actually exist
    valid_keys = [k for k in dedup_keys if k in combined.columns]
    combined = combined.drop_duplicates(subset=valid_keys, keep="first")
    return combined.reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--project", nargs="*",
        help="One or more sub-project IDs to extract. "
             f"Available: {', '.join(HELM_PROJECTS)}. Default: all.",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch data but don't write staging files.")
    args = parser.parse_args()

    project_ids = args.project if args.project else list(HELM_PROJECTS.keys())
    unknown = [p for p in project_ids if p not in HELM_PROJECTS]
    if unknown:
        parser.error(f"Unknown project IDs: {unknown}. Available: {list(HELM_PROJECTS.keys())}")

    print(f"Projects to extract: {project_ids}")
    print(f"Dry run: {args.dry_run}\n")

    # Load existing staging data
    ex_bench, ex_models, ex_results = load_existing_staging()
    print(f"Existing staging: {len(ex_bench)} benchmarks, {len(ex_models)} models, {len(ex_results)} results\n")

    all_bench_rows, all_model_rows, all_result_rows = [], [], []

    for proj_id in project_ids:
        proj = HELM_PROJECTS[proj_id]
        b, m, r = extract_project(proj_id, proj, dry_run=args.dry_run)
        all_bench_rows.extend(b)
        all_model_rows.extend(m)
        all_result_rows.extend(r)

    if args.dry_run:
        total_b = len(all_bench_rows)
        total_m = len({r["model_id"] for r in all_model_rows})
        total_r = len(all_result_rows)
        print(f"\nDry run complete — would add {total_b} benchmarks, {total_m} models, {total_r} results.")
        return 0

    # Merge into existing staging data
    new_bench = merge_new_rows(ex_bench, all_bench_rows, ["benchmark_id"])
    new_models = merge_new_rows(ex_models, all_model_rows, ["model_id"])
    new_results = merge_new_rows(
        ex_results, all_result_rows,
        ["benchmark_id", "model_name", "metric_name", "source_url"],
    )

    print(f"\nFinal staging counts:")
    print(f"  benchmarks: {len(ex_bench)} → {len(new_bench)} (+{len(new_bench) - len(ex_bench)})")
    print(f"  models:     {len(ex_models)} → {len(new_models)} (+{len(new_models) - len(ex_models)})")
    print(f"  results:    {len(ex_results)} → {len(new_results)} (+{len(new_results) - len(ex_results)})")

    save_staging(new_bench, new_models, new_results)
    print("\nDone. Review the staging files, then merge into main data files when ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
