"""Classify benchmarks.csv rows on the TEXT / NON_TEXT modality axis: does
this benchmark require the model to consume or produce a non-text
modality (vision, audio, video) to do the task at all.

Orthogonal to everything in categorize.py (which classifies models.csv
rows). benchmarks.csv has no dedicated modality column -- the pattern
knowledge base lives in config.py (NON_TEXT_MODALITY_PATTERNS,
TEXT_MODALITY_ALLOWLIST); extend that when a new benchmark or false
positive is found, not the logic here.
"""
import re

from . import config

_TEXT_COLUMNS = [
    "category", "subcategory", "task_type", "task_types", "domain",
    "benchmark_name", "description", "title",
]


def _pattern_matches(combined, pattern):
    """Whole-word match only -- a raw substring match would false-positive
    on e.g. "vision" inside "Historical Revisionism Detection"."""
    return re.search(r"\b" + re.escape(pattern) + r"\b", combined) is not None


def _matched_patterns(combined, patterns):
    return [p for p in patterns if _pattern_matches(combined, p)]


def classify_benchmark_modality(row):
    """Classify one benchmarks.csv row. Returns (modality, reason):
    modality is "TEXT" or "NON_TEXT"."""
    benchmark_id = str(row.get("benchmark_id", "") or "")

    if benchmark_id in config.TEXT_MODALITY_ALLOWLIST:
        return "TEXT", config.TEXT_MODALITY_ALLOWLIST[benchmark_id]
    if benchmark_id in config.NON_TEXT_MODALITY_OVERRIDES:
        return "NON_TEXT", config.NON_TEXT_MODALITY_OVERRIDES[benchmark_id]

    # benchmark_id joins the match text with separators spaced out so id
    # tokens (e.g. "..._video_qa") are word-boundary-matchable too --
    # several sparse-metadata imports carry their only modality signal
    # in the id itself.
    id_text = re.sub(r"[_\-]+", " ", benchmark_id)
    combined = " ".join(
        [id_text] + [str(row.get(c, "") or "") for c in _TEXT_COLUMNS]).lower()
    matched = _matched_patterns(combined, config.NON_TEXT_MODALITY_PATTERNS)
    if matched:
        return "NON_TEXT", f"Matched non-text modality pattern(s): {', '.join(matched)}"

    return "TEXT", "No non-text modality pattern matched"


def classify_benchmark_modality_all(benchmarks):
    """Run classify_benchmark_modality on every row. Returns a copy of
    `benchmarks` with `modality_category`/`modality_reason` columns
    appended."""
    out = benchmarks.copy()
    cats, reasons = [], []
    for _, row in benchmarks.iterrows():
        cat, reason = classify_benchmark_modality(row)
        cats.append(cat)
        reasons.append(reason)
    out["modality_category"] = cats
    out["modality_reason"] = reasons
    return out
