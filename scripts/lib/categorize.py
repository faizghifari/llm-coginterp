"""Classify models.csv rows into KEEP / FLAG / REMOVE, to surface
cleanup candidates: fine-tunes with no identifiable base model,
experimental/orphaned entries, etc.

The trusted-developer / from-scratch-pattern / fine-tune-keyword
knowledge base lives in config.py — extend that when a new model family
or developer needs to be taught to this classifier, not the logic here.

This module also exposes `classify_scope` / `classify_scope_all`, a
second, orthogonal classifier over a different axis (is this model a
generative LLM / LLM-backed multimodal model at all, per METHODOLOGY.md's
"Model Inclusion Criteria" — not "is its provenance traceable?", which is
all `categorize_model` above checks). A row can be KEEP on one axis and
REMOVE on the other.
"""
import re

from . import config


def is_trusted_dev(developer):
    if not developer:
        return False
    dev_lower = str(developer).lower().strip()
    return any(t.lower() in dev_lower or dev_lower in t.lower() for t in config.TRUSTED_DEVELOPERS)


def is_from_scratch_pattern(model_id, model_name, developer):
    combined = f"{model_id} {model_name} {developer or ''}".lower()
    return any(p in combined for p in config.FROM_SCRATCH_PATTERNS)


def is_fine_tuned_name(model_name):
    name_lower = str(model_name).lower()
    return any(kw in name_lower for kw in config.FINE_TUNE_KEYWORDS)


def categorize_model(row):
    """Categorize one models.csv row. Returns (category, reason)."""
    model_id = str(row.get("model_id", "") or "")
    model_name = str(row.get("model_name", "") or "")
    model_family = row.get("model_family") or ""
    developer = row.get("developer") or ""
    model_type = (row.get("model_type") or "").lower()
    base_model = row.get("base_model") or ""

    if model_type == "closed" and is_trusted_dev(developer):
        return "KEEP", f"Closed model from trusted dev ({developer})"

    if is_trusted_dev(developer) and model_family:
        return "KEEP", f"Trusted dev ({developer}) with clear family ({model_family})"

    if base_model:
        return "KEEP", f"Has base_model: {base_model}"

    if is_trusted_dev(developer):
        return "KEEP", f"Open model from trusted dev ({developer})"

    if is_from_scratch_pattern(model_id, model_name, developer):
        return "KEEP", "Known from-scratch pattern"

    if model_family:
        if is_fine_tuned_name(model_name):
            return "KEEP", f"Fine-tuned but family known: {model_family}"
        return "KEEP", f"From-scratch with family: {model_family}"

    if is_fine_tuned_name(model_name):
        return "REMOVE", f"Fine-tuned name, no family/base_model/trusted dev (dev: {developer or 'unknown'})"

    return "FLAG", f"No family/base_model, unclear origin (dev: {developer or 'unknown'}, type: {model_type or 'unknown'})"


def categorize_all(models):
    """Run categorize_model on every row. Returns a copy of `models`
    with `category` and `reason` columns appended."""
    out = models.copy()
    cats, reasons = [], []
    for _, row in models.iterrows():
        cat, reason = categorize_model(row)
        cats.append(cat)
        reasons.append(reason)
    out["category"] = cats
    out["reason"] = reasons
    return out


# ── scope (LLM-inclusion) axis ──────────────────────────────────────────

def _pattern_matches(combined, pattern):
    """Substring match, except patterns of length <= 3 (e.g. "t5", "vit")
    which are matched only against whole tokens (split on non-alphanumeric
    characters) to avoid false hits like "t5" inside "gpt5"."""
    if len(pattern) <= 3:
        tokens = re.split(r"[^a-z0-9]+", combined)
        return pattern in tokens
    return pattern in combined


def _matched_patterns(combined, patterns):
    return [p for p in patterns if _pattern_matches(combined, p)]


def classify_scope(row):
    """Classify one models.csv row on the modality/inclusion-scope axis
    (LLM-or-LLM-backed-multimodal vs not), per METHODOLOGY.md's "Model
    Inclusion Criteria". Orthogonal to categorize_model()'s fine-tune-
    provenance axis. Returns (scope_category, scope_reason)."""
    model_id = str(row.get("model_id", "") or "")
    model_name = str(row.get("model_name", "") or "")
    model_family = str(row.get("model_family", "") or "")
    developer = str(row.get("developer", "") or "")
    combined = f"{model_id} {model_name} {model_family} {developer}".lower()

    if _matched_patterns(combined, config.VLM_ALM_ALLOWLIST):
        return "KEEP", "Matches VLM/ALM allowlist (modality encoder on an LLM backbone)"

    narrow = _matched_patterns(combined, config.NARROW_TASK_PATTERNS)
    if narrow:
        return "REMOVE", f"Narrow single-modality task model (matched: {', '.join(narrow)})"

    non_gen = _matched_patterns(combined, config.NON_GENERATIVE_PATTERNS)
    if non_gen:
        return "REMOVE", f"Non-generative architecture (matched: {', '.join(non_gen)})"

    return "KEEP", "No scope-exclusion pattern matched"


def classify_scope_all(models):
    """Run classify_scope on every row. Returns a copy of `models` with
    `scope_category`/`scope_reason` columns appended — kept separate from
    categorize_all()'s `category`/`reason` so the two axes never collide
    in one DataFrame."""
    out = models.copy()
    cats, reasons = [], []
    for _, row in models.iterrows():
        cat, reason = classify_scope(row)
        cats.append(cat)
        reasons.append(reason)
    out["scope_category"] = cats
    out["scope_reason"] = reasons
    return out
