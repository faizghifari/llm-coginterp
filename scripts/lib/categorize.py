"""Classify models.csv rows into KEEP / FLAG / REMOVE, to surface
cleanup candidates: fine-tunes with no identifiable base model,
experimental/orphaned entries, etc.

The trusted-developer / from-scratch-pattern / fine-tune-keyword
knowledge base lives in config.py — extend that when a new model family
or developer needs to be taught to this classifier, not the logic here.
"""
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
