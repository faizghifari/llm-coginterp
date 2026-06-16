"""Reconcile model identity across models.csv and results.csv:
  - apply a rename map (old model_id/model_name -> canonical model_id)
  - standardize model_id formatting to one naming convention
  - suggest likely alias candidates for unresolved orphan model_names

This generalizes the one-off RENAME_MAP / fk_map / merge_groups dicts
that used to live inside fix_model_aliases.py, fix_results_fk.py, and
merge_models.py (see scripts/archive/): each of those hardcoded a
specific past cleanup pass and couldn't be reused. The functions here
take the mapping as a parameter, so the same code handles any future
cleanup pass — build a `{old_id: new_id}` JSON file and pass it to
`manage_data.py apply-aliases`.
"""
import re
from difflib import get_close_matches


def standardize_model_id(model_id):
    """Canonical naming convention: lowercase, hyphen-separated,
    no leading/trailing/duplicate hyphens, '+' spelled out as '-plus'."""
    s = str(model_id).strip().lower()
    s = s.replace(" ", "-").replace("_", "-")
    s = s.replace("+", "-plus")
    s = re.sub(r"-+", "-", s)
    return s.strip("-")


def apply_rename_map(models, results, rename_map):
    """Apply {old_id: new_id} renames to results.model_name (+
    results.model_id if present) and models.model_id. If a rename
    causes two models.csv rows to collide on the new id, the first
    occurrence is kept and the rest are merged away.

    Returns (models, results, report).
    """
    report = {"renamed_result_rows": 0, "models_renamed": 0, "models_merged": []}

    mask = results["model_name"].isin(rename_map)
    report["renamed_result_rows"] = int(mask.sum())
    results.loc[mask, "model_name"] = results.loc[mask, "model_name"].map(rename_map)
    if "model_id" in results.columns:
        mask2 = results["model_id"].isin(rename_map)
        results.loc[mask2, "model_id"] = results.loc[mask2, "model_id"].map(rename_map)

    mmask = models["model_id"].isin(rename_map)
    report["models_renamed"] = int(mmask.sum())
    models.loc[mmask, "model_id"] = models.loc[mmask, "model_id"].map(rename_map)

    dupe_ids = sorted(models.loc[models["model_id"].duplicated(keep=False), "model_id"].unique().tolist())
    report["models_merged"] = dupe_ids
    models = models.drop_duplicates(subset="model_id", keep="first").reset_index(drop=True)

    return models, results, report


def standardize_all(models, results):
    """Apply standardize_model_id() to every model_id in models.csv and
    propagate the same renames into results.csv, merging any collisions
    the normalization creates. Returns (models, results, rename_map, report)."""
    rename_map = {}
    for mid in models["model_id"]:
        new = standardize_model_id(mid)
        if new != mid:
            rename_map[mid] = new

    models, results, report = apply_rename_map(models, results, rename_map)
    return models, results, rename_map, report


def find_alias_candidates(orphan_names, candidate_ids, cutoff=0.8):
    """Fuzzy-match orphan model_names (present in results.csv but absent
    from models.csv) against known model_ids, for human review before
    building a rename map. Returns {orphan_name: [close_matches]}."""
    candidates = {}
    for name in orphan_names:
        # Case-insensitive exact match first — difflib's ratio-based
        # matching can miss "GPT-4" vs "gpt-4" if other characters differ.
        ci_matches = [c for c in candidate_ids if c.lower() == name.lower()]
        fuzzy_matches = get_close_matches(name, candidate_ids, n=3, cutoff=cutoff)
        all_matches = list(dict.fromkeys(ci_matches + fuzzy_matches))
        if all_matches:
            candidates[name] = all_matches
    return candidates
