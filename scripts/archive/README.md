# Archive

One-off scripts from past cleanup passes. Kept for audit-trail / "how did
we get here" reference — **none of these should be re-run as-is**:

- Several hardcode model names/IDs specific to a past dupe or alias fix
  (`fix_model_aliases.py`, `fix_results_fk.py`, `merge_models.py`).
- Several reference intermediate files that no longer exist
  (`data/models_updated.csv`), left over from a multi-step manual pass.
- Several reimplement the same duplicate-detection logic slightly
  differently and inconsistently (`check_dupes.py`, `check_dupes2.py`,
  `analyze_dupes.py`, `deduplicate_results.py`, `fix_results_issues.py`).
- Two reimplement the same model-categorization heuristic with diverging
  trusted-dev/from-scratch lists (`analyze_models.py`, `categorize_models.py`).

The **reusable, generic** version of everything these scripts did now
lives in `scripts/lib/` and is exposed through `scripts/manage_data.py`:

| Old one-off script(s) | Replaced by |
|---|---|
| `check_dupes.py`, `check_dupes2.py`, `analyze_dupes.py`, `analyze_source_trust.py` | `scripts/manage_data.py dupes` |
| `deduplicate_results.py`, `fix_results_issues.py` (dedup portion) | `scripts/manage_data.py dedup` |
| `find_aliases.py` | `scripts/manage_data.py find-aliases` |
| `fix_model_aliases.py`, `fix_results_fk.py`, `merge_models.py` | `scripts/manage_data.py apply-aliases --map-file <renames.json>` |
| `standardize_model_ids.py` (root) | `scripts/manage_data.py standardize-ids` |
| `analyze_models.py`, `categorize_models.py` | `scripts/manage_data.py categorize-models` |
| `remove_suspicious_orphans.py`, `keep_latest_snapshot.py`, `update_models_with_research.py`, `check_issues.py` | No direct replacement — narrow one-off passes; see git history if similar work is needed again. |

If a future cleanup pass needs new alias mappings or category rules, add
them as a JSON map / config entry and run the generic tool — don't write
a new one-off script.
