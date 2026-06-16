"""Find and resolve duplicate rows in results.csv.

Two rows are "the same evaluation" if they match on every column in
`config.RESULT_IDENTITY_KEY`. Among same-evaluation rows:
  - if every score agrees      -> pure redundancy, keep one
  - if scores disagree         -> conflict, resolve by source trust tier
                                   (lower tier wins), tie-broken by the
                                   more recent `year_evaluated`.

This replaces the old check_dupes.py / check_dupes2.py / analyze_dupes.py
/ deduplicate_results.py scripts (see scripts/archive/), which each
reimplemented a slightly different and mutually inconsistent version of
this same logic with a looser, hardcoded key.
"""
from . import config


def trust_tier(source_url):
    """Lower number = more trustworthy. See config.SOURCE_TRUST_TIERS."""
    url = str(source_url).lower()
    for tier, domains in config.SOURCE_TRUST_TIERS.items():
        if any(d in url for d in domains):
            return tier
    return config.UNKNOWN_TRUST_TIER


def _year(row):
    try:
        return int(float(row.get("year_evaluated", "") or 0))
    except (ValueError, TypeError):
        return 0


def _active_key(results, key=None):
    key = key or config.RESULT_IDENTITY_KEY
    return [k for k in key if k in results.columns]


def find_duplicate_groups(results, key=None):
    """Yield one DataFrame per group of 2+ rows sharing the same
    identity key (i.e. candidate duplicates)."""
    for _, group in results.groupby(_active_key(results, key), dropna=False):
        if len(group) > 1:
            yield group


def classify(results, key=None):
    """Classify every duplicate group as 'redundant' (all scores agree)
    or 'conflict' (scores disagree). Returns a list of
    {"kind", "scores", "rows"} dicts."""
    report = []
    for group in find_duplicate_groups(results, key):
        scores = group["score"].unique().tolist()
        report.append({
            "kind": "redundant" if len(scores) == 1 else "conflict",
            "scores": scores,
            "rows": group,
        })
    return report


def resolve(results, key=None):
    """Resolve all duplicate groups down to one row per evaluation.
    Returns (clean_df, discarded_df) — discarded_df is everything that
    should move to data/results_duplicates.csv for audit."""
    discard_indices = []
    for group in find_duplicate_groups(results, key):
        scores = group["score"].unique()
        if len(scores) == 1:
            discard_indices.extend(group.index[1:])
            continue

        ranked = sorted(
            group.index,
            key=lambda i: (trust_tier(results.loc[i, "source_url"]), -_year(results.loc[i])),
        )
        keep_idx = ranked[0]
        discard_indices.extend(i for i in group.index if i != keep_idx)

    discard_set = set(discard_indices)
    keep_mask = ~results.index.isin(discard_set)
    clean = results.loc[keep_mask].reset_index(drop=True)
    discarded = results.loc[sorted(discard_set)].reset_index(drop=True)
    return clean, discarded
