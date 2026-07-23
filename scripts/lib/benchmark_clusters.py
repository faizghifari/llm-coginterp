"""Discover benchmark_ids that look like per-language siblings of a
shared stem (e.g. `kaggle_vijitsingh1_mgsm` + `..._mgsm_chinese`), for
HUMAN review -- NOT an automatic translation-vs-distinct classifier (that
judgment requires reading each benchmark's paper/source, done in a
separate research pass).

A benchmark-level analog to `aliases.find_alias_candidates`: kept as its
own module rather than folded into aliases.py (scoped to model identity)
or dedup.py (scoped to exact-duplicate result rows), since this is a
distinct concern -- benchmark_id *naming pattern* discovery.

Over-inclusive by design: a stem+suffix match is a hint, not proof (e.g.
a `_ca`/`_cs` suffix might mean "culturally agnostic"/"culturally
sensitive", not a language) -- false positives here are expected and get
screened out by the human reviewing the output.
"""
from . import config


def _strip_language_suffix(benchmark_id):
    """Return (stem, matched_suffix) if benchmark_id ends with a
    recognized language-hint suffix (checking a two-token suffix like
    "simplified_mandarin" before a one-token suffix), else
    (benchmark_id, None)."""
    tokens = benchmark_id.split("_")
    if len(tokens) < 2:
        return benchmark_id, None

    two = "_".join(tokens[-2:]).lower()
    if two in config.LANGUAGE_SUFFIX_HINTS:
        return "_".join(tokens[:-2]), two

    one = tokens[-1].lower()
    if one in config.LANGUAGE_SUFFIX_HINTS:
        return "_".join(tokens[:-1]), one

    return benchmark_id, None


def find_language_clusters(benchmarks, results):
    """Group benchmark_ids sharing a stem after stripping a recognized
    language suffix. Returns a list of cluster dicts, largest first:
    {stem, parent_id (or None), members: [{benchmark_id, result_count,
    language_populated_pct, matched_suffix}]}. `matched_suffix` is None
    for the parent (unsuffixed) entry, if present."""
    ids = list(benchmarks["benchmark_id"])
    id_set = set(ids)

    result_counts = results.groupby("benchmark_id").size().to_dict()
    lang_populated = {}
    if "language" in results.columns:
        populated = results[results["language"].str.strip() != ""]
        lang_populated = populated.groupby("benchmark_id").size().to_dict()

    def _entry(bid, suffix):
        n = result_counts.get(bid, 0)
        pct = round(100 * lang_populated.get(bid, 0) / n, 1) if n else 0.0
        return {"benchmark_id": bid, "result_count": n,
                "language_populated_pct": pct, "matched_suffix": suffix}

    stem_map = {}
    for bid in ids:
        stem, suffix = _strip_language_suffix(bid)
        if suffix is None:
            continue
        stem_map.setdefault(stem, []).append((bid, suffix))

    clusters = []
    for stem, members in stem_map.items():
        parent_id = stem if stem in id_set else None
        if len(members) < 2 and parent_id is None:
            continue  # a single suffixed id with no sibling and no parent isn't a cluster

        member_list = []
        if parent_id:
            member_list.append(_entry(parent_id, None))
        for bid, suffix in sorted(members):
            member_list.append(_entry(bid, suffix))

        clusters.append({"stem": stem, "parent_id": parent_id, "members": member_list})

    clusters.sort(key=lambda c: -len(c["members"]))
    return clusters
