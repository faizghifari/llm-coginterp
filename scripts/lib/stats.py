"""Recompute models.csv's per-model aggregate columns
(`benchmark_count`, `total_results`, `avg_score`) from the current
contents of results.csv.

These columns are denormalized summaries -- convenient for a quick
glance at models.csv without joining results.csv, but they drift out of
sync the moment results.csv changes (new rows, renamed model_name
aliases, removed duplicates, ...) and nothing recomputes them
automatically. Run this after any batch of changes to results.csv that
adds/removes/renames rows.

Caveat: `avg_score` is a plain mean across every row for that model,
regardless of metric. Scores live on very different scales in this
dataset (most are 0-100 percentages, but Elo ratings like Chatbot
Arena's are ~1000-1500, and a few metrics are 0-1 ratios) -- so for
models evaluated on a mix of scale types, this average is not a
meaningful "typical score", just a sum-divided-by-count. This isn't
new behavior introduced here; it matches how the column was already
defined, just recomputed correctly.
"""


def compute_model_stats(results):
    """Return a DataFrame indexed by model_name with benchmark_count,
    total_results, and avg_score columns, computed from results."""
    import pandas as pd

    scores = results.copy()
    scores["_score_numeric"] = pd.to_numeric(scores["score"], errors="coerce")

    grouped = scores.groupby("model_name")
    stats = grouped.agg(
        benchmark_count=("benchmark_id", "nunique"),
        total_results=("benchmark_id", "size"),
        avg_score=("_score_numeric", "mean"),
    )
    return stats


def apply_model_stats(models, results):
    """Update models.csv's benchmark_count/total_results/avg_score
    columns in place (returns a new DataFrame; doesn't mutate `models`).
    Models with no matching results.csv rows are left at 0/0/blank.
    Returns (updated_models, report) where report has before/after
    counts for a quick diff summary."""
    stats = compute_model_stats(results)
    updated = models.copy()

    report = {"updated": 0, "zero_results": 0}
    for idx in updated.index:
        model_id = updated.at[idx, "model_id"]
        if model_id in stats.index:
            row = stats.loc[model_id]
            updated.at[idx, "benchmark_count"] = _fmt_count(row["benchmark_count"])
            updated.at[idx, "total_results"] = _fmt_count(row["total_results"])
            updated.at[idx, "avg_score"] = _fmt_score(row["avg_score"])
            report["updated"] += 1
        else:
            updated.at[idx, "benchmark_count"] = "0.0"
            updated.at[idx, "total_results"] = "0.0"
            updated.at[idx, "avg_score"] = ""
            report["zero_results"] += 1

    return updated, report


def _fmt_count(value):
    return f"{float(value):.1f}"


def _fmt_score(value):
    if value is None or value != value:  # NaN check without importing pandas
        return ""
    return f"{float(value):.2f}"
