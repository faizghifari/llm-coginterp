"""Pick ONE canonical metric per benchmark, so a matrix cell measures one thing.

The problem this solves: results.csv holds one row per model x benchmark x
evaluation, and the aggregation step averages every row for a pair into a single
cell -- regardless of `metric_name`. Where a benchmark was reported under several
metrics, the cell became a mean over incommensurable quantities (TruthfulQA
averaged "% informative" with a BLEU *delta*), and worse, WHICH metric a model
got is usually decided by which leaderboard scored it. That makes part of a
column's variance a function of its source rather than of capability -- exactly
the kind of structure a factor analysis would report as a capability factor.

The fix is to keep only one metric per benchmark. Selection order:

  1. normalize   -- strip + casefold + collapse internal whitespace, so
                    "Accuracy"/"accuracy" and "Pass@1"/"pass@1" stop counting as
                    different metrics (this alone resolves gsm8k, fever,
                    humaneval, winogrande, and would otherwise have discarded
                    ~149 gsm8k rows for a capitalisation difference).
  2. alias       -- config.METRIC_NAME_ALIASES maps provable spelling variants of
                    ONE metric onto a single name. Curated, never inferred: an
                    earlier attempt to detect aliases from small within-model
                    differences mislabelled task FACETS as aliases (sibench's
                    "cause"/"motivation", sotopia's "secret"/"social rules") and
                    misfiled CONFIG names as aliases (elephant's metric_name
                    column holds model configurations, not metrics).
  3. override    -- config.CANONICAL_METRIC_OVERRIDES pins a benchmark's metric
                    where coverage alone picks badly.
  4. coverage    -- otherwise take the metric covering the most distinct models,
                    tie-broken by row count, then by name for determinism.

Coverage is counted in MODELS, not rows: the point is to keep the widest
comparable population, and a single leaderboard can contribute many rows for few
models.
"""
import re

from . import config

_WS = re.compile(r"\s+")

BLANK = "(blank)"


def normalize_metric(name):
    """Casefold + collapse whitespace. Blank/NaN -> BLANK sentinel, so rows with
    no recorded metric group together instead of silently becoming NaN."""
    if name is None:
        return BLANK
    s = _WS.sub(" ", str(name).strip()).lower()
    if not s or s == "nan":
        return BLANK
    return s


def canonical_metric_key(name):
    """normalize_metric() then apply the curated alias map."""
    n = normalize_metric(name)
    return config.METRIC_NAME_ALIASES.get(n, n)


def add_metric_key(results):
    """Return a copy of `results` with a `_metric_key` column."""
    out = results.copy()
    out["_metric_key"] = out["metric_name"].map(canonical_metric_key)
    return out


def choose_canonical_metrics(results):
    """Decide the canonical metric for every benchmark.

    Returns (chosen, report) where `chosen` maps benchmark_id -> metric key and
    `report` is a list of per-benchmark dicts for benchmarks that had a real
    choice to make (>1 key after normalisation + aliasing)."""
    keyed = add_metric_key(results)
    chosen, report = {}, []

    for bid, sub in keyed.groupby("benchmark_id", sort=True):
        counts = (sub.groupby("_metric_key")
                     .agg(models=("model_id", "nunique"), rows=("_metric_key", "size")))
        if counts.empty:
            continue

        override = config.CANONICAL_METRIC_OVERRIDES.get(bid)
        if override is not None:
            key = canonical_metric_key(override)
            reason = "override"
            if key not in counts.index:
                # A pinned metric that no longer exists is a config bug, not a
                # silent no-op: fall back to coverage and say so loudly.
                reason = f"override '{override}' ABSENT from data -- fell back to coverage"
                key = _by_coverage(counts)
        else:
            key = _by_coverage(counts)
            reason = "coverage"

        chosen[bid] = key
        if len(counts) > 1:
            total_models = int(sub["model_id"].nunique())
            kept_models = int(counts.loc[key, "models"])
            report.append({
                "benchmark_id": bid,
                "n_metrics": int(len(counts)),
                "chosen": key,
                "reason": reason,
                "models_total": total_models,
                "models_kept": kept_models,
                "models_lost": total_models - kept_models,
                "rows_total": int(counts["rows"].sum()),
                "rows_kept": int(counts.loc[key, "rows"]),
                "candidates": ", ".join(
                    f"{m}({int(c.models)})" for m, c in
                    counts.sort_values(["models", "rows"], ascending=False).iterrows()),
            })

    return chosen, report


def _by_coverage(counts):
    """Most distinct models, tie-broken by rows, then name (deterministic)."""
    ranked = counts.sort_values(["models", "rows"], ascending=False)
    top = ranked[(ranked["models"] == ranked["models"].iloc[0])
                 & (ranked["rows"] == ranked["rows"].iloc[0])]
    return sorted(top.index)[0]


def drop_anomalous_rows(results):
    """Drop the individual rows listed in config.ANOMALOUS_RESULT_ROWS.

    Returns (results, dropped) where `dropped` lists (key, reason, found)
    tuples so a stale entry is reported rather than silently ignored."""
    dropped = []
    if not config.ANOMALOUS_RESULT_ROWS:
        return results, dropped

    drop_idx = []
    for (bid, mid), reason in config.ANOMALOUS_RESULT_ROWS.items():
        hit = results.index[(results["benchmark_id"] == bid)
                            & (results["model_id"] == mid)]
        drop_idx.extend(hit.tolist())
        dropped.append(((bid, mid), reason, len(hit)))

    if drop_idx:
        results = results.drop(index=drop_idx).reset_index(drop=True)
    return results, dropped


def filter_to_canonical_source(results):
    """Resolve source-level scale conflicts: where one metric name covers two
    incompatible scoring conventions distinguishable only by source, keep the
    source named in config.SOURCE_SCALE_CONFLICTS and drop the rest.

    This is the residue canonical-metric selection cannot reach -- it operates
    below the metric name, not on it. Returns (results, report) where `report`
    lists one dict per benchmark acted on."""
    report = []
    if not config.SOURCE_SCALE_CONFLICTS:
        return results, report

    drop_idx = []
    for bid, keep_src in config.SOURCE_SCALE_CONFLICTS.items():
        sub = results[results["benchmark_id"] == bid]
        if sub.empty:
            continue
        keeping = sub["source_name"] == keep_src
        if not keeping.any():
            # Pinning a source that contributes nothing would silently empty the
            # benchmark; refuse and say so.
            report.append({"benchmark_id": bid, "kept_source": keep_src,
                           "kept_rows": 0, "dropped_rows": 0,
                           "note": f"source '{keep_src}' absent -- NOT applied"})
            continue
        drop_idx.extend(sub.index[~keeping].tolist())
        report.append({"benchmark_id": bid, "kept_source": keep_src,
                       "kept_rows": int(keeping.sum()),
                       "dropped_rows": int((~keeping).sum()), "note": ""})

    if drop_idx:
        results = results.drop(index=drop_idx).reset_index(drop=True)
    return results, report


def filter_to_canonical(results, chosen=None):
    """Drop every result row whose metric is not its benchmark's canonical one.

    Returns (results, report_rows, dropped_count). Does NOT cascade onto the
    model axis -- the caller owns that (see make_text_only_copy.py), because
    orphan handling differs by context."""
    if chosen is None:
        chosen, _ = choose_canonical_metrics(results)
    keyed = add_metric_key(results)
    keep = keyed.apply(
        lambda row: chosen.get(row["benchmark_id"]) == row["_metric_key"], axis=1)
    dropped = int((~keep).sum())
    out = keyed[keep].drop(columns=["_metric_key"]).reset_index(drop=True)
    return out, dropped
