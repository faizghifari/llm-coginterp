"""Data integrity checks: foreign keys, orphans, exhaustion, and
pending-benchmark completeness.

This is the engine behind both `verify_data.py` and
`manage_data.py verify`. Add new checks here as functions that return
plain data (sets / DataFrames / dicts) — keep printing/formatting in
`format_report()` so the checks stay usable from other tooling too.
"""
import re

from . import config


def check_foreign_keys(benchmarks, models, results):
    """Results rows referencing a benchmark_id or model_name that
    doesn't exist. NOTE: results.model_name (not results.model_id) is
    the real FK column matched against models.model_id — that's the
    join the rest of the dataset actually relies on."""
    invalid_benchmarks = set(results["benchmark_id"]) - set(benchmarks["benchmark_id"])
    invalid_models = set(results["model_name"]) - set(models["model_id"])
    return {
        "invalid_benchmark_ids": invalid_benchmarks,
        "invalid_model_names": invalid_models,
    }


def check_orphans(benchmarks, models, results):
    """Benchmarks/models with zero result rows."""
    used_benchmarks = set(results["benchmark_id"])
    used_models = set(results["model_name"])
    return {
        "orphan_benchmarks": set(benchmarks["benchmark_id"]) - used_benchmarks,
        "orphan_models": set(models["model_id"]) - used_models,
    }


def check_exhaustion(results, threshold=5):
    """Benchmarks with fewer than `threshold` result rows — a signal of
    incomplete extraction rather than a genuinely small benchmark."""
    counts = results["benchmark_id"].value_counts()
    return counts[counts < threshold]


def check_pending_completeness(benchmarks, pending_file=None):
    """Cross-check notes/pending_benchmarks.md against benchmarks.csv to
    see which previously-pending benchmarks have since been added."""
    pending_file = pending_file or config.PENDING_BENCHMARKS_MD
    pending_names = []
    if pending_file.exists():
        with open(pending_file) as f:
            for line in f:
                match = re.match(r"^\*\*(.*?)\*\*", line)
                if match:
                    pending_names.append(match.group(1).split("(")[0].strip())

    existing_names = benchmarks["benchmark_name"].fillna("").str.lower().tolist()
    existing_ids = benchmarks["benchmark_id"].fillna("").str.lower().tolist()

    found, missing = [], []
    for expected in pending_names:
        exp_lower = expected.lower()
        if any(exp_lower in n for n in existing_names) or any(exp_lower in i for i in existing_ids):
            found.append(expected)
        else:
            missing.append(expected)
    return {"total": len(pending_names), "found": found, "missing": missing}


def run_all(benchmarks, models, results, exhaustion_threshold=5):
    """Run every check and return one structured report dict."""
    return {
        "foreign_keys": check_foreign_keys(benchmarks, models, results),
        "orphans": check_orphans(benchmarks, models, results),
        "exhaustion": check_exhaustion(results, exhaustion_threshold),
        "pending": check_pending_completeness(benchmarks),
    }


def format_report(report, max_examples=10):
    """Render a structured report dict as human-readable text."""
    lines = []
    fk = report["foreign_keys"]
    lines.append("1. Foreign Keys:")
    lines.append(f"   - Results with missing benchmark_id: {len(fk['invalid_benchmark_ids'])} distinct IDs")
    if fk["invalid_benchmark_ids"]:
        lines.append(f"     {sorted(fk['invalid_benchmark_ids'])[:max_examples]}")
    lines.append(f"   - Results with missing model_name (not in models.csv): {len(fk['invalid_model_names'])} distinct models")
    if fk["invalid_model_names"]:
        lines.append(f"     {sorted(fk['invalid_model_names'])[:max_examples]}")
    lines.append("")

    orphans = report["orphans"]
    lines.append("2. Orphans (zero result rows):")
    lines.append(f"   - Orphan benchmarks: {len(orphans['orphan_benchmarks'])}")
    if orphans["orphan_benchmarks"]:
        lines.append(f"     {sorted(orphans['orphan_benchmarks'])[:max_examples]}")
    lines.append(f"   - Orphan models: {len(orphans['orphan_models'])}")
    if orphans["orphan_models"]:
        lines.append(f"     {sorted(orphans['orphan_models'])[:max_examples]}")
    lines.append("")

    exhaustion = report["exhaustion"]
    lines.append("3. Exhaustion (benchmarks with < threshold result rows):")
    if exhaustion.empty:
        lines.append("   All benchmarks meet the threshold. Good sign for exhaustion.")
    else:
        for b_id, count in exhaustion.items():
            lines.append(f"   - {b_id}: {count} rows")
    lines.append("")

    pending = report["pending"]
    lines.append("4. Pending benchmarks completeness (notes/pending_benchmarks.md):")
    lines.append(f"   Found ~{len(pending['found'])}/{pending['total']}")
    if pending["missing"]:
        lines.append("   Still missing:")
        for m in pending["missing"][:max_examples]:
            lines.append(f"   - {m}")
        if len(pending["missing"]) > max_examples:
            lines.append(f"   ... and {len(pending['missing']) - max_examples} more.")
    lines.append("")

    return "\n".join(lines)
