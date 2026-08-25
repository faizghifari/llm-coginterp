#!/usr/bin/env python3
"""Build a derived, analysis-ready copy of the 3 canonical dataset CSVs.

Reads data/*.csv (never writes to it) and applies these orthogonal removals,
then cascades:

  1. MODALITY -- classifies every benchmark on the TEXT/NON_TEXT axis
     (scripts/lib/modality.py) and drops every NON_TEXT benchmark.
  2. BENCHMARK REMOVALS, from three maps with three different bases:
     - SCORE_REDUNDANT_BENCHMARKS: near-duplicate columns (version splits,
       per-language splits, cross-source re-imports) whose variance a kept
       benchmark already carries. Decided from pairwise-correlation audits;
       families audited but deliberately KEPT are in KEPT_DESPITE_CORRELATION.
     - TRANSLATION_DUPLICATE_BENCHMARKS: translations of an original that is
       itself in the corpus. Decided on what the benchmark IS, not on
       correlation.
     - DEFECTIVE_BENCHMARKS: columns no metric choice can make interpretable.
  2b. ANOMALOUS_RESULT_ROWS -- individual values irreconcilable with their own
     column and unrepairable at source.
  3. SOURCE_SCALE_CONFLICTS -- one metric name covering two incompatible
     scoring conventions, separable only by source.

  4. CANONICAL METRIC -- keeps only one metric per surviving benchmark
     (scripts/lib/metrics.py), because the aggregation step averages every
     row for a model x benchmark pair regardless of metric, and WHICH metric
     a model got is usually decided by which leaderboard scored it. Left
     alone, part of a column's variance is a function of its source.

Removals cascade: benchmark -> its results.csv rows -> any model left with
zero remaining results. Output goes to data/text_only/.

Rerun any time canonical data changes or any knowledge base in
scripts/lib/config.py is updated (NON_TEXT_MODALITY_PATTERNS,
TEXT_MODALITY_ALLOWLIST, NON_TEXT_MODALITY_OVERRIDES,
SCORE_REDUNDANT_BENCHMARKS, TRANSLATION_DUPLICATE_BENCHMARKS,
DEFECTIVE_BENCHMARKS, ANOMALOUS_RESULT_ROWS, SOURCE_SCALE_CONFLICTS,
METRIC_NAME_ALIASES, CANONICAL_METRIC_OVERRIDES).

NOTE: data/text_only/ is TRACKED in git, not gitignored. It is fully
reproducible from this script -- that is the invariant this script exists to
maintain, and `--check` asserts it. Never hand-edit the output: encode the
decision in config.py and regenerate, or the edit is lost on the next run.
"""
import sys
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.lib import (config, integrity, io, metrics, modality, standardise,
                         stats)

OUT_DIR = config.DATA_DIR / "text_only"


def main(check_only=False, out_dir=None):
    out_dir = out_dir or OUT_DIR
    benchmarks, models, results = io.load_data()
    before = {"benchmarks": len(benchmarks), "models": len(models), "results": len(results)}

    scoped = modality.classify_benchmark_modality_all(benchmarks)
    non_text = scoped[scoped["modality_category"] == "NON_TEXT"]
    remove_map = dict(zip(non_text["benchmark_id"], non_text["modality_reason"]))

    print(f"Classified {len(scoped)} benchmarks: "
          f"{len(scoped) - len(non_text)} TEXT, {len(non_text)} NON_TEXT")

    # Pass 2: score-redundant splits. Orthogonal to modality -- a benchmark can
    # be text-only and still be a duplicate column. Only ids actually present
    # survive into the map, so a stale entry is reported rather than silently
    # ignored.
    known = set(benchmarks["benchmark_id"])
    removal_kb = {**config.SCORE_REDUNDANT_BENCHMARKS,
                  **config.TRANSLATION_DUPLICATE_BENCHMARKS,
                  **config.DEFECTIVE_BENCHMARKS}
    redundant = {b: r for b, r in removal_kb.items() if b in known}
    stale = sorted(set(removal_kb) - known)
    already_gone = sorted(set(redundant) & set(remove_map))
    print(f"Score-redundant + defective benchmarks: {len(redundant)} of "
          f"{len(removal_kb)} listed ids present in canonical data "
          f"({len(config.SCORE_REDUNDANT_BENCHMARKS)} redundant, "
          f"{len(config.TRANSLATION_DUPLICATE_BENCHMARKS)} translation-duplicate, "
          f"{len(config.DEFECTIVE_BENCHMARKS)} defective)")
    if stale:
        print(f"  WARNING: {len(stale)} listed id(s) absent from canonical data "
              f"(prune config.SCORE_REDUNDANT_BENCHMARKS): {', '.join(stale)}")
    if already_gone:
        print(f"  note: {len(already_gone)} also removed by the modality filter: "
              f"{', '.join(already_gone)}")
    remove_map.update(redundant)

    benchmarks, models, results, report = standardise.cascade_remove_benchmarks(
        benchmarks, models, results, remove_map)

    print(f"\nRemoved {report['removed_benchmarks']} benchmarks "
          f"({report['removed_results']} result rows)")

    # Pass 2b: individual anomalous rows.
    results, anomalies = metrics.drop_anomalous_rows(results)
    for key, reason, found in anomalies:
        if found:
            print(f"\nDropped anomalous row {key[1]} / {key[0]}: {reason.split('--')[0].strip()}")
        else:
            print(f"\nWARNING: anomalous row {key} not found -- prune "
                  f"config.ANOMALOUS_RESULT_ROWS")

    # Pass 3: source-level scale conflicts. Runs BEFORE canonical-metric
    # selection so that coverage is counted over rows that share one scoring
    # convention -- otherwise a metric could win on rows we are about to drop.
    results, source_report = metrics.filter_to_canonical_source(results)
    for sr in source_report:
        if sr["note"]:
            print(f"\nSource scale conflict {sr['benchmark_id']}: {sr['note']}")
        else:
            print(f"\nSource scale conflict {sr['benchmark_id']}: kept "
                  f"{sr['kept_rows']} rows from '{sr['kept_source']}', "
                  f"dropped {sr['dropped_rows']} on an incompatible scale")

    # Pass 4: one canonical metric per benchmark. Runs AFTER the benchmark
    # removals so coverage is counted over the surviving population only.
    chosen, metric_report = metrics.choose_canonical_metrics(results)
    results, dropped_rows = metrics.filter_to_canonical(results, chosen)
    contested = [m for m in metric_report if m["n_metrics"] > 1]
    lost = sum(m["models_lost"] for m in contested)
    print(f"\nCanonical metric: {len(contested)} benchmarks had >1 metric; "
          f"dropped {dropped_rows} non-canonical result rows "
          f"({lost} model-cells lost)")
    defects = sorted(set(config.KNOWN_METRIC_COLUMN_DEFECTS) & set(chosen))
    if defects:
        print(f"  NOTE: {len(defects)} benchmark(s) have a defective metric column "
              f"that this cannot fix ({', '.join(defects)}) -- see "
              f"config.KNOWN_METRIC_COLUMN_DEFECTS")

    # Dropping rows can strand a model or a benchmark, so re-cascade to a clean
    # state rather than assuming it cannot happen.
    orphans = integrity.check_orphans(benchmarks, models, results)
    if orphans["orphan_models"] or orphans["orphan_benchmarks"]:
        benchmarks, results, _ = standardise.apply_remove_benchmark(
            benchmarks, results,
            {b: "Orphaned by canonical-metric filter"
             for b in orphans["orphan_benchmarks"]})
        models, results, _ = standardise.apply_remove(
            models, results,
            {m: "Orphaned by canonical-metric filter"
             for m in orphans["orphan_models"]})
        print(f"  cascade: removed {len(orphans['orphan_benchmarks'])} benchmark(s) "
              f"and {len(orphans['orphan_models'])} model(s) left with no rows")
    if report["orphaned_models"]:
        print(f"Removed {report['orphaned_models']} models orphaned by the cascade:")
        for m in report["orphan_model_ids"]:
            print(f"  {m}")

    models, _ = stats.apply_model_stats(models, results)

    fk = integrity.check_foreign_keys(benchmarks, models, results)
    orphans = integrity.check_orphans(benchmarks, models, results)
    ok = (not fk["invalid_benchmark_ids"] and not fk["invalid_model_names"]
          and not orphans["orphan_benchmarks"] and not orphans["orphan_models"])
    print(f"\nIntegrity check: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print(f"  invalid_benchmark_ids: {sorted(fk['invalid_benchmark_ids'])}")
        print(f"  invalid_model_names: {sorted(fk['invalid_model_names'])}")
        print(f"  orphan_benchmarks: {sorted(orphans['orphan_benchmarks'])}")
        print(f"  orphan_models: {sorted(orphans['orphan_models'])}")

    out_dir.mkdir(parents=True, exist_ok=True)
    io.save_csv(benchmarks, out_dir / "benchmarks.csv")
    io.save_csv(models, out_dir / "models.csv")
    io.save_csv(results, out_dir / "results.csv")

    if check_only:
        return _check(out_dir, ok)

    by_category = (non_text.assign(category=non_text["category"].fillna("(blank)"))
                            .groupby("category").size().sort_values(ascending=False))

    print(f"\nFinal counts: benchmarks {before['benchmarks']} -> {len(benchmarks)}  "
          f"models {before['models']} -> {len(models)}  "
          f"results {before['results']} -> {len(results)}")
    print(f"Written to: {out_dir}")

    readme = out_dir / "README.md"
    lines = [
        "# Text-only derived copy",
        "",
        f"Generated {date.today().isoformat()} by `scripts/make_text_only_copy.py` "
        "from `data/*.csv`.",
        "",
        "**This directory is tracked in git, but it is NOT hand-maintained.** It is "
        "fully reproducible from the canonical tables plus the knowledge bases in "
        "`scripts/lib/config.py`; `python3 scripts/make_text_only_copy.py --check` "
        "asserts exactly that. Never hand-edit these CSVs -- encode the decision in "
        "`config.py` and regenerate, or the edit is silently lost on the next run.",
        "",
        "Three orthogonal removals are applied, then cascades "
        "(benchmark -> its `results.csv` rows -> any model left with zero results):",
        "",
        "1. **Modality** -- every benchmark classified NON_TEXT by "
        "`scripts.lib.modality.classify_benchmark_modality` (requires the model to "
        "consume/produce image, audio, or video content).",
        "2. **Score redundancy** -- every benchmark id in "
        "`config.SCORE_REDUNDANT_BENCHMARKS`: near-duplicate columns whose variance "
        "is already carried by a kept benchmark, decided from pairwise-correlation "
        "audits rather than from names.",
        "3. **Canonical metric** -- one metric per benchmark "
        "(`scripts.lib.metrics`), so a matrix cell measures one thing. Without "
        "this the aggregation averages incommensurable metrics, and because "
        "which metric a model received is largely decided by which leaderboard "
        "scored it, part of each column's variance would be a function of its "
        "source rather than of capability.",
        "",
        f"Totals: benchmarks {before['benchmarks']} -> {len(benchmarks)}, "
        f"models {before['models']} -> {len(models)}, "
        f"results {before['results']} -> {len(results)}.",
        "",
        "## Removed benchmarks by category",
        "",
    ]
    for cat, n in by_category.items():
        lines.append(f"- {cat}: {n}")
    lines += [
        "",
        "## Allow-list exceptions (config.TEXT_MODALITY_ALLOWLIST)",
        "",
        "Benchmarks whose category/description matched a non-text pattern but were "
        "confirmed text-only on inspection:",
        "",
    ]
    for bid, reason in config.TEXT_MODALITY_ALLOWLIST.items():
        lines.append(f"- `{bid}`: {reason}")
    lines += [
        "",
        "## Non-text overrides (config.NON_TEXT_MODALITY_OVERRIDES)",
        "",
        "Benchmarks confirmed non-text on inspection despite keyword-free or "
        "mislabeled metadata (found by the 2026-07-17 audit):",
        "",
    ]
    for bid, reason in config.NON_TEXT_MODALITY_OVERRIDES.items():
        lines.append(f"- `{bid}`: {reason}")
    lines += [
        "",
        f"## Benchmarks removed ({len(redundant)}) — redundant, translated, or defective",
        "",
        "From `config.SCORE_REDUNDANT_BENCHMARKS` (near-duplicate columns, decided "
        "from full pairwise Pearson correlation over the models evaluated on both), "
        "`config.TRANSLATION_DUPLICATE_BENCHMARKS` (translations of an in-corpus "
        "original, decided on construct), and `config.DEFECTIVE_BENCHMARKS`:",
        "",
    ]
    for bid, reason in sorted(redundant.items()):
        lines.append(f"- `{bid}`: {reason}")
    lines += [
        "",
        "## Audited but deliberately KEPT (config.KEPT_DESPITE_CORRELATION)",
        "",
        "Families whose columns correlate highly but *not uniformly* -- the spread "
        "is real capability variance, not duplication. Recorded so these decisions "
        "are not silently re-litigated:",
        "",
    ]
    for family, reason in config.KEPT_DESPITE_CORRELATION.items():
        lines.append(f"- `{family}`: {reason}")

    lines += [
        "",
        f"## Canonical metric selection ({len(contested)} contested benchmarks)",
        "",
        f"{dropped_rows} result rows dropped, {lost} model-cells lost. Selection "
        "order: normalise (case/whitespace) -> alias "
        "(`config.METRIC_NAME_ALIASES`) -> override "
        "(`config.CANONICAL_METRIC_OVERRIDES`) -> most models covered.",
        "",
        "| benchmark | chosen | why | models kept | models lost | candidates |",
        "|---|---|---|---:|---:|---|",
    ]
    for m in sorted(contested, key=lambda x: -x["models_lost"]):
        lines.append(
            f"| `{m['benchmark_id']}` | {m['chosen']} | {m['reason']} | "
            f"{m['models_kept']} | {m['models_lost']} | {m['candidates']} |")
    if anomalies:
        lines += [
            "",
            "## Anomalous rows dropped (config.ANOMALOUS_RESULT_ROWS)",
            "",
            "Individual values inconsistent with their own column by a margin no "
            "plausible model difference explains, and not repairable from source. "
            "Canonical data keeps them; only this derived copy drops them:",
            "",
        ]
        for (bid, mid), reason, found in anomalies:
            state = "" if found else "  **(NOT FOUND -- stale entry)**"
            lines.append(f"- `{mid}` on `{bid}`: {reason}{state}")
    if source_report:
        lines += [
            "",
            "## Source-level scale conflicts (config.SOURCE_SCALE_CONFLICTS)",
            "",
            "One metric *name* covering two incompatible scoring conventions, "
            "separable only by source. Correlations are unchanged by a linear "
            "rescaling of a whole column, so a normalised column is fine -- "
            "provided every row shares the convention, which is what these "
            "removals enforce:",
            "",
        ]
        for sr in source_report:
            if sr["note"]:
                lines.append(f"- `{sr['benchmark_id']}`: {sr['note']}")
            else:
                lines.append(
                    f"- `{sr['benchmark_id']}`: kept {sr['kept_rows']} rows from "
                    f"**{sr['kept_source']}**, dropped {sr['dropped_rows']} on an "
                    f"incompatible scale")
    if config.DEFECTIVE_BENCHMARKS:
        lines += [
            "",
            "## Benchmarks removed as structurally defective "
            "(config.DEFECTIVE_BENCHMARKS)",
            "",
            "Columns that no metric choice can make interpretable -- the defect is "
            "in the data. Re-add if the source is re-extracted:",
            "",
        ]
        for bid, reason in config.DEFECTIVE_BENCHMARKS.items():
            lines.append(f"- `{bid}`: {reason}")
    if defects:
        lines += [
            "",
            "### Metric-column defects on record (config.KNOWN_METRIC_COLUMN_DEFECTS)",
            "",
            "Choosing a canonical metric does NOT fix these -- they need a "
            "source-level re-extraction. The choice made for them is arbitrary:",
            "",
        ]
        for bid in defects:
            lines.append(f"- `{bid}`: {config.KNOWN_METRIC_COLUMN_DEFECTS[bid]}")
    readme.write_text("\n".join(lines) + "\n")
    print(f"Wrote provenance note: {readme}")

    return 0 if ok else 1


def _check(out_dir, ok):
    """--check: assert the freshly-generated tables reproduce the committed
    ones exactly. This is the invariant that makes data/text_only/ safe to
    regenerate despite being tracked -- if it fails, the committed copy holds
    a hand-edit that is not encoded in config.py, and regenerating would
    silently lose it."""
    import filecmp

    failures = []
    for name in ("benchmarks.csv", "models.csv", "results.csv"):
        committed, regenerated = OUT_DIR / name, out_dir / name
        if not committed.exists():
            failures.append(f"{name}: no committed copy to compare against")
        elif not filecmp.cmp(committed, regenerated, shallow=False):
            a = pd.read_csv(committed, dtype=str, low_memory=False)
            b = pd.read_csv(regenerated, dtype=str, low_memory=False)
            detail = (f"{len(a)} committed rows vs {len(b)} regenerated"
                      if len(a) != len(b) else "same row count, differing content")
            failures.append(f"{name}: DIFFERS ({detail})")
        else:
            print(f"  {name}: identical")

    if failures:
        print("\n--check FAILED -- the committed copy is not reproducible:")
        for f in failures:
            print(f"  {f}")
        print("\n  A hand-edit exists that config.py does not encode. Find it and "
              "move it into config.py before regenerating, or it will be lost.")
        return 1
    print("\n--check PASSED: committed data/text_only/ is exactly reproducible.")
    return 0 if ok else 1


if __name__ == "__main__":
    check = "--check" in sys.argv
    if check:
        with tempfile.TemporaryDirectory() as tmp:
            sys.exit(main(check_only=True, out_dir=Path(tmp)))
    sys.exit(main())
