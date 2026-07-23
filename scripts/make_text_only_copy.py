#!/usr/bin/env python3
"""Build a derived, text-only-modality copy of the 3 canonical dataset CSVs.

Reads data/*.csv (never writes to it), classifies every benchmark on the
TEXT/NON_TEXT modality axis (scripts/lib/modality.py), cascade-removes every
NON_TEXT benchmark plus its results.csv rows plus any model left with zero
remaining results, and writes the result to data/text_only/. Fully
idempotent and disposable -- rerun any time canonical data changes or the
modality classifier (scripts/lib/config.py's NON_TEXT_MODALITY_PATTERNS /
TEXT_MODALITY_ALLOWLIST) is updated.
"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.lib import config, integrity, io, modality, standardise, stats

OUT_DIR = config.DATA_DIR / "text_only"


def main():
    benchmarks, models, results = io.load_data()
    before = {"benchmarks": len(benchmarks), "models": len(models), "results": len(results)}

    scoped = modality.classify_benchmark_modality_all(benchmarks)
    non_text = scoped[scoped["modality_category"] == "NON_TEXT"]
    remove_map = dict(zip(non_text["benchmark_id"], non_text["modality_reason"]))

    print(f"Classified {len(scoped)} benchmarks: "
          f"{len(scoped) - len(non_text)} TEXT, {len(non_text)} NON_TEXT")

    benchmarks, models, results, report = standardise.cascade_remove_benchmarks(
        benchmarks, models, results, remove_map)

    print(f"\nRemoved {report['removed_benchmarks']} benchmarks "
          f"({report['removed_results']} result rows)")
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

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    io.save_csv(benchmarks, OUT_DIR / "benchmarks.csv")
    io.save_csv(models, OUT_DIR / "models.csv")
    io.save_csv(results, OUT_DIR / "results.csv")

    by_category = (non_text.assign(category=non_text["category"].fillna("(blank)"))
                            .groupby("category").size().sort_values(ascending=False))

    print(f"\nFinal counts: benchmarks {before['benchmarks']} -> {len(benchmarks)}  "
          f"models {before['models']} -> {len(models)}  "
          f"results {before['results']} -> {len(results)}")
    print(f"Written to: {OUT_DIR}")

    readme = OUT_DIR / "README.md"
    lines = [
        "# Text-only derived copy",
        "",
        f"Generated {date.today().isoformat()} by `scripts/make_text_only_copy.py` "
        "from `data/*.csv`. Regenerate any time by rerunning that script -- this "
        "directory is gitignored, not hand-maintained.",
        "",
        "Every benchmark classified NON_TEXT by "
        "`scripts.lib.modality.classify_benchmark_modality` (requires the model to "
        "consume/produce image, audio, or video content) was cascade-removed, along "
        "with its `results.csv` rows and any model left with zero remaining results.",
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
    readme.write_text("\n".join(lines) + "\n")
    print(f"Wrote provenance note: {readme}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
