#!/usr/bin/env python3
"""Single CLI entry point for the dataset maintenance toolkit.

Run `python3 scripts/manage_data.py <command> --help` for per-command
options. All commands that can write data default to a dry run; pass
--write to persist changes, and always re-run `verify` afterwards.

Commands:
  verify             Run all data integrity checks (FK, orphans,
                     exhaustion, pending-benchmark completeness). Same
                     checks as verify_data.py, available from one place.
  dupes              Report duplicate result rows (redundant vs.
                     conflicting) without modifying any files.
  dedup              Resolve duplicate result rows: keep one row per
                     evaluation (trust-tier + recency for conflicts),
                     move the rest to data/results_duplicates.csv.
  find-aliases       Suggest fuzzy-match candidates for orphan
                     model_names in results.csv that have no matching
                     models.csv entry.
  apply-aliases      Apply a rename map (JSON: {"old_id": "new_id"}) to
                     both models.csv and results.csv.
  standardize-ids    Normalize every model_id to lowercase-hyphenated
                     form, merging any collisions this creates.
  categorize-models  Classify every model row as KEEP / FLAG / REMOVE,
                     to surface fine-tune/orphan cleanup candidates.
  recompute-stats    Recompute models.csv's benchmark_count/
                     total_results/avg_score from results.csv (these
                     drift out of sync the moment results.csv changes).

Examples:
  python3 scripts/manage_data.py verify
  python3 scripts/manage_data.py dupes --verbose
  python3 scripts/manage_data.py dedup --write
  python3 scripts/manage_data.py find-aliases
  python3 scripts/manage_data.py apply-aliases --map-file my_renames.json --write
  python3 scripts/manage_data.py standardize-ids --write
  python3 scripts/manage_data.py categorize-models --output data/models_categorized.csv
  python3 scripts/manage_data.py recompute-stats --write
"""
import argparse
import json
import sys
from pathlib import Path

# Allow running this script directly (`python3 scripts/manage_data.py`)
# from any working directory by putting the repo root -- this file's
# grandparent -- on sys.path so `scripts.lib` resolves as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from scripts.lib import aliases, categorize, config, dedup, integrity, io, stats


def cmd_verify(args):
    benchmarks, models, results = io.load_data()
    report = integrity.run_all(benchmarks, models, results, exhaustion_threshold=args.threshold)
    print(integrity.format_report(report))
    return 1 if report["foreign_keys"]["invalid_benchmark_ids"] else 0


def cmd_dupes(args):
    _, _, results = io.load_data()
    groups = dedup.classify(results)
    redundant = [g for g in groups if g["kind"] == "redundant"]
    conflicts = [g for g in groups if g["kind"] == "conflict"]
    print(f"Duplicate evaluation groups: {len(groups)}")
    print(f"  Redundant (same score, safe to collapse): {len(redundant)}")
    print(f"  Conflicting (different scores, needs trust-tier resolution): {len(conflicts)}")
    if args.verbose and conflicts:
        cols = [c for c in ("model_name", "benchmark_id", "metric_name", "score", "source_url", "year_evaluated") if c in results.columns]
        print("\n=== Conflicts ===")
        for g in conflicts:
            print(g["rows"][cols].to_string(index=False))
            print()
    return 0


def cmd_dedup(args):
    _, _, results = io.load_data()
    clean, discarded = dedup.resolve(results)
    print(f"Original rows: {len(results)}")
    print(f"Kept: {len(clean)}")
    print(f"Discarded (would move to results_duplicates.csv): {len(discarded)}")
    if not args.write:
        print("\nDry run only — pass --write to persist.")
        return 0

    io.save_results(clean)
    if len(discarded) > 0:
        existing = io.load_csv(config.DUPLICATES_CSV) if config.DUPLICATES_CSV.exists() else pd.DataFrame()
        combined = pd.concat([existing, discarded], ignore_index=True) if len(existing) else discarded
        io.save_csv(combined, config.DUPLICATES_CSV)
    print("\nWrote changes. Re-run `python3 scripts/manage_data.py verify` to confirm FK integrity.")
    return 0


def cmd_find_aliases(args):
    _, models, results = io.load_data()
    orphans = sorted(set(results["model_name"]) - set(models["model_id"]))
    candidates = aliases.find_alias_candidates(orphans, list(models["model_id"]), cutoff=args.cutoff)
    print(f"Orphan model_names in results.csv: {len(orphans)}")
    print(f"Candidates found for: {len(candidates)}\n")
    for name, matches in candidates.items():
        print(f"  {name!r} -> {matches}")

    unmatched = [o for o in orphans if o not in candidates]
    if unmatched:
        print(f"\nNo close match found for {len(unmatched)} (likely genuinely new models):")
        for o in unmatched:
            print(f"  {o}")
    return 0


def cmd_apply_aliases(args):
    benchmarks, models, results = io.load_data()
    with open(args.map_file) as f:
        rename_map = json.load(f)

    models, results, report = aliases.apply_rename_map(models, results, rename_map)
    print(f"Renamed {report['renamed_result_rows']} result rows")
    print(f"Renamed {report['models_renamed']} models.csv rows")
    if report["models_merged"]:
        print(f"Collisions merged (kept first occurrence): {report['models_merged']}")

    if not args.write:
        print("\nDry run only — pass --write to persist.")
        return 0

    io.save_models(models)
    io.save_results(results)
    print("\nWrote changes. Re-run `python3 scripts/manage_data.py verify` to confirm FK integrity.")
    return 0


def cmd_standardize_ids(args):
    _, models, results = io.load_data()
    models, results, rename_map, report = aliases.standardize_all(models, results)
    print(f"Standardization renames: {len(rename_map)}")
    for old, new in rename_map.items():
        print(f"  {old} -> {new}")
    print(f"\nRenamed {report['renamed_result_rows']} result rows")
    if report["models_merged"]:
        print(f"Collisions merged (kept first occurrence): {report['models_merged']}")

    if not args.write:
        print("\nDry run only — pass --write to persist.")
        return 0

    io.save_models(models)
    io.save_results(results)
    print("\nWrote changes. Re-run `python3 scripts/manage_data.py verify` to confirm FK integrity.")
    return 0


def cmd_categorize_models(args):
    _, models, _ = io.load_data()
    categorized = categorize.categorize_all(models)
    print(categorized["category"].value_counts().to_string())

    if args.output:
        io.save_csv(categorized, args.output)
        print(f"\nFull breakdown saved to: {args.output}")

    for cat in ("FLAG", "REMOVE"):
        subset = categorized[categorized["category"] == cat]
        if len(subset):
            print(f"\n=== {cat} ({len(subset)}) ===")
            for _, r in subset.iterrows():
                print(f"  {r['model_id']:40s} | {str(r.get('developer', '')):25s} | {r['reason']}")
    return 0


def cmd_recompute_stats(args):
    _, models, results = io.load_data()
    updated, report = stats.apply_model_stats(models, results)
    print(f"Models with matching results.csv rows (stats updated): {report['updated']}")
    print(f"Models with zero results (set to 0/0/blank): {report['zero_results']}")

    if not args.write:
        print("\nDry run only — pass --write to persist.")
        return 0

    io.save_models(updated)
    print("\nWrote changes to models.csv.")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        description="Dataset maintenance toolkit for benchmarks.csv / models.csv / results.csv.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("verify", help="Run all data integrity checks.")
    p.add_argument("--threshold", type=int, default=5, help="Exhaustion threshold (default: 5 rows).")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("dupes", help="Report duplicate result rows (read-only).")
    p.add_argument("--verbose", action="store_true", help="Print every conflicting group in full.")
    p.set_defaults(func=cmd_dupes)

    p = sub.add_parser("dedup", help="Resolve duplicate result rows.")
    p.add_argument("--write", action="store_true", help="Persist changes (default: dry run).")
    p.set_defaults(func=cmd_dedup)

    p = sub.add_parser("find-aliases", help="Suggest alias candidates for orphan model_names.")
    p.add_argument("--cutoff", type=float, default=0.8, help="Fuzzy-match similarity cutoff (default: 0.8).")
    p.set_defaults(func=cmd_find_aliases)

    p = sub.add_parser("apply-aliases", help="Apply a rename map to models.csv and results.csv.")
    p.add_argument("--map-file", required=True, help='JSON file: {"old_id": "new_id", ...}')
    p.add_argument("--write", action="store_true", help="Persist changes (default: dry run).")
    p.set_defaults(func=cmd_apply_aliases)

    p = sub.add_parser("standardize-ids", help="Normalize all model_ids to lowercase-hyphenated form.")
    p.add_argument("--write", action="store_true", help="Persist changes (default: dry run).")
    p.set_defaults(func=cmd_standardize_ids)

    p = sub.add_parser("categorize-models", help="Classify models as KEEP / FLAG / REMOVE.")
    p.add_argument("--output", help="Optional CSV path to save the full breakdown.")
    p.set_defaults(func=cmd_categorize_models)

    p = sub.add_parser("recompute-stats", help="Recompute models.csv's benchmark_count/total_results/avg_score from results.csv.")
    p.add_argument("--write", action="store_true", help="Persist changes (default: dry run).")
    p.set_defaults(func=cmd_recompute_stats)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
