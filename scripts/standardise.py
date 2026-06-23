#!/usr/bin/env python3
"""Normalize and standardise models, results, and benchmarks in one pass.

This single tool replaces the seven historical "standardise pass"
scripts plus deduplicate_models.py and merge_duplicate_benchmarks.py,
which each hardcoded one cleanup's map. Here the maps live in a JSON
rules file, so the same code applies any pass:

    {
      "merge_benchmark": { "merged_id":  "canonical_id", ... },
      "remove":          { "model_id":   "reason", ... },
      "rename":          { "old_id":     "new_canonical_id", ... },
      "remap":           { "old_id":     "existing_canonical_id", ... },
      "setup_extract":   { "old_id":     {"base": "base_id", "setup": "cot"}, ... }
    }

Every key is optional. Operations run in the order above (benchmark
merges first, then model removals, renames, remaps, and setup
extractions), then duplicate result rows and duplicate model rows are
collapsed, and finally models.csv's aggregate stats are recomputed.

Run with no --rules to do just the cleanup half (result + model dedup +
stat recompute) -- handy after any other edit to results.csv.

Operation meanings (see scripts/lib/standardise.py for the exact logic):
  merge_benchmark  relabel a duplicate benchmark to its canonical id,
                   porting metadata, then drop the duplicate row.
  remove           cascade-delete a model row + all its results.
  rename           rename a model_id in place (same model, canonical id).
  remap            merge a model into an existing canonical model_id.
  setup_extract    fold a technique-qualified model into its base, moving
                   the technique into results.setup.

Usage:
  python3 scripts/standardise.py --rules pass.json            # dry run
  python3 scripts/standardise.py --rules pass.json --write    # apply
  python3 scripts/standardise.py --write                      # dedup-only

Always run afterwards:
  python3 scripts/verify_data.py
"""
import argparse
import json
import sys
from pathlib import Path

# Allow running directly from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.lib import config, io, standardise, stats

KNOWN_KEYS = {"merge_benchmark", "remove", "rename", "remap", "setup_extract"}


def load_rules(path):
    if path is None:
        return {}
    with open(path) as f:
        rules = json.load(f)
    unknown = set(rules) - KNOWN_KEYS
    if unknown:
        print(f"  WARN  ignoring unknown rule keys: {sorted(unknown)}")
    return rules


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rules", help="JSON rules file (see module docstring). "
                                        "Omit to run cleanup-only (dedup + recompute).")
    parser.add_argument("--write", action="store_true",
                        help="Persist changes (default: dry run).")
    args = parser.parse_args(argv)

    rules = load_rules(args.rules)

    print("Loading files...")
    benchmarks, models, results = io.load_data()
    print(f"  benchmarks: {len(benchmarks)}  models: {len(models)}  results: {len(results)}")
    b0, m0, r0 = len(benchmarks), len(models), len(results)

    print(f"\n{'Applying' if args.write else 'Dry run —'} operations:\n")

    if rules.get("merge_benchmark"):
        benchmarks, results, _ = standardise.merge_benchmarks(
            benchmarks, results, rules["merge_benchmark"])
    if rules.get("remove"):
        models, results, _ = standardise.apply_remove(models, results, rules["remove"])
    if rules.get("rename"):
        models, results, _ = standardise.apply_rename(models, results, rules["rename"])
    if rules.get("remap"):
        models, results, _ = standardise.apply_remap(models, results, rules["remap"])
    if rules.get("setup_extract"):
        models, results, _ = standardise.apply_setup_extract(
            models, results, rules["setup_extract"])

    results, r_dropped = standardise.dedup_results(results)
    models, m_dropped = standardise.dedup_models(models)
    print(f"\n  dedup: dropped {r_dropped} duplicate result rows, "
          f"{m_dropped} duplicate model rows")

    models, _ = stats.apply_model_stats(models, results)
    print("  recomputed models.csv aggregate stats")

    print("\nSummary:")
    print(f"  benchmarks: {b0} -> {len(benchmarks)}  ({len(benchmarks)-b0:+d})")
    print(f"  models:     {m0} -> {len(models)}  ({len(models)-m0:+d})")
    print(f"  results:    {r0} -> {len(results)}  ({len(results)-r0:+d})")

    if not args.write:
        print("\nDry run only — pass --write to persist.")
        return 0

    io.save_benchmarks(benchmarks)
    io.save_models(models)
    io.save_results(results)
    print("\nWrote changes. Next:")
    print("  python3 scripts/verify_data.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
