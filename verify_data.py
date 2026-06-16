#!/usr/bin/env python3
"""Data integrity checks for benchmarks.csv / models.csv / results.csv.

Thin wrapper around the checks in scripts/lib/integrity.py — kept as its
own entry point because `python3 verify_data.py` is the command referenced
throughout README.md and METHODOLOGY.md as "run this after every change".

For everything else (duplicate detection/resolution, alias fixes, model
categorization), see `python3 manage_data.py --help`.
"""
import sys

from scripts.lib import integrity, io


def main():
    benchmarks, models, results = io.load_data()
    report = integrity.run_all(benchmarks, models, results)
    print("--- Verification Report ---")
    print(integrity.format_report(report))
    return 1 if report["foreign_keys"]["invalid_benchmark_ids"] else 0


if __name__ == "__main__":
    sys.exit(main())
