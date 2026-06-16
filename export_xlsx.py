#!/usr/bin/env python3
"""Export benchmarks.csv / models.csv / results.csv to a single Excel
workbook (one sheet each, frozen header row, auto-fit columns).

Thin wrapper around scripts/lib/export.py — kept as its own entry point
because it's referenced by name in METHODOLOGY.md. Writes
data/llm_benchmarks_export.xlsx.
"""
from scripts.lib import export


def main():
    output_path, counts = export.export_xlsx()
    print(f"Wrote {output_path}")
    for sheet, count in counts.items():
        print(f"  {sheet}: {count} rows")


if __name__ == "__main__":
    main()
