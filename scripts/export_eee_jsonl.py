#!/usr/bin/env python3
"""Export results.csv to the EEE JSONL schema (v0.2.1).

Thin wrapper around scripts/lib/export.py — kept as its own entry point
because it's referenced by name in docs/METHODOLOGY.md. Writes:
  data/eee_output/by_benchmark/{benchmark_id}.jsonl  (one file per benchmark)
  data/eee_output/all_evaluations.jsonl              (everything, consolidated)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.lib import export


def main():
    total, output_dir, consolidated_path = export.export_eee_jsonl()
    print(f"Wrote {total} records to {output_dir}/*.jsonl")
    print(f"Wrote {total} records to {consolidated_path}")


if __name__ == "__main__":
    main()
