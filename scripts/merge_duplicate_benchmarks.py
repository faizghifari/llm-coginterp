#!/usr/bin/env python3
"""
Benchmark Deduplication — Cross-Source Prefix Merge
====================================================
The PwC / Kaggle / HELM imports introduced source-prefixed benchmark_ids
(`pwc_*`) and underscore-variant IDs that duplicate existing native benchmarks
(e.g. `pwc_gsm8k` vs `gsm8k`, `med_qa` vs `medqa`). These weren't caught by the
dedup tooling because the IDs differ as strings.

This script merges each duplicate pair into one canonical benchmark_id:
  - reassigns all results.benchmark_id  merged → canonical
  - ports any non-empty benchmarks.csv metadata field from the merged row into
    an *empty* field of the canonical row (e.g. HELM's source_url)
  - deletes the merged benchmarks.csv row
  - dedupes results on RESULT_IDENTITY_KEY (HELM/PwC rows differ by source_url,
    so genuine multi-source scores are preserved as separate rows)

Canonical-key rule:
  - drop the `pwc_` source prefix where a native ID exists
  - for HELM underscore-variants vs the native concatenated acronym, keep the
    concatenated form (project's dominant convention: gsm8k, truthfulqa, ...)

Usage:
  python3 scripts/merge_duplicate_benchmarks.py           # dry run
  python3 scripts/merge_duplicate_benchmarks.py --write   # apply

Always run afterwards:
  python3 scripts/verify_data.py
  python3 scripts/manage_data.py recompute-stats --write
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.lib import io, config

# merged_id → canonical_id
MERGE = {
    # ── drop pwc_ prefix into native ID ──────────────────────────────────────
    "pwc_flores_200":      "flores_200",
    "pwc_gsm8k":           "gsm8k",
    "pwc_raft":            "raft",
    "pwc_winogrande":      "winogrande",
    "pwc_cnn_daily_mail":  "summarization_cnndm",
    "pwc_x_sum":           "summarization_xsum",
    # ── both pwc_, merge smaller spelling into larger ────────────────────────
    "pwc_story_cloze":     "pwc_storycloze",
    # ── HELM underscore-variant → native concatenated acronym ────────────────
    "arabic_mmlu":         "arabicmmlu",
    "harm_bench":          "harmbench",
    "med_mcqa":            "medmcqa",
    "med_qa":              "medqa",
    "pubmed_qa":           "pubmedqa",
    "thai_exam":           "thaiexam",
    "truthful_qa":         "truthfulqa",
}


def validate(b_df):
    existing = set(b_df["benchmark_id"])
    ok = True
    for merged, canon in MERGE.items():
        if merged not in existing:
            print(f"  WARN  merged source missing: {merged!r}")
        if canon not in existing:
            print(f"  ERROR canonical target missing: {canon!r}")
            ok = False
    return ok


def apply(b_df, r_df, write: bool):
    b = b_df.copy()
    r = r_df.copy()
    b_indexed = {row["benchmark_id"]: idx for idx, row in b.iterrows()}

    for merged, canon in MERGE.items():
        if merged not in set(b["benchmark_id"]):
            print(f"  WARN  merged source missing: {merged!r}")
            continue
        n = (r_df["benchmark_id"] == merged).sum()
        print(f"  MERGE  {merged!r} → {canon!r}  ({n}r)")

        if write:
            # port non-empty metadata from merged into empty fields of canonical
            mrow = b[b["benchmark_id"] == merged].iloc[0]
            cmask = b["benchmark_id"] == canon
            ported = []
            for col in b.columns:
                if col == "benchmark_id":
                    continue
                cval = b.loc[cmask, col].values[0]
                mval = mrow[col]
                if (cval is None or str(cval).strip() == "") and str(mval).strip() != "":
                    b.loc[cmask, col] = mval
                    ported.append(col)
            if ported:
                print(f"           ported metadata into canonical: {', '.join(ported)}")
            # reassign results, drop merged benchmark row
            r.loc[r["benchmark_id"] == merged, "benchmark_id"] = canon
            b = b[b["benchmark_id"] != merged]

    dupes_dropped = 0
    if write:
        key_cols = [c for c in config.RESULT_IDENTITY_KEY if c in r.columns]
        before = len(r)
        r = r.drop_duplicates(subset=key_cols).reset_index(drop=True)
        dupes_dropped = before - len(r)

    return b, r, dupes_dropped


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    print("Loading files...")
    b_df, m_df, r_df = io.load_data()
    print(f"  benchmarks: {len(b_df)}  results: {len(r_df)}")

    print("\nValidating maps...")
    if not validate(b_df):
        print("Validation failed — fix errors above before applying.")
        sys.exit(1)
    print("  All targets verified ✓")

    print(f"\n{'Applying' if args.write else 'Dry run —'} changes:\n")
    b_new, r_new, n_dupes = apply(b_df, r_df, write=args.write)

    if args.write:
        io.save_csv(b_new, config.BENCHMARKS_CSV)
        io.save_results(r_new)
        print(f"\nWritten.")
        print(f"  benchmarks: {len(b_df)} → {len(b_new)}  (−{len(b_df)-len(b_new)})")
        print(f"  results:    {len(r_df)} → {len(r_new)}  ({n_dupes} post-merge dupes dropped)")
        print("\nNext steps:")
        print("  python3 scripts/verify_data.py")
        print("  python3 scripts/manage_data.py recompute-stats --write")
    else:
        print(f"\nDry run summary: {len(MERGE)} benchmark pairs to merge.")
        print("Pass --write to apply.")


if __name__ == "__main__":
    sys.exit(main())
