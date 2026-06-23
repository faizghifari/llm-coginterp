#!/usr/bin/env python3
"""
Resolve the 3 remaining conflict benchmarks (opencompass / eifbench / mmar)
===========================================================================
These were the last "conflicting duplicate-evaluation groups" flagged in
notes/TODO.md. They all trace to the same root cause: the bulk leaderboard /
PwC imports squashed multi-dimensional results into a single mislabeled
`accuracy` metric, and standardise pass 5 then collapsed the resulting
same-key conflicts by arbitrarily keeping one row. On re-verification against
each benchmark's cited source the stored numbers did not match the source, so
the conflicts were symptoms of mis-extraction, not genuine score disputes.

Resolution policy: "correct + prune to source" (user-approved).

  MMAR  (arXiv 2505.13032 — audio reasoning, NOT multilingual/MT):
    - Gemini 1.5 Pro does not appear in the paper at all → DROP that result.
    - Correct the remaining 3 scores to Table 2 (Avg %):
        GPT-4o Audio 54.3→63.5, Qwen2-Audio 52.1→30.4, SALMONN-13B 38.2→33.2.
    - Fix benchmark metadata: category multilingual→audio/speech,
      subcategory/task_types machine-translation→audio-reasoning, year 2025.

  OpenCompass (rank.opencompass.org.cn — dynamic academic leaderboard):
    - The leaderboard's first/Average column was kept; the other 6 capability
      dimensions were lost and the API is inaccessible (SPA shell), so they
      cannot be recovered. The kept value is the overall average (≈ mean of the
      other 6 dimensions per model), so relabel metric_name accuracy→overall
      and record that only the aggregate score is retained.

  EIFBench (arXiv 2506.08375 — complex instruction following):
    - Paper reports ILA/CLA sub-scores (0–1) per scenario; our single
      `accuracy` values can't be verified against it, so the 15 result rows are
      left untouched. Fix benchmark metadata: category alignment/safety→
      instruction-following, add paper_url, year 2025.

Usage:
  python3 scripts/resolve_conflict_benchmarks.py           # dry run
  python3 scripts/resolve_conflict_benchmarks.py --write   # apply

Always run afterwards:
  python3 scripts/verify_data.py
  python3 scripts/manage_data.py recompute-stats --write
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.lib import io, config

# ── MMAR result fixes (benchmark_id == "mmar") ───────────────────────────────
MMAR_DROP_MODELS = ["Gemini 1.5 Pro"]            # not present in arXiv 2505.13032
MMAR_SCORE_FIX = {                               # model_name → corrected Avg %
    "GPT-4o":      "63.5",
    "Qwen2-Audio": "30.4",
    "SALMONN-13B": "33.2",
}

# ── OpenCompass result fixes (benchmark_id == "opencompass") ─────────────────
OPENCOMPASS_METRIC_RELABEL = ("accuracy", "overall")

# ── benchmark metadata corrections: benchmark_id → {col: value} ──────────────
BENCH_META = {
    "mmar": {
        "category":     "audio/speech",
        "subcategory":  "audio-reasoning",
        "task_types":   "audio reasoning",
        "task_type":    "audio reasoning",
        "domain":       "Audio",
        "year":         "2025",
        "venue":        "arXiv 2025",
        "title":        "MMAR: A Challenging Benchmark for Deep Reasoning in "
                        "Speech, Audio, Music, and Their Mix",
        "paper_title":  "MMAR: A Challenging Benchmark for Deep Reasoning in "
                        "Speech, Audio, Music, and Their Mix",
        "notes":        "Audio reasoning benchmark (1000 audio-QA triplets over "
                        "speech/audio/music/mix). Scores corrected to paper "
                        "Table 2 Avg; Gemini 1.5 Pro removed (not in paper).",
    },
    "eifbench": {
        "category":     "instruction-following",
        "paper_url":    "https://arxiv.org/abs/2506.08375",
        "year":         "2025",
        "venue":        "arXiv 2025 (EMNLP 2025)",
        "title":        "EIFBench: Extremely Complex Instruction Following "
                        "Benchmark for Large Language Models",
        "paper_title":  "EIFBench: Extremely Complex Instruction Following "
                        "Benchmark for Large Language Models",
    },
    "opencompass": {
        "notes":        "CompassAcademic leaderboard. Only the overall/average "
                        "score is retained; the 6 per-capability dimensions in "
                        "the source were lost on import and the live API is "
                        "inaccessible (SPA shell), so they cannot be recovered.",
    },
}


def apply(b_df, r_df, write: bool):
    b = b_df.copy()
    r = r_df.copy()

    # ── MMAR results ─────────────────────────────────────────────────────────
    mmar = r["benchmark_id"] == "mmar"
    for mdl in MMAR_DROP_MODELS:
        n = (mmar & (r["model_name"] == mdl)).sum()
        print(f"  MMAR  DROP   {mdl!r}  ({n}r)  [not in arXiv 2505.13032]")
        if write:
            r = r[~((r["benchmark_id"] == "mmar") & (r["model_name"] == mdl))]
    for mdl, new in MMAR_SCORE_FIX.items():
        mask = (r["benchmark_id"] == "mmar") & (r["model_name"] == mdl)
        if mask.any():
            old = r.loc[mask, "score"].values[0]
            print(f"  MMAR  FIX    {mdl!r}  score {old} → {new}")
            if write:
                r.loc[mask, "score"] = new
        else:
            print(f"  WARN  MMAR model missing: {mdl!r}")

    # ── OpenCompass results: relabel metric ──────────────────────────────────
    old_m, new_m = OPENCOMPASS_METRIC_RELABEL
    oc = (r["benchmark_id"] == "opencompass") & (r["metric_name"] == old_m)
    print(f"  OPENCOMPASS  relabel metric_name {old_m!r} → {new_m!r}  ({oc.sum()}r)")
    if write:
        r.loc[oc, "metric_name"] = new_m

    # ── benchmark metadata ───────────────────────────────────────────────────
    for bid, fields in BENCH_META.items():
        mask = b["benchmark_id"] == bid
        if not mask.any():
            print(f"  WARN  benchmark missing: {bid!r}")
            continue
        for col, val in fields.items():
            if col not in b.columns:
                print(f"  WARN  {bid}: column {col!r} not in benchmarks.csv")
                continue
            old = b.loc[mask, col].values[0]
            if str(old) != str(val):
                print(f"  META  {bid}: {col} {old!r} → {val!r}")
                if write:
                    b.loc[mask, col] = val

    return b, r


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    print("Loading files...")
    b_df, _, r_df = io.load_data()
    print(f"  benchmarks: {len(b_df)}  results: {len(r_df)}")

    print(f"\n{'Applying' if args.write else 'Dry run —'} changes:\n")
    b_new, r_new = apply(b_df, r_df, write=args.write)

    if args.write:
        io.save_benchmarks(b_new)
        io.save_results(r_new)
        print(f"\nWritten.")
        print(f"  results: {len(r_df)} → {len(r_new)}  (−{len(r_df)-len(r_new)})")
        print("\nNext steps:")
        print("  python3 scripts/verify_data.py")
        print("  python3 scripts/manage_data.py recompute-stats --write")
    else:
        print("\nDry run — pass --write to apply.")


if __name__ == "__main__":
    sys.exit(main())
