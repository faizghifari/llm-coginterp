#!/usr/bin/env python3
"""
Merge Kaggle Staging Files into Main Data Files
================================================
Merges data/staging_kaggle_*.csv into data/benchmarks.csv, models.csv, results.csv.

Dry-run by default — pass --write to persist.
Always run  python3 scripts/verify_data.py  afterwards.

Key behaviours
--------------
- 10 staging benchmark_ids are remapped to existing main benchmark_ids (same benchmark,
  sourced from Kaggle in addition to the original source).
- 49 staging model names are remapped to existing main model_ids.
- All other benchmarks/models are inserted as new rows.
- Chess Suite and Game Arena carry Elo-style absolute scores (>100) — this is correct.

Usage:
  python3 scripts/merge_kaggle_staging.py           # dry run
  python3 scripts/merge_kaggle_staging.py --write   # apply
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from scripts.lib import io, config

# ---------------------------------------------------------------------------
# Benchmark alias: staging benchmark_id → existing main benchmark_id
# These are well-known benchmarks that are already in main from other sources.
# Results are still inserted pointing to the existing benchmark_id.
# Verified 2026-06-18: all targets exist in benchmarks.csv
# ---------------------------------------------------------------------------
BENCHMARK_ALIAS = {
    "kaggle_sjmikler_mathvista":            "mathvista",
    "kaggle_aminmohamedmohami_math_500":    "math500",
    "kaggle_aminmohamedmohami_mmlu_pro":    "mmlu_prox",
    "kaggle_aminmohamedmohami_facts_search":"facts_search",
    "kaggle_aminmohamedmohami_simpleqa":    "simpleqa",
    "kaggle_andrewmingwang_facts_grounding":"facts_grounding",
    "kaggle_andrewmingwang_facts_multimodal":"facts_multimodal",
    "kaggle_andrewmingwang_gpqa":           "gpqa",
    "kaggle_andrewmingwang_multiloko":      "multiloko",
    "kaggle_vijitsingh1_aime_2025":         "aime25",
}

# ---------------------------------------------------------------------------
# Model alias: staging model_name → existing main model_id
# Verified 2026-06-18: all targets exist in models.csv
# ---------------------------------------------------------------------------
MODEL_ALIAS = {
    "Gemma 4 31B":          "Gemma 4 31B",
    "Claude Sonnet 4.6":    "Claude Sonnet 4.6",
    "DeepSeek-R1":          "DeepSeek-R1",
    "Gemma 4 26B A4B":      "Gemma 4 26B-A4B",
    "Claude Opus 4.6":      "Claude Opus 4.6",
    "GPT-5.4 mini":         "GPT-5.4 Mini",
    "gpt-oss-20b":          "gpt-oss-20b",
    "gpt-oss-120b":         "gpt-oss-120b",
    "Gemini 2.5 Pro":       "Gemini 2.5 Pro",
    "Deepseek V3.1":        "DeepSeek-V3.1",
    "Gemini 2.5 Flash":     "Gemini-2.5-Flash",
    "Claude Opus 4.5":      "Claude Opus 4.5",
    "Claude Sonnet 4.5":    "Claude Sonnet 4.5",
    "Claude Opus 4.1":      "Claude Opus 4.1",
    "Claude Sonnet 4":      "Claude Sonnet 4",
    "Gemini 2.0 Flash":     "Gemini 2.0 Flash",
    "Mistral Medium 3":     "Mistral Medium 3",
    "Gemma 3 27B":          "Gemma 3 27B",
    "Gemini 3 Pro Preview": "Gemini 3 Pro (Preview)",
    "GPT-5":                "GPT-5",
    "o4 mini":              "o4-mini",
    "Grok 4":               "Grok 4",
    "o3":                   "o3",
    "GPT-4.1":              "GPT-4.1",
    "Command A":            "Command A",
    "Claude 3.7 Sonnet":    "Claude 3.7 Sonnet",
    "o3 mini":              "o3-mini",
    "Aya Expanse 32B":      "aya-expanse-32b",
    "Grok 3 Mini":          "Grok-3-mini",
    "Claude 3.5 Haiku":     "Claude 3.5 Haiku",
    "Claude Opus 4":        "Claude Opus 4",
    "o1":                   "o1",
    "Gemini 1.5 Pro":       "Gemini-1.5-Pro",
    "Claude 3.5 Sonnet":    "Claude-3.5-Sonnet",
    "Gemini 1.5 Flash":     "Gemini 1.5 Flash",
    "GPT-4o":               "GPT-4o",
    "GPT-4o mini":          "GPT-4o-mini",
    "DeepSeek-V3":          "DeepSeek-v3",
    "o1 mini":              "o1-mini",
    "Grok 3":               "Grok-3",
    "Mistral Large 2":      "Mistral Large 2",
    "Grok 2":               "Grok-2",
    "Mixtral 8x22B":        "Mixtral (8x22B)",
    "Ministral 8B":         "Ministral 8B",
    "GPT-3.5 Turbo":        "GPT-3.5-Turbo",
    "GPT-5.1":              "GPT-5.1",
    "Llama 3.1 8B":         "Llama-3.1-8B",
    "GPT-5 mini":           "GPT-5-mini",
    "GPT-5.4 nano":         "GPT-5.4 Nano",
}

REPO_ROOT  = Path(__file__).resolve().parent.parent
STAGING_B  = REPO_ROOT / "data" / "staging_kaggle_benchmarks.csv"
STAGING_M  = REPO_ROOT / "data" / "staging_kaggle_models.csv"
STAGING_R  = REPO_ROOT / "data" / "staging_kaggle_results.csv"


def verify_alias_maps(main_b, main_m):
    ok = True
    missing_b = [(s, t) for s, t in BENCHMARK_ALIAS.items() if t not in set(main_b["benchmark_id"])]
    missing_m = [(s, t) for s, t in MODEL_ALIAS.items()     if t not in set(main_m["model_id"])]
    if missing_b:
        print("ERROR — BENCHMARK_ALIAS targets not in benchmarks.csv:")
        for s, t in missing_b: print(f"  {s!r} -> {t!r}")
        ok = False
    if missing_m:
        print("ERROR — MODEL_ALIAS targets not in models.csv:")
        for s, t in missing_m: print(f"  {s!r} -> {t!r}")
        ok = False
    if ok:
        print(f"  Alias maps verified: {len(BENCHMARK_ALIAS)} benchmark, {len(MODEL_ALIAS)} model aliases ✓")
    return ok


def apply_aliases(staging_b, staging_m, staging_r):
    staging_b = staging_b.copy()
    staging_m = staging_m.copy()
    staging_r = staging_r.copy()

    n_b = staging_r["benchmark_id"].isin(BENCHMARK_ALIAS).sum()
    staging_r["benchmark_id"] = staging_r["benchmark_id"].map(lambda x: BENCHMARK_ALIAS.get(x, x))
    staging_b["benchmark_id"] = staging_b["benchmark_id"].map(lambda x: BENCHMARK_ALIAS.get(x, x))

    n_m = staging_r["model_name"].isin(MODEL_ALIAS).sum()
    staging_r["model_name"] = staging_r["model_name"].map(lambda x: MODEL_ALIAS.get(x, x))
    staging_r["model_id"]   = staging_r["model_id"].map(  lambda x: MODEL_ALIAS.get(x, x))

    staging_m = staging_m[~staging_m["model_id"].isin(MODEL_ALIAS)].reset_index(drop=True)

    print(f"  Remapped {n_b} result rows to existing benchmark_ids")
    print(f"  Remapped {n_m} result rows to existing model_ids")
    print(f"  Staging models after alias removal: {len(staging_m)}")
    return staging_b, staging_m, staging_r


def plan_benchmarks(staging_b, main_b):
    existing = set(main_b["benchmark_id"])
    return (staging_b[~staging_b["benchmark_id"].isin(existing)],
            staging_b[ staging_b["benchmark_id"].isin(existing)])


def plan_models(staging_m, main_m):
    existing = set(main_m["model_id"])
    return (staging_m[~staging_m["model_id"].isin(existing)],
            staging_m[ staging_m["model_id"].isin(existing)])


def plan_results(staging_r, main_r, all_model_ids, all_bench_ids):
    existing_keys = set(zip(main_r["benchmark_id"], main_r["model_name"],
                            main_r["metric_name"],  main_r["source_url"]))
    mask_new = ~staging_r.apply(
        lambda row: (row["benchmark_id"], row["model_name"],
                     row["metric_name"],  row["source_url"]) in existing_keys, axis=1)
    new_rows = staging_r[mask_new]
    bad_bench = new_rows[~new_rows["benchmark_id"].isin(all_bench_ids)]
    bad_model = new_rows[~new_rows["model_name"].isin(all_model_ids)]
    fk = []
    if len(bad_bench):
        fk.append(("benchmark_id", bad_bench[["benchmark_id","model_name"]].drop_duplicates()))
    if len(bad_model):
        fk.append(("model_name",   bad_model[["model_name","benchmark_id"]].drop_duplicates()))
    return new_rows, fk


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    print("Loading files...")
    staging_b = pd.read_csv(STAGING_B, dtype=str, keep_default_na=False)
    staging_m = pd.read_csv(STAGING_M, dtype=str, keep_default_na=False)
    staging_r = pd.read_csv(STAGING_R, dtype=str, keep_default_na=False)
    main_b, main_m, main_r = io.load_data()

    print(f"\nMain:    {len(main_b)} benchmarks, {len(main_m)} models, {len(main_r)} results")
    print(f"Staging: {len(staging_b)} benchmarks, {len(staging_m)} models, {len(staging_r)} results")

    print("\n[1/4] Verifying alias maps...")
    if not verify_alias_maps(main_b, main_m):
        sys.exit(1)

    print("\n[2/4] Applying aliases...")
    staging_b, staging_m, staging_r = apply_aliases(staging_b, staging_m, staging_r)

    print("\n[3/4] Planning benchmarks & models...")
    new_bench,  skip_bench  = plan_benchmarks(staging_b, main_b)
    new_models, skip_models = plan_models(staging_m, main_m)
    print(f"  Benchmarks — skip: {len(skip_bench)}  new: {len(new_bench)}")
    print(f"  Models     — skip: {len(skip_models)}  new: {len(new_models)}")

    print("\n[4/4] Planning results...")
    all_bench_ids = set(main_b["benchmark_id"]) | set(new_bench["benchmark_id"])
    all_model_ids = set(main_m["model_id"])     | set(new_models["model_id"])
    new_results, fk = plan_results(staging_r, main_r, all_model_ids, all_bench_ids)
    print(f"  Already in main (skipped): {len(staging_r) - len(new_results)}")
    print(f"  New to add: {len(new_results)}")

    if fk:
        print("\n  ⚠ FK violations (MUST fix before write):")
        for col, df in fk:
            print(f"\n  Bad {col}:")
            print(df.to_string(index=False))
        if args.write:
            print("\nAborting --write due to FK violations.")
            return 1
    else:
        print("  FK check: ✓")

    print(f"\n{'='*55}")
    print(f"  MERGE PLAN — Kaggle")
    print(f"{'='*55}")
    print(f"  benchmarks.csv: {len(main_b)} → {len(main_b)+len(new_bench)} (+{len(new_bench)})")
    print(f"  models.csv:     {len(main_m)} → {len(main_m)+len(new_models)} (+{len(new_models)})")
    print(f"  results.csv:    {len(main_r)} → {len(main_r)+len(new_results)} (+{len(new_results)})")
    print(f"{'='*55}")

    if not args.write:
        print("\nDry run — pass --write to apply.")
        return 0
    if fk:
        return 1

    print("\nWriting...")
    merged_b = pd.concat([main_b, new_bench[main_b.columns]], ignore_index=True)
    io.save_csv(merged_b, config.BENCHMARKS_CSV)

    merged_m = pd.concat([main_m, new_models[main_m.columns]], ignore_index=True)
    io.save_csv(merged_m, config.MODELS_CSV)

    merged_r = pd.concat([main_r, new_results[main_r.columns]], ignore_index=True)
    io.save_results(merged_r)

    print("Done. Now run:")
    print("  python3 scripts/verify_data.py")
    print("  python3 scripts/manage_data.py recompute-stats --write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
