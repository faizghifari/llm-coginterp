#!/usr/bin/env python3
"""
Merge PwC Staging Files into Main Data Files
=============================================
Merges data/staging_pwc_*.csv into data/benchmarks.csv, models.csv, results.csv.

Dry-run by default — pass --write to persist.
Always run  python3 scripts/verify_data.py  afterwards.

Key behaviours
--------------
- 19 staging benchmark_ids are remapped to existing main benchmark_ids (same benchmark,
  different source).  Results are still inserted; no duplicate benchmark row is added.
- 106 staging model names are remapped to existing main model_ids.  Results still inserted;
  no duplicate model row is added.
- All other benchmarks/models are inserted as new rows.
- Deduplicates results on the full RESULT_IDENTITY_KEY before writing.

Usage:
  python3 scripts/merge_pwc_staging.py           # dry run
  python3 scripts/merge_pwc_staging.py --write   # apply
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from scripts.lib import io, config

# ---------------------------------------------------------------------------
# Benchmark alias: staging benchmark_id → existing main benchmark_id
# Verified 2026-06-18: all targets exist in benchmarks.csv
# ---------------------------------------------------------------------------
BENCHMARK_ALIAS = {
    "pwc_narrativeqa":        "narrative_qa",
    "pwc_natural_questions":  "naturalquestions",
    "pwc_pubmedqa":           "pubmed_qa",
    "pwc_medqa":              "med_qa",
    "pwc_truthfulqa":         "truthful_qa",
    "pwc_fever":              "fever",
    "pwc_boolq":              "boolq",
    "pwc_triviaqa":           "triviaqa",
    "pwc_openbookqa":         "openbookqa",
    "pwc_ai2d":               "ai2d",
    "pwc_mbpp":               "mbpp",
    "pwc_humaneval":          "humaneval",
    "pwc_tabfact":            "tab_fact",
    "pwc_the_pile":           "the_pile",
    "pwc_imdb":               "imdb",
    "pwc_math":               "math_regular",
    "pwc_chartqa":            "chartqa",
    "pwc_ifeval":             "ifeval",
    "pwc_mmbench":            "mmbench",
}

# ---------------------------------------------------------------------------
# Model alias: staging model_name → existing main model_id
# Verified 2026-06-18: all targets exist in models.csv
# NOTE: "CodeT5+ 220M" intentionally NOT mapped — it is a distinct size variant
#       from the generic "CodeT5+" entry in main.
# ---------------------------------------------------------------------------
MODEL_ALIAS = {
    "T5-11B":                        "T5 (11B)",
    "Orca 2-13B":                    "Orca-2-13b",
    "Orca 2-7B":                     "Orca-2-7b",
    "GPT-4":                         "GPT-4",
    "Flan-T5-XXL":                   "flan-t5-xxl",
    "Med-PaLM 2":                    "Med-PaLM 2",
    "LLAMA-2 (70B)":                 "Llama 2 (70B)",
    "PaLM 540B":                     "PaLM-540B",
    "OPT 175B":                      "OPT (175B)",
    "GPT-J 6B":                      "GPT-J (6B)",
    "LLaMA 65B":                     "LLaMA-65B",
    "LLaMA 13B":                     "LLaMA-13B",
    "LLaMA 7B":                      "LLaMA (7B)",
    "OPT-175B":                      "OPT (175B)",
    "BLOOMZ":                        "BLOOMZ",
    "Gpt-4":                         "GPT-4",
    "deepseek-r1":                   "DeepSeek-R1",
    "GPT-4-0613":                    "GPT-4 (0613)",
    "Gemini Ultra":                  "Gemini Ultra",
    "GPT-4o":                        "GPT-4o",
    "Qwen2.5-VL-7B":                 "Qwen2.5-VL-7B",
    "GPT-4V":                        "GPT-4V",
    "Qwen-VL-Plus":                  "Qwen-VL-Plus",
    "Qwen-VL":                       "Qwen-VL",
    "Qwen-VL-Chat":                  "Qwen-VL-Chat",
    "Gemini-Pro":                    "Gemini Pro",
    "LLaVA-1.5-13B":                 "LLaVA-1.5-13B",
    "GPT-3.5-Turbo":                 "GPT-3.5-Turbo",
    "Command":                       "Command",
    "Claude 3 Opus":                 "Claude 3 Opus",
    "Claude 3 Haiku":                "Claude 3 Haiku",
    "Claude 3 Sonnet":               "Claude 3 Sonnet",
    "Claude":                        "claude",
    "StarCoder2-15B":                "starcoder2-15b",
    "GPT-3.5 Turbo":                 "GPT-3.5-Turbo",
    "Mixtral-8x7B-Instruct":         "Mixtral-8x7B-Instruct",
    "Phi-3-mini-128k-instruct":      "Phi-3-mini-128k-instruct",
    "WizardLM-2-7B":                 "WizardLM-2-7B",
    "Llama-3-8B-Instruct":           "llama-3-8b-instruct",
    "claude-3-5-sonnet":             "Claude-3.5-Sonnet",
    "o1-mini":                       "o1-mini",
    "o1-preview":                    "o1-preview",
    "gpt-4o-2024-08-06":             "GPT-4o (2024-08-06)",
    "deepseek-v2.5":                 "DeepSeek-V2.5",
    "mistral-large-2":               "Mistral Large 2",
    "claude-3.5-sonnet":             "Claude-3.5-Sonnet",
    "deepseek-coder-v2-instruct":    "DeepSeek-Coder-V2-Instruct",
    "LLaMA 3":                       "Llama 3",
    "Phi-2":                         "phi-2",
    "Mistral 7B":                    "Mistral-7B",
    "Gemini":                        "gemini",
    "LLaVA":                         "LLaVA",
    "RWKV-4-Raven-14B":              "RWKV-4-Raven-14B",
    "GPT-3":                         "GPT-3",
    "GLM-130B":                      "GLM (130B)",
    "GPT-J-6B":                      "GPT-J (6B)",
    "Gemma-2 27B":                   "Gemma-2-27B",
    "Llama-3.2 3B":                  "Llama-3.2-3B",
    "Phi-3 14B":                     "Phi-3 (14B)",
    "Gemma-2 9B":                    "Gemma-2-9B",
    "Phi-3 7B":                      "Phi-3 (7B)",
    "Llama-3.2 1B":                  "Llama-3.2-1B",
    "GPT-Neo 2.7B":                  "gpt-neo-2.7B",
    "GPT-Neo 1.3B":                  "gpt-neo-1.3B",
    "OPT 2.7B":                      "opt-2.7b",
    "GPT-Neo 125M":                  "gpt-neo-125m",
    "OPT 1.3B":                      "opt-1.3b",
    "OPT 125M":                      "opt-125m",
    "Qwen-72B":                      "Qwen-72B",
    "Vicuna-33B":                    "Vicuna-33b",
    "Vicuna-7B":                     "Vicuna-7B",
    "GPT-J (6B)":                    "GPT-J (6B)",
    "Gemini 2.0 Flash Experimental": "Gemini 2.0 Flash (Experimental)",
    "OpenMath2-Llama3.1-8B":         "OpenMath2-Llama3.1-8B",
    "OPT (66B)":                     "OPT (66B)",
    "DeepSeek-r1":                   "DeepSeek-R1",
    "Qwen2.5-72B-Instruct":          "Qwen2.5-72B-Instruct",
    "o3":                            "o3",
    "Gemini 1.5 Pro (002)":          "Gemini 1.5 Pro (002)",
    "Claude 3.5 Sonnet":             "Claude-3.5-Sonnet",
    "MetaMath-Mistral-7B":           "MetaMath-Mistral-7B",
    "CodeT5+":                       "CodeT5+",
    "Orca 2 13B":                    "Orca-2-13b",
    "Llemma 34B":                    "llemma_34b",
    "Orca 2 7B":                     "Orca-2-7b",
    "Llemma 7B":                     "llemma_7b",
    "Gemini Pro":                    "Gemini Pro",
    "Qwen-VL-Max":                   "Qwen-VL-Max",
    "LLaVa-1.5-13B":                 "LLaVA-1.5-13B",
    "LLaVa-1.5-7B":                  "LLaVA-1.5-7B",
    "BLIP2-FLAN-T5-XXL":             "BLIP2-FLAN-T5-XXL",
    "QWEN":                          "qwen",
    "Qwen2-VL-72B":                  "Qwen2-VL-72B",
    "GLM-4V-9B":                     "GLM-4V-9B",
    "LLaVA-NeXT-34B":                "LLaVA-NeXT-34B",
    "Video-LLaVA":                   "Video-LLaVA",
    "LLaVA-1.5-7B":                  "LLaVA-1.5-7B",
    "starcoderbase":                 "starcoderbase",
    "CodeLlama-13b-hf":              "CodeLlama-13b-hf",
    "CodeLlama-34b-hf":              "CodeLlama-34b-hf",
    "CodeLlama-7b-hf":               "CodeLlama-7b-hf",
    "gpt-3.5-turbo-0301":            "gpt-3.5-turbo-0301",
    "incoder-6B":                    "InCoder-6B",
    "codegen-16B-multi":             "CodeGen-16B-multi",
    "codegen-6B-multi":              "codegen-6B-multi",
}

REPO_ROOT  = Path(__file__).resolve().parent.parent
STAGING_B  = REPO_ROOT / "data" / "staging_pwc_benchmarks.csv"
STAGING_M  = REPO_ROOT / "data" / "staging_pwc_models.csv"
STAGING_R  = REPO_ROOT / "data" / "staging_pwc_results.csv"


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

    # Remap benchmark_id in results
    n_b = staging_r["benchmark_id"].isin(BENCHMARK_ALIAS).sum()
    staging_r["benchmark_id"] = staging_r["benchmark_id"].map(
        lambda x: BENCHMARK_ALIAS.get(x, x))
    # Also remap benchmark_id in the benchmark rows (so plan_benchmarks can detect them)
    staging_b["benchmark_id"] = staging_b["benchmark_id"].map(
        lambda x: BENCHMARK_ALIAS.get(x, x))

    # Remap model_name/model_id in results
    n_m = staging_r["model_name"].isin(MODEL_ALIAS).sum()
    staging_r["model_name"] = staging_r["model_name"].map(lambda x: MODEL_ALIAS.get(x, x))
    staging_r["model_id"]   = staging_r["model_id"].map(  lambda x: MODEL_ALIAS.get(x, x))

    # Drop aliased models from staging_m
    staging_m = staging_m[~staging_m["model_id"].isin(MODEL_ALIAS)].reset_index(drop=True)

    print(f"  Remapped {n_b} result rows to existing benchmark_ids")
    print(f"  Remapped {n_m} result rows to existing model_ids")
    print(f"  Staging models after alias removal: {len(staging_m)}")
    return staging_b, staging_m, staging_r


def plan_benchmarks(staging_b, main_b):
    existing = set(main_b["benchmark_id"])
    new_rows = staging_b[~staging_b["benchmark_id"].isin(existing)]
    skipped  = staging_b[ staging_b["benchmark_id"].isin(existing)]
    return new_rows, skipped


def plan_models(staging_m, main_m):
    existing = set(main_m["model_id"])
    new_rows = staging_m[~staging_m["model_id"].isin(existing)]
    skipped  = staging_m[ staging_m["model_id"].isin(existing)]
    return new_rows, skipped


def plan_results(staging_r, main_r, all_model_ids, all_bench_ids):
    dedup_cols = ["benchmark_id", "model_name", "metric_name", "source_url"]
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
    new_bench, skip_bench = plan_benchmarks(staging_b, main_b)
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
    print(f"  MERGE PLAN — PwC")
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
