#!/usr/bin/env python3
"""
Merge HELM Staging Files into Main Data Files
==============================================
Merges data/staging_helm_*.csv into the main data/benchmarks.csv,
data/models.csv, and data/results.csv.

Dry-run by default — pass --write to persist.  Always run
  python3 scripts/verify_data.py
afterwards to confirm FK integrity.

Safety:
  - All changes are append-only (no existing rows modified or deleted)
  - Skips benchmark_ids / model_ids that already exist in the main files
  - Applies a verified alias map to remap HELM-style model names to the
    canonical model_ids already in models.csv
  - Pre-flight prints exactly what will change before writing anything

Usage:
  python3 scripts/merge_helm_staging.py           # dry run (show plan)
  python3 scripts/merge_helm_staging.py --write   # apply to main files
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from scripts.lib import io, config

# ---------------------------------------------------------------------------
# Alias map: staging model_id → existing main model_id
# Only maps when both represent EXACTLY the same model (naming difference only).
# Verified 2026-06-18: every target below exists in models.csv.
# ---------------------------------------------------------------------------
MODEL_ALIAS = {
    # HELM Classic era (name format: "Name (Size)" → "Name-Size")
    "BLOOM (176B)": "BLOOM-176B",
    "Alpaca (7B)": "Alpaca-7B",
    "LLaMA (13B)": "LLaMA-13B",
    "LLaMA (65B)": "LLaMA-65B",
    "LLaVA 1.5 (13B)": "LLaVA-1.5-13B",
    "Qwen1.5 (7B)": "Qwen1.5-7B",
    "Qwen1.5 (14B)": "Qwen1.5-14B",
    "Qwen1.5 (32B)": "Qwen1.5-32B",
    "Qwen1.5 (72B)": "Qwen1.5-72B",
    "Qwen3 (0.6B)": "Qwen3-0.6B",
    "Qwen3 (1.7B)": "Qwen3-1.7B",
    "Qwen3 (4B)": "Qwen3-4B",
    "Qwen3 (8B)": "Qwen3-8B",
    "PaLM-2 (Bison)": "PaLM-2-Bison",
    "Palmyra Med": "Palmyra-Med",
    "Phi-2": "phi-2",
    "Qwen-VL Chat": "Qwen-VL-Chat",
    "RedPajama-INCITE-Base (7B)": "RedPajama-INCITE-7B-Base",
    "RedPajama-INCITE-Base-v1 (3B)": "RedPajama-INCITE-Base-3B-v1",
    "RedPajama-INCITE-Instruct (7B)": "RedPajama-INCITE-7B-Instruct",
    "RedPajama-INCITE-Instruct-v1 (3B)": "RedPajama-INCITE-Instruct-3B-v1",
    "Arctic Instruct": "Snowflake Arctic Instruct",
    # Hyphen / space / case differences
    "Claude 2.0": "Claude-2",
    "Claude 2.1": "claude-2.1",
    "Claude Instant 1.2": "claude_instant_1.2",
    "Claude 3.5 Sonnet (20241022)": "Claude-3.5-Sonnet",
    "Claude 3.5 Haiku (20241022)": "Claude 3.5 Haiku",
    "Claude 3.7 Sonnet (20250219)": "Claude 3.7 Sonnet",
    "Command R Plus": "Command R+",
    "DeepSeek R1": "DeepSeek-R1",
    "DeepSeek v3": "DeepSeek-v3",
    "DeepSeek v3.1": "DeepSeek-V3.1",
    "DeepSeek-R1-Distill-Llama-8b": "DeepSeek-R1-Distill-Llama-8B",
    "Gemma 2 Instruct (9B)": "Gemma 2 Instruct 9B",
    "Gemini 2.5 Flash-Lite": "Gemini-2.5-Flash-Lite",
    "GPT-3.5 Turbo (1106)": "GPT-3.5-Turbo-1106",
    "GPT-4o (2024-05-13)": "GPT-4o",
    "GPT-4o (2024-11-20)": "GPT-4o",
    "GPT-4o mini (2024-07-18)": "GPT-4o-mini",
    "Grok 4 (0709)": "Grok-4-0709",
    "IBM Granite 3.3 8B Instruct": "Granite 3.3 8B Instruct",
    "Kimi K2 Instruct": "Kimi K2-Instruct",
    "Llama 4 Scout (17Bx16E) Instruct": "Llama-4-Scout-17B-16E-Instruct",
    "Mistral Instruct v0.1 (7B)": "Mistral-7B-Instruct-v0.1",
    "Mistral Instruct v0.3 (7B)": "Mistral-7B-Instruct-v0.3",
    "Mistral Large 2 (2407)": "Mistral Large 2",
    "Mistral Large 3 (2512)": "Mistral Large 3",
    "Mistral Small 3 (2501)": "Mistral Small 3 24B",
    "o1 (2024-09-12)": "o1-preview",
    "o1 (2024-12-17)": "o1-2024-12-17",
    "o1-mini (2024-09-12)": "o1-mini",
    "o3 (2025-04-16)": "o3-2025-04-16",
    "o3-mini (2025-01-31)": "o3-mini-2025-01-31",
    "o4-mini (2025-04-16)": "o4-mini-2025-04-16",
    "Qwen3-Next 80B A3B Instruct": "Qwen3-Next-80B-A3B-Instruct",
    "Qwen3.5 397B A17B": "Qwen3.5-397B-A17B",
    "Typhoon 1.5X instruct (70B)": "Typhoon-1.5X-Instruct-70B",
    "Typhoon v1.5 Instruct (72B)": "Typhoon-v1.5-Instruct-72B",
}

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGING_B = REPO_ROOT / "data" / "staging_helm_benchmarks.csv"
STAGING_M = REPO_ROOT / "data" / "staging_helm_models.csv"
STAGING_R = REPO_ROOT / "data" / "staging_helm_results.csv"


def load_all():
    staging_b = pd.read_csv(STAGING_B, dtype=str, keep_default_na=False)
    staging_m = pd.read_csv(STAGING_M, dtype=str, keep_default_na=False)
    staging_r = pd.read_csv(STAGING_R, dtype=str, keep_default_na=False)
    main_b, main_m, main_r = io.load_data()
    return staging_b, staging_m, staging_r, main_b, main_m, main_r


def verify_alias_map(staging_m, main_m):
    """Confirm every alias target exists in main models. Abort if any missing."""
    existing = set(main_m["model_id"])
    bad = [(src, tgt) for src, tgt in MODEL_ALIAS.items() if tgt not in existing]
    if bad:
        print("ERROR — alias map targets not found in models.csv:")
        for src, tgt in bad:
            print(f"  {src!r} -> {tgt!r}")
        sys.exit(1)
    print(f"  Alias map verified: {len(MODEL_ALIAS)} entries, all targets exist ✓")


def apply_aliases_to_staging(staging_m, staging_r):
    """Apply MODEL_ALIAS to staging models and results.

    For each aliased staging model_id:
      - Remove it from staging_m (target already in main)
      - Remap matching model_name values in staging_r
    """
    # Remap results rows
    staging_r = staging_r.copy()
    staging_m = staging_m.copy()

    remapped_result_rows = 0
    for old_id, new_id in MODEL_ALIAS.items():
        mask = staging_r["model_name"] == old_id
        if mask.any():
            staging_r.loc[mask, "model_name"] = new_id
            staging_r.loc[mask, "model_id"] = new_id
            remapped_result_rows += mask.sum()

    # Also remap model_developer / model_family in results (take from new_id if exists)
    # (left as-is for now — staging results carry their own metadata)

    # Drop aliased rows from staging_m (target is already in main)
    aliased_ids = set(MODEL_ALIAS.keys())
    staging_m = staging_m[~staging_m["model_id"].isin(aliased_ids)].reset_index(drop=True)

    return staging_m, staging_r, remapped_result_rows


def plan_benchmarks(staging_b, main_b):
    existing_ids = set(main_b["benchmark_id"])
    new_rows = staging_b[~staging_b["benchmark_id"].isin(existing_ids)]
    skipped = staging_b[staging_b["benchmark_id"].isin(existing_ids)]
    return new_rows, skipped


def plan_models(staging_m, main_m):
    existing_ids = set(main_m["model_id"])
    new_rows = staging_m[~staging_m["model_id"].isin(existing_ids)]
    skipped = staging_m[staging_m["model_id"].isin(existing_ids)]
    return new_rows, skipped


def plan_results(staging_r, main_r, all_model_ids, all_bench_ids):
    """Return (new_rows, fk_violations)."""
    # De-duplicate against existing results on the same key as RESULT_IDENTITY_KEY
    dedup_cols = ["benchmark_id", "model_name", "metric_name", "source_url"]
    existing_keys = set(
        zip(main_r["benchmark_id"], main_r["model_name"],
            main_r["metric_name"], main_r["source_url"])
    )
    staging_r = staging_r.copy()
    mask_new = ~staging_r.apply(
        lambda row: (row["benchmark_id"], row["model_name"],
                     row["metric_name"], row["source_url"]) in existing_keys,
        axis=1,
    )
    new_rows = staging_r[mask_new]
    # FK checks on the rows we'd insert
    bad_bench = new_rows[~new_rows["benchmark_id"].isin(all_bench_ids)]
    bad_model = new_rows[~new_rows["model_name"].isin(all_model_ids)]
    fk_violations = []
    if len(bad_bench):
        fk_violations.append(("benchmark_id", bad_bench[["benchmark_id", "model_name"]].drop_duplicates()))
    if len(bad_model):
        fk_violations.append(("model_name", bad_model[["model_name", "benchmark_id"]].drop_duplicates()))
    return new_rows, fk_violations


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true",
                        help="Write changes to main data files (default: dry run).")
    args = parser.parse_args()

    print("Loading files...")
    staging_b, staging_m, staging_r, main_b, main_m, main_r = load_all()

    print(f"\nMain files:    {len(main_b)} benchmarks, {len(main_m)} models, {len(main_r)} results")
    print(f"Staging files: {len(staging_b)} benchmarks, {len(staging_m)} models, {len(staging_r)} results")

    # ── Step 1: verify and apply alias map ──────────────────────────────────
    print("\n[1/4] Alias map verification...")
    verify_alias_map(staging_m, main_m)
    staging_m, staging_r, n_remapped = apply_aliases_to_staging(staging_m, staging_r)
    print(f"  Remapped {n_remapped} result rows to existing model_ids")
    print(f"  Staging models after alias removal: {len(staging_m)}")

    # ── Step 2: plan benchmarks ──────────────────────────────────────────────
    print("\n[2/4] Benchmarks plan...")
    new_bench, skip_bench = plan_benchmarks(staging_b, main_b)
    print(f"  Skip (already exist): {len(skip_bench)}")
    for bid in sorted(skip_bench["benchmark_id"]):
        print(f"    ↳ {bid}")
    print(f"  New to add: {len(new_bench)}")

    # ── Step 3: plan models ──────────────────────────────────────────────────
    print("\n[3/4] Models plan...")
    new_models, skip_models = plan_models(staging_m, main_m)
    print(f"  Skip (already exist or aliased): {len(skip_models)}")
    print(f"  New to add: {len(new_models)}")
    if len(new_models) <= 50:
        for mid in sorted(new_models["model_id"]):
            print(f"    + {mid}")

    # ── Step 4: plan results + FK check ─────────────────────────────────────
    print("\n[4/4] Results plan...")
    # Combined model/benchmark sets after hypothetical merge
    all_model_ids = set(main_m["model_id"]) | set(new_models["model_id"])
    all_bench_ids = set(main_b["benchmark_id"]) | set(new_bench["benchmark_id"])

    new_results, fk_violations = plan_results(staging_r, main_r, all_model_ids, all_bench_ids)
    print(f"  Already in main (skipped): {len(staging_r) - len(new_results)}")
    print(f"  New to add: {len(new_results)}")

    if fk_violations:
        print("\n  ⚠ FK violations in new result rows (MUST fix before write):")
        for col, df in fk_violations:
            print(f"\n  Bad {col} ({len(df)} unique values):")
            print(df.to_string(index=False))
        if args.write:
            print("\nAborting --write due to FK violations.")
            return 1
    else:
        print("  FK check: ✓ all benchmark_ids and model_names resolve")

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  MERGE PLAN SUMMARY")
    print(f"{'='*55}")
    print(f"  benchmarks.csv: {len(main_b)} → {len(main_b) + len(new_bench)} (+{len(new_bench)})")
    print(f"  models.csv:     {len(main_m)} → {len(main_m) + len(new_models)} (+{len(new_models)})")
    print(f"  results.csv:    {len(main_r)} → {len(main_r) + len(new_results)} (+{len(new_results)})")
    print(f"{'='*55}")

    if not args.write:
        print("\nDry run — pass --write to apply.")
        return 0

    if fk_violations:
        return 1  # already printed above

    # ── Write ────────────────────────────────────────────────────────────────
    print("\nWriting changes...")

    # benchmarks.csv
    merged_b = pd.concat([main_b, new_bench[main_b.columns]], ignore_index=True)
    io.save_csv(merged_b, config.BENCHMARKS_CSV)
    print(f"  benchmarks.csv written: {len(merged_b)} rows")

    # models.csv — append new models, then sort by model_id for readability
    merged_m = pd.concat([main_m, new_models[main_m.columns]], ignore_index=True)
    io.save_csv(merged_m, config.MODELS_CSV)
    print(f"  models.csv written: {len(merged_m)} rows")

    # results.csv
    merged_r = pd.concat([main_r, new_results[main_r.columns]], ignore_index=True)
    io.save_results(merged_r)
    print(f"  results.csv written: {len(merged_r)} rows")

    print("\nDone. Now run:")
    print("  python3 scripts/verify_data.py")
    print("  python3 scripts/manage_data.py recompute-stats --write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
