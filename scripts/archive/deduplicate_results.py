"""
Deduplicate results.csv:
- Pure redundancy (same score): keep first occurrence, discard rest
- Conflicting data (diff score): keep higher trust tier, break ties by newest source
  Move losers to data/results_duplicates.csv

Source trust hierarchy:
  Tier 1: Official sources (arxiv, aclanthology, openreview, model developer sites,
          benchmark official sites, chat.lmsys.org, official leaderboards)
  Tier 2: Reputable aggregators with known methodology (HF Open LLM Leaderboard)
  Tier 3: Third-party aggregators (llm-stats.com, vellum, emergentmind, gorilla,
          artificialanalysis, pricepertoken)
  Tier 4: Unknown/unverifiable (raw github json, etc.)
"""

import pandas as pd
import re
from datetime import datetime

def trust_tier(source_url, source_name):
    """Return trust tier (1=highest) for a source."""
    url = str(source_url).lower()
    name = str(source_name).lower() if pd.notna(source_name) else ''

    # Tier 1: Official sources
    tier1_patterns = [
        'arxiv.org', 'aclanthology.org', 'openreview.net',
        'anthropic.com', 'openai.com', 'ai.google',
        'swebench.com', 'crux-eval.github.io',
        'chat.lmsys.org', 'lmsys.org',
        'raw.githubusercontent.com/lmarena',  # arena-hard-auto official
        'raw.githubusercontent.com/IBM/mt-rag',  # mtrag official repo
    ]
    tier1_names = ['official paper', 'anthropic official', 'openai official']
    for p in tier1_patterns:
        if p in url:
            return 1
    for n in tier1_names:
        if n in name:
            return 1

    # Tier 2: Reputable aggregators with known methodology
    tier2_patterns = [
        'huggingface.co/spaces/open-llm-leaderboard',
    ]
    for p in tier2_patterns:
        if p in url:
            return 2

    # Tier 3: Third-party aggregators
    tier3_patterns = [
        'llm-stats.com', 'vellum.ai', 'emergentmind.com',
        'gorilla.cs.berkeley.edu', 'artificialanalysis.ai',
        'pricepertoken.com'
    ]
    for p in tier3_patterns:
        if p in url:
            return 3

    # Tier 4: Unknown
    return 4


def estimate_date(row):
    """Estimate relative date for tie-breaking. Higher = newer."""
    # Use year_evaluated if available
    year = row.get('year_evaluated')
    if pd.notna(year):
        try:
            return int(float(year))
        except (ValueError, TypeError):
            pass
    return 2020  # default old


def main():
    data_dir = "/Users/haznitrama/Desktop/llm-benchmarks/data"
    results = pd.read_csv(f"{data_dir}/results.csv")

    key_cols = ['model_name', 'benchmark_id', 'metric_name']

    keep_indices = []
    discard_rows = []

    for (model, bench, metric), group in results.groupby(key_cols):
        indices = group.index.tolist()

        if len(indices) == 1:
            keep_indices.append(indices[0])
            continue

        rows = group.iloc[:]

        # Check if pure redundancy (all same score)
        scores = rows['score'].unique()
        if len(scores) == 1:
            # Pure redundancy - keep first, discard rest
            keep_indices.append(indices[0])
            for idx in indices[1:]:
                discard_rows.append(idx)
            print(f"  REDUNDANT (same score={scores[0]}): {model} + {bench} + {metric}")
            continue

        # Conflicting data - pick winner by trust tier, then date
        best_idx = None
        best_tier = 999
        best_date = 0

        for idx in indices:
            row = results.loc[idx]
            tier = trust_tier(row['source_url'], row['source_name'])
            date = estimate_date(row)

            print(f"  CONFLICT: {model} + {bench} + {metric}")
            print(f"    Tier {tier} | score={row['score']} | year={date} | {row['source_url']}")

            if tier < best_tier or (tier == best_tier and date > best_date):
                best_idx = idx
                best_tier = tier
                best_date = date

        keep_indices.append(best_idx)
        for idx in indices:
            if idx != best_idx:
                discard_rows.append(idx)

    # Split
    keep_df = results.loc[keep_indices]
    discard_df = results.loc[discard_rows].reset_index(drop=True)

    print(f"\n=== Summary ===")
    print(f"Original rows: {len(results)}")
    print(f"Keeping: {len(keep_df)}")
    print(f"Discarding to duplicates: {len(discard_df)}")
    print(f"Net reduction: {len(results) - len(keep_df)} rows")

    if len(discard_df) > 0:
        print(f"\nDuplicate rows saved to: {data_dir}/results_duplicates.csv")
        discard_df.to_csv(f"{data_dir}/results_duplicates.csv", index=False)

        # Show what was discarded
        print(f"\n=== Discarded rows ({len(discard_df)}) ===")
        for _, row in discard_df.iterrows():
            print(f"  {row['model_name']} | {row['benchmark_id']} | {row['metric_name']} | score={row['score']} | {row['source_url']}")

    # Write cleaned results
    keep_df.to_csv(f"{data_dir}/results.csv", index=False)
    print(f"\nCleaned results.csv written ({len(keep_df)} rows)")


if __name__ == "__main__":
    main()
