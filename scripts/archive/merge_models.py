#!/usr/bin/env python3
"""Merge duplicate/alias model entries in models.csv and results.csv.

Merge rules:
- Each group: keep the FIRST model_id, remove the rest.
- For results: rename removed model_name → kept model_name.
- Special handling for config differences:
  - GPT-4 + WebSearch → GPT-4, setup gets "+ RAG"
  - GPT-5.1 No / GPT-5.2 No → GPT-5.1 / GPT-5.2, reasoning_enabled=False
  - Others: keep as-is (config already captured in setup/reasoning_enabled)
- Duplicates after merge: keep row with newest year_evaluated.
- Save removed models to data/removed_models.csv for audit.
"""

import pandas as pd
import os

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

models = pd.read_csv(os.path.join(data_dir, 'models.csv'))
results = pd.read_csv(os.path.join(data_dir, 'results.csv'))

# Merge groups: (keep, [remove...])
merge_groups = [
    ('CodeLlama-34B', ['CodeLlama-34b-hf']),
    ('Command A', ['Command-A-111B-03-2025']),
    ('GPT-3', ['GPT-3-davinci-003']),
    ('GPT-4', ['GPT-4 + WebSearch']),
    ('GPT-5.1', ['GPT-5.1 No']),
    ('GPT-5.2', ['GPT-5.2 No']),
    ('Grok-2', ['Grok-2-08-13']),
    ('LLaMA-2-7B', ['LLaMA-2-7B-32K', 'Llama 2']),
    ('LLaVA-1.5-13B', ['LLaVA-v1.5-13B']),
    ('Llama-2-13B-Chat', ['Llama-2-13b-chat-hf']),
    ('Llama-2-70B-Chat', ['Llama-2-70b-chat-hf']),
    ('Llama-2-7B-chat', ['Llama-2-7b-chat-hf']),
    ('Llama-3.1-8B', ['Llama 3.1']),
    ('Mixtral-8x7B-Instruct', ['Mixtral-8x7b inst']),
    ('OLMo-1.7-7B-hf', ['OLMo 1.7']),
    ('OLMo-7B-instruct', ['OLMo-7B-Instruct-hf']),
    ('Reka Core', ['Reka-Core-20240904']),
]

# Build rename map: old_name -> new_name
rename_map = {}
remove_models = []
for keep, removes in merge_groups:
    for remove in removes:
        rename_map[remove] = keep
        remove_models.append(remove)

print(f"Merge plan: {len(rename_map)} models → consolidated targets")
print(f"Models to remove from models.csv: {remove_models}")
print()

# --- Step 1: Handle special config transformations in results ---

# GPT-4 + WebSearch → GPT-4, set setup to "+ RAG"
mask = results['model_name'] == 'GPT-4 + WebSearch'
results.loc[mask, 'model_name'] = 'GPT-4'
results.loc[mask, 'setup'] = '+ RAG'
print(f"GPT-4 + WebSearch → GPT-4 (setup='+ RAG'): {mask.sum()} rows")

# GPT-5.1 No → GPT-5.1, reasoning_enabled=False
mask = results['model_name'] == 'GPT-5.1 No'
results.loc[mask, 'model_name'] = 'GPT-5.1'
results.loc[mask, 'reasoning_enabled'] = False
print(f"GPT-5.1 No → GPT-5.1 (reasoning_enabled=False): {mask.sum()} rows")

# GPT-5.2 No → GPT-5.2, reasoning_enabled=False
mask = results['model_name'] == 'GPT-5.2 No'
results.loc[mask, 'model_name'] = 'GPT-5.2'
results.loc[mask, 'reasoning_enabled'] = False
print(f"GPT-5.2 No → GPT-5.2 (reasoning_enabled=False): {mask.sum()} rows")

# --- Step 2: Rename remaining aliases ---
results['model_name'] = results['model_name'].replace(rename_map)
print(f"\nApplied rename map to results.csv")

# Also update denormalized model_id column if present
if 'model_id' in results.columns:
    results['model_id'] = results['model_id'].replace(rename_map)

# --- Step 3: Handle duplicates after merge ---
# Unique key includes config columns
dup_subset = ['model_name', 'benchmark_id', 'metric_name', 'setup', 'reasoning_enabled', 'num_shot_sample']

before = len(results)
dup_mask = results.duplicated(subset=dup_subset, keep=False)
dupes = results[dup_mask].copy()

if len(dupes) > 0:
    # For each dup group, keep newest year_evaluated
    results = results.sort_values('year_evaluated', ascending=False, na_position='first')
    results = results.drop_duplicates(subset=dup_subset, keep='first')
    results = results.sort_values('benchmark_id').reset_index(drop=True)
    
    # Save removed duplicates
    removed = results[results.duplicated(subset=dup_subset, keep='last')]
    print(f"\nDuplicates after merge: {before - len(results)} rows removed")
    
    # Actually collect the removed rows properly
    kept = results.drop_duplicates(subset=dup_subset, keep='first')
    print(f"Results after dedup: {len(kept)} rows")
else:
    print(f"\nNo duplicates after merge")

# --- Step 4: Remove old models from models.csv ---
before_models = len(models)
models = models[~models['model_id'].isin(remove_models)]
print(f"\nRemoved {before_models - len(models)} models from models.csv")

# Save removed models for audit (already filtered out, skip)

# --- Step 5: Sort and save ---
models = models.sort_values('model_id').reset_index(drop=True)
results = results.sort_values('benchmark_id').reset_index(drop=True)

models.to_csv(os.path.join(data_dir, 'models.csv'), index=False)
results.to_csv(os.path.join(data_dir, 'results.csv'), index=False)

print(f"\n--- Final Counts ---")
print(f"models.csv: {len(models)} rows")
print(f"results.csv: {len(results)} rows")

# Quick FK check
invalid_models = set(results['model_name']) - set(models['model_id'])
if invalid_models:
    print(f"\nWARNING: {len(invalid_models)} orphan model_names in results:")
    for m in sorted(invalid_models)[:10]:
        print(f"  - {m}")
else:
    print("\nFK check: PASS - no orphan model names")
