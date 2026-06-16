#!/usr/bin/env python3
"""Standardize model_id in models.csv and model_name in results.csv.

Rules:
1. Lowercase
2. Replace spaces & underscores with '-'
3. Replace '+' with '-plus'
4. Collapse multiple hyphens
5. Strip leading/trailing hyphens
"""

import pandas as pd
import os
import re
from collections import Counter

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

models = pd.read_csv(os.path.join(data_dir, 'models.csv'))
results = pd.read_csv(os.path.join(data_dir, 'results.csv'))

def standardize(mid):
    s = str(mid).strip().lower()
    s = s.replace(' ', '-').replace('_', '-')
    s = s.replace('+', '-plus')
    s = re.sub(r'-+', '-', s)
    s = s.strip('-')
    return s

# Build rename map
rename_map = {}
for mid in models['model_id']:
    new = standardize(mid)
    if new != mid:
        rename_map[mid] = new

print(f"Standardization map: {len(rename_map)} entries")

# Check collisions
new_ids = [standardize(m) for m in models['model_id']]
collisions = [k for k, v in Counter(new_ids).items() if v > 1]

if collisions:
    print(f"\nCollisions detected ({len(collisions)}):")
    for c in collisions:
        originals = [m for m in models['model_id'] if standardize(m) == c]
        print(f"  '{c}' <- {originals}")
        # Merge: keep first, remove rest
        keep = originals[0]
        for remove in originals[1:]:
            rename_map[remove] = c
    print("\nThese duplicates will be merged during standardization.")

# --- Apply to models.csv ---
before_models = len(models)
models['model_id'] = models['model_id'].map(rename_map).fillna(models['model_id'])
# Remove any duplicates created by collision merging
models = models.drop_duplicates(subset='model_id', keep='first').reset_index(drop=True)
print(f"\nmodels.csv: {before_models} -> {len(models)} rows ({before_models - len(models)} merged)")

# --- Apply to results.csv (model_name is the FK) ---
results['model_name'] = results['model_name'].map(rename_map).fillna(results['model_name'])

# Also update denormalized model_id column if it exists
if 'model_id' in results.columns:
    results['model_id'] = results['model_id'].map(rename_map).fillna(results['model_id'])

# --- Sort ---
models = models.sort_values('model_id').reset_index(drop=True)
results = results.sort_values('benchmark_id').reset_index(drop=True)

# --- Save ---
models.to_csv(os.path.join(data_dir, 'models.csv'), index=False)
results.to_csv(os.path.join(data_dir, 'results.csv'), index=False)

# --- Verify FK ---
invalid = set(results['model_name']) - set(models['model_id'])
print(f"\nFK check: {'PASS' if not invalid else 'FAIL - ' + str(len(invalid)) + ' orphans'}")
if invalid:
    for m in sorted(invalid)[:10]:
        print(f"  orphan: '{m}'")

print(f"\nFinal counts: benchmarks=200, models={len(models)}, results={len(results)}")
