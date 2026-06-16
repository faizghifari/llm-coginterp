#!/usr/bin/env python3
"""Update models.csv with research findings for all UNKNOWN models."""

import pandas as pd
import json

# Load research findings
with open('scripts/models_research_findings.json', 'r') as f:
    findings = json.load(f)['research_findings']

# Load models.csv
df = pd.read_csv('data/models.csv')
print(f"Total models: {len(df)}")

# Track updates
updated = 0
skipped = 0
errors = []

for model_id, info in findings.items():
    # Find matching row(s)
    mask = df['model_id'] == model_id
    if mask.sum() == 0:
        # Try fuzzy match on model_name
        mask = df['model_name'].str.contains(model_id, case=False, na=False) | \
               df['model_id'].str.contains(model_id, case=False, na=False)
    
    if mask.sum() == 0:
        errors.append(f"NOT FOUND: {model_id}")
        skipped += 1
        continue
    
    if mask.sum() > 1:
        errors.append(f"MULTIPLE MATCHES for {model_id}: {mask.sum()} rows")
        skipped += 1
        continue
    
    row_idx = df[mask].index[0]
    
    # Update model_family
    if info.get('family'):
        old = df.at[row_idx, 'model_family']
        df.at[row_idx, 'model_family'] = info['family']
        if old != info['family']:
            print(f"  {model_id}: model_family {old} -> {info['family']}")
    
    # Update base_model
    if info.get('base_model'):
        old = df.at[row_idx, 'base_model']
        df.at[row_idx, 'base_model'] = info['base_model']
        if old != info['base_model']:
            print(f"  {model_id}: base_model {old} -> {info['base_model']}")
    
    # Update developer
    if info.get('developer'):
        old = df.at[row_idx, 'developer']
        df.at[row_idx, 'developer'] = info['developer']
        if old != info['developer']:
            print(f"  {model_id}: developer {old} -> {info['developer']}")
    
    # Update model_type if specified
    if info.get('model_type'):
        old = df.at[row_idx, 'model_type']
        df.at[row_idx, 'model_type'] = info['model_type']
        if old != info['model_type']:
            print(f"  {model_id}: model_type {old} -> {info['model_type']}")
    
    updated += 1

print(f"\n=== SUMMARY ===")
print(f"Updated: {updated}")
print(f"Skipped: {skipped}")
if errors:
    print(f"\nErrors/Warnings:")
    for e in errors:
        print(f"  {e}")

# Save updated CSV
df.to_csv('data/models_updated.csv', index=False)
print(f"\nSaved to: data/models_updated.csv")

# Show stats
print(f"\nmodel_family empty: {df['model_family'].isna().sum()}")
print(f"base_model empty: {df['base_model'].isna().sum()}")
print(f"developer empty: {df['developer'].isna().sum()}")
