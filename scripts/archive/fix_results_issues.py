import pandas as pd
import hashlib
import shutil

results = pd.read_csv('data/results.csv')
shutil.copy2('data/results.csv', 'data/results.csv.bak.pre_clean')

# ============================================================
# 1. SEPARATE TRUE DUPLICATES into results_duplicates.csv
# ============================================================
# True dupes = same model+benchmark+metric+source but different scores (likely re-runs/snapshots)
# Keep first occurrence, move rest to duplicates file

def classify_and_split(df):
    keep_mask = pd.Series(True, index=df.index)
    dupe_rows = []
    
    for (model, bench), group in df.groupby(['model_name', 'benchmark_id']):
        if len(group) <= 1:
            continue
        
        metrics = group['metric_name'].nunique()
        sources = group['source_name'].nunique()
        
        # Skip if different sub-metrics (legitimate multiple entries)
        if metrics > 1:
            continue
        
        # Skip if different sources (legitimate — same benchmark, different providers)
        # But treat NaN source as same as non-NaN
        non_nan_sources = group['source_name'].dropna().unique()
        if len(non_nan_sources) > 1:
            continue
        
        # True duplicates — keep first, mark rest
        first_idx = group.index[0]
        for idx in group.index[1:]:
            keep_mask.loc[idx] = False
            dupe_rows.append(idx)
    
    return keep_mask, dupe_rows

keep_mask, dupe_indices = classify_and_split(results)

dupes = results.loc[dupe_indices].reset_index(drop=True)
results_clean = results[keep_mask].reset_index(drop=True)

print(f"1. DUPLICATES: {len(dupes)} rows moved to results_duplicates.csv")
print(f"   Results kept: {len(results_clean)} rows")

# Save duplicates
dupes.to_csv('data/results_duplicates.csv', index=False)

# ============================================================
# 2. GENERATE MISSING EVALUATION_IDS
# ============================================================
# MD5 of model_name + benchmark_id + source_url
def gen_eval_id(row):
    key = f"{row['model_name']}|{row['benchmark_id']}|{row.get('source_url','')}"
    return hashlib.md5(key.encode()).hexdigest()

missing_eval = results_clean['evaluation_id'].isna() | (results_clean['evaluation_id'] == '')
print(f"\n2. EVALUATION_IDS: {missing_eval.sum()} missing")

results_clean.loc[missing_eval, 'evaluation_id'] = results_clean.loc[missing_eval].apply(gen_eval_id, axis=1)
print(f"   Generated {missing_eval.sum()} evaluation_ids")

# ============================================================
# 3. CHECK ZERO SCORES
# ============================================================
zero_scores = results_clean['score'] == 0
print(f"\n3. ZERO SCORES: {zero_scores.sum()} rows")

# Check if metric_score has data (might be the real score)
zero_with_metric = results_clean.loc[zero_scores, 'metric_score'].notna()
print(f"   With metric_score filled: {zero_with_metric.sum()}")
print(f"   Truly zero: {(~zero_with_metric).sum()}")

# Show the truly zero ones by benchmark
truly_zero = results_clean.loc[zero_scores & (~zero_with_metric.reindex(results_clean.index).fillna(False))]
print(f"\n   Truly zero by benchmark:")
for bench, group in truly_zero.groupby('benchmark_id'):
    models = list(group['model_name'].unique())[:5]
    print(f"     {bench}: {len(group)} rows ({', '.join(models)}...)")

# For scores where metric_score exists but score=0, copy metric_score to score
if zero_with_metric.sum() > 0:
    mask = (results_clean['score'] == 0) & results_clean['metric_score'].notna()
    results_clean.loc[mask, 'score'] = results_clean.loc[mask, 'metric_score']
    print(f"\n   Copied metric_score -> score for {mask.sum()} rows")

# ============================================================
# SAVE
# ============================================================
results_clean.to_csv('data/results.csv', index=False)
print(f"\nSaved results.csv: {len(results_clean)} rows")

# Verify FK
models = pd.read_csv('data/models.csv')
valid = results_clean['model_name'].isin(set(models['model_id']))
print(f"FK validity: {valid.sum()}/{len(results_clean)} ({valid.mean()*100:.1f}%)")
