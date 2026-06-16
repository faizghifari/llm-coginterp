import pandas as pd

results = pd.read_csv('data/results.csv')

# 1. Duplicates - same model+benchmark
dupes = results[results.duplicated(subset=['model_name','benchmark_id'], keep='first')]
print(f'=== DUPLICATE ROWS ({len(dupes)}) ===')
print()
for (model, bench), group in dupes.groupby(['model_name','benchmark_id']):
    all_rows = results[(results['model_name']==model) & (results['benchmark_id']==bench)]
    print(f'{model} | {bench} ({len(all_rows)} entries):')
    for _, r in all_rows.iterrows():
        print(f'  score={r["score"]}, source={r["source_name"]}, metric={r["metric_name"]}, url={r.get("source_url","")}')
    print()

# 2. Empty evaluation_ids
empty_eval = results[results['evaluation_id'].isna() | (results['evaluation_id']=='')]
print(f'=== EMPTY EVALUATION_IDS ({len(empty_eval)}) ===')
print()
print('Columns available:')
print([c for c in results.columns if 'model' in c.lower() or 'benchmark' in c.lower() or 'source' in c.lower()])
print()
# Check if we can regenerate: MD5 of model_name + benchmark_id + source_url
can_regenerate = empty_eval['source_url'].notna()
print(f'Has source_url (can regenerate): {can_regenerate.sum()}/{len(empty_eval)}')
no_url = empty_eval[empty_eval['source_url'].isna() | (empty_eval['source_url']=='')]
print(f'Missing source_url: {len(no_url)}')
if len(no_url) > 0:
    print(no_url[['model_name','benchmark_id','source_name','source_url']].head(10).to_string(index=False))

# 3. Empty/zero scores
empty_score = results[(results['score'].isna()) | (results['score']==0)]
print(f'\n=== EMPTY/ZERO SCORES ({len(empty_score)}) ===')
print()
is_na = empty_score['score'].isna()
is_zero = empty_score['score'] == 0
print(f'NaN scores: {is_na.sum()}')
print(f'Zero scores: {is_zero.sum()}')
print()
print('NaN scores sample:')
nan_scores = empty_score[is_na.reindex(empty_score.index).fillna(False)]
print(nan_scores[['model_name','benchmark_id','score','metric_name','metric_score','source_name','source_url']].head(15).to_string(index=False))
print()
print('Zero scores sample:')
zero_scores = empty_score[is_zero.reindex(empty_score.index).fillna(False)]
print(zero_scores[['model_name','benchmark_id','score','metric_name','metric_score','source_name','source_url']].head(15).to_string(index=False))
