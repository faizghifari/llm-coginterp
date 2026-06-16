import pandas as pd

results = pd.read_csv('data/results.csv')

dup_mask = results.duplicated(subset=['model_name', 'benchmark_id', 'metric_name'], keep='first')
dupes = results[dup_mask]

# Categorize the dupes
same_score = 0
diff_score = 0
same_source = 0
diff_source = 0

for (model, bench, metric), group in dupes.groupby(['model_name', 'benchmark_id', 'metric_name']):
    original = results[(results['model_name'] == model) & 
                        (results['benchmark_id'] == bench) & 
                        (results['metric_name'] == metric)].iloc[0]
    
    for _, row in group.iterrows():
        if original['score'] == row['score']:
            same_score += 1
        else:
            diff_score += 1
        
        if original['source_url'] == row['source_url']:
            same_source += 1
        else:
            diff_source += 1

print('=== Dupe Summary (54 extra copies) ===')
print()
print('Score comparison:')
print('  Same score both rows:', same_score)
print('  Different scores:', diff_score)
print()
print('Source comparison:')
print('  Same source_url:', same_source)
print('  Different source_url:', diff_source)
print()

# Show the diff score ones - these are the problematic ones
print('=== Dupes with DIFFERENT scores (conflicting data) ===')
for (model, bench, metric), group in dupes.groupby(['model_name', 'benchmark_id', 'metric_name']):
    original = results[(results['model_name'] == model) & 
                        (results['benchmark_id'] == bench) & 
                        (results['metric_name'] == metric)].iloc[0]
    
    for _, row in group.iterrows():
        if original['score'] != row['score']:
            print(f'  {model} + {bench} + {metric}')
            print(f'    A: score={original["score"]} src={original["source_url"]}')
            print(f'    B: score={row["score"]}     src={row["source_url"]}')
            print()

print('=== Dupes with SAME score (redundant copies) ===')
for (model, bench, metric), group in dupes.groupby(['model_name', 'benchmark_id', 'metric_name']):
    original = results[(results['model_name'] == model) & 
                        (results['benchmark_id'] == bench) & 
                        (results['metric_name'] == metric)].iloc[0]
    
    for _, row in group.iterrows():
        if original['score'] == row['score']:
            print(f'  {model} + {bench} + {metric} = {original["score"]}')
            print(f'    A: {original["source_url"]}')
            print(f'    B: {row["source_url"]}')
            print()
