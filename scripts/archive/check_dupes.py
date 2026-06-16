import pandas as pd

results = pd.read_csv('data/results.csv')

dup_mask = results.duplicated(subset=['model_name', 'benchmark_id', 'metric_name'], keep='first')
dupes = results[dup_mask]

print('Total extra copies:', len(dupes))
print()

key_cols = ['model_name', 'benchmark_id', 'metric_name']
count = 0
for (model, bench, metric), group in dupes.groupby(key_cols):
    original = results[(results['model_name'] == model) & 
                        (results['benchmark_id'] == bench) & 
                        (results['metric_name'] == metric)].iloc[0]
    
    for _, row in group.iterrows():
        print('Model:', model)
        print('Benchmark:', bench, 'Metric:', metric)
        print('  Original: score=', original['score'], 'source=', original['source_url'], 'name=', original['source_name'])
        print('  Dupe:     score=', row['score'], '     source=', row['source_url'], 'name=', row['source_name'])
        print()
        count += 1
        if count >= 15:
            print('... (stopping after 15 examples)')
            break
    if count >= 15:
        break
