import pandas as pd

results = pd.read_csv('data/results.csv')

# Find all model+benchmark combos with multiple rows
grouped = results.groupby(['model_name','benchmark_id'])
multi = grouped.filter(lambda x: len(x) > 1)

print(f"Total rows in dup groups: {len(multi)}")
print(f"Unique model+bench combos with multiple rows: {multi.groupby(['model_name','benchmark_id']).ngroups}")

# Classify: true dupes (same metric+source) vs different metrics/subtasks
true_dupes = []
different_metrics = []
different_sources = []

for (model, bench), group in results.groupby(['model_name','benchmark_id']):
    if len(group) <= 1:
        continue
    
    # Check if metric_name differs
    metrics = group['metric_name'].unique()
    sources = group['source_name'].unique()
    
    # True duplicates: same metric, same source, same everything
    exact_dupes = group.duplicated(keep=False)
    if exact_dupes.any() and group.drop_duplicates().shape[0] == 1:
        true_dupes.append((model, bench, len(group)))
    elif len(metrics) > 1:
        different_metrics.append((model, bench, len(group), len(metrics)))
    elif len(sources) > 1:
        different_sources.append((model, bench, len(group), list(sources)))
    else:
        # Same metric, same source, but different scores
        true_dupes.append((model, bench, len(group)))

print(f"\n=== CLASSIFICATION ===")
print(f"True duplicates (same metric/source): {len(true_dupes)} groups")
print(f"Different sub-metrics: {len(different_metrics)} groups")
print(f"Different sources: {len(different_sources)} groups")

print(f"\n=== TRUE DUPLICATES (should move to results_duplicates.csv) ===")
rows_to_move = 0
for model, bench, count in true_dupes:
    print(f"  {model} | {bench} ({count} rows)")
    rows_to_move += count - 1
print(f"Total rows to move (keeping first): {rows_to_move}")

print(f"\n=== DIFFERENT SUB-METRICS (keep as is) ===")
for model, bench, count, n_metrics in different_metrics[:10]:
    print(f"  {model} | {bench} ({count} rows, {n_metrics} metrics)")
print(f"  ... and {len(different_metrics)-10} more" if len(different_metrics) > 10 else "")

print(f"\n=== DIFFERENT SOURCES (keep as is) ===")
for model, bench, count, sources in different_sources[:10]:
    print(f"  {model} | {bench} ({count} rows, sources: {sources})")
print(f"  ... and {len(different_sources)-10} more" if len(different_sources) > 10 else "")
