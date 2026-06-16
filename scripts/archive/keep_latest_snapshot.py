import pandas as pd

results = pd.read_csv('data/results.csv')
dupes = pd.read_csv('data/results_duplicates.csv')

# Combine
combined = pd.concat([results, dupes], ignore_index=True)

# Parse dates
combined['date_recorded'] = pd.to_datetime(combined['date_recorded'], errors='coerce')

# For each model+benchmark+metric+source group, keep the latest date
# If no date, keep the last one (arbitrary — they're unordered)
combined = combined.sort_values('date_recorded', ascending=False, na_position='last')

# Keep first per group (latest date first, NaN dates last)
kept = combined.drop_duplicates(subset=['model_name','benchmark_id','metric_name','source_name'], keep='first')

# Sort back
kept = kept.sort_index()

# Separate back
# Find which rows came from dupes that were dropped
dropped = combined.drop_duplicates(subset=['model_name','benchmark_id','metric_name','source_name'], keep='last')
dropped = dropped[~dropped.index.isin(kept.index)]

print(f"Combined: {len(combined)}")
print(f"Kept (latest): {len(kept)}")
print(f"Dropped (older): {len(combined) - len(kept)}")

# Check date distribution of kept vs dropped
has_date = kept['date_recorded'].notna()
print(f"\nKept with date: {has_date.sum()}")
print(f"Kept without date: {(~has_date).sum()}")

# Save
kept = kept.drop(columns=['date_recorded'])
kept.to_csv('data/results.csv', index=False)

# Verify
models = pd.read_csv('data/models.csv')
benchmarks = pd.read_csv('data/benchmarks.csv')
valid_m = kept['model_name'].isin(set(models['model_id']))
valid_b = kept['benchmark_id'].isin(set(benchmarks['benchmark_id']))
print(f"\nFinal results: {len(kept)}")
print(f"Model FK: {valid_m.mean()*100:.1f}%")
print(f"Benchmark FK: {valid_b.mean()*100:.1f}%")
print(f"Duplicates: {kept.duplicated(subset=['model_name','benchmark_id','metric_name','source_name']).sum()}")
