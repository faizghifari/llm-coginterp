import pandas as pd

models = pd.read_csv('data/models_updated.csv')

suspicious_devs = ['7B', '30B', '40B', '13B']
to_remove = models[(models['total_results'] == 0) & (models['developer'].isin(suspicious_devs))]

print('REMOVING:')
print(to_remove[['model_name','model_id','developer']].to_string(index=False))
print(f'Count: {len(to_remove)}')
print()

keep_mask = ~((models['total_results'] == 0) & (models['developer'].isin(suspicious_devs)))
models_clean = models[keep_mask]
print(f'Before: {len(models)}')
print(f'After: {len(models_clean)}')
print(f'Removed: {len(to_remove)}')

models_clean.to_csv('data/models_updated.csv', index=False)
print('Saved.')
