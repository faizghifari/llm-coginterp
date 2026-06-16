import pandas as pd

models = pd.read_csv('data/models_updated.csv')
results = pd.read_csv('data/results.csv')

valid_ids = set(models['model_id'].unique())

# 1. Delete rows referencing removed models
removed = ['Qwen2-Audio','Qwen-Audio','Qwen-Audio-Chat','snorkel-mistral-pairrm-dpo',
           'NusaMT-7B','NusaMT-7B-Minang','M2UGen','MusiLingo',
           'Falcon-7B','Falcon-40B','MPT-7B','MPT-30B','DeepSeek-chat']
before = len(results)
results = results[~results['model_name'].isin(removed)]
print(f'Deleted {before - len(results)} rows referencing removed models')

# 2. Fix all FK mappings
fk_map = {
    # Kept orphans (model_name uses display name, need model_id)
    'ChatGPT-5': 'chatgpt5',
    'ChatGPT-5-mini': 'chatgpt5_mini',
    'Qwen3-VL-235B-A22B-Instruct': 'qwen3_vl_235b',
    'ERNIE-Bot': 'ernie_bot',
    'Claude-instant-1.2': 'claude_instant_1.2',
    'XuanYuan-70B': 'xuanYuan_70b',
    'educhat-base-002-13B': 'educhat_base_13b',
    'MuLLaMa': 'mullama',
    'Qwen3-Max': 'qwen3_max',
    # Alias fixes
    'DeepSeek-V3': 'DeepSeek-v3',
    'GLM-4-9B': 'GLM-4-9B-Chat',
    'Llama-3.1-70B': 'Llama-3.1-70B-Instruct',
    'Llama-2-13B': 'Llama-2-13B-Chat',
    'OpenAI o1': 'o1',
    'Gemini-2.5-Pro': 'Gemini 2.5 Pro',
    'GPT-4.1-Mini': 'GPT-4.1 mini',
    'GPT-4.1-Nano': 'GPT-4.1 nano',
    'DeepSeek-R1-7B': 'DeepSeek-R1-Distill-Qwen-7B',
}

for old, new in fk_map.items():
    mask = results['model_name'] == old
    count = mask.sum()
    if count > 0:
        results.loc[mask, 'model_name'] = new
        print(f'Mapped {old} -> {new} ({count} rows)')

# Check for Llama-4-Large - not in models, flag
remaining_orphans = results[~results['model_name'].isin(valid_ids)]
print(f'\nRemaining orphans after fix: {len(remaining_orphans)}')
if len(remaining_orphans) > 0:
    print(remaining_orphans['model_name'].value_counts().to_string())

# Verify
final_valid = results['model_name'].isin(valid_ids)
print(f'\nFinal FK validity: {final_valid.sum()}/{len(results)} ({final_valid.mean()*100:.1f}%)')

results.to_csv('data/results.csv', index=False)
print('Saved results.csv')
