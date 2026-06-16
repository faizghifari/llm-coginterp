import pandas as pd

results = pd.read_csv('data/results.csv')

# Source trustworthiness hierarchy
# Tier 1: Official sources (paper authors, benchmark official site, model developer)
# Tier 2: Reputable aggregators with methodology (HF Open LLM Leaderboard, Chat Arena)
# Tier 3: Third-party aggregators (llm-stats.com, vellum, emergentmind, gorilla, artificialanalysis)
# Tier 4: Unknown/unverifiable

def trust_tier(source_url, source_name):
    url = str(source_url).lower()
    name = str(source_name).lower() if pd.notna(source_name) else ''

    # Tier 1: Official
    official_domains = [
        'arxiv.org', 'aclanthology.org', 'openreview.net',
        'anthropic.com', 'openai.com', 'ai.google', 'google.com',
        'meta.com', 'microsoft.com', 'openai.com',
        'swebench.com', 'crux-eval.github.io', 'lmsys.org', 'chat.lmsys.org',
        'lmarena.ai', 'arena-hard-auto',
    ]
    official_names = ['official', 'paper', 'anthropic', 'openai', 'benchmark']
    for d in official_domains:
        if d in url:
            return 1
    for n in official_names:
        if n in name:
            return 1

    # Tier 2: Reputable aggregators with known methodology
    tier2_domains = [
        'huggingface.co/spaces/open-llm-leaderboard',
        'chat.lmsys.org',
    ]
    for d in tier2_domains:
        if d in url:
            return 2

    # Tier 3: Third-party aggregators
    tier3_domains = [
        'llm-stats.com', 'vellum.ai', 'emergentmind.com',
        'gorilla.cs.berkeley.edu', 'artificialanalysis.ai',
        'pricepertoken.com'
    ]
    for d in tier3_domains:
        if d in url:
            return 3

    # Tier 4: Unknown
    return 4

# Get all dupes
dup_mask = results.duplicated(subset=['model_name', 'benchmark_id', 'metric_name'], keep=False)
all_dupes = results[dup_mask]

print('=== Source trust analysis for conflicting dupes ===\n')

for (model, bench, metric), group in all_dupes.groupby(['model_name', 'benchmark_id', 'metric_name']):
    rows = group.sort_values('score')
    scores = rows['score'].unique()
    
    if len(scores) < 2:
        continue  # skip same-score (pure redundancy)
    
    print(f'CONFLICT: {model} + {bench} + {metric}')
    for _, row in rows.iterrows():
        tier = trust_tier(row['source_url'], row['source_name'])
        print(f'  Tier {tier}: score={row["score"]}  src={row["source_url"]}  name={row["source_name"]}')
    print()
