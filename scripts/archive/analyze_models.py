#!/usr/bin/env python3
"""Analyze models.csv to categorize models for cleanup.

Categories:
- KEEP: Closed/proprietary from trusted devs, or open models trained from scratch with clear family
- FLAG: Unclear if fine-tuned or from scratch, needs manual review
- REMOVE: Fine-tuned models with no identifiable base model/family
"""

import pandas as pd
import json
import sys

# Trusted developers that publish proprietary or from-scratch models
TRUSTED_DEVS = {
    'Meta', 'OpenAI', 'Anthropic', 'Google', 'Mistral AI', 'DeepSeek',
    'Alibaba Cloud', 'Microsoft', 'nvidia', '01.AI', 'Zhipu AI',
    'TII UAE', 'stabilityai', 'ibm-granite', 'Cohere', 'xAI',
    'AI21 Labs', 'Baichuan', 'Baidu', 'Tencent', 'ByteDance',
    'Nvidia', 'Moonshot AI', '01ai', 'deepseek-ai', 'Qwen Team',
    'Core42', 'Sakana AI', 'Upstage', 'NVIDIA',
}

# Known from-scratch model families (open weights)
FROM_SCRATCH_FAMILIES = {
    'llama', 'llama 2', 'llama 3', 'llama 3.1', 'llama 3.2', 'llama 3.3',
    'mistral', 'mixtral', 'gemma', 'gemma 2', 'qwen', 'qwen2', 'qwen2.5', 'qwen3',
    'yi', 'yi 1.5', 'olmo', 'olmo 2', 'olmo 1.7',
    'phi', 'phi-3', 'phi-4',
    'bloom', 'bloomz',
    'pythia', 'gpt-j', 'gpt-neox',
    'falcon', 'falcon-7b', 'falcon-40b',
    'dbrx', 'dbrx-instruct',
    'grok', 'grok-2',
    'claude', 'gpt', 'gemini',
    'jais', 'jais-chat',
    'exaone', 'k-exaone',
    'command-r', 'command-r+',
    'granite', 'granite-3',
    'bactrian', 'bactrian-2',
    'sea-lion',
    'internlm',
    'deepseek-v2', 'deepseek-v3', 'deepseek-coder',
    'qwen-audio', 'qwen-vl',
    'aya', 'aya-23',
    'glm', 'glm-4', 'chatglm',
    'hyperclova',
    'med-palm', 'flan-palm',
    'kimi', 'kimi k2',
    'minicpm',
    'internvl',
    'qwen1.5', 'qwen1.5-chat',
    'llama-3.1', 'llama-3.2', 'llama-3.3',
}

# Keywords suggesting fine-tuning
FT_KEYWORDS = [
    'instruct', 'chat', 'fine-tuned', 'ft-', '-ft', '-chat', '-instruct',
    'align', 'dpo', 'rlhf', 'orpo', 'sft', 'alpaca', 'vicuna', 'gemma-it',
    'hermes', 'openhermes', 'zephyr', 'openchat', 'open-orca', 'orca',
    'dpo', 'ppo', 'kto', 'simpo', 'ppo',
    '-grpo', '-dpo', '-sft', '-rlhf',
    'grit', 'nous-hermes', 'teknium', 'cognitivecomputations',
    'firefunction', 'wizard', 'magist', 'magpie',
    'lmsys', 'lmsys-chat', 'lmms',
    'blip', 'salmonn', 'sota',
    'solar', 'solar-10.7b',
    'exaone',
]

def is_fine_tuned_name(name):
    """Check if model name suggests it's fine-tuned."""
    name_lower = name.lower()
    return any(kw in name_lower for kw in FT_KEYWORDS)

def is_trusted_dev(developer):
    """Check if developer is a trusted company/org."""
    if pd.isna(developer):
        return False
    dev_lower = developer.lower().strip()
    return any(t.lower() in dev_lower or dev_lower in t.lower() for t in TRUSTED_DEVS)

def has_clear_family(family):
    """Check if model_family is populated and meaningful."""
    if pd.isna(family):
        return False
    return len(str(family).strip()) > 0

def categorize_model(row):
    """Categorize a model row."""
    model_id = row['model_id']
    model_name = str(row['model_name'])
    model_family = row.get('model_family', pd.NA)
    developer = row.get('developer', pd.NA)
    model_type = row.get('model_type', pd.NA)
    base_model = row.get('base_model', pd.NA)
    url = row.get('url', pd.NA)

    reasons = []

    # Closed/proprietary models - KEEP
    if model_type and str(model_type).lower() in ['closed', 'unknown']:
        if is_trusted_dev(developer):
            return 'KEEP', f'Closed model from trusted dev ({developer})'

    # Trusted developer with clear family - KEEP
    if is_trusted_dev(developer) and has_clear_family(model_family):
        return 'KEEP', f'Trusted dev ({developer}) with clear family ({model_family})'

    # Has base_model filled - KEEP (base is known)
    if not pd.isna(base_model):
        return 'KEEP', f'Has base_model: {base_model}'

    # Open model from trusted dev - likely from scratch, KEEP
    if is_trusted_dev(developer):
        return 'KEEP', f'Open model from trusted dev ({developer})'

    # Has model_family filled
    if has_clear_family(model_family):
        # Check if name suggests fine-tuning but family is set
        if is_fine_tuned_name(model_name):
            # Fine-tuned but family is known - KEEP
            return 'KEEP', f'Fine-tuned but family known: {model_family}'
        else:
            # Likely from-scratch with family - KEEP
            return 'KEEP', f'From-scratch with family: {model_family}'

    # No family, no trusted dev, no base_model
    if is_fine_tuned_name(model_name):
        return 'REMOVE', f'Fine-tuned (name: {model_name}), no family/base_model, dev: {developer}'

    # Unknown - could be from-scratch or fine-tuned
    return 'FLAG', f'No family/base_model, unclear origin. Dev: {developer}, Type: {model_type}'

def main():
    df = pd.read_csv('data/models.csv')
    print(f"Total models: {len(df)}")

    results = []
    for _, row in df.iterrows():
        category, reason = categorize_model(row)
        results.append({
            'model_id': row['model_id'],
            'model_name': row['model_name'],
            'model_family': row.get('model_family', ''),
            'developer': row.get('developer', ''),
            'model_type': row.get('model_type', ''),
            'base_model': row.get('base_model', ''),
            'category': category,
            'reason': reason,
        })

    results_df = pd.DataFrame(results)

    # Counts
    print("\n=== CATEGORIZATION SUMMARY ===")
    print(results_df['category'].value_counts())

    # Save detailed output
    output_file = 'data/models_cleanup_analysis.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nDetailed analysis saved to: {output_file}")

    def safe(val):
        return '' if pd.isna(val) else str(val)

    # Print FLAG models for review
    print("\n=== FLAGGED FOR REVIEW ===")
    flagged = results_df[results_df['category'] == 'FLAG']
    for _, r in flagged.iterrows():
        print(f"  {safe(r['model_id']):40s} | {safe(r['developer']):25s} | {safe(r['model_family'])} | {r['reason']}")

    # Print REMOVE candidates
    print("\n=== REMOVE CANDIDATES ===")
    remove = results_df[results_df['category'] == 'REMOVE']
    for _, r in remove.iterrows():
        print(f"  {safe(r['model_id']):40s} | {safe(r['developer']):25s} | {r['reason']}")

if __name__ == '__main__':
    main()
