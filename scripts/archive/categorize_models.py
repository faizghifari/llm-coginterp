#!/usr/bin/env python3
"""Categorize models into KEEP (closed/proprietary/from-scratch) vs UNKNOWN (needs research).

Approach:
- KEEP: Closed/proprietary models from trusted companies
- KEEP: Open models clearly trained from scratch (base models from major orgs)
- UNKNOWN: Everything else (fine-tunes, community models, unclear origin)
"""

import pandas as pd
import json

# Trusted developers that publish proprietary or from-scratch models
TRUSTED_DEVS = {
    'Meta', 'OpenAI', 'Anthropic', 'Google', 'Mistral AI', 'DeepSeek',
    'Alibaba Cloud', 'Microsoft', 'nvidia', '01.AI', 'Zhipu AI',
    'TII UAE', 'stabilityai', 'ibm-granite', 'Cohere', 'xAI',
    'AI21 Labs', 'Baichuan', 'Baidu', 'Tencent', 'ByteDance',
    'Nvidia', 'Moonshot AI', '01ai', 'deepseek-ai', 'Qwen Team',
    'Core42', 'Sakana AI', 'Upstage', 'NVIDIA', 'facebook',
}

# Known from-scratch model families/prefixes
FROM_SCRATCH_PATTERNS = {
    # Meta
    'llama', 'llama-2', 'llama-3', 'llama-3.1', 'llama-3.2', 'llama-3.3', 'llama-4',
    # Mistral
    'mistral', 'mixtral',
    # Google
    'gemma', 'gemma-2', 'gemma2', 'palm', 'gemini', 'flan-palm', 'med-palm',
    # Qwen
    'qwen', 'qwen2', 'qwen2.5', 'qwen3', 'qwen1.5',
    # Microsoft
    'phi', 'phi-3', 'phi-4', 'phi-3.5',
    # DeepSeek
    'deepseek-v2', 'deepseek-v3', 'deepseek-coder', 'deepseek-r1',
    # Yi
    'yi', 'yi-1.5', 'yi 1.5',
    # EleutherAI - from scratch research models
    'pythia', 'gpt-j', 'gpt-neox', 'gpt-neo', 'llemma',
    # BigScience
    'bloom', 'bloomz',
    # Falcon
    'falcon',
    # Databricks
    'dbrx',
    # AllenAI
    'olmo', 'olmo-2', 'olmo 2', 'olmo 1.7',
    # Together
    'redpajama', 'gpt-jt',
    # Salesforce
    'codegen', 'xLAM', 'sfr',
    # HuggingFace
    'smollm', 'smollm2',
    # MosaicML
    'mpt',
    # Facebook
    'opt', 'xglm', 'dolly',
    # InternLM
    'internlm', 'internlm2', 'internlm2.5',
    # Exaone
    'exaone', 'k-exaone',
    # Jais
    'jais',
    # Granite
    'granite',
    # GLM
    'glm', 'glm-4', 'chatglm',
    # Hyperclova
    'hyperclova',
    # Command-R
    'command-r',
    # Grok
    'grok',
    # Aya
    'aya',
    # MiniCPM
    'minicpm',
    # Baichuan
    'baichuan',
    # AceGPT
    'acegpt',
    # NexusRaven
    'nexusraven',
    # LLM360
    'amber', 'k2',
    # TinyLlama
    'tinyllama',
    # BigCode
    'starcoder', 'starcoder2', 'santacoder', 'gpt_bigcode',
    # Distil
    'distilgpt',
    # IDEA Research
    'ziya',
    # PrimeIntellect
    'intellect',
    # Refuel
    'refueled',
    # Rhymes
    'aria',
    # Speakleash
    'bielik',
    # ABACUS AI
    'smaug', 'dracarys', 'bigyi', 'bigstral', 'liberated',
    # mlabonne
    'daredevil', 'neuralbeagle', 'neuraldaredevil', 'chimera', 'alphamonarch', 'beyonder', 'phixtral',
    # LMSYS
    'vicuna', 'longchat',
    # OpenChat
    'openchat', 'opencoder',
    # Nous Research
    'hermes', 'nous-hermes', 'puffin', 'redmond', 'yarn', 'capybara',
    # Open-Orca
    'orca', 'openorca', 'slimorca', 'platypus',
    # WizardLM
    'wizardlm',
    # teknium
    'openhermes', 'collective', 'trismegistus',
    # cognitivecomputations
    'dolphin',
    # Pygmalion
    'pygmalion',
    # Deci
    'decicoder',
    # Gradient
    'gradient',
    # argilla
    'notus', 'notux', 'magpie',
    # AllenAI Tulu
    'tulu',
    # FreedomIntelligence
    'phoenix', 'acegpt',
    # UW / Haotian Liu
    'llava',
    # Shanghai AI Lab
    'internvl', 'sota',
    # OpenBMB
    'minicpm',
    # AI Singapore
    'sea-lion',
    # BigCode
    'codegen',
    # OpenAssistant
    'oasst',
    # HuggingFaceH4
    'zephyr', 'starchat',
    # Intel
    'neural-chat',
    # lmsys
    'vicuna', 'longchat',
    # mlabonne BigQwen
    'bigqwen',
    # QwQ
    'qwq',
    # DeepSeek distill
    'deepseek-r1-distill',
    # PolyLM
    'polylm',
    # SALMONN
    'salmonn',
    # BLIP
    'blip',
    # Codex
    'codex',
    # HyperCLOVA
    'hyperclova',
    # K-EXAONE
    'k-exaone',
    # Med-PaLM
    'med-palm',
    # Flan-PaLM
    'flan-palm',
    # Kimi
    'kimi',
    # MiniCPM-V
    'minicpm-v',
    # SOTA
    'sota',
    # SALMONN
    'salmonn',
    # PolyLM
    'polylm',
    # Codex
    'codex',
    # InternVL
    'internvl',
    # LLaVA
    'llava',
    # semcoder
    'semcoder',
}

def is_trusted_dev(developer):
    """Check if developer is a trusted company/org."""
    if pd.isna(developer):
        return False
    dev_lower = developer.lower().strip()
    return any(t.lower() in dev_lower or dev_lower in t.lower() for t in TRUSTED_DEVS)

def is_from_scratch_pattern(model_id, model_name, developer):
    """Check if model matches a known from-scratch pattern."""
    combined = f"{model_id} {model_name}".lower()
    if developer and not pd.isna(developer):
        combined += f" {developer.lower()}"
    return any(pattern in combined for pattern in FROM_SCRATCH_PATTERNS)

def is_closed_model(model_type, developer):
    """Check if model is closed/proprietary."""
    if model_type and str(model_type).lower() in ['closed']:
        return True
    # Models from API-only companies
    api_only_devs = {'OpenAI', 'Anthropic', 'Google', 'Cohere', 'xAI', 'AI21 Labs'}
    if developer and not pd.isna(developer):
        dev_lower = developer.lower().strip()
        if any(d.lower() in dev_lower or dev_lower in d.lower() for d in api_only_devs):
            # Check if it's an API model (not open weights)
            if model_type and str(model_type).lower() not in ['open', 'open-weights']:
                return True
    return False

def categorize_model(row):
    """Categorize a model row into KEEP or UNKNOWN."""
    model_id = str(row['model_id'])
    model_name = str(row['model_name'])
    model_family = row.get('model_family', pd.NA)
    developer = row.get('developer', pd.NA)
    model_type = row.get('model_type', pd.NA)

    # Closed/proprietary models - KEEP
    if is_closed_model(model_type, developer):
        return 'KEEP', f'Closed/proprietary model'

    # Trusted dev with identifiable from-scratch pattern - KEEP
    if is_trusted_dev(developer) and is_from_scratch_pattern(model_id, model_name, developer):
        return 'KEEP', f'From-scratch from trusted dev ({developer})'

    # Trusted dev but unclear if from-scratch or fine-tune - KEEP (conservative)
    if is_trusted_dev(developer):
        return 'KEEP', f'Trusted dev ({developer}), likely from-scratch or official fine-tune'

    # Known from-scratch patterns from community orgs
    if is_from_scratch_pattern(model_id, model_name, developer):
        return 'KEEP', f'Known from-scratch pattern'

    # Everything else - UNKNOWN
    return 'UNKNOWN', f'Needs research. Dev: {developer if not pd.isna(developer) else "nan"}'

def main():
    df = pd.read_csv('data/models.csv')
    print(f"Total models: {len(df)}")

    results = []
    for _, row in df.iterrows():
        category, reason = categorize_model(row)
        results.append({
            'model_id': row['model_id'],
            'model_name': row['model_name'],
            'developer': row.get('developer', ''),
            'model_type': row.get('model_type', ''),
            'category': category,
            'reason': reason,
        })

    results_df = pd.DataFrame(results)

    # Counts
    print("\n=== CATEGORIZATION ===")
    print(results_df['category'].value_counts())

    # Save
    output_file = 'data/models_initial_categorization.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nSaved to: {output_file}")

    # Print UNKNOWN models
    print(f"\n=== UNKNOWN MODELS (need research) ===")
    unknown = results_df[results_df['category'] == 'UNKNOWN']
    print(f"Count: {len(unknown)}")
    for _, r in unknown.iterrows():
        dev = '' if pd.isna(r['developer']) else str(r['developer'])
        print(f"  {r['model_id']:45s} | {dev:25s}")

if __name__ == '__main__':
    main()
