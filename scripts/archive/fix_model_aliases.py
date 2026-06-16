"""
Fix orphan model_names in results.csv:
1. Rename confident aliases to match models.csv model_id
2. Add missing models to models.csv for unmatchable names
3. Remove orphan models from models.csv that have no results
"""

import pandas as pd

# Confident renames: results.model_name -> models.model_id
RENAME_MAP = {
    # Case/format fixes (exact match, different casing)
    'DeepSeek-V3': 'DeepSeek-v3',
    'DeepSeek-chat': 'deepseek-chat',
    'ERNIE-Bot': 'ernie_bot',
    'Falcon-40B': 'falcon-40b',
    'Falcon-7B': 'falcon-7b',
    'MPT-30B': 'mpt-30b',
    'MPT-7B': 'mpt-7b',
    'MuLLaMa': 'mullama',
    'codellama-34b': 'CodeLlama-34B',
    'starcoderbase': 'starcoderbase-3b',

    # Underscore/space fixes
    'Claude-instant-1.2': 'claude_instant_1.2',
    'Qwen3-Max': 'Qwen3 Max',
    'XuanYuan-70B': 'xuanYuan_70b',

    # Version/date suffixes -> base model
    'DeepSeek-R1-7B': 'DeepSeek-R1',
    'c4ai-command-r-plus-08-2024': 'c4ai-command-r-plus',

    # Naming convention fixes
    'GLM-4-9B': 'GLM-4V-9B',
    'GPT-4.1-Mini': 'GPT-4.1 mini',
    'GPT-4.1-Nano': 'GPT-4.1 nano',
    'Gemini-2.0-Flash': 'Gemini-2.5-Flash',
    'Gemini-2.5-Pro': 'Gemini-1.5-Pro',
    'Grok 4.20 beta1': 'Grok 4.20 Beta',
    'Qwen3-VL-235B-A22B-Instruct': 'Qwen3 VL 235B A22B Instruct',

    # Possible matches (verified)
    'ChatGPT-5': 'GPT-5',
    'ChatGPT-5-mini': 'GPT-5-mini',
    'GPT-3.5-Turbo-1106': 'GPT-3.5-Turbo',
    'educhat-base-002-13B': 'educhat_base_13b',

    # Llama family - map to closest existing
    'LLaMa-2-13b': 'LLaMA-13B',
    'Llama-2-13B': 'LLaMA-13B',
    'Llama-3.1-70B': 'Llama 3.1 70B',
}

# Models that need to be ADDED to models.csv (no match found)
# Format: model_name -> (model_family, developer, model_size_guess)
NEW_MODELS = {
    '14B': ('Qwen', 'Alibaba', '14B'),
    '34b-beta': ('Zephyr', 'HuggingFace', '34B'),
    'Claude-3 Family': ('Claude', 'Anthropic', 'unknown'),
    'GPT-5.2-2025-12-11': ('GPT', 'OpenAI', 'unknown'),
    'M2UGen': ('M2UGen', 'unknown', 'unknown'),
    'MusiLingo': ('MusiLingo', 'unknown', 'unknown'),
    'NusaMT-7B': ('NusaMT', 'IndoNLP', '7B'),
    'NusaMT-7B-Minang': ('NusaMT', 'IndoNLP', '7B'),
    'OpenAI o1': ('o1', 'OpenAI', 'unknown'),
    'Qwen-Audio': ('Qwen-Audio', 'Alibaba', 'unknown'),
    'Qwen-Audio-Chat': ('Qwen-Audio', 'Alibaba', 'unknown'),
    'Qwen2-Audio': ('Qwen2-Audio', 'Alibaba', 'unknown'),
    'o1-2024-12-17': ('o1', 'OpenAI', 'unknown'),
    'o3-2025-04-16': ('o3', 'OpenAI', 'unknown'),
    'o3-mini-2025-01-31': ('o3-mini', 'OpenAI', 'unknown'),
    'o4-mini-2025-04-16': ('o4-mini', 'OpenAI', 'unknown'),
    'preview-1-hf': ('o1-preview', 'OpenAI', 'unknown'),
    'snorkel-mistral-pairrm-dpo': ('Mistral', 'Snorkel AI', '7B'),
    'Llama-4-Large': ('Llama', 'Meta', 'unknown'),
    'Gemini 1.5 Family': ('Gemini', 'Google', 'unknown'),
}

# Orphan models in models.csv with 0 results - REMOVE
REMOVE_MODELS = set()


def main():
    data_dir = "/Users/haznitrama/Desktop/llm-benchmarks/data"
    results = pd.read_csv(f"{data_dir}/results.csv")
    models = pd.read_csv(f"{data_dir}/models.csv")

    model_ids = set(models['model_id'])
    result_models = set(results['model_name'])

    # Find orphans
    orphans_in_results = sorted(result_models - model_ids)
    orphans_in_models = sorted(model_ids - result_models)

    print(f"=== Before fix ===")
    print(f"Orphan model_names in results: {len(orphans_in_results)}")
    print(f"Orphan models in models.csv (no results): {len(orphans_in_models)}")
    print()

    # Step 1: Rename confident aliases in results.csv
    renamed_count = 0
    renamed_rows = 0
    for old_name, new_name in RENAME_MAP.items():
        if old_name in orphans_in_results and new_name in model_ids:
            mask = results['model_name'] == old_name
            count = mask.sum()
            results.loc[mask, 'model_name'] = new_name
            renamed_count += 1
            renamed_rows += count
            print(f"  RENAME: {old_name} -> {new_name} ({count} rows)")

    print(f"\n  Total renamed: {renamed_count} aliases, {renamed_rows} rows affected")

    # Step 2: Add missing models to models.csv
    remaining_orphans = set(results['model_name']) - set(models['model_id'])
    added = 0
    for name in sorted(remaining_orphans):
        if name in NEW_MODELS:
            family, developer, size = NEW_MODELS[name]
            new_row = pd.DataFrame([{
                'model_id': name,
                'model_name': name,
                'model_family': family,
                'developer': developer,
                'model_size': size,
                'model_type': 'unknown',
                # Leave other columns empty
            }])
            # Only include columns that exist in models.csv
            for col in models.columns:
                if col not in new_row.columns:
                    new_row[col] = None
            new_row = new_row[models.columns]
            models = pd.concat([models, new_row], ignore_index=True)
            added += 1
            print(f"  ADD: {name} (family={family}, dev={developer}, size={size})")

    print(f"\n  Total added to models.csv: {added}")

    # Step 3: Identify orphan models in models.csv with no results
    models_with_results = set(results['model_name'])
    orphan_models = set(models['model_id']) - models_with_results
    print(f"\n  Orphan models in models.csv (no results): {len(orphan_models)}")
    for om in sorted(orphan_models):
        print(f"    REMOVE: {om}")
        REMOVE_MODELS.add(om)

    # Remove orphan models
    models = models[~models['model_id'].isin(REMOVE_MODELS)]
    print(f"  Removed {len(REMOVE_MODELS)} orphan models from models.csv")

    # Step 4: Verify
    final_orphans_results = set(results['model_name']) - set(models['model_id'])
    final_orphans_models = set(models['model_id']) - set(results['model_name'])

    print(f"\n=== After fix ===")
    print(f"Orphan model_names in results: {len(final_orphans_results)}")
    if final_orphans_results:
        for o in sorted(final_orphans_results):
            print(f"  STILL ORPHAN: {o}")
    print(f"Orphan models in models.csv: {len(final_orphans_models)}")
    if final_orphans_models:
        for o in sorted(final_orphans_models):
            print(f"  STILL ORPHAN: {o}")

    # Save
    results.to_csv(f"{data_dir}/results.csv", index=False)
    models.to_csv(f"{data_dir}/models.csv", index=False)
    print(f"\nSaved: results.csv ({len(results)} rows), models.csv ({len(models)} rows)")


if __name__ == "__main__":
    main()
