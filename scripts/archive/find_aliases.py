import pandas as pd

models = pd.read_csv('data/models_updated.csv')

targets = ['Gemini-2.5-Pro','GPT-4.1-Mini','GPT-4.1-Nano','Llama-4-Large','DeepSeek-R1-7B','Llama-2-13B','OpenAI o1']

for t in targets:
    if t == 'Gemini-2.5-Pro':
        match = models[models['model_name'].str.contains('Gemini.*2\.5.*Pro', case=False, na=False) | models['model_id'].str.contains('gemini.*2.5.*pro', case=False, na=False)]
    elif t == 'GPT-4.1-Mini':
        match = models[models['model_name'].str.contains('GPT.*4\.1.*mini', case=False, na=False) | models['model_id'].str.contains('gpt.*4.1.*mini', case=False, na=False)]
    elif t == 'GPT-4.1-Nano':
        match = models[models['model_name'].str.contains('GPT.*4\.1.*nano', case=False, na=False) | models['model_id'].str.contains('gpt.*4.1.*nano', case=False, na=False)]
    elif t == 'Llama-4-Large':
        match = models[models['model_name'].str.contains('Llama.*4.*Large', case=False, na=False) | models['model_id'].str.contains('llama.*4.*large', case=False, na=False)]
    elif t == 'DeepSeek-R1-7B':
        match = models[models['model_name'].str.contains('DeepSeek.*R1.*7B', case=False, na=False) | models['model_id'].str.contains('deepseek.*r1.*7b', case=False, na=False)]
    elif t == 'Llama-2-13B':
        match = models[(models['model_name'].str.contains('Llama.*2.*13B', case=False, na=False)) | (models['model_id'].str.contains('llama.*2.*13b', case=False, na=False))].drop_duplicates()
    elif t == 'OpenAI o1':
        match = models[models['model_id'].str.contains('o1', case=False, na=False) & ~models['model_id'].str.contains('preview')]
    
    print(f'{t}:')
    if len(match) > 0:
        for _, row in match.head(3).iterrows():
            print(f'  model_id={row["model_id"]}, model_name={row["model_name"]}')
    else:
        print('  NOT FOUND')
    print()
