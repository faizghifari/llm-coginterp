import pandas as pd
import os

base_dir = os.path.expanduser('~/Desktop/llm-benchmarks/')
data_dir = os.path.join(base_dir, 'data')
output_path = os.path.join(data_dir, 'llm_benchmarks_export.xlsx')

benchmarks = pd.read_csv(os.path.join(data_dir, 'benchmarks.csv'))
models     = pd.read_csv(os.path.join(data_dir, 'models.csv'))
results    = pd.read_csv(os.path.join(data_dir, 'results.csv'))

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    for df, sheet in [(benchmarks, 'benchmarks'), (models, 'models'), (results, 'results')]:
        df.to_excel(writer, sheet_name=sheet, index=False)
        ws = writer.sheets[sheet]

        # Freeze header row
        ws.freeze_panes = 'A2'

        # Auto-fit column widths (capped at 60)
        for col in ws.columns:
            max_len = max(
                len(str(col[0].value) or ''),
                *(len(str(c.value) or '') for c in col[1:21])  # sample first 20 rows
            )
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 60)

print(f"Wrote {output_path}")
print(f"  benchmarks: {len(benchmarks)} rows")
print(f"  models:     {len(models)} rows")
print(f"  results:    {len(results)} rows")
