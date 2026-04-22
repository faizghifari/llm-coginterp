import csv
import json
import os
import time
import hashlib

base_dir = os.path.expanduser('~/Desktop/llm-benchmarks/')
bmarks_path = os.path.join(base_dir, 'data/benchmarks.csv')
results_path = os.path.join(base_dir, 'data/results.csv')
output_dir = os.path.join(base_dir, 'data/eee_output/by_benchmark')
consolidated_path = os.path.join(base_dir, 'data/eee_output/all_evaluations.jsonl')
os.makedirs(output_dir, exist_ok=True)

bmarks_dict = {}
with open(bmarks_path, 'r') as f:
    for row in csv.DictReader(f):
        b_id = row.get('benchmark_id', row.get('abbreviation', '')).lower()
        bmarks_dict[b_id] = row

def generate_eval_id(row):
    unique_string = f"{row.get('model_name')}_{row.get('benchmark_id')}_{row.get('source_url')}"
    return hashlib.md5(unique_string.encode()).hexdigest()

grouped_results = {}
with open(results_path, 'r') as f:
    for row in csv.DictReader(f):
        b_id = row.get('benchmark_id', '').lower()
        if b_id not in grouped_results:
            grouped_results[b_id] = []
        grouped_results[b_id].append(row)

current_time = str(time.time())
total_written = 0

def build_record(row, bmark_meta, current_time):
    gen_args = {}
    if row.get('generation_temperature'): gen_args['temperature'] = float(row['generation_temperature'])
    if row.get('generation_max_tokens'): gen_args['max_tokens'] = int(float(row["generation_max_tokens"]))
    if row.get('generation_top_p'): gen_args['top_p'] = float(row['generation_top_p'])

    record = {
        "schema_version": "0.2.1",
        "evaluation_id": generate_eval_id(row),
        "retrieved_timestamp": current_time,
        "source_metadata": {
            "source_name": row.get('source_url', 'unknown'),
            "source_type": "evaluation_run"
        },
        "model_info": {
            "name": row.get('model_name', ''),
            "id": row.get('model_name', '').lower().replace(' ', '-'),
            "developer": row.get('developer', 'unknown'),
            "inference_platform": row.get('inference_platform') or None,
            "inference_engine": {
                "name": row.get('inference_engine') or None
            } if row.get('inference_engine') else None
        },
        "evaluation_results": [{
            "evaluation_name": bmark_meta.get('benchmark_name', row.get('benchmark_id')),
            "evaluation_timestamp": current_time,
            "source_data": {
                "dataset_name": bmark_meta.get('benchmark_name', row.get('benchmark_id')),
                "hf_repo": bmark_meta.get('hf_url', '').replace('https://huggingface.co/datasets/', '') if 'huggingface.co' in bmark_meta.get('hf_url', '') else None
            },
            "metric_config": {
                "evaluation_description": row.get('metric', 'Score')
            },
            "score_details": {
                "score": float(row.get('score', 0) or 0)
            },
            "generation_config": {"generation_args": gen_args} if gen_args else None
        }]
    }
    # Clean up Nones
    if not record["model_info"]["inference_platform"]: del record["model_info"]["inference_platform"]
    if not record["model_info"]["inference_engine"]: del record["model_info"]["inference_engine"]
    if not record["evaluation_results"][0]["source_data"]["hf_repo"]: del record["evaluation_results"][0]["source_data"]["hf_repo"]
    if not record["evaluation_results"][0]["generation_config"]: del record["evaluation_results"][0]["generation_config"]
    return record

with open(consolidated_path, 'w') as fall:
    for b_id, rows in grouped_results.items():
        bmark_meta = bmarks_dict.get(b_id, {})
        output_file = os.path.join(output_dir, f"{b_id}.jsonl")

        with open(output_file, 'w') as fout:
            for row in rows:
                record = build_record(row, bmark_meta, current_time)
                line = json.dumps(record) + '\n'
                fout.write(line)
                fall.write(line)
                total_written += 1

print(f"Wrote {total_written} records to {output_dir}/*.jsonl")
print(f"Wrote {total_written} records to {consolidated_path}")
