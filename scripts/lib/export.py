"""Export results.csv / benchmarks.csv to downstream formats:
  - EEE JSONL (one file per benchmark + one consolidated file)
  - Excel workbook (one sheet per CSV)

Used by export_eee_jsonl.py and export_xlsx.py at the repo root — those
are kept as separate, named entry points (matching METHODOLOGY.md's
references to them) but contain no logic of their own beyond argument
parsing; everything reusable lives here.
"""
import hashlib
import json
import time

import pandas as pd

from . import config


def generate_eval_id(row):
    key = f"{row.get('model_name', '')}_{row.get('benchmark_id', '')}_{row.get('source_url', '')}"
    return hashlib.md5(key.encode()).hexdigest()


def build_eee_record(row, benchmark_meta, timestamp):
    """Build one EEE schema v0.2.1 record from a results.csv row."""
    gen_args = {}
    if row.get("generation_temperature"):
        gen_args["temperature"] = float(row["generation_temperature"])
    if row.get("generation_max_tokens"):
        gen_args["max_tokens"] = int(float(row["generation_max_tokens"]))
    if row.get("generation_top_p"):
        gen_args["top_p"] = float(row["generation_top_p"])

    hf_url = benchmark_meta.get("hf_url", "") or ""
    hf_repo = hf_url.replace("https://huggingface.co/datasets/", "") if "huggingface.co" in hf_url else None

    record = {
        "schema_version": "0.2.1",
        "evaluation_id": generate_eval_id(row),
        "retrieved_timestamp": timestamp,
        "source_metadata": {
            "source_name": row.get("source_url") or "unknown",
            "source_type": "evaluation_run",
        },
        "model_info": {
            "name": row.get("model_name", ""),
            "id": str(row.get("model_name", "")).lower().replace(" ", "-"),
            # NOTE: results.csv's developer column is `model_developer`,
            # not `developer` -- a past version of this script read the
            # wrong column name and always fell back to "unknown".
            "developer": row.get("model_developer") or "unknown",
            "inference_platform": row.get("inference_platform") or None,
            "inference_engine": {"name": row.get("inference_engine")} if row.get("inference_engine") else None,
        },
        "evaluation_results": [{
            "evaluation_name": benchmark_meta.get("benchmark_name", row.get("benchmark_id")),
            "evaluation_timestamp": timestamp,
            "source_data": {
                "dataset_name": benchmark_meta.get("benchmark_name", row.get("benchmark_id")),
                "hf_repo": hf_repo,
            },
            "metric_config": {
                "evaluation_description": row.get("metric") or "Score",
            },
            "score_details": {
                "score": float(row.get("score") or 0),
            },
            "generation_config": {"generation_args": gen_args} if gen_args else None,
        }],
    }

    if not record["model_info"]["inference_platform"]:
        del record["model_info"]["inference_platform"]
    if not record["model_info"]["inference_engine"]:
        del record["model_info"]["inference_engine"]
    if not record["evaluation_results"][0]["source_data"]["hf_repo"]:
        del record["evaluation_results"][0]["source_data"]["hf_repo"]
    if not record["evaluation_results"][0]["generation_config"]:
        del record["evaluation_results"][0]["generation_config"]
    return record


def export_eee_jsonl(output_dir=None, consolidated_path=None):
    """Write one EEE JSONL file per benchmark plus one consolidated file.
    Returns the total number of records written."""
    output_dir = output_dir or (config.DATA_DIR / "eee_output" / "by_benchmark")
    consolidated_path = consolidated_path or (config.DATA_DIR / "eee_output" / "all_evaluations.jsonl")
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmarks, _, results = _load_for_export()
    benchmarks_by_id = {
        str(row.get("benchmark_id") or row.get("abbreviation", "")).lower(): row
        for row in benchmarks.to_dict(orient="records")
    }

    timestamp = str(time.time())
    total = 0
    with open(consolidated_path, "w") as fall:
        for benchmark_id, group in results.groupby(results["benchmark_id"].str.lower()):
            benchmark_meta = benchmarks_by_id.get(benchmark_id, {})
            output_file = output_dir / f"{benchmark_id}.jsonl"
            with open(output_file, "w") as fout:
                for row in group.to_dict(orient="records"):
                    record = build_eee_record(row, benchmark_meta, timestamp)
                    line = json.dumps(record) + "\n"
                    fout.write(line)
                    fall.write(line)
                    total += 1
    return total, output_dir, consolidated_path


def export_xlsx(output_path=None):
    """Write benchmarks/models/results to a 3-sheet Excel workbook.
    Returns {sheet_name: row_count}."""
    output_path = output_path or (config.DATA_DIR / "llm_benchmarks_export.xlsx")
    benchmarks, models, results = _load_for_export()

    counts = {}
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for df, sheet in [(benchmarks, "benchmarks"), (models, "models"), (results, "results")]:
            df.to_excel(writer, sheet_name=sheet, index=False)
            ws = writer.sheets[sheet]
            ws.freeze_panes = "A2"
            for col in ws.columns:
                max_len = max(
                    len(str(col[0].value) or ""),
                    *(len(str(c.value) or "") for c in col[1:21]),
                )
                ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 60)
            counts[sheet] = len(df)
    return output_path, counts


def _load_for_export():
    from . import io
    return io.load_data()
