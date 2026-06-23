# Data Extraction Guide — Kaggle & Papers With Code
*For use by extraction agents. Read entirely before writing a single line of code.*

---

## 0. Non-negotiable rules (violations cause data corruption)

1. **Never write directly to `data/benchmarks.csv`, `data/models.csv`, or `data/results.csv`.**
   Always write to staging files first:
   - `data/staging_{source}_benchmarks.csv`
   - `data/staging_{source}_models.csv`
   - `data/staging_{source}_results.csv`
   Where `{source}` is `pwc` or `kaggle`.

2. **Staging files must be idempotent.** If you run the script twice, the staging files must be identical after the second run (dedup on `[benchmark_id, model_name, metric_name, source_url]` before writing).

3. **Scores must be on a 0–100 scale** unless the metric is an absolute measure (see §3.2). A raw score of `0.75` means **75.0**, not `0.75`.

4. **`results.model_name` is a foreign key to `models.model_id`.** Every model_name you write into `staging_{source}_results.csv` must also appear in `staging_{source}_models.csv` (or already exist in the main `data/models.csv`). Never write a result row for a model you haven't registered.

5. **CSV I/O conventions must be exact.** Read: `pd.read_csv(..., dtype=str, keep_default_na=False)`. Write: `df.to_csv(..., index=False, lineterminator="\r\n")`.

6. **After staging is complete, do NOT merge.** The merge step is run separately by the user via `python3 scripts/merge_{source}_staging.py`.

---

## 1. Repository layout

```
data/
  benchmarks.csv          ← main benchmarks file (DO NOT TOUCH)
  models.csv              ← main models file (DO NOT TOUCH)
  results.csv             ← main results file (DO NOT TOUCH)
  staging_pwc_benchmarks.csv     ← your output for PwC
  staging_pwc_models.csv
  staging_pwc_results.csv
  staging_kaggle_benchmarks.csv  ← your output for Kaggle
  staging_kaggle_models.csv
  staging_kaggle_results.csv

scripts/
  lib/
    config.py   ← canonical paths + RESULT_IDENTITY_KEY
    io.py       ← load_data(), save_csv(), save_results()
  verify_data.py
  manage_data.py
```

`scripts/lib/io.load_data()` returns `(benchmarks_df, models_df, results_df)`.
Use it to check what already exists before deciding what is "new".

---

## 2. Exact file schemas

All three files use `dtype=str` for all columns. Missing values are empty strings `""`, never `NaN`.

### 2.1 `benchmarks.csv` — 37 columns
```
benchmark_id, benchmark_name, abbreviation, year, venue, category, subcategory,
paper_url, github_url, hf_url, languages_count, languages, cultures_regions,
task_types, metrics, dataset_size, annotation_method, key_finding, notes, other_url,
description, task_type, domain, source_url, title, acronym, huggingface_url,
language_count, geography, metric_type, data_source, organization, name,
release_date, paper_title, source_type, source_name
```

**Required fields (non-empty for every new row you add):**
- `benchmark_id`: lowercase, alphanumeric + underscores/hyphens, globally unique. Derive from the benchmark name: `"GSM8K"` → `"gsm8k"`, `"TriviaQA"` → `"triviaqa"`. If name has spaces, use underscores.
- `benchmark_name`: human-readable display name
- `source_url`: the canonical leaderboard URL where you pulled data from
- `source_name`: e.g. `"Papers With Code"` or `"Kaggle AI Benchmarks"`
- `source_type`: `"leaderboard"` or `"paper"` or `"dataset"`
- `organization`: the org responsible (e.g. `"Papers With Code"`, `"Meta AI"`, `"Kaggle"`)

**Do NOT populate:** `benchmark_count`, `total_results`, `avg_score` — these are computed by `manage_data.py recompute-stats`.

### 2.2 `models.csv` — 24 columns
```
model_id, model_name, model_family, developer, sizes, types, benchmark_count,
total_results, avg_score, model_type, model_size, provider, type, base_model,
year_evaluated, inference_platform, inference_engine, organization, release_date,
architecture, context_window, license, url, parameters_billion
```

**Required fields:**
- `model_id`: exact same value as `model_name` unless you're remapping a known alias. Keep the model's common name as used by the source (PwC or Kaggle). You will clean up aliases in the merge script.
- `model_name`: same as `model_id` for new entries
- `developer`: the model's maker if known (e.g. `"OpenAI"`, `"Meta"`, `"Google"`)

**Do NOT populate:** `benchmark_count`, `total_results`, `avg_score` — computed separately.

### 2.3 `results.csv` — 38 columns
```
benchmark_id, model_name, model_family, model_size, active_size, model_type,
score, metric_name, setup, num_shot_sample, language, source_url, year_evaluated,
notes, evaluation_id, source_name, source_type, source_organization,
model_developer, model_id, hf_repo, hf_split, samples_number, metric_lower_is_better,
metric_score_type, metric_min_score, metric_max_score, reasoning_enabled,
inference_platform, inference_engine, generation_temperature, generation_max_tokens,
generation_top_p, benchmark_name, metric_score, metric, score_type, date_recorded
```

**Required fields:**
- `benchmark_id`: must match a row in your staging benchmarks file OR already exist in `data/benchmarks.csv`
- `model_name`: must match a row in your staging models file OR already exist in `data/models.csv`
- `score`: float as string, on 0–100 scale (see §3.2). Never empty. Skip the row if no score exists.
- `metric_name`: the name of the metric (e.g. `"Accuracy"`, `"F1"`, `"EM"`)
- `source_url`: the page/URL where this score was found

**Dedup key** (`RESULT_IDENTITY_KEY` in `scripts/lib/config.py`):
```python
["model_name", "model_id", "benchmark_id", "metric_name",
 "setup", "reasoning_enabled", "num_shot_sample", "source_url", "language"]
```
Two rows are "the same evaluation" if all these fields match — deduplicate before writing.

---

## 3. Score normalization

### 3.1 The rule
PwC and Kaggle report scores in various formats. You must normalize to **0–100 scale**:
- Raw `0.75` → store as `"75.0"`
- Raw `75.2` → store as `"75.2"` (already on 0–100 scale)
- Raw `1.0` → store as `"100.0"`

Detection: if the raw value is in `[0.0, 1.0]`, multiply by 100. If it's already above 1.0, keep as-is. Cap at 100.0 to handle floating-point noise.

```python
def normalize_score(raw, metric_name=""):
    """Normalize a score to 0-100 scale. Returns None if unparseable."""
    NO_SCALE_METRICS = {"Perplexity", "perplexity", "BPB", "Bits/byte", "bits/byte",
                        "BLEURT", "BERTScore", "Elo", "elo", "# eval"}
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if metric_name in NO_SCALE_METRICS:
        return round(val, 6)          # keep absolute
    if 0.0 <= val <= 1.0:
        return min(round(val * 100, 4), 100.0)
    return round(val, 4)              # already on 0-100 scale or absolute
```

### 3.2 Exceptions — keep absolute (do NOT multiply by 100)
- `Perplexity` / `perplexity` (typical range: 5–200)
- `BPB` / `Bits/byte` (typical range: 0.5–4)
- `BLEURT` (typical range: −1 to 1, but reported on −100 to 100 sometimes — check)
- `BERTScore` (typical range: 0.7–1.0 — but this usually IS a fraction, so it DOES get ×100 under the rule above unless the values you see are already >1)
- `Elo` (typical range: 800–1600)
- `# eval`, `count` (counts, not quality scores)

### 3.3 lower_is_better
Do not skip rows or change the scale because `lower_is_better=True`. Store the direction in the `metric_lower_is_better` column (`"True"` or `"False"`) and apply ×100 normalization normally. Example: WER (word error rate) is lower-is-better and is a fraction → normalize to percentage.

---

## 4. Model naming rules

### 4.1 No special characters
Strip status markers before storing: `☠`, `⚠`, `†`, `★`, `✗`, `✓`, circled numbers `⓪①②③④⑤`.

```python
import re
_STATUS_MARKERS = re.compile(r'[☠⚠†★✗✓⓪①②③④⑤]')
def clean_model_name(name):
    return _STATUS_MARKERS.sub("", name).strip()
```

### 4.2 Keep the source's naming convention
Do NOT try to normalize `"GPT-4o (2024-05-13)"` to `"GPT-4o"` yourself during extraction. Extract the exact name the source uses. The merge script's alias map handles remapping. If you see a model that's clearly the same as one already in `data/models.csv` under a slightly different name, log it as a comment but still extract under the source's exact name.

### 4.3 One model_id per real model
If the same model appears under two names in the source data (e.g. `"GPT-4o"` and `"gpt-4o"`), pick the more canonical one and use it consistently. Don't create two model entries for the same model.

---

## 5. Benchmark ID collision handling

Before writing a new benchmark row, check if `benchmark_id` already exists in `data/benchmarks.csv`:

```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.lib import io
_, _, main_r = io.load_data()  # or load benchmarks specifically
```

If the ID already exists:
- **Do NOT add a new benchmarks.csv row** (skip it silently)
- **DO add result rows** pointing to the new `source_url` — these are valid because the result came from a different source

---

## 6. Papers With Code (PwC) — extraction guide

### 6.1 Data source

PwC's full evaluation table data is published daily as a public parquet dataset on HuggingFace (no auth required):

```
Dataset: pwc-archive/evaluation-tables
Files:
  data/train-00000-of-00004.parquet
  data/train-00001-of-00004.parquet
  data/train-00002-of-00004.parquet
  data/train-00003-of-00004.parquet

URL pattern:
  https://huggingface.co/datasets/pwc-archive/evaluation-tables/resolve/main/data/train-{NNNN}-of-00004.parquet
```

Download all 4 shards with `urllib.request` or `requests`. **Do not try to use the `paperswithcode.com` API** — the domain fully redirects to Hugging Face and no longer serves API responses.

```python
import urllib.request, pandas as pd
from pathlib import Path

SHARDS = [
    "https://huggingface.co/datasets/pwc-archive/evaluation-tables/resolve/main/data/train-00000-of-00004.parquet",
    "https://huggingface.co/datasets/pwc-archive/evaluation-tables/resolve/main/data/train-00001-of-00004.parquet",
    "https://huggingface.co/datasets/pwc-archive/evaluation-tables/resolve/main/data/train-00002-of-00004.parquet",
    "https://huggingface.co/datasets/pwc-archive/evaluation-tables/resolve/main/data/train-00003-of-00004.parquet",
]

dfs = []
for url in SHARDS:
    local = Path(f"/tmp/pwc_eval_{url.split('train-')[1][:2]}.parquet")
    if not local.exists():
        print(f"Downloading {url}...")
        with urllib.request.urlopen(url) as resp:
            local.write_bytes(resp.read())
    dfs.append(pd.read_parquet(local))

df = pd.concat(dfs, ignore_index=True)
print(f"Total rows: {len(df)}")
```

### 6.2 Parquet schema

Each row is one **task**. Nested inside are multiple datasets, each with its own leaderboard.

```
df columns:
  task          — task name (e.g. "Question Answering", "Code Generation")
  categories    — list of category strings
  description   — markdown description
  subtasks      — nested list of sub-tasks (same structure as top-level)
  synonyms      — alternate task names
  source_link   — usually NaN
  datasets      — array of dataset objects (the benchmarks + results)
```

Each element of `datasets` is a dict:
```python
{
  "dataset": "GSM8K",           # benchmark name
  "dataset_citations": [...],
  "dataset_links": [            # list of {title, url} — look for paperswithcode.com/sota/
    {"title": "Papers with Code Leaderboard URL",
     "url": "https://paperswithcode.com/sota/arithmetic-reasoning-on-gsm8k"}
  ],
  "description": "...",
  "sota": {
    "metrics": ["Accuracy"],    # list of metric names for this benchmark
    "rows": [                   # list of result rows
      {
        "model_name": "GPT-4",
        "paper_url": "https://arxiv.org/abs/2303.08774",
        "paper_title": "GPT-4 Technical Report",
        "paper_date": "2023-03-15",
        "metrics": {"Accuracy": "92.0"},
        "code_links": [...],
        "model_links": [...],
        "uses_additional_data": "False"
      },
      ...
    ]
  },
  "subdatasets": [...]
}
```

### 6.3 Flattening to result rows

```python
import numpy as np

def iter_results(df):
    """Yield (task_name, dataset_name, leaderboard_url, metric_name, model_name, score, paper_url, paper_date)."""
    for _, row in df.iterrows():
        task = row["task"]
        datasets = row["datasets"]
        if not isinstance(datasets, np.ndarray):
            continue
        for d in datasets:
            if not isinstance(d, dict):
                continue
            dataset_name = d.get("dataset", "")
            # Get the leaderboard URL
            links = d.get("dataset_links", [])
            if isinstance(links, np.ndarray):
                links = list(links)
            leaderboard_url = ""
            for lnk in links:
                if isinstance(lnk, dict) and "paperswithcode.com/sota/" in str(lnk.get("url", "")):
                    leaderboard_url = lnk["url"]
                    break
            # Get sota data
            sota = d.get("sota")
            if not isinstance(sota, dict):
                continue
            metrics = sota.get("metrics", [])
            if isinstance(metrics, np.ndarray):
                metrics = list(metrics)
            result_rows = sota.get("rows", [])
            if not isinstance(result_rows, np.ndarray):
                continue
            for r in result_rows:
                if not isinstance(r, dict):
                    continue
                model_name = clean_model_name(str(r.get("model_name", "") or ""))
                if not model_name:
                    continue
                paper_url = str(r.get("paper_url", "") or "")
                paper_date = str(r.get("paper_date", "") or "")
                row_metrics = r.get("metrics", {})
                if not isinstance(row_metrics, dict):
                    continue
                for metric_name, raw_score in row_metrics.items():
                    metric_name = metric_name.strip()
                    if not metric_name or raw_score is None:
                        continue
                    score = normalize_score(raw_score, metric_name)
                    if score is None:
                        continue
                    yield task, dataset_name, leaderboard_url, metric_name, model_name, score, paper_url, paper_date
```

### 6.4 Scope — which tasks to include

**Include**: tasks directly related to LLM/language model evaluation:
- question answering, reading comprehension, commonsense reasoning, mathematical reasoning, arithmetic reasoning, code generation, text summarization, machine translation, natural language inference, language modeling, dialogue, information extraction, text classification, named entity recognition, sentiment analysis, knowledge graph completion, fact verification, instruction following, safety/alignment, multimodal (vision-language)

**Exclude**: pure computer vision tasks (image classification, object detection, etc.) and audio tasks unless they directly test language model understanding.

Use a keyword filter on the `task` column:
```python
INCLUDE_KEYWORDS = {
    "question answering", "reading comprehension", "commonsense", "reasoning",
    "math", "arithmetic", "code", "language model", "summarization", "translation",
    "nlp", "natural language", "dialogue", "information extraction", "text classification",
    "sentiment", "named entity", "knowledge", "fact", "instruction", "safety",
    "alignment", "multimodal", "vision-language", "language understanding",
    "language generation", "inference"
}

EXCLUDE_KEYWORDS = {
    "image classification", "object detection", "image segmentation", "depth estimation",
    "image generation", "speech recognition", "audio", "video", "3d", "point cloud",
    "autonomous driving", "medical imaging", "drug discovery"
}

def is_relevant_task(task_name):
    t = task_name.lower()
    if any(kw in t for kw in EXCLUDE_KEYWORDS):
        return False
    return any(kw in t for kw in INCLUDE_KEYWORDS)
```

### 6.5 Benchmark ID generation

Generate a stable `benchmark_id` from the dataset name + task:
```python
import re

def make_benchmark_id(dataset_name, task_name):
    """Make a stable, unique benchmark_id from PwC dataset name."""
    s = dataset_name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)  # non-alphanumeric → underscore
    s = re.sub(r"_+", "_", s).strip("_")
    return f"pwc_{s}"   # prefix to avoid collisions with existing IDs
```

Before writing, check the existing `data/benchmarks.csv` for duplicates on `benchmark_id`. If there's a collision with an existing non-PwC entry, append `_pwc` to differentiate.

### 6.6 Staging row template

```python
def make_benchmark_row(dataset_name, task_name, leaderboard_url, paper_url=""):
    bid = make_benchmark_id(dataset_name, task_name)
    return {
        "benchmark_id":   bid,
        "benchmark_name": dataset_name,
        "description":    f"PwC benchmark: {dataset_name} (task: {task_name})",
        "source_url":     leaderboard_url or f"https://paperswithcode.com/task/{task_name.lower().replace(' ', '-')}",
        "source_name":    "Papers With Code",
        "source_type":    "leaderboard",
        "organization":   "Papers With Code",
        "category":       task_name,
        "paper_url":      paper_url,
        # Everything else: empty string ""
    }

def make_model_row(model_name, developer=""):
    return {
        "model_id":   model_name,
        "model_name": model_name,
        "developer":  developer,
        # Everything else: empty string ""
    }

def make_result_row(bid, benchmark_name, model_name, metric_name, score,
                    leaderboard_url, paper_url="", paper_date=""):
    return {
        "benchmark_id":       bid,
        "model_name":         model_name,
        "model_id":           model_name,
        "score":              str(score),
        "metric_name":        metric_name,
        "source_url":         leaderboard_url,
        "source_name":        "Papers With Code",
        "source_type":        "leaderboard",
        "source_organization": "Papers With Code",
        "benchmark_name":     benchmark_name,
        "notes":              f"Paper: {paper_url}" if paper_url else "",
        "date_recorded":      paper_date[:10] if paper_date else "",
        # Everything else: empty string ""
    }
```

### 6.7 Writing staging files

Always fill ALL 37/24/38 columns, using `""` for anything you don't have:

```python
import pandas as pd

# Load main files to check existing IDs
from scripts.lib import io as data_io
main_b, main_m, main_r = data_io.load_data()
existing_bench_ids = set(main_b["benchmark_id"])
existing_model_ids = set(main_m["model_id"])

BENCH_COLS  = list(main_b.columns)   # 37 columns
MODEL_COLS  = list(main_m.columns)   # 24 columns
RESULT_COLS = list(main_r.columns)   # 38 columns

def write_staging(benchmarks, models, results, source_prefix):
    def to_df(rows, cols):
        df = pd.DataFrame(rows)
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        return df[cols].fillna("")

    b_df = to_df(benchmarks, BENCH_COLS)
    m_df = to_df(models,     MODEL_COLS)
    r_df = to_df(results,    RESULT_COLS)

    # Dedup results on identity key
    id_key = ["benchmark_id", "model_name", "metric_name", "source_url"]
    r_df = r_df.drop_duplicates(subset=id_key)

    b_df.to_csv(f"data/staging_{source_prefix}_benchmarks.csv", index=False, lineterminator="\r\n")
    m_df.to_csv(f"data/staging_{source_prefix}_models.csv",     index=False, lineterminator="\r\n")
    r_df.to_csv(f"data/staging_{source_prefix}_results.csv",    index=False, lineterminator="\r\n")
    print(f"Wrote: {len(b_df)} benchmarks, {len(m_df)} models, {len(r_df)} results")
```

---

## 7. Kaggle AI Benchmarks — extraction guide

### 7.1 Data access

The Kaggle AI Benchmarks platform (`https://www.kaggle.com/benchmarks`) uses an
internal gRPC-gateway API. **No account or API key is required** — all three steps
below work with an anonymous session cookie.

**Critical implementation note:** The leaderboard endpoint requires a *benchmark
version ID* (from `GetBenchmark`), NOT the task version ID from `ListBenchmarks`.
Using the wrong ID returns HTTP 500 or 403. The correct 3-step flow is:
1. `ListBenchmarks` → get `benchmarkId`
2. `GetBenchmark` → get `version.id` (the benchmark version ID)
3. `GetBenchmarkLeaderboard` → use that `version.id` in `versionIdSelector`

#### Step 1 — Get session tokens (anonymous, no account needed)

```python
import urllib.request, json, http.cookiejar

cj = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

# Load the page to get session + XSRF cookies
opener.open(urllib.request.Request(
    "https://www.kaggle.com/benchmarks",
    headers={"User-Agent": "Mozilla/5.0"}
), timeout=15).read()

xsrf = next((c.value for c in cj if c.name == "XSRF-TOKEN"), "")

def kaggle_post(endpoint, payload, opener=opener, xsrf=xsrf):
    req = urllib.request.Request(
        f"https://www.kaggle.com/api/i/{endpoint}",
        data=json.dumps(payload).encode(),
        headers={
            "User-Agent":    "Mozilla/5.0",
            "Accept":        "application/json",
            "Content-Type":  "application/json",
            "X-Xsrf-Token":  xsrf,
        },
        method="POST"
    )
    with opener.open(req, timeout=30) as resp:
        return json.loads(resp.read())
```

#### Step 2 — List all benchmarks

```python
all_benchmarks = []
token = None
while True:
    payload = {"pageSize": 100}
    if token:
        payload["pageToken"] = token
    data = kaggle_post("benchmarks.BenchmarkService/ListBenchmarks", payload)
    all_benchmarks.extend(data.get("benchmarks", []))
    token = data.get("nextPageToken")
    if not token:
        break
    import time; time.sleep(0.5)  # be polite

print(f"Total benchmarks: {len(all_benchmarks)}")  # expect ~1000+
```

Each benchmark object has:
```python
{
  "id": 4974,
  "name": "ParseBench",
  "slug": "parsebench",
  "type": "PERSONAL",    # or "INDIVIDUAL", "GAME"
  "published": True,
  "viewCount": 7488,
  "voteCount": 34,
  "ownerUser": {"userName": "llamaindex-org", "displayName": "LlamaIndex"},
  "media": [
    {"type": "PAPER",   "url": "https://arxiv.org/abs/2604.08538"},
    {"type": "WEBSITE", "url": "https://www.parsebench.ai/"},
    {"type": "KAGGLE_DATASET", "url": "https://www.kaggle.com/datasets/..."},
    {"type": "KAGGLE_NOTEBOOK", "url": "https://www.kaggle.com/code/..."}
  ],
  "task": {
    "id": 19043,
    "version": {
      "id": 30741,
      "aggregationType": "NOOP",   # or "PERCENTAGE_PASSED", "MEAN", etc.
      "displayType": "PERCENTAGES",
      "sortOrder": "DESCENDING",
    }
  },
  "categories": {"type": "TAG_TYPE_BENCHMARK"},
}
```

#### Step 3 — Filter for LLM-relevant research benchmarks

The Kaggle benchmark list contains ~1000 benchmarks. Many are hobby/personal tasks, games, or domain-specific benchmarks. Filter to LLM-relevant ones:

```python
LLM_KEYWORDS = {
    "llm", "language model", "reasoning", "coding", "math", "nlp",
    "question answering", "benchmark", "evaluation", "eval", "knowledge",
    "instruction", "alignment", "safety", "gsm", "mmlu", "humaneval",
    "commonsense", "text", "chat", "agent", "retrieval", "rag"
}

def is_llm_relevant(b):
    name = b.get("name", "").lower()
    desc = str(b.get("description", "")).lower()
    return any(kw in name or kw in desc for kw in LLM_KEYWORDS)

relevant = [b for b in all_benchmarks if is_llm_relevant(b)]
print(f"LLM-relevant: {len(relevant)}")
```

#### Step 4 — Get the benchmark version ID (CRITICAL — different from task version ID)

The `GetBenchmarkLeaderboard` endpoint requires a *benchmark version ID* that comes from
`GetBenchmark`. This is **NOT** the `task.version.id` you see in `ListBenchmarks`. They are
different integers. Using the task version ID returns HTTP 500 or 403.

**No API key or account is required for any of these calls.**

```python
def get_benchmark_version_id(bench_id):
    """Get the benchmark version ID needed for GetBenchmarkLeaderboard.
    bench_id: the integer id from ListBenchmarks (b["id"])
    """
    bm = kaggle_post("benchmarks.BenchmarkService/GetBenchmark", {
        "benchmarkIdentifier": {"id": bench_id},
        "versionIdentifier": {
            "publishedLatestSelector": {
                "parentBenchmarkIdentifier": {"id": bench_id}
            }
        }
    })
    return bm["version"]["id"]  # e.g. 5514 for GSM8K (NOT the task version 32363)
```

#### Step 5 — Get leaderboard (fully public, no auth required)

```python
def get_leaderboard(benchmark_version_id, page_size=100):
    """Returns leaderboard rows. Completely public — no API key needed."""
    lb = kaggle_post("benchmarks.BenchmarkService/GetBenchmarkLeaderboard", {
        "versionIdentifier": {"versionIdSelector": {"id": benchmark_version_id}},
        "pageSize": page_size
    })
    return lb.get("rows", [])
```

Each row looks like:
```python
{
  "modelVersion": {
    "name": "Claude Fable 5",
    "organization": {"name": "Anthropic"}
  },
  "aggregateScore": "0.9234",    # string float, or None for NOOP-type benchmarks
  "results": [                    # per-child-task scores (if benchmark has subtasks)
    {
      "childTaskVersion": {"name": "Standard"},
      "numericResult": {"value": "0.9234"}
    }
  ],
  "rank": 1
}
```

Use `row["modelVersion"]["name"]` as both `model_name` and `model_id`.
Use `row["modelVersion"]["organization"]["name"]` as `developer` (when present).

#### Step 5b — Score extraction

```python
def extract_score(row, agg_type, num_child_tasks):
    """
    Extract and normalize score from a leaderboard row.
    Returns (score_float_or_None, metric_name_str).

    agg_type: from task.version.aggregationType in ListBenchmarks
              ("PERCENTAGE_PASSED", "MEAN", "NOOP", etc.)
    num_child_tasks: number of child tasks this benchmark has (default 1 if unknown)
    """
    agg = row.get("aggregateScore")
    if agg_type != "NOOP" and agg is not None:
        val = float(agg)
        if 0.0 <= val <= 1.0:
            return round(val * 100, 2), "score"       # fraction → percentage
        elif 1.0 < val <= 100.0 and num_child_tasks <= 1:
            return round(val, 2), "score"             # already on 0-100 scale
        else:
            return None, None   # raw problem count across multiple child tasks — skip
    # NOOP type or missing aggregateScore: fall back to per-result scores
    for result in row.get("results", []):
        v = (result.get("numericResult") or {}).get("value")
        if v is not None:
            val = float(v)
            if 0.0 <= val <= 1.0:
                return round(val * 100, 2), "score"
            return round(val, 2), "score"
    return None, None
```

**Complete per-benchmark loop:**

```python
seen_models  = set()
models_rows  = []
bench_rows   = []
results_rows = []

for b in relevant:
    bench_id   = b["id"]
    slug       = b["slug"]
    owner      = b["ownerUser"]["userName"]
    source_url = f"https://www.kaggle.com/benchmarks/{owner}/{slug}"
    agg_type   = b.get("task", {}).get("version", {}).get("aggregationType", "")

    try:
        bv_id = get_benchmark_version_id(bench_id)
    except Exception as e:
        print(f"  SKIP {slug}: GetBenchmark failed: {e}")
        continue

    try:
        rows = get_leaderboard(bv_id)
    except Exception as e:
        print(f"  SKIP {slug}: GetBenchmarkLeaderboard failed: {e}")
        continue

    if not rows:
        print(f"  SKIP {slug}: empty leaderboard")
        continue

    bench_rows.append(make_kaggle_benchmark_row(b))

    for row in rows:
        model_name = (row.get("modelVersion") or {}).get("name", "")
        if not model_name:
            continue
        model_name = clean_model_name(model_name)
        developer  = ((row.get("modelVersion") or {}).get("organization") or {}).get("name", "")

        score, metric = extract_score(row, agg_type, len(rows))
        if score is None:
            continue

        if model_name not in seen_models:
            seen_models.add(model_name)
            models_rows.append(make_model_row(model_name, developer))

        results_rows.append(make_kaggle_result_row(
            make_kaggle_benchmark_id(slug),
            b["name"], source_url,
            model_name, model_name, score, metric
        ))

    import time; time.sleep(0.3)  # be polite
```

#### Step 6 — Benchmark page URL

The benchmark page is at: `https://www.kaggle.com/benchmarks/{ownerUser.userName}/{slug}`
Use this as the `source_url` in your staging results.

#### Step 7 — Benchmark ID generation

```python
def make_kaggle_benchmark_id(slug):
    s = slug.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return f"kaggle_{s}"
```

#### Step 8 — Staging row templates (Kaggle)

```python
def make_kaggle_benchmark_row(b):
    slug = b["slug"]
    owner = b["ownerUser"]["userName"]
    media = {m["type"]: m["url"] for m in b.get("media", [])}
    return {
        "benchmark_id":   make_kaggle_benchmark_id(slug),
        "benchmark_name": b["name"],
        "description":    b.get("description", ""),
        "source_url":     f"https://www.kaggle.com/benchmarks/{owner}/{slug}",
        "source_name":    "Kaggle AI Benchmarks",
        "source_type":    "leaderboard",
        "organization":   b["ownerUser"].get("displayName", owner),
        "paper_url":      media.get("PAPER", ""),
        "github_url":     media.get("GITHUB", ""),
        "hf_url":         media.get("HUGGINGFACE", ""),
    }

def make_kaggle_result_row(benchmark_id, benchmark_name, source_url,
                            model_name, model_id, score, metric_name, date=""):
    return {
        "benchmark_id":        benchmark_id,
        "model_name":          model_name,
        "model_id":            model_id,
        "score":               str(score),
        "metric_name":         metric_name,
        "source_url":          source_url,
        "source_name":         "Kaggle AI Benchmarks",
        "source_type":         "leaderboard",
        "source_organization": "Kaggle",
        "benchmark_name":      benchmark_name,
        "date_recorded":       date[:10] if date else "",
    }
```

---

## 8. Validation checklist — run before declaring staging complete

After building all three staging files, run these checks. Fix everything before finishing.

```python
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.lib import io as data_io

def validate_staging(source_prefix):
    b = pd.read_csv(f"data/staging_{source_prefix}_benchmarks.csv", dtype=str, keep_default_na=False)
    m = pd.read_csv(f"data/staging_{source_prefix}_models.csv",     dtype=str, keep_default_na=False)
    r = pd.read_csv(f"data/staging_{source_prefix}_results.csv",    dtype=str, keep_default_na=False)
    main_b, main_m, _ = data_io.load_data()

    print(f"=== {source_prefix} staging validation ===")
    print(f"  benchmarks: {len(b)}, models: {len(m)}, results: {len(r)}")

    # 1. No empty scores
    empty_scores = r[r["score"].str.strip() == ""]
    if len(empty_scores):
        print(f"  ERROR: {len(empty_scores)} results with empty score")

    # 2. Scores must be numeric and in a sane range
    def try_float(x):
        try: return float(x)
        except: return None
    numeric_scores = r["score"].apply(try_float)
    bad_numeric = r[numeric_scores.isna()]
    if len(bad_numeric):
        print(f"  ERROR: {len(bad_numeric)} results with non-numeric score")
    out_of_range = r[numeric_scores.apply(lambda x: x is not None and (x < 0 or x > 100))]
    if len(out_of_range):
        # This is a WARNING not an error — some metrics are absolute (BPB, Perplexity, Elo)
        print(f"  WARN: {len(out_of_range)} results with score outside [0,100] — verify these are absolute metrics")

    # 3. FK: results.model_name in staging_models or main_models
    all_model_ids = set(m["model_id"]) | set(main_m["model_id"])
    bad_model = r[~r["model_name"].isin(all_model_ids)]
    if len(bad_model):
        print(f"  ERROR: {len(bad_model)} results reference unknown model_name:")
        for mn in sorted(bad_model["model_name"].unique())[:10]:
            print(f"    {mn!r}")

    # 4. FK: results.benchmark_id in staging_benchmarks or main_benchmarks
    all_bench_ids = set(b["benchmark_id"]) | set(main_b["benchmark_id"])
    bad_bench = r[~r["benchmark_id"].isin(all_bench_ids)]
    if len(bad_bench):
        print(f"  ERROR: {len(bad_bench)} results reference unknown benchmark_id:")
        for bid in sorted(bad_bench["benchmark_id"].unique())[:10]:
            print(f"    {bid!r}")

    # 5. No duplicate results
    id_key = ["benchmark_id", "model_name", "metric_name", "source_url"]
    dupes = r[r.duplicated(subset=id_key, keep=False)]
    if len(dupes):
        print(f"  ERROR: {len(dupes)} duplicate result rows on identity key")

    # 6. Schema completeness
    from scripts.lib import io as data_io
    main_b_cols  = list(pd.read_csv("data/benchmarks.csv", nrows=0, dtype=str).columns)
    main_m_cols  = list(pd.read_csv("data/models.csv",     nrows=0, dtype=str).columns)
    main_r_cols  = list(pd.read_csv("data/results.csv",    nrows=0, dtype=str).columns)
    for staging_df, main_cols, label in [(b, main_b_cols, "benchmarks"), (m, main_m_cols, "models"), (r, main_r_cols, "results")]:
        missing = set(main_cols) - set(staging_df.columns)
        extra   = set(staging_df.columns) - set(main_cols)
        if missing:
            print(f"  ERROR: staging_{label} missing columns: {missing}")
        if extra:
            print(f"  WARN:  staging_{label} has extra columns: {extra}")

    print("  Validation complete.")
```

---

## 9. What the merge script must do

After staging is validated, write `scripts/merge_{source}_staging.py` (same pattern as `scripts/merge_helm_staging.py`):

1. **Alias map**: identify staging model names that are the same model as an existing `models.csv` entry but with a different naming convention. Add them to a `MODEL_ALIAS` dict and verify all targets exist in main models.
2. **Plan benchmarks**: skip IDs already in main, add new ones.
3. **Plan models**: apply alias map, skip IDs already in main or aliased, add new ones.
4. **Plan results**: dedup against existing 4-tuple key, FK check on new rows.
5. **Print full plan** before writing anything.
6. **Dry run by default** — require `--write` flag to actually modify files.
7. **Abort on FK violations** even with `--write`.

Reference: the HELM/PwC/Kaggle merge scripts were one-off pipelines that have
been applied and removed from the tree now that those sources are merged. For a
complete, working example to follow, see `scripts/merge_helm_staging.py` in git
history (commit `086363c`, or the PwC/Kaggle merge in `3de712a`).

---

## 10. Post-merge steps (user runs these, not the extraction agent)

```bash
python3 scripts/merge_{source}_staging.py           # dry run first
python3 scripts/merge_{source}_staging.py --write   # apply
python3 scripts/verify_data.py                      # must show 0 FK violations
python3 scripts/manage_data.py recompute-stats --write  # update model aggregate stats
git add data/benchmarks.csv data/models.csv data/results.csv \
        scripts/merge_{source}_staging.py && \
git commit -m "Merge {source} data: +N benchmarks, +M models, +K results"
```
