#!/usr/bin/env python3
"""Clean and collapse benchmark results into standard and aggressive combinations."""
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "results.csv"
OUT_DIR = ROOT / "data" / "combinations"

# normalize tokens
STRIP_TOKENS = {
    "instruct", "inst", "instruction", "chat", "chatbot", "base", "base-model",
    "ft", "sft", "dpo", "rlhf", "ppo", "trained", "tuned", "finetuned",
    "eval", "evals", "orca", "openorca", "neural", "aligned", "deduped", "prompt", "fc", "cot"
}
LANG_REGION_TOKENS = {"en","ch","zh","korean","kr","ja","ar","es","de","fr","ru","multi","multilingual","turbo","davinci","curie","babbage","ada","dbrx","gpt4all","ggml","gguf"}
MONTH_TOKENS = {
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    "january", "february", "march", "april", "june", "july", "august", "september", "october", "november", "december"
}
ALL_STRIP = STRIP_TOKENS | LANG_REGION_TOKENS | MONTH_TOKENS
VARIANT_TOKENS = {"opus", "sonnet", "haiku", "mythos", "fable", "pro", "mini", "flash", "maverick", "scout", "vl"}


def strip_dates_and_metadata(s):
    # Match trailing junk starting with hyphens and ampersand (e.g. ---&---)
    s = re.split(r'-+&', s)[0].strip('-_. ')
    # Match YYYY-MM-DD, YYYY_MM_DD, YYYY/MM/DD
    s = re.sub(r'\b\d{4}[-_/]\d{2}[-_/]\d{2}\b', '', s)
    # Match YYYYMMDD (like 20240620)
    s = re.sub(r'\b20\d{6}\b', '', s)
    # Match MM/DD (e.g. 05/13) or MM-DD or YY/MM
    s = re.sub(r'\b\d{2}[-_/]\d{2}\b', '', s)
    # Match any 4-digit numbers: release years/cutoffs (like 2024, 2025, 1106, 0613, 2507) or context lengths (like 2048)
    s = re.sub(r'\b\d{4}\b', '', s)
    return s


def clean_parentheses(s):
    def repl(match):
        content = match.group(1)
        size_match = re.search(r'\b\d+(?:\.\d+)?[mbMB]\b', content)
        if size_match:
            return size_match.group(0)
        return ""
    return re.sub(r'\(([^)]*)\)', repl, s)


def normalize_standard(row):
    fam = row.get('model_family')
    size = row.get('model_size')
    mid = row.get('model_id')
    m = str(mid).lower().strip() if pd.notna(mid) else ""

    # Strip organization prefix first (everything before the first slash)
    m_clean = m.split('/', 1)[-1] if '/' in m else m
    
    # Clean parentheses
    m_clean = clean_parentheses(m_clean)
    
    # Strip dates and metadata
    m_clean = strip_dates_and_metadata(m_clean)
    
    # Normalize version hyphens (e.g. 3-5 to 3.5, 3-1 to 3.1)
    m_clean = re.sub(r'\b(\d+)[-_](\d+)\b', r'\1.\2', m_clean)

    if pd.notna(fam) and str(fam).strip() != "":
        fam_clean = re.sub(r"[^A-Za-z0-9\s.\-]", "", str(fam)).strip()
        # Skip if family is numeric-only (likely data error)
        if re.match(r"^\d+(?:\.\d+)?$", fam_clean):
            fam_clean = None
            
        if fam_clean:
            fam_base = re.sub(r"\s+", "-", fam_clean).lower()
            # Extract versions in family name
            fam_versions = set(re.findall(r"(\d+(?:\.\d+)?)", fam_base))
            lower_fam_base = re.split(r"[-\s]", fam_clean.lower())[0]
            
            # Find family base in model name to extract suffix
            idx = m_clean.find(lower_fam_base)
            if idx != -1:
                suffix_source = m_clean[idx:]
                suffix = suffix_source[len(lower_fam_base):].strip("-_. ")
            else:
                suffix = m_clean

            # Tokenize suffix using non-alphanumeric chars as delimiters (preserve dot for decimal versions/sizes)
            tokens = re.split(r"[^A-Za-z0-9.]+", suffix)
            tokens = [t.strip('. ') for t in tokens if t.strip()]
            
            preserved = []
            size_part = None
            seen_versions = set()
            
            for tok in tokens:
                if tok.lower() in ALL_STRIP:
                    continue
                if tok.lower() in VARIANT_TOKENS:
                    preserved.append(tok.lower())
                    continue
                    
                # Check if it is a size part (e.g. 70b, 1.5b, 8B)
                is_size = False
                size_val = None
                
                num_b_match = re.match(r"^(\d+(?:\.\d+)?)[bB]$", tok)
                if num_b_match:
                    is_size = True
                    size_val = num_b_match.group(1)
                else:
                    num_match = re.match(r"^(\d+(?:\.\d+)?)$", tok)
                    if num_match:
                        val_str = num_match.group(1)
                        val_float = float(val_str)
                        # Check if it matches canonical size metadata
                        canonical_size_matches = False
                        if pd.notna(size) and str(size).strip() != "":
                            try:
                                if abs(float(str(size).strip()) - val_float) < 0.01:
                                    canonical_size_matches = True
                            except ValueError:
                                pass
                        
                        # Check if it is a common size
                        is_common_size = False
                        try:
                            val_int = int(val_float)
                            if val_float == val_int:
                                COMMON_SIZES = {7, 8, 9, 11, 12, 13, 14, 22, 26, 27, 30, 32, 33, 34, 70, 72, 80, 90, 235, 405}
                                if val_int in COMMON_SIZES:
                                    is_claude_or_gpt = any(x in fam_base for x in ('claude', 'gpt')) if fam_base else False
                                    if not (is_claude_or_gpt and val_int in (3, 4, 20, 21)):
                                        is_common_size = True
                        except ValueError:
                            pass
                            
                        if canonical_size_matches or is_common_size:
                            is_size = True
                            size_val = val_str
                            
                if is_size:
                    formatted_size = str(int(float(size_val))) if float(size_val).is_integer() else str(float(size_val))
                    size_part = f"{formatted_size}B"
                    continue
                    
                # Check version number
                if re.match(r"^\d+(\.\d+)?$", tok):
                    # check compatibility with fam_versions
                    merged = False
                    for fv in list(fam_versions):
                        # if either starts with/extends the other
                        if tok.startswith(fv) or fv.startswith(tok):
                            longer_v = tok if len(tok) > len(fv) else fv
                            fam_base = fam_base.replace(fv, longer_v)
                            fam_versions.remove(fv)
                            fam_versions.add(longer_v)
                            merged = True
                            break
                    if not merged and tok not in seen_versions:
                        preserved.append(tok)
                        seen_versions.add(tok)
                    continue
                    
                if len(tok) > 2 and tok.lower() not in ALL_STRIP:
                    preserved.append(tok.lower())
                    
            if pd.notna(size) and str(size).strip() != "" and not size_part:
                if re.match(r"^\d+$", str(size).strip()):
                    size_part = f"{int(float(size))}B"
                    
            parts = [fam_base] + preserved + ([size_part] if size_part else [])
            return '-'.join(parts)

    if pd.isna(mid) or str(mid).strip() == "":
        return 'unknown'

    # Fallback for no family: tokenize m_clean directly
    tokens = re.split(r"[^A-Za-z0-9.]+", m_clean)
    tokens = [t.strip('. ') for t in tokens if t.strip()]
    preserved = []
    size_part = None
    seen_versions = set()
    
    for tok in tokens:
        if tok.lower() in ALL_STRIP:
            continue
        if tok.lower() in VARIANT_TOKENS:
            preserved.append(tok.lower())
            continue
            
        # Check if size part
        is_size = False
        size_val = None
        num_b_match = re.match(r"^(\d+(?:\.\d+)?)[bB]$", tok)
        if num_b_match:
            is_size = True
            size_val = num_b_match.group(1)
        else:
            num_match = re.match(r"^(\d+(?:\.\d+)?)$", tok)
            if num_match:
                val_str = num_match.group(1)
                val_float = float(val_str)
                # Check canonical size
                canonical_size_matches = False
                if pd.notna(size) and str(size).strip() != "":
                    try:
                        if abs(float(str(size).strip()) - val_float) < 0.01:
                            canonical_size_matches = True
                    except ValueError:
                        pass
                
                # Check common sizes
                is_common_size = False
                try:
                    val_int = int(val_float)
                    if val_float == val_int:
                        COMMON_SIZES = {7, 8, 9, 11, 12, 13, 14, 22, 26, 27, 30, 32, 33, 34, 70, 72, 80, 90, 235, 405}
                        if val_int in COMMON_SIZES:
                            is_common_size = True
                except ValueError:
                    pass
                    
                if canonical_size_matches or is_common_size:
                    is_size = True
                    size_val = val_str
                    
        if is_size:
            formatted_size = str(int(float(size_val))) if float(size_val).is_integer() else str(float(size_val))
            size_part = f"{formatted_size}B"
            continue
            
        # Check version number
        if re.match(r"^\d+(\.\d+)?$", tok):
            if tok not in seen_versions:
                preserved.append(tok)
                seen_versions.add(tok)
            continue
            
        if len(tok) > 2 and tok.lower() not in ALL_STRIP:
            preserved.append(tok.lower())
            
    if pd.notna(size) and str(size).strip() != "" and not size_part:
        if re.match(r"^\d+$", str(size).strip()):
            size_part = f"{int(float(size))}B"
            
    parts = preserved + ([size_part] if size_part else [])
    return '-'.join(parts) if parts else str(mid)


def normalize_aggressive(row):
    mid = row.get('model_id')
    fam = row.get('model_family')
    
    # 1. Prefer model_family if it is a valid family name
    if pd.notna(fam) and str(fam).strip() != "":
        fam_str = str(fam).strip().lower()
        # Skip if purely numeric
        if not re.match(r"^\d+(?:\.\d+)?[bB]?$", fam_str):
            # Take first token split by common separators
            tokens = re.split(r"[\s\-_/]", fam_str)
            for token in tokens:
                token = re.sub(r"[^a-z0-9]", "", token)
                if token and re.search(r"[a-z]", token):
                    return token
                    
    # 2. Fallback: parse model_id
    if pd.notna(mid) and str(mid).strip() != "":
        mid_str = str(mid).strip().lower()
        # Strip organization prefix first
        m_clean = mid_str.split('/', 1)[-1] if '/' in mid_str else mid_str
        # Clean parentheses
        m_clean = clean_parentheses(m_clean)
        # Also remove dates and cutoff strings
        m_clean = strip_dates_and_metadata(m_clean)
        # Take first alphabetic token from the cleaned model ID
        tokens = re.split(r"[\-_/\s]", m_clean)
        for token in tokens:
            token = re.sub(r"[^a-z0-9]", "", token)
            if token and re.search(r"[a-z]", token):
                return token
                
    return 'unknown'


def first_nonnull(series):
    for val in series:
        if pd.notna(val) and str(val).strip() != "" and str(val).strip().lower() != "nan":
            return str(val).strip()
    return None


def main():
    if not DATA_PATH.exists():
        print(f"Error: input file {DATA_PATH} not found.")
        return

    print(f"Loading raw results from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, dtype=str)

    # Coerce score to numeric
    df['score_numeric'] = pd.to_numeric(df.get('score'), errors='coerce')
    if df['score_numeric'].isna().all() and 'metric_score' in df.columns:
        df['score_numeric'] = pd.to_numeric(df.get('metric_score'), errors='coerce')

    print(f"Total raw evaluations: {len(df)}")

    # 1. First, canonicalize metadata (model_family, model_size) per model_id
    print("Canonicalizing model metadata...")
    model_meta = df.groupby('model_id').agg({
        'model_family': first_nonnull,
        'model_size': first_nonnull
    }).reset_index()

    # 2. Average out multiple results of the same (model_id, benchmark_id)
    print("Averaging multiple sources/setups for the same model and benchmark...")
    df_avg = df.groupby(['model_id', 'benchmark_id'])['score_numeric'].mean().reset_index()
    print(f"Unique (model_id, benchmark_id) evaluations: {len(df_avg)}")

    # Merge averaged scores back with canonical model metadata
    df_clean = df_avg.merge(model_meta, on='model_id', how='left')

    # Prepare combinations directory
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Baseline: no collapse statistics calculation
    baseline_pivot = df_avg.pivot_table(
        index='model_id',
        columns='benchmark_id',
        values='score_numeric',
        aggfunc='mean'
    ).reset_index()

    def get_stats(pivot, name):
        models = pivot.shape[0]
        benches = len(pivot.columns) - 1
        total = models * benches
        val_cols = [c for c in pivot.columns if c not in ('collapse_key', 'model_id')]
        non_null = int(pivot[val_cols].notna().sum().sum())
        density = 100 * non_null / total if total > 0 else 0.0
        sparsity = 100 * (1 - non_null / total) if total > 0 else 100.0
        return {
            'strategy': name,
            'models': models,
            'benchmarks': benches,
            'matrix_size': f"{models}x{benches}",
            'total_cells': total,
            'non_null': non_null,
            'density_pct': f"{density:.4f}",
            'sparsity_pct': f"{sparsity:.4f}"
        }

    summary_rows = [get_stats(baseline_pivot, 'original_no_collapse')]

    # Strategies to run
    strategies = {
        "all_standard": normalize_standard,
        "all_aggressive": normalize_aggressive
    }

    for name, norm_func in strategies.items():
        print(f"\nProcessing strategy: {name}...")
        df_strat = df_clean.copy()
        df_strat['collapse_key'] = df_strat.apply(norm_func, axis=1)

        # Save mapping of model_id -> collapse_key
        strategy_dir = OUT_DIR / name
        strategy_dir.mkdir(parents=True, exist_ok=True)
        
        mapping_file = strategy_dir / "collapse_mapping.csv"
        mapping_df = df_strat[['model_id', 'collapse_key']].drop_duplicates().sort_values(['collapse_key', 'model_id'])
        mapping_df.to_csv(mapping_file, index=False)
        print(f"  Saved mapping to: {mapping_file}")

        # Average duplicate evaluations for the same (collapse_key, benchmark_id)
        df_collapsed = df_strat.groupby(['collapse_key', 'benchmark_id'])['score_numeric'].mean().reset_index()

        # Pivot to model x benchmark table (rows = collapse_key, cols = benchmark_id)
        pivot_df = df_collapsed.pivot_table(
            index='collapse_key',
            columns='benchmark_id',
            values='score_numeric',
            aggfunc='mean'
        ).reset_index()

        # Write output
        out_file = strategy_dir / "model_benchmark_table.csv"
        pivot_df.to_csv(out_file, index=False)
        print(f"  Wrote results to: {out_file}")

        # Collect stats
        summary_rows.append(get_stats(pivot_df, "aggressive_collapse_all" if name == "all_aggressive" else "standard_collapse_all"))

    # Print final summary comparison table
    print("\n" + "="*40 + "\nCOLLAPSE STRATEGY COMPARISON SUMMARY\n" + "="*40)
    summary_df = pd.DataFrame(summary_rows)
    cols = ['strategy', 'models', 'benchmarks', 'matrix_size', 'total_cells', 'non_null', 'density_pct', 'sparsity_pct']
    summary_df = summary_df[cols]
    print(summary_df.to_string(index=False))
    print("="*40)


if __name__ == "__main__":
    main()
