import pandas as pd
import re
import os

def main():
    data_dir = "/Users/haznitrama/Desktop/llm-benchmarks/data"
    notes_dir = "/Users/haznitrama/Desktop/llm-benchmarks/notes"
    
    # Load CSVs
    try:
        benchmarks = pd.read_csv(os.path.join(data_dir, "benchmarks.csv"))
        models = pd.read_csv(os.path.join(data_dir, "models.csv"))
        results = pd.read_csv(os.path.join(data_dir, "results.csv"))
    except Exception as e:
        print(f"Error loading CSVs: {e}")
        return

    # Parse pending_benchmarks.md
    pending_file = os.path.join(notes_dir, "pending_benchmarks.md")
    pending_names = []
    if os.path.exists(pending_file):
        with open(pending_file, "r") as f:
            for line in f:
                match = re.match(r'^\*\*(.*?)\*\*', line)
                if match:
                    # Clean up the name a bit (e.g., removing parentheticals if included inside)
                    name = match.group(1).split('(')[0].strip()
                    pending_names.append(name)
    
    print(f"--- Verification Report ---")
    print(f"Total benchmarks expected from pending file: {len(pending_names)}\n")

    # 1. Completeness Check
    # Lowercase for easier matching
    existing_b_names = benchmarks['benchmark_name'].fillna('').str.lower().tolist()
    existing_b_ids = benchmarks['benchmark_id'].fillna('').str.lower().tolist()
    
    missing = []
    found_count = 0
    for expected in pending_names:
        exp_lower = expected.lower()
        # Check if expected is in name or id
        if any(exp_lower in name for name in existing_b_names) or any(exp_lower in bid for bid in existing_b_ids):
            found_count += 1
        else:
            missing.append(expected)
            
    print(f"1. Completeness: Found ~{found_count}/{len(pending_names)} benchmarks.")
    if missing:
        print(f"   Potential Missing (or named differently):")
        for m in missing[:10]:
            print(f"   - {m}")
        if len(missing) > 10:
            print(f"   ... and {len(missing) - 10} more.")
    print("")

    # 2. Consistency Check (Foreign Keys)
    invalid_benchmarks = set(results['benchmark_id']) - set(benchmarks['benchmark_id'])
    invalid_models = set(results['model_name']) - set(models['model_id'])
    
    print(f"2. Consistency:")
    print(f"   - Results with missing benchmark_id: {len(invalid_benchmarks)} distinct IDs")
    if invalid_benchmarks: print(f"     {list(invalid_benchmarks)[:5]}")
    print(f"   - Results with missing model_name (in models.csv): {len(invalid_models)} distinct models")
    if invalid_models: print(f"     {list(invalid_models)[:5]}")
    print("")

    # 3. Exhaustion Check
    print(f"3. Exhaustion (Results per benchmark):")
    # Get recently added benchmarks (rough heuristic: look at the bottom of the dataframe, or check ones with low counts)
    result_counts = results['benchmark_id'].value_csv() if hasattr(results['benchmark_id'], 'value_csv') else results['benchmark_id'].value_counts()
    
    low_count_benchmarks = result_counts[result_counts < 5]
    print(f"   Benchmarks with < 5 result rows (Potential incomplete extraction):")
    for b_id, count in low_count_benchmarks.items():
        print(f"   - {b_id}: {count} rows")
        
    if low_count_benchmarks.empty:
        print("   All benchmarks have >= 5 result rows. Good sign for exhaustion.")

if __name__ == "__main__":
    main()
