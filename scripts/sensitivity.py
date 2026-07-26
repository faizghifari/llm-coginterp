#!/usr/bin/env python3
"""LOCO delta omega_h vs benchmark frequency: pooled correlation table.

For each (method, dataset) group in the loco table, pool delta_omega_h
across runs (elementwise mean, NaN-safe), then compute the Pearson
correlation between each benchmark's non-missing frequency in the original
sparse/unimputed dataset and its pooled delta_omega_h. Prints one row per
(method, dataset) group.
"""

import argparse
import csv
import json
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
RESULTS_ROOT = REPO_ROOT / "results"


def parse_dataset(dataset: str) -> tuple[str, str]:
    """'C_all_standard' -> ('C', 'all_standard'), 'raw_all_aggressive' -> ('raw', 'all_aggressive')."""
    idx = dataset.index("_")
    return dataset[:idx], dataset[idx + 1 :]


def sparse_csv_path(data_root: Path, dz: str, st: str) -> Path:
    sub = "combinations" if dz == "raw" else f"combinations_{dz}"
    return data_root / sub / st / "model_benchmark_table.csv"


def bench_frequencies(csv_path: Path, min_obs: int = 2) -> tuple[list[str], list[float]]:
    """Read sparse CSV, drop collapse_key, filter like prep_matrix (min obs + non-zero
    variance), return (names, non_missing_proportion) for kept columns."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    bench_cols = [c for c in fieldnames if c != "collapse_key"]
    if not rows:
        return [], []

    n = len(rows)
    kept_names = []
    kept_freqs = []

    for col in bench_cols:
        values = []
        for row in rows:
            v = row.get(col, "").strip()
            if v == "" or v.upper() == "NA":
                continue
            try:
                values.append(float(v))
            except ValueError:
                pass
        n_obs = len(values)
        if n_obs < min_obs:
            continue
        if n_obs > 1:
            mean = sum(values) / n_obs
            var = sum((x - mean) ** 2 for x in values) / (n_obs - 1)
        else:
            var = 0.0
        if var == 0.0:
            continue
        kept_names.append(col)
        kept_freqs.append(n_obs / n)

    return kept_names, kept_freqs


def pearson(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Pearson r and t-statistic. Returns (nan, nan) if n < 3 or zero variance."""
    n = len(xs)
    if n < 3:
        return float("nan"), float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs) / (n - 1))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys) / (n - 1))
    if sx == 0.0 or sy == 0.0:
        return float("nan"), float("nan")
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n - 1)
    r = cov / (sx * sy)
    t = r * math.sqrt((n - 2) / (1.0 - r * r)) if abs(r) < 1.0 else float("inf")
    return r, t


def pool_deltas(delta_lists: list[list]) -> list[float]:
    """Elementwise mean across runs, ignoring None/NaN at each position.
    Raises ValueError on length mismatch."""
    lengths = {len(d) for d in delta_lists}
    if len(lengths) != 1:
        raise ValueError(f"delta array length mismatch across runs: {sorted(lengths)}")
    n_cols = lengths.pop()
    pooled = []
    for i in range(n_cols):
        vals = [
            d[i]
            for d in delta_lists
            if d[i] is not None and not (isinstance(d[i], float) and math.isnan(d[i]))
        ]
        pooled.append(sum(vals) / len(vals) if vals else float("nan"))
    return pooled


def main() -> None:
    parser = argparse.ArgumentParser(description="LOCO delta omega_h vs frequency: pooled correlation table")
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    args = parser.parse_args()

    data_root = Path(args.data_root)
    db_path = Path(args.results_root) / "database.db"

    if not db_path.exists():
        print(f"No database at {db_path}", file=sys.stderr)
        sys.exit(1)

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    cur = con.execute(
        "SELECT dataset, method, run, delta_omega_h FROM loco ORDER BY dataset, method, run"
    )
    rows = cur.fetchall()
    con.close()

    if not rows:
        print("No rows in loco table.")
        return

    groups: dict[tuple[str, str], list[list]] = defaultdict(list)
    for r in rows:
        key = (r["dataset"], r["method"])
        groups[key].append(json.loads(r["delta_omega_h"]))

    results = []  # (method, dataset, r, t, df, n, note)
    for (dataset, method), delta_lists in groups.items():
        if any(len(d) == 0 for d in delta_lists):
            results.append((method, dataset, None, None, None, None, "empty delta array"))
            continue

        try:
            pooled = pool_deltas(delta_lists)
        except ValueError as e:
            results.append((method, dataset, None, None, None, None, str(e)))
            continue

        dz, st = parse_dataset(dataset)
        csv_path = sparse_csv_path(data_root, dz, st)
        if not csv_path.exists():
            results.append((method, dataset, None, None, None, None, "sparse CSV not found"))
            continue

        bench_names, freqs = bench_frequencies(csv_path)
        if len(freqs) != len(pooled):
            results.append((method, dataset, None, None, None, None, "column-filter mismatch"))
            continue

        valid_pairs = [(f, d) for f, d in zip(freqs, pooled) if not math.isnan(d)]
        if len(valid_pairs) < 3:
            results.append((method, dataset, None, None, None, len(valid_pairs), "not enough pairs"))
            continue

        f_vals = [p[0] for p in valid_pairs]
        d_vals = [p[1] for p in valid_pairs]
        r_val, t_val = pearson(f_vals, d_vals)
        df = len(valid_pairs) - 2
        results.append((method, dataset, r_val, t_val, df, len(valid_pairs), ""))

    results.sort(key=lambda x: (x[0], x[1]))

    header = f"{'method':15s} {'dataset':20s} {'r':>8s} {'t':>8s} {'df':>5s} {'n':>5s}"
    print(header)
    print("-" * len(header))
    for method, dataset, r_val, t_val, df, n, _ in results:
        r_str = f"{r_val:+.4f}" if r_val is not None else "   -"
        t_str = f"{t_val:+.3f}" if t_val is not None else "   -"
        df_str = f"{df}" if df is not None else "-"
        n_str = f"{n}" if n is not None else "-"
        print(f"{method:15s} {dataset:20s} {r_str:>8s} {t_str:>8s} {df_str:>5s} {n_str:>5s}")

    valid_r = [r for _, _, r, _, _, _, _ in results if r is not None and not math.isnan(r)]
    print("-" * len(header))
    if valid_r:
        avg_r = sum(valid_r) / len(valid_r)
        print(f"{'average':15s} {'':20s} {avg_r:+8.4f} {'':>8s} {'':>5s} {len(valid_r):>5d}  (mean r over {len(valid_r)} groups)")
    else:
        print("average: no valid r values")


if __name__ == "__main__":
    main()
