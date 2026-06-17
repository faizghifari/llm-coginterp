#!/usr/bin/env python3
"""Compare factor loadings across imputation methods — do the loadings agree, not
just the held-out R²?

Brute-searches results/ for every <method>_<densifier>_<strategy>_loadings.csv,
groups them by DATASET (densifier × strategy), then by SHAPE (benchmark set +
n_factors), and computes pairwise factor-congruence (|cosine|) between every pair
of methods within a group. Only computes for results that exist, so it's safe to
run before all methods/runs have finished.

Alignment (per the loadings format):
  - rows  : factors are over benchmarks; sort rows by benchmark name so they line
            up across runs.
  - cols  : psych does NOT save factors in a consistent order (headers like
            "PA3,PA1,PA2" prove it), so sort factor columns by SSQ (sum of
            squared loadings) descending — the rotated analogue of eigenvalue
            order — then match positionally (1st-vs-1st, …).
  - sign  : factors are sign-indeterminate, so similarity uses |cosine|.

Per-pair similarity = mean over matched factors of |cosine(f_a, f_b)| (Tucker's
congruence on aligned, sorted factors). Writes a verbose markdown report:
average pairwise similarity + the full pairwise table, per dataset group.

Usage:  python3 scripts/compare_loadings.py [--results DIR] [--out FILE]
        (defaults: results/, results/loadings_congruence.md)
"""

import argparse
import itertools
import re
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parent.parent

# Three kinds of loadings file, each compared only against its own kind:
#   <m>_<dz>_<st>_loadings.csv             -> first-order  (rows = benchmarks)
#   <m>_<dz>_<st>_secondorder_loadings.csv -> secondorder (rows = first-order factors)
#   <m>_<dz>_<st>_bifactor_loadings.csv    -> bifactor    (rows = benchmarks; g + groups)
# strategy may contain underscores (all_standard), so anchor densifier to the set
# and the kind suffix explicitly.
DENSIFIERS = ("raw", "C", "S", "R")
# rowname column + non-loading columns to drop, per kind.
KIND_KEY = {"first": "benchmark", "secondorder": "first_order_factor",
            "bifactor": "benchmark"}
BIFACTOR_DROP = ("h2", "u2", "p2")  # diagnostics in om$schmid$sl, not loadings
FNAME_RE = re.compile(
    r"^(?P<method>.+?)_(?P<densifier>" + "|".join(DENSIFIERS) +
    r")_(?P<strategy>.+?)(?:_(?P<kind>secondorder|bifactor))?_loadings\.csv$"
)


def parse_name(path: Path):
    m = FNAME_RE.match(path.name)
    if not m:
        return None
    kind = m.group("kind") or "first"
    return m.group("method"), m.group("densifier"), m.group("strategy"), kind


def load_loadings(path: Path, kind: str):
    """Return (rownames_sorted, L) with rows sorted by name and columns sorted by
    SSQ descending. Drops bifactor diagnostic columns. None on failure."""
    df = pl.read_csv(path)
    key = KIND_KEY[kind]
    if key not in df.columns:
        return None
    drop = {key}
    if kind == "bifactor":
        drop |= {c for c in df.columns if c in BIFACTOR_DROP}
    factor_cols = [c for c in df.columns if c not in drop]
    if not factor_cols:
        return None
    df = df.sort(key)  # align rows across runs
    names = df[key].to_list()
    L = df.select(factor_cols).to_numpy()
    ssq = (L ** 2).sum(axis=0)               # SSQ-sort columns (factor alignment)
    L = L[:, np.argsort(-ssq)]
    return names, L


def factor_congruence(La, Lb):
    """Mean |cosine| over positionally-matched factors of two aligned, SSQ-sorted
    loadings matrices of identical shape."""
    cong = []
    for j in range(La.shape[1]):
        a, b = La[:, j], Lb[:, j]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            cong.append(np.nan)
        else:
            cong.append(abs(float(a @ b) / (na * nb)))
    return float(np.nanmean(cong)) if cong else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "results"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    results_dir = Path(args.results)
    out_path = Path(args.out) if args.out else results_dir / "loadings_congruence.md"

    # collect every loadings csv (first-order, secondorder, bifactor)
    entries = []
    for path in sorted(results_dir.rglob("*_loadings.csv")):
        parsed = parse_name(path)
        if parsed is None:
            continue
        method, dz, st, kind = parsed
        loaded = load_loadings(path, kind)
        if loaded is None:
            continue
        names, L = loaded
        entries.append(dict(method=method, dz=dz, st=st, kind=kind, path=path,
                            rows=tuple(names), nf=L.shape[1], L=L))

    if not entries:
        print(f"No loadings CSVs found under {results_dir}")
        return

    # group by dataset (densifier × strategy) × KIND, then by shape (row set + nf)
    lines = ["# Cross-method loadings congruence",
             "",
             "Per-factor |cosine| (SSQ-sorted, sign-invariant), averaged over "
             "factors. 1.0 = identical factor structure, 0 = orthogonal.",
             "Grouped by dataset (densifier × strategy) × loadings kind "
             "(first-order / secondorder / bifactor), then shape; only same-kind, "
             "same-shape runs are comparable.",
             ""]

    KIND_LABEL = {"first": "first-order", "secondorder": "second-order",
                  "bifactor": "bifactor (Schmid-Leiman)"}

    datasets = {}
    for e in entries:
        datasets.setdefault((e["st"], e["dz"], e["kind"]), []).append(e)

    for (st, dz, kind) in sorted(datasets):
        group = datasets[(st, dz, kind)]
        # sub-group by exact shape (row set + nf)
        shapes = {}
        for e in group:
            shapes.setdefault((e["rows"], e["nf"]), []).append(e)

        lines.append(f"## {st} / densifier {dz} / {KIND_LABEL[kind]}")
        lines.append("")
        row_label = "factors" if kind == "secondorder" else "benchmarks"
        for (rows, nf), runs in shapes.items():
            n_row = len(rows)
            methods = sorted(r["method"] for r in runs)
            tag = f"shape {n_row} {row_label} × {nf} factors"
            if len(runs) < 2:
                only = methods[0] if methods else "?"
                lines.append(f"### {tag}")
                lines.append(f"- only one method present (`{only}`) — nothing to compare.")
                lines.append("")
                continue

            by_method = {r["method"]: r["L"] for r in runs}
            ms = sorted(by_method)
            # pairwise matrix
            pair_vals = []
            mat = {a: {} for a in ms}
            for a in ms:
                mat[a][a] = 1.0
            for a, b in itertools.combinations(ms, 2):
                c = factor_congruence(by_method[a], by_method[b])
                mat[a][b] = mat[b][a] = c
                pair_vals.append(c)

            avg = float(np.nanmean(pair_vals)) if pair_vals else np.nan
            lines.append(f"### {tag}")
            lines.append(f"- methods: {', '.join('`'+m+'`' for m in ms)}")
            lines.append(f"- **average pairwise congruence: {avg:.3f}**")
            lines.append("")
            # full table
            header = "| | " + " | ".join(ms) + " |"
            sep = "|" + "---|" * (len(ms) + 1)
            lines.append(header)
            lines.append(sep)
            for a in ms:
                row = [f"`{a}`"]
                for b in ms:
                    v = mat[a].get(b)
                    row.append("—" if v is None or np.isnan(v) else f"{v:.3f}")
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")

    out_path.write_text("\n".join(lines))
    kinds = sorted({e["kind"] for e in entries})
    print(f"Compared {len(entries)} loadings files across {len(datasets)} "
          f"dataset×kind groups (kinds: {', '.join(kinds)}) -> {out_path}")


if __name__ == "__main__":
    main()
