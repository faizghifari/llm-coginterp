#!/usr/bin/env python3
"""Composite benchmark-similarity embedding for the factor-loading viewer.

Reads every *_bifactor_{pa,2f}_loadings.csv under results/text_only/, builds a
consensus cosine-distance matrix between benchmarks per (densifier, tag) by
averaging the within-cell cosine distances (rotation/sign invariant per cell,
hence well-defined to average), then UMAP-embeds each composite distance
matrix and writes viewer/positions.json.

Each benchmark's vector within a cell is [g, F1..Fk] (group + general
factors). Cells where a benchmark is absent simply don't contribute to that
benchmark's pairwise means.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
from umap import UMAP

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "text_only"
OUT = Path(__file__).resolve().parent / "positions.json"

NAME_RE = re.compile(
    r"^(?P<method>.+)_(?P<dz>C|R|S|raw)_(?P<st>all_standard|all_aggressive)"
    r"_bifactor_(?P<tag>pa|2f)_loadings\.csv$"
)
FACTOR_COLS = ["g"] + [f"F{i}*" for i in range(1, 32)]


def load_cells() -> dict[tuple[str, str], list[tuple[str, np.ndarray, list[str]]]]:
    """{ (dz, tag): [ (cell_key, factor_matrix, [benchmarks]) ] }"""
    cells: dict[tuple[str, str], list] = defaultdict(list)
    for path in sorted(RESULTS.glob("*/*_loadings.csv")):
        m = NAME_RE.match(path.name)
        if not m:
            continue
        df = pl.read_csv(path, infer_schema_length=0)
        cols = [c for c in FACTOR_COLS if c in df.columns]
        if not cols:
            continue
        # skip corrupt cells whose benchmark names were lost upstream
        # (e.g. softimpute_corr surrogate wrote numeric column headers)
        bench_all = df["benchmark"].to_list()
        if any(b.strip().isdigit() for b in bench_all):
            print(f"skipping {path.name}: numeric benchmark names (corrupt)")
            continue
        bench = bench_all
        mat = df.select([pl.col(c).cast(pl.Float64, strict=False) for c in cols]).to_numpy()
        if not np.isfinite(mat).all():
            continue
        key = f"{m['method']}_{m['dz']}_{m['st']}"
        cells[(m["dz"], m["tag"])].append((key, mat, bench))
    return cells


def composite_distance(
    cell_list: list[tuple[str, np.ndarray, list[str]]],
) -> tuple[np.ndarray, list[str]]:
    """Average pairwise cosine distance across cells (union of benchmarks)."""
    bench_union: list[str] = []
    seen: set[str] = set()
    for _, _, bench in cell_list:
        for b in bench:
            if b not in seen:
                seen.add(b)
                bench_union.append(b)
    idx = {b: i for i, b in enumerate(bench_union)}
    n = len(bench_union)

    dist_sum = np.zeros((n, n))
    count = np.zeros((n, n))
    for _, mat, bench in cell_list:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        unit = np.where(norms > 0, mat / np.maximum(norms, 1e-12), 0.0)
        cos = unit @ unit.T
        d = np.clip(1.0 - cos, 0.0, 2.0)
        rows = [idx[b] for b in bench]
        r = np.array(rows)
        dist_sum[np.ix_(r, r)] += d
        count[np.ix_(r, r)] += 1.0

    with np.errstate(invalid="ignore", divide="ignore"):
        dist = np.where(count > 0, dist_sum / np.maximum(count, 1e-12), np.nan)
    # Cells missing one benchmark leave NaNs; fill with the mean of the
    # benchmark's known distances (or global mean) so UMAP gets a full matrix.
    known = ~np.isnan(dist)
    global_mean = dist[known].mean()
    for i in range(n):
        for j in range(i + 1, n):
            if math.isnan(dist[i, j]):
                cand = [dist[i, k] for k in range(n) if known[i, k]] or [
                    dist[k, j] for k in range(n) if known[k, j]
                ]
                v = float(np.mean(cand)) if cand else global_mean
                dist[i, j] = dist[j, i] = v
    np.fill_diagonal(dist, 0.0)
    # symmetrize + PSD-ish for UMAP precomputed use
    dist = (dist + dist.T) / 2.0
    return dist, bench_union


def embed(dist: np.ndarray) -> np.ndarray:
    return UMAP(
        n_components=2,
        metric="precomputed",
        n_neighbors=10,
        min_dist=0.15,
        random_state=42,
    ).fit_transform(dist)


def main() -> None:
    cells = load_cells()
    if not cells:
        raise SystemExit(f"no loadings found under {RESULTS}")
    out = {}
    for (dz, tag), cell_list in sorted(cells.items()):
        dist, bench = composite_distance(cell_list)
        xy = embed(dist)
        out[f"{dz}|{tag}"] = {
            "densifier": dz,
            "tag": tag,
            "n_cells": len(cell_list),
            "benchmarks": bench,
            "points": [
                {"benchmark": b, "x": round(float(x), 5), "y": round(float(y), 5)}
                for b, (x, y) in zip(bench, xy)
            ],
        }
        print(f"{dz}|{tag}: {len(bench)} benchmarks, {len(cell_list)} cells")
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
