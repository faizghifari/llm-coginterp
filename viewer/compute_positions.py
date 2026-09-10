#!/usr/bin/env python3
"""Composite benchmark-similarity embedding for the factor-loading viewer.

Reads every *_bifactor_*_loadings.csv under results/text_only/, builds a
consensus cosine-distance matrix between benchmarks per (densifier, tag,
year) by averaging the within-cell cosine distances (rotation/sign invariant
per cell, hence well-defined to average), then UMAP-embeds each composite
distance matrix and writes viewer/positions.json. Year-less loadings files
are the aggregate embedding (key "dz|tag"); per-release-year cohorts from
factor.R --timed are embedded separately (key "dz|tag|y<year>").

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
    r"_bifactor_(?P<tag>pa|2f|forced2f)(?:_y(?P<year>\d{4}))?_loadings\.csv$"
)
# Year-less files are the aggregate (all-years) embedding; `..._y<year>_` files
# come from factor.R --timed (release-year cohorts). forced2f is the DB run
# name leaked into timed filenames — normalize it to the combined 2f tag.
FACTOR_COLS = ["g"] + [f"F{i}*" for i in range(1, 32)]


def load_cells() -> dict[tuple[str, str, int | None], list[tuple[str, np.ndarray, list[str]]]]:
    """{ (dz, tag, year|None): [ (cell_key, factor_matrix, [benchmarks]) ] }"""
    cells: dict[tuple[str, str, int | None], list] = defaultdict(list)
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
        tag = "2f" if m["tag"] == "forced2f" else m["tag"]
        year = int(m["year"]) if m["year"] else None
        cells[(m["dz"], tag, year)].append((key, mat, bench))
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
    for (dz, tag, year), cell_list in sorted(
        cells.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] is not None, kv[0][2] or 0)
    ):
        dist, bench = composite_distance(cell_list)
        xy = embed(dist)
        # aggregate key stays "dz|tag" (backwards compatible); cohorts get |y<year>
        key = f"{dz}|{tag}" if year is None else f"{dz}|{tag}|y{year}"
        out[key] = {
            "densifier": dz,
            "tag": tag,
            **({} if year is None else {"year": year}),
            "n_cells": len(cell_list),
            "benchmarks": bench,
            "points": [
                {"benchmark": b, "x": round(float(x), 5), "y": round(float(y), 5)}
                for b, (x, y) in zip(bench, xy)
            ],
        }
        cohort = "all years" if year is None else f"cohort {year}"
        print(f"{key}: {len(bench)} benchmarks, {len(cell_list)} cells ({cohort})")
    OUT.write_text(json.dumps(out, indent=1))
    (Path(__file__).resolve().parent / "positions.js").write_text(
        "window.POSITIONS = " + json.dumps(out) + ";\n"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
