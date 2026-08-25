"""Densify the super-sparse model x benchmark tables.

The aggregated tables are MNAR-supersparse (~3-5% filled). A single densifier
cannot fix that, so this emits a *family* of densified slices with different
bias profiles. Downstream we cross {densifier} x {imputation method}; the
agreement (or not) between slices is the sensitivity analysis.

All densifiers target the same density (TARGET); they differ only in which
axis is privileged when peeling, which spans the bias space:

  C  column-primary peel : drop sparsest benchmark, then emptied models.
                           -> famous benchmarks x wide model coverage.
  R  row-primary peel    : drop sparsest model, then emptied benchmarks.
                           -> saturated models x wide benchmarks (incl. obscure).
  S  symmetric peel      : drop whichever marginal has the lowest FILL-RATE.
                           -> balanced, neither axis privileged.

Density is the only target (imputer-agnostic). Pairwise-overlap / PD-ness are
SoftImpute-specific notions of "well-posed" and are deliberately NOT optimized
here, so the densifier stays neutral across imputation methods.

Defaults to the TEXT-ONLY analysis view, which is the corpus the analysis is run
on. The multimodal-inclusive corpus is retained and reachable with an explicit
root:

    python3 scripts/densify.py                       # text-only (default)
    python3 scripts/densify.py --data-root data      # multimodal-inclusive

Reads:  <root>/combinations/<strategy>/model_benchmark_table.csv
Writes: <root>/combinations_<densifier>/<strategy>/model_benchmark_table.csv
        <root>/combinations_<densifier>/<strategy>/summary.csv
        <root>/combinations_<densifier>/summary.csv   (rollup per densifier)
        <root>/densify_summary.csv                    (rollup across all)
where <root> defaults to data/text_only.
"""

from pathlib import Path

import numpy as np
import polars as pl

# Anchor to the repo root (parent of src/) so paths resolve from any CWD.
REPO = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO / "data" / "text_only"   # text-only is the analysis default
SRC = DATA_ROOT / "combinations"
DST_ROOT = DATA_ROOT
TARGET = 0.10  # target density (fraction)
MIN_OBS = 3  # floor: every kept model AND benchmark must have >= MIN_OBS scores.
# Needed so all downstream methods are well-posed: prep_matrix drops <2-obs cols,
# and OneSidedMC's pairwise-product loss needs >=2 obs per row (1 obs = 0 pairs).
KEY = "collapse_key"

STRATEGIES = ["all_standard", "all_aggressive"]
DENSIFIERS = ["C", "R", "S"]


def peel(mask: np.ndarray, target: float, axis: str):
    """Greedy peel toward target density.

    axis='col': drop sparsest column (benchmark) each step, then emptied rows.
    axis='row': drop sparsest row (model) each step, then emptied columns.
    Returns (row_keep, col_keep) boolean masks.
    """
    rkeep = np.ones(mask.shape[0], dtype=bool)
    ckeep = np.ones(mask.shape[1], dtype=bool)
    while True:
        sub = mask[np.ix_(rkeep, ckeep)]
        if sub.size == 0 or sub.sum() / sub.size >= target:
            break
        if axis == "col":
            gcols = np.where(ckeep)[0]
            ckeep[gcols[np.argmin(sub.sum(axis=0))]] = False
            sub = mask[np.ix_(rkeep, ckeep)]
            grows = np.where(rkeep)[0]
            rkeep[grows[sub.sum(axis=1) == 0]] = False
        else:  # row
            grows = np.where(rkeep)[0]
            rkeep[grows[np.argmin(sub.sum(axis=1))]] = False
            sub = mask[np.ix_(rkeep, ckeep)]
            gcols = np.where(ckeep)[0]
            ckeep[gcols[sub.sum(axis=0) == 0]] = False
        if ckeep.sum() == 0 or rkeep.sum() == 0:
            break
    return rkeep, ckeep


def symmetric_peel(mask: np.ndarray, target: float):
    """Drop whichever marginal has the lowest fill-RATE (not raw count), so
    neither axis is privileged. Then clear fully-empty rows/cols."""
    rkeep = np.ones(mask.shape[0], dtype=bool)
    ckeep = np.ones(mask.shape[1], dtype=bool)
    while True:
        sub = mask[np.ix_(rkeep, ckeep)]
        if sub.size == 0 or sub.sum() / sub.size >= target:
            break
        col_rate = sub.sum(axis=0) / rkeep.sum()
        row_rate = sub.sum(axis=1) / ckeep.sum()
        if col_rate.min() <= row_rate.min():
            gcols = np.where(ckeep)[0]
            ckeep[gcols[np.argmin(col_rate)]] = False
        else:
            grows = np.where(rkeep)[0]
            rkeep[grows[np.argmin(row_rate)]] = False
        sub = mask[np.ix_(rkeep, ckeep)]
        grows = np.where(rkeep)[0]
        rkeep[grows[sub.sum(axis=1) == 0]] = False
        gcols = np.where(ckeep)[0]
        ckeep[gcols[sub.sum(axis=0) == 0]] = False
        if ckeep.sum() == 0 or rkeep.sum() == 0:
            break
    return rkeep, ckeep


def enforce_min_obs(mask, rkeep, ckeep, min_obs):
    """Iteratively drop any kept row/col below min_obs observations (within the
    currently-kept submatrix) until both axes satisfy the floor. Dropping a
    sparse row can starve a column and vice versa, so loop to a fixed point."""
    rkeep = rkeep.copy()
    ckeep = ckeep.copy()
    while True:
        sub = mask[np.ix_(rkeep, ckeep)]
        if sub.size == 0:
            break
        row_ok = sub.sum(axis=1) >= min_obs
        col_ok = sub.sum(axis=0) >= min_obs
        if row_ok.all() and col_ok.all():
            break
        grows = np.where(rkeep)[0]
        gcols = np.where(ckeep)[0]
        rkeep[grows[~row_ok]] = False
        ckeep[gcols[~col_ok]] = False
        if rkeep.sum() == 0 or ckeep.sum() == 0:
            break
    return rkeep, ckeep


def drop_degenerate_cols(df, value_cols, min_obs=2):
    """Drop columns that prep_matrix / drop_degenerate_cols would drop at runtime:
    fewer than min_obs observed values, zero variance among observed values, or
    only 1 observation (sd is undefined). Returns (df, dropped_names)."""
    drop = []
    for c in value_cols:
        col = df[c]
        vals = col.drop_nulls()
        n = len(vals)
        if n < min_obs or n <= 1:
            drop.append(c)
        elif vals.n_unique() == 1:
            drop.append(c)
    if drop:
        df = df.drop(drop)
    return df, drop


def col_obs_counts(df, value_cols):
    """Count non-null values per column. Returns list of obs counts (len = ncols)."""
    return [df[c].len() - df[c].null_count() for c in value_cols]


def run_densifier(mask: np.ndarray, densifier: str):
    if densifier == "C":
        return peel(mask, TARGET, "col")
    if densifier == "R":
        return peel(mask, TARGET, "row")
    if densifier == "S":
        return symmetric_peel(mask, TARGET)
    raise ValueError(f"unknown densifier {densifier!r}")


def densify_one(strategy: str, densifier: str, peek: bool = False) -> dict:
    df = pl.read_csv(SRC / strategy / "model_benchmark_table.csv")
    value_cols = [c for c in df.columns if c != KEY]

    # ── column obs distribution (raw input) ────────────────────────────────
    raw_obs = col_obs_counts(df, value_cols)
    raw_obs2 = sum(1 for n in raw_obs if n == 2)
    raw_degen = sum(1 for n in raw_obs if n < MIN_OBS)
    raw_const = sum(1 for c, n in zip(value_cols, raw_obs)
                    if n >= MIN_OBS and df[c].drop_nulls().n_unique() == 1)

    mask = df.select(value_cols).to_pandas().notna().to_numpy()  # models x benchmarks
    orig_models, orig_bench = mask.shape
    orig_filled = int(mask.sum())

    rkeep, ckeep = run_densifier(mask, densifier)
    rkeep, ckeep = enforce_min_obs(mask, rkeep, ckeep, MIN_OBS)

    kept_cols = [KEY] + [c for c, k in zip(value_cols, ckeep) if k]
    out = df.select(kept_cols).filter(pl.Series(rkeep))

    # ── column degeneracy guard (parity with prep_matrix / drop_degenerate_cols) ──
    out_value_cols = [c for c in out.columns if c != KEY]
    out, dropped = drop_degenerate_cols(out, out_value_cols, MIN_OBS)
    out_value_cols = [c for c in out.columns if c != KEY]
    if dropped:
        for c in dropped:
            ckeep[value_cols.index(c)] = False

    # ── column obs distribution (densified output) ─────────────────────────
    out_obs = col_obs_counts(out, out_value_cols)
    out_obs2 = sum(1 for n in out_obs if n == MIN_OBS)

    sub = mask[np.ix_(rkeep, ckeep)]
    kept_filled = int(sub.sum())
    density = kept_filled / sub.size if sub.size else 0.0

    if not peek:  # --peek: compute + report only, write nothing
        out_dir = DST_ROOT / f"combinations_{densifier}" / strategy
        out_dir.mkdir(parents=True, exist_ok=True)
        out.write_csv(out_dir / "model_benchmark_table.csv")

    row = {
        "strategy": strategy,
        "densifier": densifier,
        "orig_models": orig_models,
        "orig_benchmarks": orig_bench,
        "orig_density_pct": round(100 * orig_filled / (orig_models * orig_bench), 4),
        "kept_models": out.height,
        "kept_benchmarks": len(out_value_cols),
        "min_obs": MIN_OBS,
        "density_pct": round(100 * density, 4),
        "sparsity_pct": round(100 * (1 - density), 4),
        "data_retained_pct": round(100 * kept_filled / orig_filled, 2) if orig_filled else 0.0,
    }
    if not peek:
        pl.DataFrame([row]).write_csv(out_dir / "summary.csv")

    # ── diagnostic print ───────────────────────────────────────────────────
    print(
        f"[{densifier}] {strategy:16} "
        f"{orig_models}x{orig_bench} "
        f"({orig_filled} cells, {100*orig_filled/(orig_models*orig_bench):.1f}%) -> "
        f"{out.height}x{len(out_value_cols)} "
        f"({100*density:.1f}%)  retained {100*kept_filled/orig_filled:.0f}%"
        f"  out obs={MIN_OBS}: {out_obs2}"
    )
    # parts = []
    # if raw_obs2: parts.append(f"raw obs=2: {raw_obs2}")
    # if raw_degen: parts.append(f"raw <{MIN_OBS}: {raw_degen}")
    # if raw_const: parts.append(f"raw zero-var: {raw_const}")
    # if dropped: parts.append(f"dropped degen: {len(dropped)}")
    # if out_obs2: parts.append(f"out obs=2: {out_obs2}")
    # if parts:
    #     print("  | " + "  ".join(parts))
    # else:
    #     print()

    return row


def main(peek: bool = False):
    all_rows = []

    # ── raw table column-obs diagnostics (undensified inputs) ──────────────
    for strategy in STRATEGIES:
        rdf = pl.read_csv(SRC / strategy / "model_benchmark_table.csv")
        rvc = [c for c in rdf.columns if c != KEY]
        raw_obs = col_obs_counts(rdf, rvc)
        raw_obs2 = sum(1 for n in raw_obs if n == 2)
        raw_degen = sum(1 for n in raw_obs if n < MIN_OBS)
        raw_const = sum(1 for c, n in zip(rvc, raw_obs)
                        if n >= MIN_OBS and rdf[c].drop_nulls().n_unique() == 1)
        print(
            f"[raw] {strategy:16} "
            f"{rdf.height}x{len(rvc)} "
            # f"| raw obs=2: {raw_obs2}  raw <{MIN_OBS}: {raw_degen}"
            # + (f"  raw zero-var: {raw_const}" if raw_const else "")
        )

    for densifier in DENSIFIERS:
        rows = []
        for strategy in STRATEGIES:
            r = densify_one(strategy, densifier, peek=peek)
            rows.append(r)
            all_rows.append(r)
        if not peek:
            pl.DataFrame(rows).write_csv(DST_ROOT / f"combinations_{densifier}" / "summary.csv")
    if peek:
        print(
            f"\n[peek] {len(all_rows)} tables previewed "
            f"({len(STRATEGIES)} strategies x {len(DENSIFIERS)} densifiers) "
            f"— nothing written."
        )
    else:
        pl.DataFrame(all_rows).write_csv(DST_ROOT / "densify_summary.csv")
        print(
            f"\nWrote {len(all_rows)} densified tables "
            f"({len(STRATEGIES)} strategies x {len(DENSIFIERS)} densifiers) "
            f"to data/combinations_<C|R|S>/ + rollup data/densify_summary.csv"
        )


if __name__ == "__main__":
    import sys
    argv = sys.argv[1:]
    if "--data-root" in argv:
        _root = Path(argv[argv.index("--data-root") + 1])
        _root = _root if _root.is_absolute() else REPO / _root
        SRC = _root / "combinations"
        DST_ROOT = _root
    main(peek="--peek" in argv)
