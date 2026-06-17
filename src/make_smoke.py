"""Generate a tiny smoke-test fixture under data/smoke/ that exercises the whole
pipeline (densifier -> all 3 imputers -> factoring) fast.

Builds two small synthetic model x benchmark tables with real low-rank factor
structure + MNAR-ish sparsity, then runs them through the SAME densifier (C/R/S)
used in production, writing data/smoke/combinations_<C|R|S>/<strategy>/.

Run:  python3 src/make_smoke.py
Then: Rscript src/run/main.R --smoke
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))  # find densify from any CWD
import densify  # reuse run_densifier + enforce_min_obs + KEY

# Anchor to the repo root (parent of src/) so paths resolve from any CWD.
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "smoke"
KEY = densify.KEY

# Small, but big enough that factoring + rank sweep are meaningful and fast.
SPECS = {
    "all_standard": dict(n_models=200, n_bench=24, rank=3, base_density=0.12),
    "all_aggressive": dict(n_models=140, n_bench=20, rank=3, base_density=0.15),
}


def synth_table(n_models, n_bench, rank, base_density, seed):
    """Low-rank score matrix U V' + noise, on a 0-100ish scale per column, then
    punch MNAR-ish holes: popular benchmarks (low index) observed more often."""
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(n_models, rank))
    V = rng.normal(size=(n_bench, rank))
    X = U @ V.T
    # per-column scale/offset so columns live on different scales (like real data)
    scale = rng.uniform(3, 30, size=n_bench)
    offset = rng.uniform(20, 90, size=n_bench)
    X = X * scale + offset + rng.normal(scale=1.0, size=X.shape)

    # MNAR observation prob: decays with benchmark index (fame) and varies by
    # model. Gentle decay (0.5 floor) so col-primary peel keeps a usable number
    # of benchmarks rather than collapsing to a tiny famous core.
    col_fame = np.linspace(1.0, 0.5, n_bench)
    row_act = rng.uniform(0.4, 1.0, size=n_models)
    p_obs = np.outer(row_act, col_fame)
    p_obs *= base_density / p_obs.mean()
    p_obs = np.clip(p_obs, 0, 1)
    mask = rng.random(X.shape) < p_obs

    Xm = X.copy()
    Xm[~mask] = np.nan
    keys = [f"model_{i:03d}" for i in range(n_models)]
    cols = [f"bench_{j:02d}" for j in range(n_bench)]
    df = pl.DataFrame({KEY: keys})
    for j, c in enumerate(cols):
        df = df.with_columns(pl.Series(c, Xm[:, j]))
    return df


def main():
    for si, (strategy, spec) in enumerate(SPECS.items()):
        df = synth_table(seed=100 + si, **spec)
        value_cols = [c for c in df.columns if c != KEY]
        mask = df.select(value_cols).to_pandas().notna().to_numpy()

        # raw (undensified) table — the pipeline's 4th densifier level reads it.
        raw_dir = OUT / "combinations" / strategy
        raw_dir.mkdir(parents=True, exist_ok=True)
        df.write_csv(raw_dir / "model_benchmark_table.csv")
        print(f"[raw] {strategy:14} -> {df.shape[0]}x{df.shape[1] - 1} "
              f"({100 * mask.sum() / mask.size:.0f}% dense)")

        for dz in densify.DENSIFIERS:
            rk, ck = densify.run_densifier(mask, dz)
            rk, ck = densify.enforce_min_obs(mask, rk, ck, densify.MIN_OBS)
            kept_cols = [KEY] + [c for c, k in zip(value_cols, ck) if k]
            out = df.select(kept_cols).filter(pl.Series(rk))

            d = OUT / f"combinations_{dz}" / strategy
            d.mkdir(parents=True, exist_ok=True)
            out.write_csv(d / "model_benchmark_table.csv")
            dens = 100 * mask[np.ix_(rk, ck)].sum() / mask[np.ix_(rk, ck)].size
            print(f"[{dz}] {strategy:14} -> {out.shape[0]}x{out.shape[1] - 1} "
                  f"({dens:.0f}% dense)")
    print(f"\nSmoke fixture written under {OUT}/")
    print("Run: Rscript src/run/main.R --smoke")


if __name__ == "__main__":
    main()
