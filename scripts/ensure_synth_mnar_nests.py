"""Verify that the synthetic MNAR masks nest across density slices.

For every (truth cell, mask mechanism, table tag) it checks, walking the
density slices from densest to sparsest:

  1. the denser slice's grid (models x benchmarks) is a subset of the
     sparser slice's grid, and
  2. every cell observed in the denser slice is also observed in the
     sparser slice.

i.e. mask(0.5) subset of mask(0.3) subset of mask(0.1), the property that
makes the synthetic corpus a faithful stand-in for densify.py's real slices.

Reads:
    data/synth/<density>/<cell>/<mask>/s0/truth/<tag>_mask.npz
    data/synth/<density>/<cell>/<mask>/s0/combinations_<dz>/<st>/model_benchmark_table.csv
`tag` is <dz>_<st>_<cell>. The npz mask must agree with the CSV's non-null
pattern (checked too, so a stale CSV or npz is reported).

Usage:  uv run python scripts/ensure_synth_mnar_nests.py [--root data/synth]
Exit code 1 if any pair violates nesting (for example data built with
make_mnar.py --no-nest).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parent.parent
KEY = "collapse_key"


def _is_num(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def _table_for(s0, tag, cell):
    """Locate the model_benchmark_table.csv that owns a truth tag."""
    dz = tag.split("_")[0]
    st = tag[len(dz) + 1:-(len(cell) + 1)]
    return s0 / f"combinations_{dz}" / st / "model_benchmark_table.csv"


def collect(root):
    """(mask_name, tag) -> [(density, keys, cols, observed), ...]"""
    groups = {}
    problems = []
    for dens_dir in sorted((p for p in root.iterdir()
                            if p.is_dir() and _is_num(p.name)),
                           key=lambda p: float(p.name)):
        for s0 in sorted(dens_dir.glob("*/*/s0")):
            cell = s0.parent.parent.name
            mask_name = s0.parent.name
            truth_dir = s0 / "truth"
            if not truth_dir.is_dir():
                continue
            for npz in sorted(truth_dir.glob("*_mask.npz")):
                tag = npz.name[:-len("_mask.npz")]
                csv = _table_for(s0, tag, cell)
                if not csv.exists():
                    problems.append(f"{npz.relative_to(root)}: no table {csv}")
                    continue
                df = pl.read_csv(csv)
                keys = [str(k) for k in df[KEY].to_list()]
                cols = [c for c in df.columns if c != KEY]
                obs = df.select(cols).to_numpy()
                obs = np.isfinite(obs.astype(np.float64))
                mask = np.load(npz)["mask"].astype(bool)
                if mask.shape != obs.shape:
                    problems.append(
                        f"{npz.relative_to(root)}: mask {mask.shape} != "
                        f"table {obs.shape}")
                    continue
                if not np.array_equal(mask, obs):
                    problems.append(
                        f"{npz.relative_to(root)}: npz mask disagrees with "
                        f"CSV non-null pattern ({(mask != obs).sum()} cells)")
                groups.setdefault((mask_name, tag), []).append(
                    (float(dens_dir.name), keys, cols, mask))
    return groups, problems


def main():
    ap = argparse.ArgumentParser(
        description="check synthetic MNAR mask nesting across densities")
    ap.add_argument("--root", default="data/synth",
                    help="synth corpus root (default data/synth)")
    a = ap.parse_args()
    root = Path(a.root)
    root = root if root.is_absolute() else REPO / root
    if not root.is_dir():
        sys.exit(f"ensure_synth_mnar_nests: {root} is not a directory "
                 f"(run scripts/make_mnar.py first)")

    groups, problems = collect(root)
    if not groups:
        sys.exit(f"ensure_synth_mnar_nests: no mask npz files under {root}")

    checked = 0
    for (mask_name, tag), rows in sorted(groups.items()):
        rows.sort(key=lambda r: r[0], reverse=True)  # densest first
        for (d1, k1, c1, m1), (d2, k2, c2, m2) in zip(rows, rows[1:]):
            checked += 1
            label = f"{mask_name}/{tag}  {d1:g} -> {d2:g}"
            rk = {k: i for i, k in enumerate(k2)}
            ck = {c: i for i, c in enumerate(c2)}
            miss_r = [k for k in k1 if k not in rk]
            miss_c = [c for c in c1 if c not in ck]
            if miss_r or miss_c:
                problems.append(
                    f"{label}: grid not nested "
                    f"({len(miss_r)} models, {len(miss_c)} benchmarks absent "
                    f"from {d2:g}); e.g. {miss_r[:3]} / {miss_c[:3]}")
                continue
            ri = np.array([rk[k] for k in k1])
            ci = np.array([ck[c] for c in c1])
            leaked = m1 & ~m2[np.ix_(ri, ci)]
            if leaked.any():
                problems.append(
                    f"{label}: {int(leaked.sum())} observed cells of {d1:g} "
                    f"are missing at {d2:g} (of {int(m1.sum())})")
            else:
                print(f"OK   {label}: {int(m1.sum())} observed cells nested")

    if problems:
        print(f"\nFAIL: {len(problems)} nesting violation(s) across "
              f"{checked} density pair(s)")
        for p in problems:
            print("  " + p)
        sys.exit(1)
    print(f"\nPASS: {checked} density pair(s) nested, "
          f"{len(groups)} mask/tag group(s), root {root.relative_to(REPO)
          if root.is_relative_to(REPO) else root}")


if __name__ == "__main__":
    main()
