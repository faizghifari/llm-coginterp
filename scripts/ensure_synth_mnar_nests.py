"""Verify that the synthetic MNAR masks nest across density slices.

For every (truth cell, mask mechanism, table tag) it checks, walking the
density slices from densest to sparsest:

  1. the denser slice's grid (models x benchmarks) is a subset of the
     sparser slice's grid, and
  2. every cell observed in the denser slice is also observed in the
     sparser slice.

i.e. mask(0.5) subset of mask(0.3) subset of mask(0.1), the property that
makes the synthetic corpus a faithful stand-in for densify.py's real slices.

It also validates every planted truth (per density slice's truth/ dir): the
planted correlation is symmetric, unit-diagonal and PSD, its spectrum is
decreasing over the planted factors (general first, then group factors in
decreasing-communalities order), the achieved omega_h matches its wh_target,
and the written complete matrix reproduces the planted mean off-diagonal
correlation.

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
import json
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


def check_truth(root, problems):
    """Truth-file validity per <tag>_<cell>: the planted correlation is
    symmetric, unit-diagonal and PSD (all eigenvalues >= 0), its spectrum is
    DECREASING over the planted factors (general factor first, then group
    factors in decreasing-communalities order), the achieved omega_h matches
    its wh_target, and the written complete matrix reproduces the planted
    mean off-diagonal correlation (rounding-safe)."""
    for dens_dir in sorted((p for p in root.iterdir()
                            if p.is_dir() and _is_num(p.name)),
                           key=lambda p: float(p.name)):
        truth_dir = dens_dir / "truth"
        if not truth_dir.is_dir():
            continue
        for corr_path in sorted(truth_dir.glob("*_correlation.csv")):
            base = corr_path.name[:-len("_correlation.csv")]
            rel = corr_path.relative_to(root)
            complete_path = truth_dir / f"{base}_complete.csv"
            meta_path = truth_dir / f"{base}_meta.json"
            if not complete_path.is_file() or not meta_path.is_file():
                problems.append(f"{rel}: missing complete.csv or meta.json")
                continue
            Rdf = pl.read_csv(corr_path)
            cols = [c for c in Rdf.columns if c != KEY]
            R = Rdf.select(cols).to_numpy().astype(np.float64)
            if R.shape[0] != R.shape[1] or R.shape[0] != len(cols):
                problems.append(f"{rel}: correlation not square ({R.shape})")
                continue
            if (np.abs(R - R.T).max() > 1e-9
                    or np.abs(np.diag(R) - 1.0).max() > 1e-9):
                problems.append(f"{rel}: correlation not "
                                "symmetric/unit-diagonal")
                continue
            eig = np.linalg.eigvalsh(R)[::-1]   # descending
            if eig[-1] < -1e-8:
                problems.append(f"{rel}: negative eigenvalue {eig[-1]:.3g} "
                                "(planted correlation is not PSD)")
                continue
            meta = json.loads(meta_path.read_text())
            nfac = int(meta["true_nfactors"])
            top = eig[:nfac]
            if not np.all(top[:-1] > top[1:]):
                problems.append(
                    f"{rel}: spectrum not decreasing over the first "
                    f"{nfac} factors: {np.round(top, 4).tolist()}")
                continue
            if "wh_target" in meta and abs(
                    meta["omega_hierarchical"] - meta["wh_target"]) > 0.02:
                problems.append(
                    f"{rel}: achieved omega_h "
                    f"{meta['omega_hierarchical']:.4f} misses target "
                    f"{meta['wh_target']:.2f}")
                continue
            df = pl.read_csv(complete_path)
            M = df.select(cols).to_numpy().astype(np.float64)
            if M.shape[1] != R.shape[1] or M.shape[0] != int(meta["n_models"]):
                problems.append(f"{rel}: complete {M.shape} != correlation "
                                f"{R.shape} (columns must match)")
                continue
            off = ~np.eye(len(cols), dtype=bool)
            r_emp = float(np.corrcoef(M, rowvar=False)[off].mean())
            r_planted = float(R[off].mean())
            if abs(r_emp - r_planted) > 0.02:
                problems.append(
                    f"{rel}: complete matrix mean r {r_emp:.4f} != planted "
                    f"{r_planted:.4f}")
                continue
            print(f"OK   truth {rel}: PSD, decreasing spectrum, "
                  f"omega_h={meta['omega_hierarchical']:.3f}")


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

    check_truth(root, problems)
    checked = 0
    for (mask_name, tag), rows in sorted(groups.items()):
        rows.sort(key=lambda r: r[0], reverse=True)  # densest first
        chain = " -> ".join(f"{r[0]:g}" for r in rows)
        label = f"{mask_name}/{tag}  {chain}"
        chain_ok = True
        for (d1, k1, c1, m1), (d2, k2, c2, m2) in zip(rows, rows[1:]):
            checked += 1
            rk = {k: i for i, k in enumerate(k2)}
            ck = {c: i for i, c in enumerate(c2)}
            miss_r = [k for k in k1 if k not in rk]
            miss_c = [c for c in c1 if c not in ck]
            if miss_r or miss_c:
                problems.append(
                    f"{mask_name}/{tag}  {d1:g} -> {d2:g}: grid not nested "
                    f"({len(miss_r)} models, {len(miss_c)} benchmarks absent "
                    f"from {d2:g}); e.g. {miss_r[:3]} / {miss_c[:3]}")
                chain_ok = False
                continue
            ri = np.array([rk[k] for k in k1])
            ci = np.array([ck[c] for c in c1])
            leaked = m1 & ~m2[np.ix_(ri, ci)]
            if leaked.any():
                problems.append(
                    f"{mask_name}/{tag}  {d1:g} -> {d2:g}: "
                    f"{int(leaked.sum())} observed cells of {d1:g} "
                    f"are missing at {d2:g} (of {int(m1.sum())})")
                chain_ok = False
        if chain_ok:
            print(f"OK   {label}: {int(rows[0][3].sum())} observed cells "
                  f"nested")

    if problems:
        print(f"\nFAIL: {len(problems)} problem(s) (nesting + truth checks) "
              f"across {checked} density pair(s)")
        for p in problems:
            print("  " + p)
        sys.exit(1)
    print(f"\nPASS: {checked} density pair(s) nested, "
          f"{len(groups)} mask/tag group(s), root {root.relative_to(REPO)
          if root.is_relative_to(REPO) else root}")


if __name__ == "__main__":
    main()
