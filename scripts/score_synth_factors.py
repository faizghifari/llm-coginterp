#!/usr/bin/env python3
"""Score factor-level recovery on the synthetic MNAR corpus against planted truth.

Reads results/synth/factor/rows_*.csv + per-run SL loadings written by
src/run/factor_synth.R, and the planted truth in data/synth/truth/. Per
(method, truth cell, mask, table):

  omega_h_true / omega_h_est / omega_bias   -- same estimand both sides:
      (sum g-loadings)^2 / 1'R1 (psych::omega; make_mnar.py 'omega_hierarchical')
  nf_true / nf_est                          -- 1 + planted group factors vs Horn PA
  g_congruence    -- |Tucker phi| between estimated g loadings and planted lam_g
  domain_ari      -- adjusted Rand index between each benchmark's argmax
                     |group loading| and its planted group (domain recovery)
  imp_r2          -- held-out imputation R² from results/synth/database.db
                     (imputation_synth), so the R² >= 0.45 gate can be applied

Writes results/synth/factor/scored.csv and prints per-method/mask summaries.
"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

REPO = Path(__file__).resolve().parent.parent
SYNTH = REPO / "data" / "synth"
FDIR = REPO / "results" / "synth" / "factor"
DB = REPO / "results" / "synth" / "database.db"
ST = "all_standard"


def truth_for(dz, cell):
    base = SYNTH / "truth" / f"{dz}_{ST}_{cell}"
    meta = json.loads(Path(f"{base}_meta.json").read_text())
    npz = np.load(f"{base}_factors.npz")
    cols = pd.read_csv(f"{base}_complete.csv", nrows=0).columns[1:].tolist()
    return meta, pd.Series(npz["lam_g"], index=cols), pd.Series(npz["fcols"], index=cols)


def imputation_r2():
    if not DB.exists():
        return None
    with sqlite3.connect(DB) as con:
        tabs = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
        if "imputation_synth" not in tabs:
            return None
        df = pd.read_sql("select * from imputation_synth", con)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gate", type=float, default=0.45, help="R² gate for the gated summary")
    a = ap.parse_args()

    rows = pd.concat([pd.read_csv(p) for p in sorted(FDIR.glob("rows_*.csv"))], ignore_index=True)
    out = []
    for r in rows.itertuples():
        meta, lam_g, fcols = truth_for(r.dz, r.cell)
        lp = FDIR / r.cell / r.mask / f"{r.method}_{r.dz}_loadings.csv"
        rec = dict(method=r.method, cell=r.cell, mask=r.mask, dz=r.dz,
                   nf_true=meta["true_nfactors"], nf_est=r.nf,
                   omega_h_true=meta["omega_hierarchical"], omega_h_est=r.omega_h,
                   omega_total_est=r.omega_total,
                   g_congruence=np.nan, domain_ari=np.nan)
        if lp.exists():
            L = pd.read_csv(lp).set_index("benchmark")
            common = [b for b in lam_g.index if b in L.index]
            g = L.loc[common, "g"].to_numpy()
            t = lam_g[common].to_numpy()
            rec["g_congruence"] = abs(g @ t) / (np.linalg.norm(g) * np.linalg.norm(t))
            fc = [c for c in L.columns if c.startswith("F")]
            if fc:
                assign = L.loc[common, fc].abs().to_numpy().argmax(1)
                rec["domain_ari"] = adjusted_rand_score(fcols[common].to_numpy(), assign)
        out.append(rec)
    df = pd.DataFrame(out)
    df["omega_bias"] = df.omega_h_est - df.omega_h_true
    df["nf_err"] = df.nf_est - df.nf_true

    r2 = imputation_r2()
    if r2 is not None:
        # imputation_synth: one row per (method, dataset=<dz>_<st>, cell, mask)
        r2 = r2.assign(dz=r2["dataset"].str.split("_").str[0])
        keep = [c for c in ("method", "cell", "mask", "dz", "r2", "gt_r2") if c in r2.columns]
        df = df.merge(r2[keep].drop_duplicates(["method", "cell", "mask", "dz"]),
                      on=["method", "cell", "mask", "dz"], how="left")
        df = df.rename(columns={"r2": "imp_r2"})
    df.to_csv(FDIR / "scored.csv", index=False)

    pd.set_option("display.width", 220)
    cols = ["omega_h_true", "omega_h_est", "omega_bias", "nf_true", "nf_est",
            "g_congruence", "domain_ari"] + (["imp_r2"] if "imp_r2" in df else [])
    print("\n== by method x mask (mean over 4 truth cells x 2 tables) ==")
    print(df.groupby(["method", "mask"])[cols].mean().round(3).to_string())
    print("\n== by method x truth cell (mean over masks/tables) ==")
    print(df.groupby(["method", "cell"])[cols].mean().round(3).to_string())
    if "imp_r2" in df:
        g = df[(df.method == "oracle") | (df.imp_r2 >= a.gate)]
        print(f"\n== gated (imputation R² >= {a.gate}) by method x mask ==")
        print(g.groupby(["method", "mask"])[cols].agg("mean").round(3).to_string())
        print(g.groupby(["method", "mask"]).size().rename("n").to_string())
    print(f"\nwrote {FDIR / 'scored.csv'}")


if __name__ == "__main__":
    sys.exit(main())
