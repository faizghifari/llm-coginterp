#!/usr/bin/env python3
"""Score factor-level recovery on the synthetic MNAR corpus against planted truth.

Reads results/synth/<density>/factor/rows_*.csv + per-run SL loadings written by
src/run/factor_synth.R, and the planted truth in data/synth/<density>/truth/. Per
(method, truth cell, mask, table):

  omega_h_est / omega_h_oracle / omega_bias -- omega bias vs the ORACLE
                     arm's own estimate for the same (dz, cell): the
                     full-data run of the exact factor.R recipe. The
                     generator plants NO general factor and targets NO
                     omega (mean_r is the only calibrated truth quantity),
                     so there is no planted omega to score against -- the
                     full-data oracle is the attainable reference. Oracle
                     rows themselves get bias 0.
  nf_true / nf_est                          -- 1 + planted suites vs Horn PA
  g_congruence    -- |Tucker phi| between estimated g loadings and the
                     ORACLE arm's g loadings (no planted lam_g exists)
  domain_ari      -- adjusted Rand index between each benchmark's argmax
                     |group loading| and its planted suite (domain recovery)
  imp_r2          -- held-out imputation R² from
                     results/synth/<density>/database.db
                     (imputation_synth), so the R² >= 0.45 gate can be applied

Writes results/synth/<density>/factor/scored.csv and prints per-method/mask
summaries.
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
ST = "all_standard"
HIER = False   # --density 0: score the hierarchical (impute.R --hier --synth) end result
WR2_GATE = 0.25

def resolve_density_0():
    """density 0 aliases the SPARSEST density slice (final step of the hier descent)."""
    slices = []
    for p in (REPO / "data" / "synth").iterdir():
        if not p.is_dir():
            continue
        try:
            slices.append(float(p.name))
        except ValueError:
            continue
    if not slices:
        sys.exit("density 0: no numeric density slices under data/synth/")
    return f"{min(slices):g}"


def truth_for(dz, cell):
    base = SYNTH / "truth" / f"{dz}_{ST}_{cell}"
    meta = json.loads(Path(f"{base}_meta.json").read_text())
    npz = np.load(f"{base}_factors.npz")
    cols = pd.read_csv(f"{base}_complete.csv", nrows=0).columns[1:].tolist()
    return meta, pd.Series(npz["fcols"], index=cols)


def oracle_g(dz, cell):
    """The oracle arm's SL g loadings for a (dz, cell) -- the congruence
    reference now that no planted lam_g exists."""
    p = FDIR / cell / "complete" / f"oracle_{dz}_loadings.csv"
    if not p.exists():
        return None
    return pd.read_csv(p).set_index("benchmark")["g"]


def imputation_r2():
    if not DB.exists():
        return None
    with sqlite3.connect(DB) as con:
        tabs = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
        table = "imputation_synth_hier" if HIER else "imputation_synth"
        if table not in tabs:
            return None
        df = pd.read_sql(f"select * from {table}", con)
    if HIER:
        # keep only the final hand-off row of the descent (density "0") and use
        # wr2 (the cumulative weighted R2 the factoring gate reads) — no fallback
        # to the per-step r2
        df = df[df["density"] == "0"]
        df = df.assign(r2=df["wr2"])
    return df


def main():
    global SYNTH, FDIR, DB, HIER
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--density", required=True, metavar="FRAC",
                    help="synth density slice, e.g. 0.1 (data/synth/<density>/); "
                         "0 aliases the sparsest slice's hierarchical end result")
    ap.add_argument("--gate", type=float, default=WR2_GATE, help="R² gate for the gated summary")
    a = ap.parse_args()
    HIER = a.density == "0"
    density = resolve_density_0() if HIER else a.density
    SYNTH = REPO / "data" / "synth" / density
    FDIR = REPO / "results" / "synth" / ("0" if HIER else a.density) / "factor"
    DB = (REPO / "results" / "synth" / "hierarchical" / "database.db" if HIER
          else REPO / "results" / "synth" / a.density / "database.db")

    rows = pd.concat([pd.read_csv(p) for p in sorted(FDIR.glob("rows_*.csv"))], ignore_index=True)
    out = []
    for r in rows.itertuples():
        meta, fcols = truth_for(r.dz, r.cell)
        lp = FDIR / r.cell / r.mask / f"{r.method}_{r.dz}_loadings.csv"
        rec = dict(method=r.method, cell=r.cell, mask=r.mask, dz=r.dz,
                   nf_true=meta["true_nfactors"], nf_est=r.nf,
                   omega_h_est=r.omega_h, omega_total_est=r.omega_total,
                   g_congruence=np.nan, domain_ari=np.nan)
        if lp.exists():
            L = pd.read_csv(lp).set_index("benchmark")
            og = oracle_g(r.dz, r.cell)
            if og is not None:
                common = [b for b in og.index if b in L.index]
                g = L.loc[common, "g"].to_numpy()
                o = og.loc[common].to_numpy()
                rec["g_congruence"] = abs(g @ o) / (np.linalg.norm(g)
                                                    * np.linalg.norm(o))
            fc = [c for c in L.columns if c.startswith("F")]
            if fc:
                common = [b for b in fcols.index if b in L.index]
                assign = L.loc[common, fc].abs().to_numpy().argmax(1)
                rec["domain_ari"] = adjusted_rand_score(
                    fcols[common].to_numpy(), assign)
        out.append(rec)
    df = pd.DataFrame(out)
    df["nf_err"] = df.nf_est - df.nf_true

    # Rebase the omega comparison on the oracle arm: the full-data run of the
    # same recipe is the attainable reference, since the planted structure is
    # not representable by the PA/SL estimator (see the oracle bias itself).
    orc = df[(df["method"] == "oracle") & (df["mask"] == "complete")][
        ["dz", "cell", "omega_h_est"]].rename(columns={"omega_h_est": "omega_h_oracle"})
    df = df.merge(orc.drop_duplicates(["dz", "cell"]), on=["dz", "cell"], how="left")
    df["omega_bias"] = df.omega_h_est - df.omega_h_oracle

    r2 = imputation_r2()
    if r2 is not None:
        # imputation_synth: one row per (method, dataset=<dz>_<st>, cell, mask)
        r2 = r2.assign(dz=r2["dataset"].str.split("_").str[0])
        keep = [c for c in ("method", "cell", "mask", "dz", "r2", "gt_r2") if c in r2.columns]
        df = df.merge(r2[keep].drop_duplicates(["method", "cell", "mask", "dz"]),
                      on=["method", "cell", "mask", "dz"], how="left")
        df = df.rename(columns={"r2": "imp_r2"})
    df.to_csv(FDIR / "scored.csv", index=False)

    def md(d):
        """Markdown-render a frame, no tabulate needed; floats at 4 decimals."""
        head = "| " + " | ".join(map(str, d.columns)) + " |"
        sep = "|" + "---|" * len(d.columns)
        def cell(v):
            if pd.isna(v):
                return ""
            if isinstance(v, float):
                return f"{v:.4f}"
            return str(v)
        rows = ["| " + " | ".join(cell(v) for v in row) + " |"
                for row in d.itertuples(index=False, name=None)]
        return "\n".join([head, sep, *rows])

    cols = ["method", "ds", "mask", "cell", "ωh_orac", "ωh_est", "Δωh",
            "g_cong", "ari"]
    ren = {"dataset": "ds", "omega_h_oracle": "ωh_orac", "omega_h_est": "ωh_est",
           "omega_bias": "Δωh", "g_congruence": "g_cong", "domain_ari": "ari"}
    if "imp_r2" in df:
        ren["imp_r2"] = "wr2"
        cols.append("wr2")
    df["dataset"] = df["dz"] + "_" + ST
    raw = ["method", "mask", "cell", "dataset", "omega_h_oracle", "omega_h_est",
           "omega_bias", "g_congruence", "domain_ari"]
    if "imp_r2" in df:
        raw.append("imp_r2")
    if "imp_r2" in df:
        g = df[df.imp_r2 >= a.gate]
        print(f"== gated (imputation R² >= {a.gate}) — one row per run, no averaging ==")
    else:
        g = df
        print("== no imputation DB — UNGATED, one row per run ==")
    print(md(g.sort_values(["method", "dataset", "mask", "cell"])[raw]
             .rename(columns=ren)[cols]))
    print(f"\nwrote {FDIR / 'scored.csv'}")


if __name__ == "__main__":
    sys.exit(main())
