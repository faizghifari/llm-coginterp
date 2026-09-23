#!/usr/bin/env python3
"""Release-date analysis: does the factor structure move with model release date,
and does a benchmark's own release date relate to its place in that structure?

Two subcommands.

``run`` analyses one data/results root (one target density) and writes
everything to ``<results-root>/release_date/`` (or ``--out``):

  year_map.csv           collapse_key -> release year, from collapse_mapping.csv
                         (same parsing as factor.R --timed)
  coverage.csv           per densifier x cohort: models, observed share,
                         benchmarks with no observed score in the cohort
  cohort_fits.csv        per-cohort EFA plus size-matched random subsets
                         (src/run/release_cohorts.R; the slow step)
  cohort_vs_random.csv   each cohort's omega_h against its subsets' 5-95 % band
  gscore.csv             mean pooled general-factor score per cohort
  shared_benchmarks.csv  imputation-free check: first-eigenvalue share of the
                         observed correlations over benchmarks every cohort took
  bench_date.csv         benchmark release year vs the pooled solutions: g
                         loading, era cohesion, year gap vs shared missingness
  bench_pairs.csv        observed pairwise correlations vs release-year gap
  summary.md             all of the above as tables

``report`` combines several ``run`` outputs (e.g. the 10 % and 20 % densities)
into the paper's Appendix tables and the omega_h figure.

    uv run python scripts/release_date.py run                     # text-only defaults
    uv run python scripts/release_date.py run --data-root D --results-root R
    uv run python scripts/release_date.py report --run 10%=R10/release_date \\
        --run 20%=R20/release_date --out results/text_only/release_date_report

Methods: cohorts use the row-preserving imputers only (surrogate-emitting ones
emit rows that are not the labelled models), benchmark-date uses every pooled
solution except the fill-smooth ones. Both apply factor.R's own R2_GATE.
Everything is seeded per cell, so reruns reproduce exactly.

The shared imputation assumption and its caveat are written up in the paper's
release-date appendix: the matrix is imputed once over all models, then split.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

REPO = Path(__file__).resolve().parents[1]
FACTOR_R = REPO / "src" / "run" / "factor.R"
COHORT_R = REPO / "src" / "run" / "release_cohorts.R"

# Keep COHORTS in step with bins() in src/run/release_cohorts.R.
COHORTS = {2022: "<=2022", 2023: "2023", 2024: "2024", 2025: ">=2025"}
ERAS = [(0, 2019, "<=2019"), (2020, 2021, "2020-21"), (2022, 2023, "2022-23"), (2024, 9999, ">=2024")]
ROW_METHODS = ["softimpute", "missforest", "knn", "mice", "iterativepca"]
FILL_SMOOTH = {"default", "zeros"}
YEAR_RX = r"((?:19|20)\d\d)"
MIN_SHARED = 10      # observed models per cohort for a benchmark to count as shared
MIN_PAIR = 30        # co-observed models for an observed pairwise correlation
BOOT = 1000
ERA_PERM = 2000
CONFOUND_PERM = 500
METHOD_LABEL = {"softimpute": "SoftImpute", "missforest": "missForest", "knn": "k-NN",
                "mice": "MICE", "iterativepca": "IterativePCA", "onesidedmc": "OneSidedMC",
                "softimpute_corr": "SoftImpute (corr.)", "optspace": "OptSpace",
                "usvt": "USVT", "cvxr": "CVXR", "ggm": "GGM"}


def cohort_of(year: float) -> int | None:
    if pd.isna(year):
        return None
    return 2022 if year <= 2022 else 2025 if year >= 2025 else int(year)


def era_of(year: float) -> str:
    for lo, hi, lab in ERAS:
        if lo <= year <= hi:
            return lab
    raise ValueError(year)


def parse_year(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype("string").str.extract(YEAR_RX)[0], errors="coerce")


def r2_gate() -> float:
    m = re.search(r"^R2_GATE\s*<-\s*([0-9.]+)", FACTOR_R.read_text(), re.M)
    if not m:
        sys.exit(f"could not read R2_GATE from {FACTOR_R}")
    return float(m.group(1))


def seed_of(*parts) -> int:
    """Stable per-cell seed, independent of loop order."""
    return zlib.crc32("|".join(map(str, parts)).encode())


# ── inputs ───────────────────────────────────────────────────────────────────
def year_map(data: Path, st: str) -> pd.Series:
    path = data / "combinations" / st / "collapse_mapping.csv"
    if not path.exists():
        sys.exit(f"{path} missing: run `make preproc` first")
    m = pd.read_csv(path, dtype=str).drop_duplicates("collapse_key")
    return pd.Series(parse_year(m["release_date"]).values, index=m["collapse_key"], name="year")


def bench_years(data: Path) -> pd.Series:
    b = pd.read_csv(data / "benchmarks.csv", dtype=str)
    y = parse_year(b["release_date"]).fillna(parse_year(b["year"]))
    return pd.Series(y.values, index=b["benchmark_id"]).dropna()


def sparse(data: Path, dz: str, st: str) -> pd.DataFrame:
    return pd.read_csv(data / f"combinations_{dz}" / st / "model_benchmark_table.csv", index_col=0)


def completed(data: Path, m: str, dz: str, st: str) -> Path:
    return data / "imputed" / m / dz / st / "imputed_model_benchmark_table.csv"


def pooled_loadings(res: Path, m: str, dz: str, st: str) -> Path:
    return res / m / f"{m}_{dz}_{st}_bifactor_pa_loadings.csv"


def imputation_r2(res: Path) -> dict:
    db = res / "database.db"
    if not db.exists():
        return {}
    with sqlite3.connect(db) as c:
        im = pd.read_sql("select dataset, method, r2 from imputation", c)
    return {(m, d): r for m, d, r in zip(im.method, im.dataset, im.r2)}


# ── analyses ─────────────────────────────────────────────────────────────────
def coverage(M: pd.DataFrame, cohort: pd.Series) -> list[dict]:
    rows = []
    for k, lab in COHORTS.items():
        S = M.loc[cohort.index[cohort == k]].notna()
        rows.append(dict(cohort=lab, models=len(S), observed=S.values.mean() if len(S) else np.nan,
                         unobserved_benchmarks=int((S.sum() == 0).sum())))
    return rows


def cohort_vs_random(fits: pd.DataFrame) -> pd.DataFrame:
    obs = fits[fits.rep == 0].set_index(["dz", "method", "cohort"])
    nul = fits[fits.rep > 0].groupby(["dz", "method", "cohort"])
    t = obs[["n", "p", "nf", "omega_h", "tucker_g"]].join(
        nul.omega_h.quantile(.05).rename("omega_lo")).join(
        nul.omega_h.quantile(.95).rename("omega_hi")).join(
        nul.tucker_g.median().rename("tucker_random")).join(
        nul.omega_h.apply(lambda s: s.isna().sum()).rename("random_failed"))
    t["flag"] = np.where(t.omega_h > t.omega_hi, "above",
                         np.where(t.omega_h < t.omega_lo, "below", ""))
    return t.reset_index()


def gscore(X: pd.DataFrame, g: pd.Series, cohort: pd.Series) -> dict:
    """Loading-weighted sum of standardised completed scores (mean 0 over models)."""
    X = X[g.index]
    s = ((X - X.mean()) / X.std()) @ g / (g ** 2).sum()
    c = cohort.reindex(s.index)
    return {lab: s[c == k].mean() for k, lab in COHORTS.items()}


def shared_benchmarks(M: pd.DataFrame, cohort: pd.Series, seed: int) -> tuple[list, list[dict]]:
    c = cohort.reindex(M.index)
    cnt = pd.DataFrame({k: M[c == k].notna().sum() for k in COHORTS})
    keep = list(cnt.index[(cnt >= MIN_SHARED).all(axis=1)])
    rows = []
    if len(keep) < 2:
        return keep, rows
    O = M[keep].dropna()
    cc = c.reindex(O.index)
    rng = np.random.default_rng(seed)

    def stats(X):
        R = np.corrcoef(X.T)
        return np.linalg.eigvalsh(R)[-1] / len(R), R[np.triu_indices(len(R), 1)].mean()

    for k, lab in COHORTS.items():
        X = O[cc == k].values
        if len(X) < 8:
            rows.append(dict(cohort=lab, models=len(X)))
            continue
        share, mean_r = stats(X)
        bs = [stats(X[rng.integers(0, len(X), len(X))])[0] for _ in range(BOOT)]
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        rows.append(dict(cohort=lab, models=len(X), first_eig_share=share, ci_lo=lo, ci_hi=hi,
                         mean_r=mean_r))
    return keep, rows


def era_cohesion(L: np.ndarray, eras: np.ndarray, seed: int) -> tuple[float, float]:
    """Within-era mean Euclidean loading distance, z and p against permuted eras."""
    D = np.sqrt(((L[:, None, :] - L[None, :, :]) ** 2).sum(-1))

    def within(lab):
        tot, cnt = 0.0, 0
        for e in np.unique(lab):
            idx = np.where(lab == e)[0]
            if len(idx) > 1:
                tot += D[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)].sum()
                cnt += len(idx) * (len(idx) - 1) // 2
        return tot / cnt
    rng = np.random.default_rng(seed)
    obs = within(eras)
    null = np.array([within(rng.permutation(eras)) for _ in range(ERA_PERM)])
    return (obs - null.mean()) / null.std(), (null <= obs).mean()


def _rank_coefs(y, a, b):
    y, a, b = (rankdata(v) for v in (y, a, b))
    z = lambda v: (v - v.mean()) / v.std()
    X = np.column_stack([np.ones_like(a), z(a), z(b)])
    return np.linalg.lstsq(X, z(y), rcond=None)[0][1:]


def gap_vs_coobs(L: np.ndarray, years: np.ndarray, O: np.ndarray, seed: int) -> dict:
    """Rank regression of pairwise loading distance on release-year gap and on the
    Jaccard distance between the two benchmarks' observed-model sets; permutation p
    for the year-gap term (benchmarks relabelled jointly)."""
    n = len(L)
    D = np.sqrt(((L[:, None] - L[None]) ** 2).sum(-1))
    G = np.abs(years[:, None] - years[None])
    inter = O.T @ O
    cnt = O.sum(0)
    J = 1 - inter / (cnt[:, None] + cnt[None] - inter)
    iu = np.triu_indices(n, 1)
    b_gap, b_co = _rank_coefs(D[iu], G[iu], J[iu])
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(CONFOUND_PERM):
        p = rng.permutation(n)
        null.append(_rank_coefs(D[iu], G[np.ix_(p, p)][iu], J[iu])[0])
    return dict(beta_year_gap=b_gap, p_year_gap=(np.array(null) >= b_gap).mean(), beta_coobs=b_co)


def observed_pairs(M: pd.DataFrame, years: pd.Series) -> dict:
    cols = [c for c in M.columns if c in years.index]
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            m = M[a].notna() & M[b].notna()
            if m.sum() >= MIN_PAIR and M.loc[m, a].std() > 0 and M.loc[m, b].std() > 0:
                rows.append((abs(years[a] - years[b]), spearmanr(M.loc[m, a], M.loc[m, b])[0]))
    if not rows:
        return dict(pairs=0)
    pr = pd.DataFrame(rows, columns=["gap", "r"])
    rho, p = spearmanr(pr.gap, pr.r)
    return dict(pairs=len(pr), rho_gap_r=rho, p=p, r_gap_le1=pr.r[pr.gap <= 1].mean(),
                r_gap_2_3=pr.r[pr.gap.between(2, 3)].mean(), r_gap_ge4=pr.r[pr.gap >= 4].mean())


# ── run ──────────────────────────────────────────────────────────────────────
def cmd_run(a) -> None:
    data, res = REPO / a.data_root, REPO / a.results_root
    out = REPO / a.out if a.out else res / "release_date"
    out.mkdir(parents=True, exist_ok=True)
    dzs, st = a.dz.split(","), a.strategy
    gate, r2 = r2_gate(), imputation_r2(res)
    passes = lambda m, dz: r2.get((m, f"{dz}_{st}"), -np.inf) >= gate

    ym = year_map(data, st)
    ym.to_frame().rename_axis("collapse_key").to_csv(out / "year_map.csv")
    coh = ym.map(cohort_of).dropna()
    by = bench_years(data)
    print(f"release_date: {ym.notna().sum()}/{len(ym)} collapse keys dated, R2 gate {gate}", flush=True)

    # which methods enter
    cohort_methods = {dz: [m for m in ROW_METHODS if passes(m, dz)
                           and completed(data, m, dz, st).exists()
                           and pooled_loadings(res, m, dz, st).exists()] for dz in dzs}
    pooled_methods = {dz: sorted(p.name for p in res.iterdir() if p.is_dir()
                                 and p.name not in FILL_SMOOTH and passes(p.name, dz)
                                 and pooled_loadings(res, p.name, dz, st).exists()) for dz in dzs}
    for dz in dzs:
        print(f"  {dz}: cohorts {cohort_methods[dz]}; benchmark-date {pooled_methods[dz]}", flush=True)

    # coverage, gscore, shared benchmarks, benchmark date
    cov, gs, shared, bench, pairs = [], [], [], [], []
    for dz in dzs:
        M = sparse(data, dz, st)
        dated = ym.reindex(M.index).notna().mean()
        for r in coverage(M, coh.reindex(M.index).dropna()):
            cov.append(dict(dz=dz, benchmarks=M.shape[1], dated=dated, **r))
        for m in cohort_methods[dz]:
            g = pd.read_csv(pooled_loadings(res, m, dz, st), index_col=0)["g"]
            X = pd.read_csv(completed(data, m, dz, st), index_col=0)
            gs.append(dict(dz=dz, method=m, **gscore(X, g, coh)))
        keep, rows = shared_benchmarks(M, coh, seed_of("shared", dz))
        shared += [dict(dz=dz, benchmarks=" ".join(keep), **r) for r in rows]

        yr = by.reindex(M.columns).dropna()
        mean_model_year = pd.Series({c: ym.reindex(M.index)[M[c].notna()].mean() for c in yr.index})
        pairs.append(dict(dz=dz, benchmark_years=f"{int(yr.min())}-{int(yr.max())}",
                          rho_bench_year_model_year=spearmanr(yr, mean_model_year[yr.index])[0],
                          **observed_pairs(M, yr)))
        for m in pooled_methods[dz]:
            L = pd.read_csv(pooled_loadings(res, m, dz, st), index_col=0)
            L = L[L.index.isin(yr.index)]
            y = yr[L.index]
            fac = [c for c in L.columns if c == "g" or c.startswith("F")]
            rho, p = spearmanr(y, L["g"].abs())
            row = dict(dz=dz, method=m, benchmarks=len(L), rho_year_absg=rho, p_year_absg=p)
            eras = np.array([era_of(v) for v in y])
            for variant, cs in [("with_g", fac), ("without_g", [c for c in fac if c != "g"])]:
                z, pz = era_cohesion(L[cs].values, eras, seed_of("era", dz, m, variant))
                row[f"era_z_{variant}"], row[f"era_p_{variant}"] = z, pz
            O = M[L.index].notna().values.astype(float)
            row.update(gap_vs_coobs(L[fac].values, y.values, O, seed_of("gap", dz, m)))
            bench.append(row)

    pd.DataFrame(cov).to_csv(out / "coverage.csv", index=False)
    pd.DataFrame(gs).to_csv(out / "gscore.csv", index=False)
    pd.DataFrame(shared).to_csv(out / "shared_benchmarks.csv", index=False)
    pd.DataFrame(bench).to_csv(out / "bench_date.csv", index=False)
    pd.DataFrame(pairs).to_csv(out / "bench_pairs.csv", index=False)

    # cohort EFA (R)
    methods = sorted({m for v in cohort_methods.values() for m in v})
    if a.skip_efa and (out / "cohort_fits.csv").exists():
        print("  --skip-efa: reusing cohort_fits.csv")
    elif methods:
        cmd = ["Rscript", str(COHORT_R), "--data-root", str(data), "--results-root", str(res),
               "--year-map", str(out / "year_map.csv"), "--out", str(out),
               "--methods", ",".join(methods), "--dz", a.dz, "--strategy", st,
               "--reps", str(a.reps)] + (["--cores", str(a.cores)] if a.cores else [])
        log = out / "release_cohorts.log"
        print(f"  cohort EFA for {methods} ({a.reps} random subsets per cohort), log: {log}", flush=True)
        with open(log, "w") as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
        if rc:
            sys.exit(f"release_cohorts.R failed (exit {rc}); see {log}")
    fits = pd.read_csv(out / "cohort_fits.csv")
    # a method can pass the gate on one densifier and not the other
    fits = fits[[m in cohort_methods.get(d, []) for m, d in zip(fits.method, fits.dz)]]
    fits["cohort"] = fits.cohort.map(COHORTS)
    cvr = cohort_vs_random(fits)
    cvr.to_csv(out / "cohort_vs_random.csv", index=False)

    write_summary(out, gate, st, cov, cvr, fits, gs, shared, bench, pairs)
    plot_omega([("", out)], out / "release_cohort_omega.png")
    print(f"wrote {out}")


def fmt(df: pd.DataFrame, digits: int = 2) -> str:
    """Markdown table without the optional tabulate dependency."""
    if not len(df):
        return "(none)"
    df = df.round(digits).astype(object).where(df.notna(), "")
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + " --- |" * df.shape[1]]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def write_summary(out, gate, st, cov, cvr, fits, gs, shared, bench, pairs) -> None:
    rnd = fits[fits.rep > 0]
    n_out = (cvr.flag != "").sum()
    s = [f"# Release-date analysis ({st}, R2 gate {gate})", "",
         "## Coverage per cohort", fmt(pd.DataFrame(cov), 3), "",
         "## Cohort omega_h against size-matched random subsets",
         f"{n_out} of {len(cvr)} cohort fits outside the 5-95 % band "
         f"({(cvr.flag == 'above').sum()} above, {(cvr.flag == 'below').sum()} below). "
         f"Median congruence of cohort g with pooled g: {cvr.tucker_g.median():.2f}, "
         f"random subsets: {rnd.tucker_g.median():.2f}.", "", fmt(cvr, 3), "",
         "## Mean general-factor score per cohort", fmt(pd.DataFrame(gs)), "",
         "## Imputation-free check (benchmarks observed in every cohort)",
         fmt(pd.DataFrame(shared)), "",
         "## Benchmark release date against pooled solutions", fmt(pd.DataFrame(bench), 3), "",
         "## Observed pairwise correlations against release-year gap", fmt(pd.DataFrame(pairs), 3), ""]
    (out / "summary.md").write_text("\n".join(s))


# ── report ───────────────────────────────────────────────────────────────────
def plot_omega(runs: list[tuple[str, Path]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fits = {lab: pd.read_csv(d / "cohort_vs_random.csv") for lab, d in runs}
    raw = {lab: pd.read_csv(d / "cohort_fits.csv") for lab, d in runs}
    dzs = sorted({dz for f in fits.values() for dz in f.dz})
    meths = [m for m in ROW_METHODS if any(m in set(f.method) for f in fits.values())]
    colors = dict(zip(ROW_METHODS, ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b"]))
    fig, axes = plt.subplots(len(runs), len(dzs), figsize=(3.5 * len(dzs), 2.5 * len(runs)),
                             sharey=True, squeeze=False)
    for i, (lab, _) in enumerate(runs):
        for j, dz in enumerate(dzs):
            ax = axes[i, j]
            f, r = fits[lab], raw[lab]
            for q, m in enumerate(meths):
                off = (q - (len(meths) - 1) / 2) * 0.22
                for x, (k, cl) in enumerate(COHORTS.items()):
                    row = f[(f.dz == dz) & (f.method == m) & (f.cohort == cl)]
                    if row.empty:
                        continue
                    ax.plot([x + off] * 2, [row.omega_lo.iloc[0], row.omega_hi.iloc[0]],
                            color=colors[m], alpha=.35, lw=5, solid_capstyle="butt")
                    ax.plot(x + off, row.omega_h.iloc[0], "o", color=colors[m], ms=5,
                            label=METHOD_LABEL[m] if x == 0 else None)
            ax.set_xticks(range(len(COHORTS)), [c.replace("<=", "≤").replace(">=", "≥")
                                                for c in COHORTS.values()])
            ax.set_title(f"{lab} target density, {dz}" if lab else dz, fontsize=10)
            ax.grid(axis="y", alpha=.3)
        axes[i, 0].set_ylabel(r"$\omega_h$")
    axes[-1, -1].legend(fontsize=7, loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def cmd_report(a) -> None:
    runs = []
    for spec in a.run:
        lab, sep, d = spec.partition("=")
        runs.append((lab, REPO / d) if sep else ("", REPO / spec))
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    tex = lambda c: {"<=2022": r"$\leq 2022$", ">=2025": r"$\geq 2025$"}.get(c, c)
    L = lambda m: METHOD_LABEL.get(m, m)
    cl = list(COHORTS.values())
    md = ["# Release-date tables", ""]

    md += ["## Coverage (models / observed / benchmarks with no observed score)", "",
           "| Density | Densifier | Benchmarks | " + " | ".join(map(tex, cl)) + " |",
           "| --- | --- | ---: |" + " --- |" * len(cl)]
    for lab, d in runs:
        c = pd.read_csv(d / "coverage.csv")
        for dz, g in c.groupby("dz", sort=False):
            g = g.set_index("cohort")
            cells = [f"{g.loc[k, 'models']}/{g.loc[k, 'observed']:.0%}/{g.loc[k, 'unobserved_benchmarks']}"
                     for k in cl]
            md.append(f"| {lab} | {dz} | {g.benchmarks.iloc[0]} | " + " | ".join(cells) + " |")

    md += ["", "## Cohorts against random subsets", "",
           "| Density | Cohort fits | Above | Below | Congruence, cohorts | Congruence, random subsets |",
           "| --- | ---: | ---: | ---: | ---: | ---: |"]
    counts = []
    for lab, d in runs:
        v = pd.read_csv(d / "cohort_vs_random.csv")
        f = pd.read_csv(d / "cohort_fits.csv")
        f = f[f.rep > 0].merge(v[["dz", "method"]].drop_duplicates(), on=["dz", "method"])
        above, below = (v.flag == "above").sum(), (v.flag == "below").sum()
        md.append(f"| {lab} | {len(v)} | {above} | {below} | {v.tucker_g.median():.2f} | "
                  f"{f.tucker_g.median():.2f} |")
        counts.append(f"{lab}: {above + below} of {len(v)} cohort fits outside the band")

    md += ["", "## Mean general-factor score", "",
           "| Density | Densifier | Imputer | " + " | ".join(map(tex, cl)) + " |",
           "| --- | --- | --- |" + " ---: |" * len(cl)]
    for lab, d in runs:
        for _, r in pd.read_csv(d / "gscore.csv").iterrows():
            md.append(f"| {lab} | {r.dz} | {L(r.method)} | " + " | ".join(f"{r[k]:.2f}" for k in cl) + " |")

    md += ["", "## Imputation-free check", ""]
    # The observed scores do not depend on the densifier once the shared benchmarks
    # and their complete-case models coincide, so identical tables are shown once.
    tables = {}
    for lab, d in runs:
        s = pd.read_csv(d / "shared_benchmarks.csv")
        for dz, g in s.groupby("dz", sort=False):
            body = ["| Cohort | Models | First-eigenvalue share | 95% interval | Mean correlation |",
                    "| --- | ---: | ---: | --- | ---: |"]
            for _, r in g.iterrows():
                if pd.isna(r.get("first_eig_share")):
                    body.append(f"| {tex(r.cohort)} | {r.models} | | | |")
                else:
                    body.append(f"| {tex(r.cohort)} | {r.models} | {r.first_eig_share:.2f} | "
                                f"[{r.ci_lo:.2f}, {r.ci_hi:.2f}] | {r.mean_r:.2f} |")
            key = (g.benchmarks.iloc[0], "\n".join(body))
            tables.setdefault(key, []).append(f"{lab} {dz}")
    for (bench, body), where in tables.items():
        md += [f"{', '.join(where)}: {bench}", "", body, ""]

    md += ["## Benchmark release date", "",
           "| Density | Densifier | Imputer | $\\rho$ | Era $z$ | Year gap ($p$) | Co-obs. |",
           "| --- | --- | --- | ---: | ---: | --- | ---: |"]
    allb = []
    order = {m: i for i, m in enumerate(METHOD_LABEL)}
    for lab, d in runs:
        b = pd.read_csv(d / "bench_date.csv")
        b = b.sort_values(["dz", "method"], key=lambda c: c.map(order) if c.name == "method" else c)
        allb.append(b)
        for _, r in b.iterrows():
            md.append(f"| {lab} | {r.dz} | {L(r.method)} | {r.rho_year_absg:.2f} | "
                      f"{r.era_z_with_g:.1f} | {r.beta_year_gap:.2f} ({r.p_year_gap:.3f}) | "
                      f"{r.beta_coobs:.2f} |")
    b = pd.concat(allb)
    sig = b.p_year_absg < .05
    counts += [f"g loading vs year significant in {sig.sum()} of {len(b)} "
               f"({(sig & (b.rho_year_absg > 0)).sum()} positive)",
               f"era z below -1.96 in {(b.era_z_with_g < -1.96).sum()} of {len(b)}",
               f"co-observation term larger in {(b.beta_coobs > b.beta_year_gap).sum()} of {len(b)} "
               f"(co-obs {b.beta_coobs.min():.2f} to {b.beta_coobs.max():.2f}, "
               f"year gap {b.beta_year_gap.min():.2f} to {b.beta_year_gap.max():.2f})",
               f"year gap p < 0.05 in {(b.p_year_gap < .05).sum()} of {len(b)}"]
    md += ["", "## Observed pairwise correlations", ""]
    for lab, d in runs:
        md += [f"{lab}", "", fmt(pd.read_csv(d / "bench_pairs.csv")), ""]
    md += ["## Counts quoted in the text", ""] + [f"- {c}" for c in counts]

    (out / "release_date_tables.md").write_text("\n".join(md) + "\n")
    plot_omega(runs, out / "release-cohort-omega.png")
    print(f"wrote {out / 'release_date_tables.md'} and {out / 'release-cohort-omega.png'}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="analyse one data/results root")
    r.add_argument("--data-root", default="data/text_only")
    r.add_argument("--results-root", default="results/text_only")
    r.add_argument("--out", help="default <results-root>/release_date")
    r.add_argument("--dz", default="C,S")
    r.add_argument("--strategy", default="all_standard")
    r.add_argument("--reps", type=int, default=50, help="random subsets per cohort")
    r.add_argument("--cores", type=int, help="R worker cores (default: all but one)")
    r.add_argument("--skip-efa", action="store_true", help="reuse an existing cohort_fits.csv")
    r.set_defaults(func=cmd_run)
    p = sub.add_parser("report", help="combine run outputs into the paper tables and figure")
    p.add_argument("--run", action="append", required=True, metavar="LABEL=DIR",
                   help="a run output dir with its label, e.g. 10%%=results/x/release_date")
    p.add_argument("--out", default="results/text_only/release_date_report")
    p.set_defaults(func=cmd_report)
    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
