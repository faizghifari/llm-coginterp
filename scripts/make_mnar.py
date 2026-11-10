"""Generate synthetic MNAR stress-test datasets under data/mnar/.

For every real (densifier x strategy) table in data/text_only/ we plant a
ground-truth benchmark correlation structure (bifactor-shaped), draw a
complete Gaussian surrogate matrix with the SAME rows, columns and keys as
the real table, then punch holes under several missingness mechanisms whose
expected row AND column observation counts are calibrated to match the real
table. Every scenario masks the SAME planted complete matrices; only the
mask differs, so downstream differences are attributable to the mechanism.

Mask model, per cell (i, j):
    logit P(obs_ij) = a_i + b_j + s_ij
(a_i, b_j) are fit by iterative bisection so that expected row/column
observation counts equal the real table's. Pure main effects are therefore
absorbed by the offsets; the scenario term s_ij is the MNAR part.

Scenarios (name, mechanism, parameter):
    mcar           s = 0                          popularity-only control
    cell_g05/10    s = gam * z_ij                 cells reported only when good
    ability_d05/10 s = del * niche_j * g_i        niche benchmarks gatekeep
                                                  on model strength
    benchmaxx_k05/10 s = kap * eta_i,f(j)         models observed on their
                                                  strong suites; highly
                                                  correlated benchmarks
                                                  co-occur
    combined       0.5 * (cell + ability + benchmaxx)
    real_mask      no model: the real mask itself, planted values underneath

Outputs (masked roots and truth are stored separately):
    data/mnar/truth/<dz>_<st>_complete.csv      unmasked complete matrix
    data/mnar/truth/<dz>_<st>_correlation.csv   planted true correlation
    data/mnar/truth/<dz>_<st>_factors.npz       g_i, eta_i,f, group per column
    data/mnar/truth/benchmark_params.csv        per-benchmark planted params
    data/mnar/<scenario>/s<seed>/combinations[_<dz>]/<st>/model_benchmark_table.csv
    data/mnar/<scenario>/s<seed>/truth/<dz>_<st>_mask.npz + _meta.json
    data/mnar/summary.csv                       one row per generated table

Each <scenario>/s<seed> directory is a drop-in --data-root for
src/run/impute.R / factor.R. Pair-level diagnostics are computed for the
first mask seed only (they are seed-stable summaries of the mechanism).

Run:  uv run python scripts/make_mnar.py
The imputation evaluation is deferred: point the pipeline at
data/mnar/<scenario>/s<seed> and score against data/mnar/truth/.
"""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import densify  # KEY convention only

REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "data" / "text_only"
OUT = REPO / "data" / "mnar"
KEY = densify.KEY

TRUTH_SEED = 20261110
MASK_SEEDS = [0, 1, 2]
N_GROUPS = 5
LAM_G = (0.55, 0.80)          # general-factor loading range
LAM_F = (0.35, 0.60)          # group-factor loading range
MAX_COMMUNALITY = 0.85        # cap lam_g^2 + lam_f^2
MIN_COOBS = 10                # min co-observations for a pair diagnostic
IPF_SWEEPS = 10
IPF_BISECT = 40

DENSIFIERS = [("raw", "combinations"), ("C", "combinations_C"),
              ("S", "combinations_S"), ("R", "combinations_R")]
STRATEGIES = ["all_standard", "all_aggressive"]

SCENARIOS = [
    ("mcar", None, 0.0),
    ("cell_g05", "cell", 0.5),
    ("cell_g10", "cell", 1.0),
    ("ability_d05", "ability", 0.5),
    ("ability_d10", "ability", 1.0),
    ("benchmaxx_k05", "benchmaxx", 0.5),
    ("benchmaxx_k10", "benchmaxx", 1.0),
    ("combined", "combined", 0.5),
    ("real_mask", "real", 0.0),
]


# ---------------------------------------------------------------- helpers --

def load_table(path):
    """Read a model x benchmark table -> (keys, cols, matrix with NaN holes)."""
    df = pl.read_csv(path)
    keys = df[KEY].to_list()
    cols = [c for c in df.columns if c != KEY]
    M = df.select(cols).to_numpy().astype(np.float64)
    M[~np.isfinite(M)] = np.nan
    return keys, cols, M


def standardize(v):
    v = np.asarray(v, dtype=np.float64)
    sd = v.std()
    if sd < 1e-12:
        return np.zeros_like(v)
    return (v - v.mean()) / sd


def benchmark_params(names):
    """Global planted parameters, keyed by benchmark name (shared across
    tables so benchmarks appearing in several tables get consistent truth)."""
    rng = np.random.default_rng(TRUTH_SEED)
    n = len(names)
    perm = rng.permutation(n)
    group = np.zeros(n, dtype=np.int64)
    for gi, chunk in enumerate(np.array_split(perm, N_GROUPS)):
        group[chunk] = gi
    lam_g = rng.uniform(*LAM_G, size=n)
    lam_f = rng.uniform(*LAM_F, size=n)
    communal = lam_g**2 + lam_f**2
    over = communal > MAX_COMMUNALITY
    lam_g[over] *= np.sqrt(MAX_COMMUNALITY / communal[over])
    lam_f[over] *= np.sqrt(MAX_COMMUNALITY / communal[over])
    return {"names": list(names), "group": group,
            "lam_g": lam_g, "lam_f": lam_f}


def plant_truth(seed, keys, cols, bp):
    """Complete surrogate matrix + planted correlation for one table.

    z_ij = lam_g_j g_i + lam_f_j eta_i,f(j) + sqrt(psi_j) eps_ij, factors
    independent; the population correlation of the columns is exactly
    R = lam_g lam_g' + F F' (+ unit diagonal).
    """
    rng = np.random.default_rng(seed)
    n, p = len(keys), len(cols)
    idx = {name: i for i, name in enumerate(bp["names"])}
    ci = np.array([idx[c] for c in cols])
    lg = bp["lam_g"][ci]
    lf = bp["lam_f"][ci]
    fcols = bp["group"][ci]

    g = standardize(rng.normal(size=n))
    eta = rng.normal(size=(n, N_GROUPS))
    for f in range(N_GROUPS):
        eta[:, f] = standardize(eta[:, f])
    psi = np.maximum(1.0 - lg**2 - lf**2, 1e-6)

    z = (g[:, None] * lg[None, :]
         + eta[:, fcols] * lf[None, :]
         + rng.normal(scale=np.sqrt(psi)[None, :], size=(n, p)))

    Fmat = np.zeros((p, N_GROUPS))
    Fmat[np.arange(p), fcols] = lf
    R = np.outer(lg, lg) + Fmat @ Fmat.T
    np.fill_diagonal(R, 1.0)

    X = 50.0 + 15.0 * z  # cosmetic 0-100-ish scale; correlations unchanged
    return {"X": X, "z": z, "R": R, "g": g, "eta": eta, "fcols": fcols}


def mechanism_term(kind, param, z, g, eta_mapped, niche):
    """s_ij for the mask model (the part NOT absorbed by the calibrated
    row/column offsets)."""
    if kind is None:
        return np.zeros_like(z)
    if kind == "cell":
        return param * z
    if kind == "ability":
        return param * niche[None, :] * g[:, None]
    if kind == "benchmaxx":
        return param * eta_mapped
    if kind == "combined":
        return (0.5 * z
                + 0.5 * niche[None, :] * g[:, None]
                + 0.5 * eta_mapped)
    raise ValueError(kind)


def fit_offsets(s, row_t, col_t):
    """Find per-row a_i and per-column b_j so that the expected observation
    counts of expit(a_i + b_j + s_ij) match the real table's margins."""
    n, p = s.shape
    row_t = np.maximum(row_t, 0.05)
    col_t = np.maximum(col_t, 0.05)
    a = np.zeros(n)
    b = np.zeros(p)
    for sweep in range(1, IPF_SWEEPS + 1):
        base = b[None, :] + s
        lo = np.full(n, -40.0)
        hi = np.full(n, 40.0)
        for _ in range(IPF_BISECT):
            mid = 0.5 * (lo + hi)
            e = expit(mid[:, None] + base).sum(1)
            hit = e > row_t
            lo = np.where(hit, mid, lo)
            hi = np.where(hit, hi, mid)
        a = 0.5 * (lo + hi)

        base = a[:, None] + s
        lo = np.full(p, -40.0)
        hi = np.full(p, 40.0)
        for _ in range(IPF_BISECT):
            mid = 0.5 * (lo + hi)
            e = expit(base + mid[None, :]).sum(0)
            hit = e > col_t
            lo = np.where(hit, mid, lo)
            hi = np.where(hit, hi, mid)
        b = 0.5 * (lo + hi)

        E = expit(a[:, None] + b[None, :] + s)
        if (np.abs(E.sum(1) - row_t).max() < 0.1
                and np.abs(E.sum(0) - col_t).max() < 0.1):
            break
    return a, b, sweep


def pairwise_obs_corr(z, mask):
    """Pairwise correlations of z on co-observed rows (what the corr-matrix
    imputers see), NaN where a pair is never co-observed or degenerate."""
    n, p = z.shape
    z2 = z * z
    R = np.full((p, p), np.nan)
    for j in range(p):
        m = mask[:, j]
        if not m.any():
            continue
        C = m[:, None] & mask
        cnt = C.sum(0)
        zj = np.where(m, z[:, j], 0.0)
        Czj = C * zj[:, None]
        s_x = Czj.sum(0)
        s_y = (C * z).sum(0)
        s_xy = (Czj * z).sum(0)
        s_yy = (C * z2).sum(0)
        s_xx = (Czj * zj[:, None]).sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mx = s_x / cnt
            my = s_y / cnt
            vx = s_xx / cnt - mx * mx
            vy = s_yy / cnt - my * my
            r = (s_xy / cnt - mx * my) / np.sqrt(vx * vy)
        R[j] = r
    return R


def diagnostics(mask, z, R, fcols, real_mask):
    p = mask.shape[1]
    d = {
        "density": float(mask.mean()),
        "row_mae": float(np.abs(mask.sum(1) - real_mask.sum(1)).mean()),
        "col_mae": float(np.abs(mask.sum(0) - real_mask.sum(0)).mean()),
        "mean_z_obs": float(z[mask].mean()) if mask.any() else None,
        "mean_z_miss": float(z[~mask].mean()) if (~mask).any() else None,
    }
    Robs = pairwise_obs_corr(z, mask)
    coobs = mask.T.astype(np.float64) @ mask
    same = fcols[:, None] == fcols[None, :]
    off = ~np.eye(p, dtype=bool)
    D = np.abs(Robs - R)
    usable = off & np.isfinite(D) & (coobs >= MIN_COOBS)
    within = usable & same
    cross = usable & ~same
    d["within_pair_bias"] = float(D[within].mean()) if within.any() else None
    d["cross_pair_bias"] = float(D[cross].mean()) if cross.any() else None
    d["within_coobs"] = float(coobs[same & off].mean())
    d["cross_coobs"] = float(coobs[~same].mean())
    d["n_pairs_within"] = int(within.sum())
    d["n_pairs_cross"] = int(cross.sum())
    return d


def write_masked_table(path, keys, cols, X, mask):
    Xm = np.where(mask, X, np.nan)
    data = {KEY: pl.Series(keys)}
    for j, c in enumerate(cols):
        data[c] = pl.Series(np.round(Xm[:, j], 4)).fill_nan(None)
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(data).write_csv(path)


def jwrite(path, obj):
    def cast(o):
        if hasattr(o, "item"):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    path.write_text(json.dumps(obj, indent=2, default=cast))


# ------------------------------------------------------------------- main --

def main():
    tables = []
    for dz, sub in DENSIFIERS:
        for st in STRATEGIES:
            p = SOURCE / sub / st / "model_benchmark_table.csv"
            if p.exists():
                tables.append((dz, st, sub, p))
    loaded = {(dz, st): load_table(p) for dz, st, _, p in tables}
    print(f"make_mnar: {len(tables)} source tables from {SOURCE}")

    all_bench = sorted({c for _, cols, _ in loaded.values() for c in cols})
    bp = benchmark_params(all_bench)

    truth_dir = OUT / "truth"
    truth_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({
        "benchmark": bp["names"],
        "group": bp["group"],
        "lam_g": bp["lam_g"],
        "lam_f": bp["lam_f"],
    }).write_csv(truth_dir / "benchmark_params.csv")

    truth_cache = {}
    summary = []

    for t_idx, (dz, st, sub, src_path) in enumerate(tables):
        keys, cols, M = loaded[(dz, st)]
        real_mask = ~np.isnan(M)
        n, p = real_mask.shape
        tag = f"{dz}_{st}"

        # ---- planted truth (shared by every scenario and seed) ----
        if (dz, st) not in truth_cache:
            truth = plant_truth(TRUTH_SEED + t_idx, keys, cols, bp)
            truth_cache[(dz, st)] = truth
            pl.DataFrame({KEY: pl.Series(keys),
                          **{c: pl.Series(np.round(truth["X"][:, j], 4))
                             for j, c in enumerate(cols)}}
                         ).write_csv(truth_dir / f"{tag}_complete.csv")
            Rdf = pl.DataFrame({KEY: pl.Series(cols),
                                **{c: pl.Series(truth["R"][:, j])
                                   for j, c in enumerate(cols)}})
            Rdf.write_csv(truth_dir / f"{tag}_correlation.csv")
            np.savez(truth_dir / f"{tag}_factors.npz",
                     g=truth["g"], eta=truth["eta"], fcols=truth["fcols"])
            jwrite(truth_dir / f"{tag}_meta.json", {
                "truth_seed": TRUTH_SEED + t_idx,
                "n_models": n, "n_benchmarks": p,
                "group_sizes": np.bincount(truth["fcols"],
                                           minlength=N_GROUPS).tolist(),
                "lam_g_range": LAM_G, "lam_f_range": LAM_F,
                "max_communality": MAX_COMMUNALITY,
            })
            print(f"[truth] {tag:22} {n}x{p} planted R + complete matrix")

        truth = truth_cache[(dz, st)]
        col_rate = real_mask.mean(0)
        niche = standardize(1.0 - col_rate)
        eta_mapped = truth["eta"][:, truth["fcols"]]

        for scen, kind, param in SCENARIOS:
            for seed in MASK_SEEDS:
                if kind == "real":
                    mask = real_mask.copy()
                    s = np.zeros_like(truth["z"])
                    sweeps = 0
                else:
                    s = mechanism_term(kind, param, truth["z"], truth["g"],
                                       eta_mapped, niche)
                    a, b, sweeps = fit_offsets(s, real_mask.sum(1),
                                               real_mask.sum(0))
                    P = expit(a[:, None] + b[None, :] + s)
                    rng = np.random.default_rng(seed)
                    mask = rng.random((n, p)) < P

                run_dir = OUT / scen / f"s{seed}"
                write_masked_table(
                    run_dir / sub / st / "model_benchmark_table.csv",
                    keys, cols, truth["X"], mask)
                run_truth = run_dir / "truth"
                run_truth.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(run_truth / f"{tag}_mask.npz", mask=mask)

                diag = (diagnostics(mask, truth["z"], truth["R"],
                                    truth["fcols"], real_mask)
                        if seed == MASK_SEEDS[0] else {})
                jwrite(run_truth / f"{tag}_meta.json", {
                    "scenario": scen, "mechanism": kind, "param": param,
                    "mask_seed": seed, "truth_tag": tag,
                    "truth_dir": str(truth_dir.relative_to(REPO)),
                    "ipf_sweeps": sweeps,
                    "n_models": n, "n_benchmarks": p,
                    "real_density": float(real_mask.mean()),
                    "diagnostics_note": ("pair-level diagnostics computed for "
                                         "the first mask seed only")
                    if diag else "see s0 run for diagnostics",
                    **diag,
                })
                row = {"scenario": scen, "seed": seed, "dz": dz,
                       "strategy": st, "n_models": n, "n_benchmarks": p,
                       "real_density": round(float(real_mask.mean()), 4),
                       **{k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in diag.items()}}
                summary.append(row)
                d = (f" density={row['density']}"
                     if "density" in row else "")
                print(f"[mask] {scen:14} s{seed} {tag:22}"
                      f"{d} row_mae={row.get('row_mae', '-')} "
                      f"col_mae={row.get('col_mae', '-')}")

    pl.DataFrame(summary).write_csv(OUT / "summary.csv")
    print(f"make_mnar: wrote {len(summary)} tables to {OUT} "
          f"(+ {OUT / 'summary.csv'})")


if __name__ == "__main__":
    main()
