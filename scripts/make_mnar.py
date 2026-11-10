"""Generate synthetic MNAR stress-test datasets under data/mnar/.

Design axes (no single planted truth: the truth grid spans weak-to-loud
factor structure so no imputer's inductive bias is favored by
construction; rankings are only reported per truth cell):

  truth cells : F group factors x group-loading range
                F in {5, 10}, loadings ~ U(0.1, 0.5) "weak" or
                U(0.5, 0.9) "loud" -> 4 planted truths per table
  masks       : mcar        calibrated control (s = 0)
                cell        good cells observed (s = gam * z_ij)
                popularity  observability tied to a benchmark's expected
                            correlation, interacting with model ability so
                            the effect survives offset absorption
                            (s = kap * rho_std_j * g_i)
                real_mask   the real table's own mask over planted truth
  seed        : one mask seed; common random numbers (the uniform behind
                the Bernoulli mask draw is identical across truth cells
                and masks for a fixed table), so every cross-condition
                comparison is paired
  tables      : C_all_standard (671x78), S_all_standard (669x124)

Total: 4 truths x 4 masks x 1 seed x 2 tables = 32 masked datasets.
All imputers see identical masked input per dataset, so imputer
comparisons are paired by construction.

Truth model per (table, truth cell) -- bifactor, all loadings positive so
R has only positive entries, matching the real tables:
    z_ij = lam_g_j g_i + lam_f_j eta_i,f(j) + sqrt(psi_j) eps_ij
    R    = lam_g lam_g' + F F'   (unit diagonal)
lam_g is solved per table so the planted mean off-diagonal R matches the
real table's observed mean pairwise correlation (co-observed rows);
lam_f ~ U(lo, hi) is drawn once per cell over the union of benchmark
names, so benchmarks in both tables get consistent truth. Communalities
are capped at 0.95 (both loadings scaled jointly); the ACHIEVED mean r,
loading ranges and omega are always recorded, never silently distorted.
If the loud cell's group contribution alone exceeds the observed target,
lam_g floors near 0 and the achieved mean r is reported as-is.

Mask model per entry (i, j):
    logit P(obs_ij) = a_i + b_j + s_ij
(a_i, b_j) are fit by damped iterated bisection so that expected row and
column observation counts match the real table's margins. Pure main
effects are absorbed by the offsets; s_ij is the MNAR part. Margin
targets are clipped to [0.05, p-0.05] per row (columns analogously), so a
row with zero real observations gains ~0.05 expected observations instead
of exactly zero. If a mechanism is too strong for the real margins to be
jointly feasible, s is scaled down by halves until calibration converges
(mech_scale < 1 in the meta/summary records how much survived).

Outputs:
    data/mnar/truth/<tag>_<cell>_complete.csv      unmasked complete matrix
    data/mnar/truth/<tag>_<cell>_correlation.csv   planted true correlation
    data/mnar/truth/<tag>_<cell>_factors.npz       g, eta, groups, loadings, psi
    data/mnar/truth/<tag>_<cell>_meta.json         targets, achieved, omega_h
    data/mnar/<cell>/<mask>/s0/<sub>/<st>/model_benchmark_table.csv
    data/mnar/<cell>/<mask>/s0/truth/<tag>_<cell>_mask.npz + _meta.json
    data/mnar/summary.csv                          one row per dataset

Each data/mnar/<cell>/<mask>/s0 directory is a drop-in --data-root for
src/run/impute.R / factor.R (containing the C and S all_standard tables).
True quantities for scoring: R, true_nfactors = 1 + F, and
omega_hierarchical = (sum lam_g)^2 / ((sum lam_g)^2 + sum(lam_f^2 + psi)).

Run:  uv run python scripts/make_mnar.py
The layout is incompatible with the previous 9-scenario output; remove an
old data/mnar first (rm -rf data/mnar).
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
MASK_SEED = 0

TRUTH_CELLS = [  # (n_group_factors, loading_lo, loading_hi, cell_name)
    (5, 0.10, 0.50, "f5_weak"),
    (5, 0.50, 0.90, "f5_loud"),
    (10, 0.10, 0.50, "f10_weak"),
    (10, 0.50, 0.90, "f10_loud"),
]
TABLES = [("C", "combinations_C", "all_standard"),
          ("S", "combinations_S", "all_standard")]
MASKS = [("mcar", None), ("cell", "cell"), ("popularity", "popularity"),
         ("real_mask", "real")]
CELL_GAMMA = 1.0               # cell mask strength
POP_KAPPA = 1.0                # popularity mask strength
MAX_COMMUNALITY = 0.95         # cap lam_g^2 + lam_f^2
MIN_COOBS = 10                 # min co-observations for a pair diagnostic
IPF_SWEEPS = 10
IPF_BISECT = 40
IPF_DAMP = 0.7                 # offset update damping per sweep
IPF_BOUND = 20.0               # |offset| clamp; saturation means infeasible


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


def planted_params(rng, names, nf, lo, hi):
    """Group assignments and group-factor loadings for one truth cell,
    drawn once over the union of benchmark names so shared benchmarks get
    consistent truth across tables."""
    n = len(names)
    perm = rng.permutation(n)
    group = np.zeros(n, dtype=np.int64)
    for gi, chunk in enumerate(np.array_split(perm, nf)):
        group[chunk] = gi
    lam_f = rng.uniform(lo, hi, size=n)
    return {"names": list(names), "group": group, "lam_f": lam_f}


def plant_truth(seed, keys, cols, pp, nf, target_mean_r):
    """Complete surrogate matrix + planted correlation for one (table,
    truth cell).

    lam_g is a constant solved so the planted mean off-diagonal R hits
    target_mean_r with the group-factor contribution included, then
    communalities are capped at MAX_COMMUNALITY (both loadings scaled
    jointly). Achieved values are computed from the final R and returned;
    nothing is assumed.
    """
    rng = np.random.default_rng(seed)
    n, p = len(keys), len(cols)
    idx = {name: i for i, name in enumerate(pp["names"])}
    ci = np.array([idx[c] for c in cols])
    lam_f = pp["lam_f"][ci]
    fcols = pp["group"][ci]

    # group-factor contribution to the mean off-diagonal correlation
    same = fcols[:, None] == fcols[None, :]
    off = ~np.eye(p, dtype=bool)
    c_group = float((lam_f[:, None] * lam_f[None, :] * (same & off)).sum()
                    / max(off.sum(), 1))
    lam_g = np.full(p, np.sqrt(max(target_mean_r - c_group, 1e-4)))

    communal = lam_g**2 + lam_f**2
    over = communal > MAX_COMMUNALITY
    shrink = np.sqrt(MAX_COMMUNALITY / communal[over])
    lam_g[over] *= shrink
    lam_f[over] *= shrink
    psi = np.maximum(1.0 - lam_g**2 - lam_f**2, 1e-6)

    g = standardize(rng.normal(size=n))
    eta = rng.normal(size=(n, nf))
    for f in range(nf):
        eta[:, f] = standardize(eta[:, f])
    z = (g[:, None] * lam_g[None, :]
         + eta[:, fcols] * lam_f[None, :]
         + rng.normal(scale=np.sqrt(psi)[None, :], size=(n, p)))

    Fmat = np.zeros((p, nf))
    Fmat[np.arange(p), fcols] = lam_f
    R = np.outer(lam_g, lam_g) + Fmat @ Fmat.T
    np.fill_diagonal(R, 1.0)

    mean_r = float(R[off].mean())
    omega_h = float(lam_g.sum()**2
                    / (lam_g.sum()**2 + (lam_f**2 + psi).sum()))
    X = 50.0 + 15.0 * z  # cosmetic 0-100-ish scale; correlations unchanged
    return {"X": X, "z": z, "R": R, "g": g, "eta": eta, "fcols": fcols,
            "lam_g": lam_g, "lam_f": lam_f, "psi": psi,
            "target_mean_r": target_mean_r, "mean_r": mean_r,
            "omega_h": omega_h}


def mechanism_term(kind, z, g, rho_std):
    """s_ij for the mask model (the part NOT absorbed by the calibrated
    row/column offsets)."""
    if kind is None:
        return np.zeros_like(z)
    if kind == "cell":
        return CELL_GAMMA * z
    if kind == "popularity":
        # high-expected-correlation benchmarks observed by able models;
        # the row x column interaction survives offset absorption
        return POP_KAPPA * rho_std[None, :] * g[:, None]
    raise ValueError(kind)


def fit_offsets(s, row_t, col_t):
    """Find per-row a_i and per-column b_j so that the expected observation
    counts of expit(a_i + b_j + s_ij) match the real table's margins.

    Damped and bounded: strong mechanism terms can make exact margin
    matching infeasible, in which case the undamped alternation oscillates
    and drives the offsets to saturation (P -> 1). Damping keeps the search
    stable; the caller scales s down and retries if this still fails.

    Targets are clipped to [0.05, p-0.05] (rows) / [0.05, n-0.05]
    (columns): a fully-missing real row gets a small positive target
    instead of 0, so the bisection stays well-posed; such rows gain ~0.05
    expected observations.
    """
    n, p = s.shape
    row_t = np.clip(row_t, 0.05, p - 0.05)
    col_t = np.clip(col_t, 0.05, n - 0.05)
    a = np.zeros(n)
    b = np.zeros(p)
    for sweep in range(1, IPF_SWEEPS + 1):
        base = b[None, :] + s
        lo = np.full(n, -IPF_BOUND)
        hi = np.full(n, IPF_BOUND)
        for _ in range(IPF_BISECT):
            mid = 0.5 * (lo + hi)
            e = expit(mid[:, None] + base).sum(1)
            hit = e > row_t          # expected too high -> root below mid
            hi = np.where(hit, mid, hi)
            lo = np.where(hit, lo, mid)
        a = IPF_DAMP * (0.5 * (lo + hi)) + (1 - IPF_DAMP) * a
        np.clip(a, -IPF_BOUND, IPF_BOUND, out=a)

        base = a[:, None] + s
        lo = np.full(p, -IPF_BOUND)
        hi = np.full(p, IPF_BOUND)
        for _ in range(IPF_BISECT):
            mid = 0.5 * (lo + hi)
            e = expit(base + mid[None, :]).sum(0)
            hit = e > col_t          # expected too high -> root below mid
            hi = np.where(hit, mid, hi)
            lo = np.where(hit, lo, mid)
        b = IPF_DAMP * (0.5 * (lo + hi)) + (1 - IPF_DAMP) * b
        np.clip(b, -IPF_BOUND, IPF_BOUND, out=b)

        E = expit(a[:, None] + b[None, :] + s)
        if (np.abs(E.sum(1) - row_t).max() < 0.1
                and np.abs(E.sum(0) - col_t).max() < 0.1):
            return a, b, sweep, True
    return a, b, IPF_SWEEPS, False


def calibrated_mask(s, row_t, col_t, rng):
    """Draw a mask under expit(a_i + b_j + s_ij) with margins calibrated to
    the real table. If the mechanism term is too strong for the margins to
    be jointly feasible, s is scaled down by halves until calibration
    converges; the applied scale is returned (1.0 = full strength)."""
    n, p = s.shape
    lam = 1.0
    for _ in range(7):
        a, b, sweeps, ok = fit_offsets(lam * s, row_t, col_t)
        if ok:
            break
        lam *= 0.5
    else:
        lam = 0.0  # s = 0 always calibrates: fall back to the control
        a, b, sweeps, ok = fit_offsets(np.zeros_like(s), row_t, col_t)
    P = expit(a[:, None] + b[None, :] + lam * s)
    return rng.random((n, p)) < P, lam, ok, sweeps


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


def observed_mean_r(M, real_mask):
    """Mean pairwise correlation of a real incomplete table over
    off-diagonal pairs with co-observations: the calibration target for
    the planted general factor."""
    R = pairwise_obs_corr(M, real_mask)
    p = R.shape[0]
    off = ~np.eye(p, dtype=bool)
    vals = R[off & np.isfinite(R)]
    return float(vals.mean()) if vals.size else 0.0


def margin_diagnostics(mask, real_mask, z):
    """Cheap O(n*p) per-mask stats: how far the drawn mask's margins land
    from the real table's, plus the mean planted z of observed vs missing
    cells (the raw selection effect)."""
    return {
        "density": float(mask.mean()),
        "row_mae": float(np.abs(mask.sum(1) - real_mask.sum(1)).mean()),
        "col_mae": float(np.abs(mask.sum(0) - real_mask.sum(0)).mean()),
        "mean_z_obs": float(z[mask].mean()) if mask.any() else None,
        "mean_z_miss": float(z[~mask].mean()) if (~mask).any() else None,
    }


def pair_diagnostics(mask, z, R, fcols):
    """O(p^2 n) pair-level stats: observed pairwise correlation vs planted
    truth, split within vs across planted groups."""
    p = mask.shape[1]
    d = {}
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
    for dz, sub, st in TABLES:
        p = SOURCE / sub / st / "model_benchmark_table.csv"
        if not p.exists():
            sys.exit(f"make_mnar: missing source table {p}")
        tables.append((dz, sub, st, p))
    loaded = {(dz, st): load_table(p) for dz, sub, st, p in tables}
    print(f"make_mnar: {len(tables)} source tables from {SOURCE}")

    all_bench = sorted({c for _, cols, _ in loaded.values() for c in cols})
    truth_dir = OUT / "truth"
    truth_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for cell_idx, (nf, lo, hi, cell) in enumerate(TRUTH_CELLS):
        pp = planted_params(
            np.random.default_rng(TRUTH_SEED + 1000 + cell_idx),
            all_bench, nf, lo, hi)

        for t_idx, (dz, sub, st, src_path) in enumerate(tables):
            keys, cols, M = loaded[(dz, st)]
            real_mask = ~np.isnan(M)
            n, p = real_mask.shape
            tag = f"{dz}_{st}"
            base = f"{tag}_{cell}"

            # ---- planted truth (shared by every mask) ----
            target = observed_mean_r(M, real_mask)
            truth = plant_truth(TRUTH_SEED + cell_idx * 10 + t_idx,
                                keys, cols, pp, nf, target)
            pl.DataFrame({KEY: pl.Series(keys),
                          **{c: pl.Series(np.round(truth["X"][:, j], 4))
                             for j, c in enumerate(cols)}}
                         ).write_csv(truth_dir / f"{base}_complete.csv")
            Rdf = pl.DataFrame({KEY: pl.Series(cols),
                                **{c: pl.Series(truth["R"][:, j])
                                   for j, c in enumerate(cols)}})
            Rdf.write_csv(truth_dir / f"{base}_correlation.csv")
            np.savez(truth_dir / f"{base}_factors.npz",
                     g=truth["g"], eta=truth["eta"], fcols=truth["fcols"],
                     lam_g=truth["lam_g"], lam_f=truth["lam_f"],
                     psi=truth["psi"])
            jwrite(truth_dir / f"{base}_meta.json", {
                "table": tag, "truth_cell": cell,
                "n_group_factors": nf, "true_nfactors": nf + 1,
                "loading_range": [lo, hi],
                "lam_g_min": float(truth["lam_g"].min()),
                "lam_g_max": float(truth["lam_g"].max()),
                "lam_f_min": float(truth["lam_f"].min()),
                "lam_f_max": float(truth["lam_f"].max()),
                "target_mean_r": target,
                "achieved_mean_r": truth["mean_r"],
                "omega_hierarchical": truth["omega_h"],
                "max_communality": MAX_COMMUNALITY,
                "n_models": n, "n_benchmarks": p,
                "truth_seed": TRUTH_SEED + cell_idx * 10 + t_idx,
                "group_sizes": np.bincount(truth["fcols"],
                                           minlength=nf).tolist(),
            })
            print(f"[truth] {base:26} {n}x{p}"
                  f" mean_r={truth['mean_r']:.3f} (target {target:.3f})"
                  f" omega_h={truth['omega_h']:.3f}")

            # expected correlation per benchmark (row mean of planted R),
            # the popularity handle for the mask
            rho = (truth["R"].sum(1) - 1.0) / (p - 1)
            rho_std = standardize(rho)

            for mask_name, kind in MASKS:
                if kind == "real":
                    mask = real_mask.copy()
                    lam, ok, sweeps = 1.0, True, 0
                else:
                    s = mechanism_term(kind, truth["z"], truth["g"],
                                       rho_std)
                    # fresh rng with the same seed for every draw: common
                    # random numbers across truth cells and masks (the
                    # table shape fixes the uniform matrix)
                    mask, lam, ok, sweeps = calibrated_mask(
                        s, real_mask.sum(1), real_mask.sum(0),
                        np.random.default_rng(MASK_SEED))

                run_dir = OUT / cell / mask_name / f"s{MASK_SEED}"
                write_masked_table(
                    run_dir / sub / st / "model_benchmark_table.csv",
                    keys, cols, truth["X"], mask)
                run_truth = run_dir / "truth"
                run_truth.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(run_truth / f"{base}_mask.npz",
                                    mask=mask)

                diag = margin_diagnostics(mask, real_mask, truth["z"])
                diag.update(pair_diagnostics(mask, truth["z"], truth["R"],
                                             truth["fcols"]))
                jwrite(run_truth / f"{base}_meta.json", {
                    "truth_cell": cell, "mask": mask_name,
                    "mechanism": kind,
                    "cell_gamma": CELL_GAMMA if kind == "cell" else None,
                    "pop_kappa": POP_KAPPA if kind == "popularity" else None,
                    "mask_seed": MASK_SEED, "truth_tag": base,
                    "truth_dir": str(truth_dir.relative_to(REPO)),
                    "mech_scale": lam, "ipf_converged": ok,
                    "ipf_sweeps": sweeps,
                    "n_models": n, "n_benchmarks": p,
                    "real_density": float(real_mask.mean()),
                    **diag,
                })
                row = {"truth_cell": cell, "n_group_factors": nf,
                       "true_nfactors": nf + 1,
                       "loading_range": f"{lo:g}-{hi:g}",
                       "mask": mask_name, "dz": dz, "strategy": st,
                       "n_models": n, "n_benchmarks": p,
                       "real_density": round(float(real_mask.mean()), 4),
                       "target_mean_r": round(target, 4),
                       "achieved_mean_r": round(truth["mean_r"], 4),
                       "omega_h": round(truth["omega_h"], 4),
                       "mech_scale": lam, "ipf_converged": ok,
                       **{k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in diag.items()}}
                summary.append(row)
                scale = "" if lam == 1.0 else f" scale={lam}"
                print(f"[mask] {cell:9} {mask_name:11} {tag:16}"
                      f" density={row['density']}{scale}"
                      f" row_mae={row['row_mae']} col_mae={row['col_mae']}")

    pl.DataFrame(summary).write_csv(OUT / "summary.csv")
    print(f"make_mnar: wrote {len(summary)} datasets to {OUT} "
          f"(+ {OUT / 'summary.csv'})")


if __name__ == "__main__":
    main()
