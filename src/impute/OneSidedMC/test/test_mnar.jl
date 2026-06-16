# ─────────────────────────────────────────────────────────────────────────────
# MNAR (Missing Not At Random) stress tests.
#
# For each MNAR flavor we sweep a small parameter grid that controls the
# severity / shape of the missingness pattern. For each (m, d, params) combo
# we run NSEEDS independent dataset realizations — different X each time, same
# MNAR mechanism — and report the resulting error distribution.
#
# The three flavors:
#
# (1) Column popularity bias.
#     Per-column weights w_j ∝ 1/j^p with `p` controlling concentration.
#     p=0.3: nearly flat. p=1.5: very concentrated on low-index columns.
#
# (2) Block / group structure.
#     Partition rows into G equal-size groups and columns into G blocks.
#     Each group samples with weight α on its own block and 1 elsewhere.
#
# (3) Value-thresholded MNAR.
#     P(observe col j in row i) ∝ |X[i,j]|^p. Selection depends on values,
#     so this is the regime that violates the paper's theory.
#
# Per-cell output reports min/median/max/std across NSEEDS, plus the
# small-angle equivalent (in degrees) of the median error for interpretability.
# ─────────────────────────────────────────────────────────────────────────────

const MNAR_MS = (3_000, 6_000, 12_000, 25_000)
const MNAR_DS = (50, 100, 200)
const MNAR_R = 1
const MNAR_K = 2
const NSEEDS = parse(Int, get(ENV, "NSEEDS", "10"))

# Parameter grids per flavor.
const POPULARITY_PS = (0.3, 0.7, 1.5)
const BLOCK_PARAMS = ((G=2, α=2.0), (G=4, α=3.0), (G=6, α=5.0))
const VALUE_PS = (0.5, 1.0, 2.0, 3.0)

# ── Weight-function builders ─────────────────────────────────────────────────

popularity_weights(d::Int, p::Float64) =
    (
        let w = [1.0 / (j^p) for j in 1:d]
            (_i, _X) -> w
        end
    )

function block_weights(d::Int, m::Int, G::Int, α::Float64)
    col_block_size = cld(d, G)
    row_block_size = cld(m, G)
    function _wfn(i::Int, _X)
        g = min(G, cld(i, row_block_size))
        lo = (g - 1) * col_block_size + 1
        hi = min(g * col_block_size, d)
        w = ones(d)
        @inbounds for j in lo:hi
            w[j] = α
        end
        return w
    end
    return _wfn
end

value_thresholded_weights(p::Float64) =
    (i::Int, X) -> abs.(@view X[i, :]) .^ p

# ── Per-(m, d, params) driver ────────────────────────────────────────────────

"""
    run_param_cell(m, d, r, k, weight_fn, base_seed, n_seeds)

Run the algorithm on `n_seeds` independent dataset realizations (each seed
generates a fresh X and a fresh sampled mask from the SAME `weight_fn`). Also
runs the uniform-sampling baseline on each seed's data for comparison, and
factorization-without-diagonal on the MNAR samples.

Returns vectors of errors (one entry per seed) for ours-under-MNAR,
ours-under-uniform, and FWD-under-MNAR.
"""
function run_param_cell(m, d, r, k, weight_fn, base_seed, n_seeds)
    err_mnar = Float64[]
    err_uniform = Float64[]
    err_fwd = Float64[]
    for s in 1:n_seeds
        rng = MersenneTwister(base_seed + s)
        X, Q = gaussian_factors(m, d, r; rng=rng)

        idx_u, vals_u = sample_observations(X, k; rng=rng)
        V_u, _ = fit_onesided(idx_u, vals_u, d, r; iters=1200, lr=0.05, rng=rng)
        push!(err_uniform, rowspace_error(right_singular_vectors(V_u; r=r), Q))

        idx_m, vals_m = sample_observations_weighted(X, k, weight_fn; rng=rng)
        V_m, _ = fit_onesided(idx_m, vals_m, d, r; iters=1200, lr=0.05, rng=rng)
        push!(err_mnar, rowspace_error(right_singular_vectors(V_m; r=r), Q))

        push!(err_fwd,
            rowspace_error(factorization_without_diagonal(idx_m, vals_m, m, d, r), Q))
    end
    return (mnar=err_mnar, uniform=err_uniform, fwd=err_fwd)
end

"""
    summarise(errs, r)

Compact distribution summary across seeds. Reports the first four standardised
moments (mean, variance, skewness, kurtosis) plus min / median / max and the
small-angle equivalent of the median error in degrees (`θ̄ ≈ √(err / r)` rad).

Skewness and kurtosis use the standard sample estimators:
    skew = (1/n) Σ ((x − μ) / σ)³
    kurt = (1/n) Σ ((x − μ) / σ)⁴       (raw, not excess — Gaussian gives ~3)
With small `n` these are noisy; treat as eyeball-only.
"""
function summarise(errs::AbstractVector{<:Real}, r::Int)
    n = length(errs)
    μ = Statistics.mean(errs)
    σ = n > 1 ? Statistics.std(errs) : 0.0
    var = σ * σ
    # Sample skewness/kurtosis. Guard against σ = 0 (all-equal samples).
    if σ > 0
        z = (errs .- μ) ./ σ
        skew = sum(z .^ 3) / n
        kurt = sum(z .^ 4) / n
    else
        skew = 0.0
        kurt = 0.0
    end
    med = Statistics.median(errs)
    return (
        min=minimum(errs),
        med=med,
        max=maximum(errs),
        mean=μ,
        var=var,
        skew=skew,
        kurt=kurt,
        approx_angle_deg=rad2deg(sqrt(max(med, 0.0) / r)),
    )
end

# ── The MNAR sweeps ──────────────────────────────────────────────────────────

@testset "MNAR — column popularity bias" begin
    @info "MNAR popularity: $(NSEEDS) seeds per (m, d, p)"
    for (im, m) in enumerate(MNAR_MS), (id, d) in enumerate(MNAR_DS), (ip, p) in enumerate(POPULARITY_PS)
        base = 2_000_000 + 100_000 * im + 1_000 * id + 10 * ip
        wfn = popularity_weights(d, p)
        res = run_param_cell(m, d, MNAR_R, MNAR_K, wfn, base, NSEEDS)

        s_m = summarise(res.mnar, MNAR_R)
        s_u = summarise(res.uniform, MNAR_R)
        s_f = summarise(res.fwd, MNAR_R)
        @info "MNAR popularity" m d p mnar = s_m uniform = s_u fwd = s_f

        @test all(isfinite, res.mnar)
        # Ours beats FWD on a majority of seeds.
        @test sum(res.mnar .< res.fwd) >= cld(2 * NSEEDS, 3)
        # Bounded median degradation vs uniform.
        @test s_m.med < 4 * s_u.med + 0.5
    end
end

@testset "MNAR — block / group structure" begin
    @info "MNAR block: $(NSEEDS) seeds per (m, d, G, α)"
    for (im, m) in enumerate(MNAR_MS), (id, d) in enumerate(MNAR_DS), (ip, p) in enumerate(BLOCK_PARAMS)
        base = 3_000_000 + 100_000 * im + 1_000 * id + 10 * ip
        wfn = block_weights(d, m, p.G, p.α)
        res = run_param_cell(m, d, MNAR_R, MNAR_K, wfn, base, NSEEDS)

        s_m = summarise(res.mnar, MNAR_R)
        s_u = summarise(res.uniform, MNAR_R)
        s_f = summarise(res.fwd, MNAR_R)
        @info "MNAR block" m d G = p.G α = p.α mnar = s_m uniform = s_u fwd = s_f

        @test all(isfinite, res.mnar)
        @test sum(res.mnar .< res.fwd) >= cld(2 * NSEEDS, 3)
        @test s_m.med < 5 * s_u.med + 0.7
    end
end

@testset "MNAR — value-thresholded" begin
    # No accuracy assertions — this flavor violates the theory. We just want
    # the distribution surfaced in the logs.
    @info "MNAR value-thresholded: $(NSEEDS) seeds per (m, d, p)"
    for (im, m) in enumerate(MNAR_MS), (id, d) in enumerate(MNAR_DS), (ip, p) in enumerate(VALUE_PS)
        base = 4_000_000 + 100_000 * im + 1_000 * id + 10 * ip
        wfn = value_thresholded_weights(p)
        res = run_param_cell(m, d, MNAR_R, MNAR_K, wfn, base, NSEEDS)

        s_m = summarise(res.mnar, MNAR_R)
        s_u = summarise(res.uniform, MNAR_R)
        @info "MNAR value-thresholded" m d p mnar = s_m uniform = s_u

        @test all(isfinite, res.mnar)
    end
end
