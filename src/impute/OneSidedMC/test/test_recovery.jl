# ─────────────────────────────────────────────────────────────────────────────
# Recovery tests under uniform i.i.d. sampling (the paper's theoretical setting).
#
# Two scopes:
#   - quick: a single small problem (m=3k, d=50). Sanity check + baselines.
#   - full:  Cartesian sweep over (m, d) at fixed r=3, k=2. Verifies the
#            algorithm wins across scales and that error decreases with m.
# Both always run; full takes a few minutes.
#
# Each numeric error is logged alongside its mean principal-angle equivalent
# (in degrees) for easier interpretation. 0° = identical subspaces, 90° =
# orthogonal (worst possible).
# ─────────────────────────────────────────────────────────────────────────────

@testset "recovery (quick) — m=3k, d=50, r=3, k=2" begin
    rng = MersenneTwister(42)
    m, d, r = 3_000, 50, 3
    X, Q = gaussian_factors(m, d, r; rng=rng)
    idx, vals = sample_observations(X, 2; rng=rng)

    V, _    = fit_onesided(idx, vals, d, r; iters=1500, lr=0.05, rng=rng)
    Q̂_ours  = right_singular_vectors(V; r=r)
    Q̂_df    = direct_factorization(idx, vals, m, d, r)
    Q̂_fwd   = factorization_without_diagonal(idx, vals, m, d, r)
    Q̂_fmc   = full_matrix_completion(idx, vals, m, d, r; iters=300, lr=0.05, rng=rng)

    err_ours = rowspace_error(Q̂_ours, Q); ang_ours = mean_principal_angle_deg(Q̂_ours, Q)
    err_df   = rowspace_error(Q̂_df,   Q); ang_df   = mean_principal_angle_deg(Q̂_df,   Q)
    err_fwd  = rowspace_error(Q̂_fwd,  Q); ang_fwd  = mean_principal_angle_deg(Q̂_fwd,  Q)
    err_fmc  = rowspace_error(Q̂_fmc,  Q); ang_fmc  = mean_principal_angle_deg(Q̂_fmc,  Q)
    # pseudo-R^2: 1 - rowspace_error / (2r). Not a true R^2 — it's the subspace
    # recovery error normalised by its worst-case value 2r (fully orthogonal
    # subspace). 1 = perfect recovery, 0 = orthogonal/useless; reads as relative
    # improvement over the worst case. Logged only, for interpretability.
    pr2(e) = 1 - e / (2r)
    @info "quick" err_ours ang_ours pseudo_r2_ours=pr2(err_ours) err_df ang_df pseudo_r2_df=pr2(err_df) err_fwd ang_fwd pseudo_r2_fwd=pr2(err_fwd) err_fmc ang_fmc pseudo_r2_fmc=pr2(err_fmc)

    # At m=3k we're well below the paper's scale, so this is just a
    # "much better than random" check. Max possible error = 2r = 6.
    @test err_ours < 1.5
    @test err_ours < err_df
    @test err_ours < err_fwd
    @test err_ours < err_fmc
end

@testset "recovery (full) — (m, d) sweep at r=3, k=2" begin
    ms = (3_000, 6_000, 12_000, 25_000)
    ds = (50, 100, 200)
    r, k = 3, 2

    err_ours_grid = zeros(length(ms), length(ds))

    for (im, m) in enumerate(ms), (id, d) in enumerate(ds)
        rng = MersenneTwister(1000 + 31 * im + id)
        X, Q = gaussian_factors(m, d, r; rng=rng)
        idx, vals = sample_observations(X, k; rng=rng)

        V, _   = fit_onesided(idx, vals, d, r; iters=1200, lr=0.05, rng=rng)
        Q̂_ours = right_singular_vectors(V; r=r)
        Q̂_df   = direct_factorization(idx, vals, m, d, r)
        Q̂_fwd  = factorization_without_diagonal(idx, vals, m, d, r)

        err_ours = rowspace_error(Q̂_ours, Q); ang_ours = mean_principal_angle_deg(Q̂_ours, Q)
        err_df   = rowspace_error(Q̂_df,   Q); ang_df   = mean_principal_angle_deg(Q̂_df,   Q)
        err_fwd  = rowspace_error(Q̂_fwd,  Q); ang_fwd  = mean_principal_angle_deg(Q̂_fwd,  Q)
        # pseudo-R^2 = 1 - rowspace_error/(2r); see the quick testset above.
        pr2(e) = 1 - e / (2r)
        @info "full cell" m d err_ours ang_ours pseudo_r2_ours=pr2(err_ours) err_df ang_df pseudo_r2_df=pr2(err_df) err_fwd ang_fwd pseudo_r2_fwd=pr2(err_fwd)

        err_ours_grid[im, id] = err_ours
        @test err_ours < err_df
        @test err_ours < err_fwd
    end

    decreasing = sum(err_ours_grid[end, id] < err_ours_grid[1, id]
                     for id in eachindex(ds))
    @info "full m-monotonicity" decreasing_d_columns=decreasing total=length(ds)
    @test decreasing >= length(ds) - 1
end
