@testset "loss gradient (finite differences)" begin
    # The loss is the most bug-prone piece. We sanity-check the analytic gradient
    # against a finite-difference approximation on a tiny problem.
    rng = MersenneTwister(2)
    m, d, r = 50, 8, 3
    X, _ = gaussian_factors(m, d, r; rng=rng)

    for k in (2, 4)
        idx, vals = sample_observations(X, k; rng=rng)
        V = 0.1 .* randn(rng, d, r)

        loss0, grad = loss_and_grad(V, idx, vals)
        @test isfinite(loss0)
        @test all(isfinite, grad)

        # Probe a handful of random coordinates.
        h = 1e-6
        for _ in 1:6
            i = rand(rng, 1:d)
            j = rand(rng, 1:r)
            Vp = copy(V); Vp[i, j] += h
            Vm = copy(V); Vm[i, j] -= h
            lp, _ = loss_and_grad(Vp, idx, vals)
            lm, _ = loss_and_grad(Vm, idx, vals)
            numerical = (lp - lm) / (2h)
            # Mix of absolute and relative tolerance: loss values can be small,
            # so a relative-only check is too tight.
            @test isapprox(numerical, grad[i, j]; atol=1e-5, rtol=1e-3)
        end
    end
end
