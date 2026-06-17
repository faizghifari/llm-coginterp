@testset "data sampling" begin
    rng = MersenneTwister(1)
    m, d, r = 1_000, 40, 4
    X, Q = gaussian_factors(m, d, r; rng=rng)

    @test size(X) == (m, d)
    @test size(Q) == (d, r)
    # Orthonormality of Q.
    @test norm(Q' * Q - I) < 1e-8

    for k in (2, 5, 10)
        idx, vals = sample_observations(X, k; rng=rng)
        @test size(idx) == (m, k)
        @test size(vals) == (m, k)
        @test all(1 .<= idx .<= d)
        # Distinct columns per row.
        for i in 1:m
            @test length(unique(idx[i, :])) == k
        end
        # Values agree with X.
        for i in 1:m, j in 1:k
            @test vals[i, j] == X[i, idx[i, j]]
        end
    end
end
