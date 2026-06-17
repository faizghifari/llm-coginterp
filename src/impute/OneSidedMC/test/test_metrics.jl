@testset "rowspace_error" begin
    rng = MersenneTwister(0)
    d, r = 30, 5
    # Build an arbitrary Q with orthonormal columns.
    Q = Matrix(qr(randn(rng, d, r)).Q)[:, 1:r]

    # Identity case: error should be zero.
    @test rowspace_error(Q, Q) < 1e-12

    # Rotational invariance: any orthogonal R applied to Q yields the same subspace.
    R = Matrix(qr(randn(rng, r, r)).Q)
    @test rowspace_error(Q * R, Q) < 1e-10

    # Sensitivity: bigger perturbations → bigger error.
    errs = Float64[]
    for ε in (0.01, 0.05, 0.2)
        N = randn(rng, d, r)
        Q̂ = Matrix(qr(Q + ε * N).Q)[:, 1:r]
        push!(errs, rowspace_error(Q̂, Q))
    end
    @test errs[1] < errs[2] < errs[3]

    # Sanity: rowspace_error is bounded above by ‖Q̂‖_F² + ‖Q‖_F² = 2r.
    @test errs[3] < 2r + 1e-8
end
