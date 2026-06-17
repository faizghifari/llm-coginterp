# ─────────────────────────────────────────────────────────────────────────────
# Our algorithm: fit Θ̂ = V Vᵀ by Adam on the loss in loss.jl.
#
# This implements the "non-convex factored" solver mentioned in §5 of the paper
# (footnote 6 / §5.1). We drop the nuclear-norm regulariser because:
#   - The factored form Θ = V Vᵀ already enforces rank ≤ r and PSD.
#   - The diagonal terms in the loss penalise V having too-large rows, which
#     plays the role the nuclear-norm regulariser plays in the convex form.
# After fitting V, the top-r right singular vectors of X are recovered as the
# top-r eigenvectors of Θ̂ = V Vᵀ (see right_singular_vectors below).
# ─────────────────────────────────────────────────────────────────────────────

"""
    fit_onesided(idx, vals, d, r; iters=2000, lr=0.01, init_scale=0.1,
                 β1=0.9, β2=0.999, ϵ=1e-8, rng=Random.default_rng(), verbose=false)

Run Adam on `V → L(V Vᵀ)` for the loss defined in `loss_and_grad`. Returns
`(V, trace)` where `V::Matrix{Float64}` of shape `(d, r)` is the fitted factor
and `trace::Vector{Float64}` is the loss every 50 iterations (for debugging
and convergence checks; not used by tests).

Hyperparameters:
- `iters`: number of gradient steps.
- `lr`: Adam learning rate.
- `init_scale`: stdev of the Gaussian initialiser for `V`.
- `β1, β2, ϵ`: standard Adam parameters.
"""
function fit_onesided(idx::AbstractMatrix{<:Integer},
                       vals::AbstractMatrix{<:Real},
                       d::Int, r::Int;
                       iters::Int = 2000,
                       lr::Float64 = 0.01,
                       init_scale::Float64 = 0.1,
                       β1::Float64 = 0.9,
                       β2::Float64 = 0.999,
                       ϵ::Float64 = 1e-8,
                       rng::AbstractRNG = Random.default_rng(),
                       verbose::Bool = false)

    V = init_scale .* randn(rng, d, r)
    # Adam moment buffers (first and second moments of the gradient).
    mvec = zeros(d, r)
    vvec = zeros(d, r)
    trace = Float64[]

    for t in 1:iters
        loss, grad = loss_and_grad(V, idx, vals)

        # Standard Adam update.
        @. mvec = β1 * mvec + (1 - β1) * grad
        @. vvec = β2 * vvec + (1 - β2) * grad * grad
        # Bias-corrected estimates.
        m̂_correction = 1 - β1^t
        v̂_correction = 1 - β2^t
        @. V -= lr * (mvec / m̂_correction) / (sqrt(vvec / v̂_correction) + ϵ)

        if t == 1 || t % 50 == 0 || t == iters
            push!(trace, loss)
            if verbose
                @printf("[fit_onesided] iter %4d  loss = %.6e\n", t, loss)
            end
        end
    end

    return V, trace
end

"""
    right_singular_vectors(V; r=size(V, 2))

Given the factor `V` returned by `fit_onesided`, return the top-`r`
eigenvectors of Θ̂ = V Vᵀ as a `d × r` matrix with orthonormal columns. These
are our estimates of the top-`r` right singular vectors of the underlying `X`.

Implementation: form `Θ̂` explicitly (it is `d × d`, which is small) and call
`eigen` on the `Symmetric` view. `eigen` returns eigenvalues in ascending
order, so we take the last `r` columns of the eigenvector matrix.
"""
function right_singular_vectors(V::AbstractMatrix; r::Int = size(V, 2))
    Θ = Symmetric(V * V')
    F = eigen(Θ)
    # Largest r eigenvalues / eigenvectors.
    return F.vectors[:, end - r + 1:end]
end
