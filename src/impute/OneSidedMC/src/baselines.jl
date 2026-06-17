# ─────────────────────────────────────────────────────────────────────────────
# Baselines we compare against in §5 of the paper.
#
# Let P_E(X) ∈ R^{m × d} denote the observed matrix: P_E(X)[i, j] = X[i, j] if
# (i, j) was observed and 0 otherwise. All baselines compute some surrogate of
# Θ* = (1/m) Xᵀ X from P_E(X) and then extract its top-r eigenvectors.
#
# Baselines:
#   (b) Direct factorisation: top-r eigenvectors of P_E(X)ᵀ P_E(X).
#   (c) Factorisation without diagonal (Cai et al. 2021): same as (b) but zero
#       out the diagonal of P_E(X)ᵀ P_E(X) before the eigendecomposition. This
#       fixes the obvious bias on the diagonal (sums of squares are always
#       positive and have a different scale than off-diagonal cross-products).
#   (a) Full matrix completion: directly fit a low-rank UVᵀ to P_E(X) by
#       Frobenius loss on observed entries, regularised by ‖U‖_F² + ‖V‖_F².
#       With only k = 2 observations per row this is severely underdetermined,
#       which is exactly the regime where our algorithm wins.
# ─────────────────────────────────────────────────────────────────────────────

"""
    _build_observed_matrix(idx, vals, m, d) -> Matrix{Float64}

Materialise `P_E(X) ∈ R^{m × d}` densely. We only do this for the baselines;
the main algorithm never builds it.
"""
function _build_observed_matrix(idx::AbstractMatrix{<:Integer},
                                 vals::AbstractMatrix{<:Real},
                                 m::Int, d::Int)
    PEX = zeros(m, d)
    k = size(idx, 2)
    @inbounds for i in 1:m, j in 1:k
        PEX[i, idx[i, j]] = vals[i, j]
    end
    return PEX
end

"""
    direct_factorization(idx, vals, m, d, r) -> Q̂

Baseline (b): top-`r` eigenvectors of `P_E(X)ᵀ P_E(X)`.
"""
function direct_factorization(idx::AbstractMatrix{<:Integer},
                               vals::AbstractMatrix{<:Real},
                               m::Int, d::Int, r::Int)
    PEX = _build_observed_matrix(idx, vals, m, d)
    M = Symmetric(PEX' * PEX)
    F = eigen(M)
    return F.vectors[:, end - r + 1:end]
end

"""
    factorization_without_diagonal(idx, vals, m, d, r) -> Q̂

Baseline (c) from Cai et al. (2021): zero out the diagonal of
`P_E(X)ᵀ P_E(X)` before taking the top-`r` eigenvectors. Removes the diagonal
bias from the sums of squares.
"""
function factorization_without_diagonal(idx::AbstractMatrix{<:Integer},
                                         vals::AbstractMatrix{<:Real},
                                         m::Int, d::Int, r::Int)
    PEX = _build_observed_matrix(idx, vals, m, d)
    M = PEX' * PEX
    for j in 1:d
        M[j, j] = 0.0
    end
    F = eigen(Symmetric(M))
    return F.vectors[:, end - r + 1:end]
end

"""
    full_matrix_completion(idx, vals, m, d, r; iters=500, lr=0.01,
                           λ=0.1, init_scale=0.1, rng=Random.default_rng())

Baseline (a): non-convex matrix completion. Minimises

    (1/|E|) Σ_{(i,j) ∈ E} (U_i · V_j − X_{i,j})² + λ (‖U‖_F²/m + ‖V‖_F²/d)

via Adam, where `U ∈ R^{m × r}` and `V ∈ R^{d × r}`. Returns the top-`r` right
singular vectors of `U Vᵀ` (which, with both factors orthogonalised, coincide
with the column-space of `V`).

`λ = 0.1` is the default the paper picks via grid search.
"""
function full_matrix_completion(idx::AbstractMatrix{<:Integer},
                                 vals::AbstractMatrix{<:Real},
                                 m::Int, d::Int, r::Int;
                                 iters::Int = 500,
                                 lr::Float64 = 0.01,
                                 λ::Float64 = 0.1,
                                 init_scale::Float64 = 0.1,
                                 β1::Float64 = 0.9,
                                 β2::Float64 = 0.999,
                                 ϵ::Float64 = 1e-8,
                                 rng::AbstractRNG = Random.default_rng())
    k = size(idx, 2)
    U = init_scale .* randn(rng, m, r)
    V = init_scale .* randn(rng, d, r)

    # Adam buffers for U and V.
    mU = zeros(m, r); vU = zeros(m, r)
    mV = zeros(d, r); vV = zeros(d, r)

    inv_nobs = 1.0 / (m * k)  # |E| = m * k

    for t in 1:iters
        gU = zeros(m, r)
        gV = zeros(d, r)

        # Data term: for each (i, j) in E, residual = U_i · V_j - X_{i,j}.
        # Gradients: ∂/∂U_i contributes residual * V_j; ∂/∂V_j contributes residual * U_i.
        @inbounds for i in 1:m, jj in 1:k
            j = idx[i, jj]
            xij = vals[i, jj]
            # Inner product U_i · V_j.
            dotuv = 0.0
            for c in 1:r
                dotuv += U[i, c] * V[j, c]
            end
            residual = dotuv - xij
            g = 2 * inv_nobs * residual
            for c in 1:r
                gU[i, c] += g * V[j, c]
                gV[j, c] += g * U[i, c]
            end
        end

        # Regulariser gradients.
        @. gU += 2 * λ / m * U
        @. gV += 2 * λ / d * V

        # Adam updates.
        @. mU = β1 * mU + (1 - β1) * gU
        @. vU = β2 * vU + (1 - β2) * gU * gU
        @. mV = β1 * mV + (1 - β1) * gV
        @. vV = β2 * vV + (1 - β2) * gV * gV
        bc1 = 1 - β1^t
        bc2 = 1 - β2^t
        @. U -= lr * (mU / bc1) / (sqrt(vU / bc2) + ϵ)
        @. V -= lr * (mV / bc1) / (sqrt(vV / bc2) + ϵ)
    end

    # Right singular vectors of UVᵀ live in the column-space of V, so we read
    # them off via an SVD of V (cheap: V is d × r).
    F = svd(V)
    return F.U[:, 1:r]
end
