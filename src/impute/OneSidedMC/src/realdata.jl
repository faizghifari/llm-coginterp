# ─────────────────────────────────────────────────────────────────────────────
# Real-data driver for OneSidedMC.
#
# The paper's core (data.jl/loss.jl) assumes a FIXED k observations per row. Real
# densified tables have a VARIABLE number of observed benchmarks per model, so
# this file provides:
#   - a ragged loss/gradient (all k_i*(k_i-1) ordered pairs per row i),
#   - column center+scale on observed cells (so Theta-hat approximates a
#     correlation matrix, which is what psych factors downstream),
#   - held-out pairwise-product RMSE for rank selection (OneSidedMC has no
#     built-in rank selector; this is the analog of softimpute's RMSE sweep),
#   - synthesis of a surrogate data matrix whose covariance equals Theta-hat,
#     handed to the shared R factoring (psych never sees that it is synthetic).
#
# Theta* = (1/m) X'X is the d x d benchmark Gram matrix; on column-standardized
# data that is the benchmark correlation matrix. OneSidedMC recovers Theta-hat =
# V V' even when the cells themselves cannot be completed (k can be < r).
# ─────────────────────────────────────────────────────────────────────────────

using LinearAlgebra
using Random
using Statistics
using Printf

# Ragged observation container: obs[i] = (cols, vals) for row i, where cols and
# vals are equal-length vectors of that row's observed column indices and
# (standardized) values.
const RaggedObs = Vector{Tuple{Vector{Int}, Vector{Float64}}}

"""
    ragged_loss_and_grad(V, obs) -> (loss, grad)

Squared loss of Theta = V V' against observed pairwise products, summed over all
rows and all ordered off-diagonal pairs plus diagonal terms, then normalised by
the total number of pairs/diagonals (so rows with more observations are weighted
by how much evidence they carry). Mirrors loss.jl but for variable-length rows.
"""
function ragged_loss_and_grad(V::AbstractMatrix{<:Real}, obs::RaggedObs)
    d, r = size(V)
    grad = zeros(d, r)
    loss = 0.0
    n_off = 0      # total ordered off-diagonal pairs across all rows
    n_dia = 0      # total diagonal terms across all rows

    # First pass to count normalisers (cheap; keeps loss scale comparable across
    # datasets with different observation counts).
    @inbounds for (cols, _) in obs
        k = length(cols)
        n_off += k * (k - 1)
        n_dia += k
    end
    inv_off = n_off > 0 ? 1.0 / n_off : 0.0
    inv_dia = n_dia > 0 ? 1.0 / n_dia : 0.0

    @inbounds for (cols, vals) in obs
        k = length(cols)
        # Off-diagonal: each unordered pair counted twice (factor 2 below).
        for j1 in 1:k
            a = cols[j1]; xa = vals[j1]
            for j2 in (j1 + 1):k
                b = cols[j2]; xb = vals[j2]
                dot_ab = 0.0
                for c in 1:r
                    dot_ab += V[a, c] * V[b, c]
                end
                residual = dot_ab - xa * xb
                loss += 2 * inv_off * residual * residual
                g = 4 * inv_off * residual
                for c in 1:r
                    grad[a, c] += g * V[b, c]
                    grad[b, c] += g * V[a, c]
                end
            end
        end
        # Diagonal: residual = <V_a, V_a> - x_a^2.
        for j1 in 1:k
            a = cols[j1]; xa = vals[j1]
            dot_aa = 0.0
            for c in 1:r
                dot_aa += V[a, c] * V[a, c]
            end
            residual = dot_aa - xa * xa
            loss += inv_dia * residual * residual
            g = 4 * inv_dia * residual
            for c in 1:r
                grad[a, c] += g * V[a, c]
            end
        end
    end
    return loss, grad
end

"""
    fit_onesided_ragged(obs, d, r; iters, lr, ...) -> (V, trace)

Adam on `V -> ragged_loss_and_grad(V, obs)`. Same optimiser as algorithm.jl's
`fit_onesided`, adapted to the ragged loss.
"""
function fit_onesided_ragged(obs::RaggedObs, d::Int, r::Int;
                              iters::Int = 2000, lr::Float64 = 0.01,
                              init_scale::Float64 = 0.1,
                              β1::Float64 = 0.9, β2::Float64 = 0.999,
                              ϵ::Float64 = 1e-8,
                              rng::AbstractRNG = Random.default_rng(),
                              verbose::Bool = false)
    V = init_scale .* randn(rng, d, r)
    mvec = zeros(d, r); vvec = zeros(d, r)
    trace = Float64[]
    for t in 1:iters
        loss, grad = ragged_loss_and_grad(V, obs)
        @. mvec = β1 * mvec + (1 - β1) * grad
        @. vvec = β2 * vvec + (1 - β2) * grad * grad
        @. V -= lr * (mvec / (1 - β1^t)) / (sqrt(vvec / (1 - β2^t)) + ϵ)
        if t == 1 || t % 50 == 0 || t == iters
            push!(trace, loss)
            verbose && @printf("[osmc] iter %4d  loss %.6e\n", t, loss)
        end
    end
    return V, trace
end

# Theta-hat = V V' (d x d, PSD by construction).
theta_hat(V::AbstractMatrix) = Symmetric(V * V')
