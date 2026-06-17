# ─────────────────────────────────────────────────────────────────────────────
# Propensity score matrix estimation via 1-Bit Matrix Completion
#
# Implements the approach of Ma & Chen (2019) "Missing Not at Random in Matrix
# Completion: The Effectiveness of Estimating Missingness Probabilities Under a
# Low Nuclear Norm Assumption" (NeurIPS 2019).
#
# Given a fully-observed binary missingness mask M ∈ {0,1}^{m×d} where
# M[i,j] = 1 iff entry (i,j) was observed, we estimate the propensity score
# matrix P ∈ [0,1]^{m×d} where P[i,j] = probability that (i,j) is observed.
#
# The estimation solves a nuclear-norm-constrained Bernoulli MLE (1-Bit MC
# from Davenport et al. 2014):
#
#   max_Γ  Σ_{i,j} [M_{i,j} log σ(Γ_{i,j}) + (1-M_{i,j}) log(1-σ(Γ_{i,j}))]
#   s.t.   ‖Γ‖_* ≤ τ√(md),  ‖Γ‖_max ≤ γ
#
# then sets P̂_{i,j} = σ(Γ̂_{i,j}) where σ is the standard logistic function.
#
# Implementation: projected gradient descent with nuclear norm thresholding
# and entry-wise clipping.
# ─────────────────────────────────────────────────────────────────────────────

"""
    logistic(x)

Standard logistic function σ(x) = 1 / (1 + e⁻ˣ).
"""
logistic(x::Real) = 1.0 / (1.0 + exp(-x))

"""
    log_logistic(x)

Compute log(σ(x)) = log(1 / (1 + e⁻ˣ)) numerically stably.
For x ≥ 0: -log1p(exp(-x))
For x < 0:  x - log1p(exp(x))
"""
function log_logistic(x::Real)
    if x >= 0
        return -log1p(exp(-x))
    else
        return x - log1p(exp(x))
    end
end

"""
    log_one_minus_logistic(x)

Compute log(1 - σ(x)) = log(e⁻ˣ / (1 + e⁻ˣ)) numerically stably.
For x ≥ 0: -x - log1p(exp(-x))
For x < 0:  -log1p(exp(x))
"""
function log_one_minus_logistic(x::Real)
    if x >= 0
        return -x - log1p(exp(-x))
    else
        return -log1p(exp(x))
    end
end

"""
    soft_threshold!(A, λ)

Apply element-wise soft thresholding to the singular values of A:
    S_λ(A) = U · diag(max(σ_i - λ, 0)) · Vᵀ

Mutates A in place for the SVD buffer. Returns the thresholded matrix.
"""
function soft_threshold!(A::AbstractMatrix, λ::Float64)
    F = svd(A)
    sthresh = max.(F.S .- λ, 0.0)
    return F.U * Diagonal(sthresh) * F.Vt
end

"""
    nuclear_norm_projection(A, τ, m, d; max_iter=20, tol=1e-6)

Project A onto the nuclear norm ball ‖·‖_* ≤ τ√(md) via iterative
soft-thresholding (the dual of nuclear norm regularisation).

Returns the projected matrix.
"""
function nuclear_norm_projection(A::AbstractMatrix, τ::Float64, m::Int, d::Int;
                                 max_iter::Int=50, tol::Float64=1e-8)
    radius = τ * sqrt(m * d)
    # Check if already feasible.
    F = svd(A)
    if sum(F.S) <= radius + tol
        return copy(A)
    end

    # Binary search for λ such that sum(max(σ_i - λ, 0)) = radius.
    λ_lo = 0.0
    λ_hi = maximum(F.S)
    for _ in 1:max_iter
        λ = (λ_lo + λ_hi) / 2.0
        s = max.(F.S .- λ, 0.0)
        total = sum(s)
        if abs(total - radius) <= tol
            break
        elseif total > radius
            λ_lo = λ
        else
            λ_hi = λ
        end
    end
    λ = (λ_lo + λ_hi) / 2.0
    s = max.(F.S .- λ, 0.0)
    return F.U * Diagonal(s) * F.Vt
end

"""
    estimate_propensity_1bitmc(M::AbstractMatrix{<:Integer};
                               τ::Real=1.0, γ::Real=5.0,
                               iters::Int=1000, lr::Float64=1.0,
                               verbose::Bool=false,
                               rng::AbstractRNG=Random.default_rng())

Estimate the propensity score matrix P from a binary missingness mask M via
1-Bit Matrix Completion (Davenport et al. 2014).

Arguments:
- `M`: binary matrix of shape (m, d), M[i,j] = 1 if entry (i,j) was observed.
- `τ`: nuclear norm constraint parameter (see Assumption A1).
- `γ`: entry-wise max norm constraint (see Assumption A2).
- `iters`: number of projected gradient descent iterations.
- `lr`: learning rate for gradient descent on the negative log-likelihood.
- `verbose`: print loss every 100 iterations.

Returns:
- `P̂::Matrix{Float64}` of shape (m, d) with P̂[i,j] = estimated propensity.
- `Γ̂::Matrix{Float64}` of shape (m, d) — the fitted log-odds matrix (useful
  for diagnostics).
"""
function estimate_propensity_1bitmc(M::AbstractMatrix{<:Integer};
                                    τ::Float64=1.0, γ::Real=5.0,
                                    iters::Int=1000, lr::Float64=1.0,
                                    verbose::Bool=false,
                                    rng::AbstractRNG=Random.default_rng())
    m, d = size(M)
    Mf = float(M)

    # Initialise Γ from a small random matrix.
    Γ = 0.1 .* randn(rng, m, d)

    for t in 1:iters
        # Negative log-likelihood and gradient for Bernoulli model:
        #   L = -Σ [M·logσ(Γ) + (1-M)·log(1-σ(Γ))]
        #   ∇L = σ(Γ) - M   (gradient of negative log-likelihood w.r.t. Γ)
        σΓ = logistic.(Γ)
        grad = σΓ .- Mf

        # Gradient step.
        @. Γ -= lr * grad

        # Project onto constraints.
        Γ .= nuclear_norm_projection(Γ, Float64(τ), m, d)
        # Clip entries to [-γ, γ].
        clamp!(Γ, -Float64(γ), Float64(γ))

        if verbose && (t == 1 || t % 100 == 0 || t == iters)
            # Actually compute the loss for logging.
            loss = 0.0
            for j in 1:d, i in 1:m
                loss -= M[i,j] == 1 ? log_logistic(Γ[i,j]) : log_one_minus_logistic(Γ[i,j])
            end
            @printf("[propensity 1bitmc] iter %5d  neg-loglik = %.6e  nuc_norm = %.4f\n",
                    t, loss, sum(svdvals(Γ)))
        end
    end

    P̂ = logistic.(Γ)
    return P̂, Γ
end

"""
    estimate_propensity_1bitmc_simple(M; kwargs...)

Convenience wrapper that returns only the propensity matrix P̂ (discarding Γ̂).
"""
function estimate_propensity_1bitmc_simple(M::AbstractMatrix{<:Integer}; kwargs...)
    P̂, _ = estimate_propensity_1bitmc(M; kwargs...)
    return P̂
end
