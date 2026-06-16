# ─────────────────────────────────────────────────────────────────────────────
# Loss for one-sided matrix completion
#
# We parametrise Θ ∈ R^{d × d} as Θ = V Vᵀ for some V ∈ R^{d × r}. This factored
# form has two advantages:
#   1. It bakes in the rank-r constraint and PSD structure of Θ* = (1/m) XᵀX.
#   2. The optimisation becomes unconstrained — no projection step needed —
#      which is why §5 of the paper picks this form for the practical solver.
#
# For each row i we observe k column indices and the corresponding entries of X.
# For every ordered pair (j1, j2) of observed columns in the same row, the
# product X_{i,j1} * X_{i,j2} is an unbiased estimator of Θ*_{j1,j2}, so we fit
# Θ to these products with a squared loss. Off-diagonal pairs (j1 ≠ j2) target
# off-diagonal entries; diagonal "pairs" (j1 = j2) — i.e. the squared
# observations X_{i,j}² — target the diagonal of Θ*.
#
# For k = 2 this reduces exactly to Eq. 2 in the paper (four terms per row:
# two off-diagonal residuals and two diagonal residuals). For k > 2 we average
# over all k*(k-1) ordered off-diagonal pairs and all k diagonal terms — the
# natural generalisation that the paper uses for its k > 2 experiments.
# ─────────────────────────────────────────────────────────────────────────────

"""
    loss_and_grad(V, idx, vals)

Compute the value of L(V Vᵀ) and its gradient ∇_V L, where L is the squared
loss against observed pairwise products.

Arguments:
- `V::Matrix{Float64}` of shape `(d, r)` — current factor.
- `idx::Matrix{Int}` of shape `(m, k)` — observed column indices per row.
- `vals::Matrix{Float64}` of shape `(m, k)` — observed entry values.

Returns `(loss, grad)` where `loss::Float64` and `grad::Matrix{Float64}` has
the same shape as `V`.

Implementation notes:
- We never form Θ = V Vᵀ as a `d × d` matrix in the hot loop. Instead, for each
  row we only touch the `k × r` slice of `V` indexed by that row's observed
  columns; the residuals then live in a `k × k` matrix per row, with the
  diagonal handled separately from the off-diagonal.
- Gradient derivation: since Θ_{ab} = ⟨V_a, V_b⟩, where V_a is the a-th row of
  V, we have ∂Θ_{ab}/∂V_c = δ_{ac} V_b + δ_{bc} V_a. So a residual r at
  position (a, b) contributes r * V_b to row a of grad and r * V_a to row b
  of grad. (The diagonal case a = b just doubles that, but we account for the
  factor explicitly below to match the loss scaling.)
"""
function loss_and_grad(V::AbstractMatrix{<:Real},
                       idx::AbstractMatrix{<:Integer},
                       vals::AbstractMatrix{<:Real})
    d, r = size(V)
    m, k = size(idx)
    @assert size(vals) == (m, k)

    grad = zeros(d, r)
    loss = 0.0

    # Normalisation: there are k*(k-1) ordered off-diagonal pairs per row, so
    # off-diagonal contributions are scaled by 1/(m * k*(k-1)). Diagonal
    # contributions are scaled by 1/(m * k). For k = 2 these reduce to
    # 1/(2m) and 1/(2m) respectively, matching the 1/(4m) prefactor in Eq. 2
    # once we sum the two off-diagonal residuals (j1,j2) and (j2,j1) which
    # the paper writes out as separate terms.
    inv_off = 1.0 / (m * k * (k - 1))
    inv_dia = 1.0 / (m * k)

    # Scratch buffer reused per row to avoid allocations in the hot loop.
    Vslice = zeros(k, r)

    @inbounds for i in 1:m
        # Copy out the k × r submatrix V[idx[i, :], :].
        for jj in 1:k, c in 1:r
            Vslice[jj, c] = V[idx[i, jj], c]
        end

        # ── Off-diagonal terms ──────────────────────────────────────────────
        # For each ordered pair (j1, j2) with j1 ≠ j2:
        #     residual = ⟨V_a, V_b⟩ - vals[i, j1] * vals[i, j2]
        #     loss   += inv_off * residual²
        #     grad   += 2 * inv_off * residual * (∂Θ_{ab}/∂V)
        # We exploit symmetry: pair (j1, j2) and (j2, j1) yield the same
        # residual, so we iterate j1 < j2 and double the contribution.
        for j1 in 1:k
            a = idx[i, j1]
            xj1 = vals[i, j1]
            for j2 in (j1 + 1):k
                b = idx[i, j2]
                xj2 = vals[i, j2]
                # ⟨V_a, V_b⟩
                dot_ab = 0.0
                for c in 1:r
                    dot_ab += Vslice[j1, c] * Vslice[j2, c]
                end
                residual = dot_ab - xj1 * xj2
                # Both (j1,j2) and (j2,j1) pairs → factor of 2.
                loss += 2 * inv_off * residual * residual
                # Gradient: 2 * (2 * inv_off) * residual * V_b into row a,
                # and the symmetric counterpart into row b. The leading 2 is
                # from d(residual²)/d(residual), the second 2 collapses the
                # two symmetric pairs.
                g = 4 * inv_off * residual
                for c in 1:r
                    grad[a, c] += g * Vslice[j2, c]
                    grad[b, c] += g * Vslice[j1, c]
                end
            end
        end

        # ── Diagonal terms ──────────────────────────────────────────────────
        # residual = ⟨V_a, V_a⟩ - X_{i,a}²
        # ∂Θ_{aa}/∂V_a = 2 V_a, so gradient contribution = 2 * (2 * inv_dia) * residual * V_a.
        for j1 in 1:k
            a = idx[i, j1]
            xa = vals[i, j1]
            dot_aa = 0.0
            for c in 1:r
                dot_aa += Vslice[j1, c] * Vslice[j1, c]
            end
            residual = dot_aa - xa * xa
            loss += inv_dia * residual * residual
            g = 4 * inv_dia * residual
            for c in 1:r
                grad[a, c] += g * Vslice[j1, c]
            end
        end
    end

    return loss, grad
end
