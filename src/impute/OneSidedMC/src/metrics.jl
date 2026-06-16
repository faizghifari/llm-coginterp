"""
    rowspace_error(Q̂, Q)

Compute the rowspace recovery error used throughout the paper (Equation 4):

    err = min_{R: RᵀR = I_r} ‖Q̂ R − Q‖_F²

i.e. the best squared Frobenius distance between `Q̂` and `Q` after an optimal
orthogonal rotation `R` of `Q̂`. This is the orthogonal Procrustes problem; it
has a closed-form solution via the SVD of `Q̂ᵀ Q`.

If `Q̂ᵀ Q = A Σ Bᵀ`, then `R* = A Bᵀ` minimises the loss and the residual is
`‖Q̂‖_F² + ‖Q‖_F² − 2 tr(Σ)`.

Both `Q̂` and `Q` are expected to have orthonormal columns; the function does
not enforce this — caller's responsibility.
"""
function rowspace_error(Q̂::AbstractMatrix, Q::AbstractMatrix)
    @assert size(Q̂) == size(Q) "Q̂ and Q must have the same shape"
    F = svd(Q̂' * Q)
    R = F.U * F.Vt
    return norm(Q̂ * R - Q)^2
end

"""
    mean_principal_angle_deg(Q̂, Q)

Return the mean principal angle between the column-spans of `Q̂` and `Q`, in
degrees. The principal angles `θ_i` between two `r`-dim subspaces satisfy
`cos θ_i = σ_i`, where `σ_i` are the singular values of `Q̂ᵀ Q`. We average
over the `r` principal angles.

This is a more interpretable summary than `rowspace_error`:
- 0° = identical subspaces
- 90° = orthogonal subspaces (worst possible)
Useful for human consumption alongside the Frobenius-based error.
"""
function mean_principal_angle_deg(Q̂::AbstractMatrix, Q::AbstractMatrix)
    @assert size(Q̂) == size(Q) "Q̂ and Q must have the same shape"
    σ = svdvals(Q̂' * Q)
    # Clamp to handle floating-point overshoot before acos.
    θ_rad = mean(acos.(clamp.(σ, -1.0, 1.0)))
    return rad2deg(θ_rad)
end
