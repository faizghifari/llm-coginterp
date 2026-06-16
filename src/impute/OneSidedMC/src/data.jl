"""
    gaussian_factors(m, d, r; rng=Random.default_rng())

Generate a synthetic rank-`r` matrix `X = U * Vᵀ ∈ R^{m × d}` with i.i.d. Gaussian
factors, following the setup in §5.1 of Cao, Liang & Valiant (2023).

`U ∈ R^{m × r}` has entries from `N(0, 1)` and `V ∈ R^{d × r}` has entries from
`N(0, 1/√r)`. With this scaling, `E[X_{ij}²] = 1`, so the expected entry magnitude
is O(1) regardless of the rank.

Returns `(X, Q)` where `Q ∈ R^{d × r}` is the matrix of ground-truth top-`r` right
singular vectors of `X` (the rowspace we will try to recover from sparse
observations).
"""
function gaussian_factors(m::Int, d::Int, r::Int; rng::AbstractRNG=Random.default_rng())
    U = randn(rng, m, r)
    V = randn(rng, d, r) ./ sqrt(r)
    X = U * V'
    Q = svd(X).V[:, 1:r]
    return X, Q
end

"""
    sample_observations(X, k; rng=Random.default_rng())

For each of the `m` rows of `X`, sample `k` distinct column indices uniformly at
random without replacement. This is the "k observations per row" sampling model
from the paper (`k = 2` is the headline case).

Returns `(idx, vals)`:
- `idx::Matrix{Int}` of shape `(m, k)` — the sampled column indices.
- `vals::Matrix{Float64}` of shape `(m, k)` — the observed entries
  `vals[i, j] = X[i, idx[i, j]]`.
"""
function sample_observations(X::AbstractMatrix, k::Int;
                              rng::AbstractRNG=Random.default_rng())
    m, d = size(X)
    @assert 1 <= k <= d "k must satisfy 1 ≤ k ≤ d (got k=$k, d=$d)"
    idx = Matrix{Int}(undef, m, k)
    vals = Matrix{Float64}(undef, m, k)
    cols = collect(1:d)
    for i in 1:m
        # Partial Fisher–Yates: shuffle the first k positions of `cols` only.
        for j in 1:k
            swap = rand(rng, j:d)
            cols[j], cols[swap] = cols[swap], cols[j]
            idx[i, j] = cols[j]
            vals[i, j] = X[i, cols[j]]
        end
    end
    return idx, vals
end

"""
    sample_observations_weighted(X, k, weight_fn; rng=Random.default_rng())

Like `sample_observations`, but for each row `i` the `k` distinct column
indices are drawn without replacement from a categorical distribution with
weights `w = weight_fn(i, X)::Vector{Float64}` (length `d`, all entries
nonnegative, not all zero). Used by the MNAR test suite to inject non-uniform
sampling.

Implementation: at each of the `k` draws, normalise the current weights over
the remaining columns and sample via the cumulative-distribution / inverse-CDF
method. Once a column is picked we zero its weight so subsequent draws cannot
re-pick it (sampling without replacement).
"""
function sample_observations_weighted(X::AbstractMatrix, k::Int, weight_fn;
                                       rng::AbstractRNG=Random.default_rng())
    m, d = size(X)
    @assert 1 <= k <= d "k must satisfy 1 ≤ k ≤ d (got k=$k, d=$d)"
    idx = Matrix{Int}(undef, m, k)
    vals = Matrix{Float64}(undef, m, k)
    w = zeros(d)
    for i in 1:m
        w_i = weight_fn(i, X)
        @assert length(w_i) == d "weight_fn must return a length-$d vector"
        @inbounds for j in 1:d
            w[j] = max(0.0, float(w_i[j]))
        end
        for jj in 1:k
            total = sum(w)
            @assert total > 0 "weights for row $i exhausted before $k columns drawn"
            u = rand(rng) * total
            # Inverse-CDF: find smallest c such that cumulative ≥ u.
            cum = 0.0
            chosen = 0
            @inbounds for c in 1:d
                cum += w[c]
                if cum >= u
                    chosen = c
                    break
                end
            end
            # Numerical fallback (shouldn't happen, but guard against fp drift).
            if chosen == 0
                @inbounds for c in d:-1:1
                    if w[c] > 0
                        chosen = c
                        break
                    end
                end
            end
            idx[i, jj] = chosen
            vals[i, jj] = X[i, chosen]
            w[chosen] = 0.0  # sample without replacement
        end
    end
    return idx, vals
end
