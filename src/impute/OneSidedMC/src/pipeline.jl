# ─────────────────────────────────────────────────────────────────────────────
# CSV-driven OneSidedMC pipeline: densified table -> Theta-hat -> surrogate CSV.
#
# Output surrogate (data/imputed/onesidedmc/<densifier>/<strategy>/...) is a
# synthetic n x p matrix whose covariance equals the recovered Theta-hat, on the
# ORIGINAL column scale, so the shared R factoring treats it like any imputed
# matrix. It is NOT an imputation of the real cells — OneSidedMC's claim is that
# Theta-hat is recoverable while the cells are not.
# ─────────────────────────────────────────────────────────────────────────────

# realdata.jl is part of the OneSidedMC module (ragged loss/fit live there).
# Load it the same way regardless of whether the package is dev-activated.
using OneSidedMC
using LinearAlgebra, Random, Statistics

# Split a CSV line into fields, respecting RFC4180-style double-quoting (a
# quoted field may contain commas; "" inside a quoted field is an escaped
# quote). collapse_key values like `"RL-IP/TT (Choubey et al., 2021)"` have a
# comma inside quotes, which a naive split(line, ',') would break on.
function split_csv_line(ln::AbstractString)
    fields = String[]
    buf = IOBuffer()
    in_quotes = false
    chars = collect(ln)
    i, n = 1, length(chars)
    while i <= n
        c = chars[i]
        if in_quotes
            if c == '"'
                if i < n && chars[i + 1] == '"'
                    write(buf, '"'); i += 1
                else
                    in_quotes = false
                end
            else
                write(buf, c)
            end
        else
            if c == '"'
                in_quotes = true
            elseif c == ','
                push!(fields, String(take!(buf)))
            else
                write(buf, c)
            end
        end
        i += 1
    end
    push!(fields, String(take!(buf)))
    return fields
end

# Minimal CSV reader for the model x benchmark table: first column is the string
# key `collapse_key`, the rest are numeric with empty cells = missing.
function read_table(path::AbstractString)
    lines = readlines(path)
    header = split_csv_line(strip(lines[1]))
    bench = String.(header[2:end])
    p = length(bench)
    keys = String[]
    rows = Vector{Vector{Union{Missing, Float64}}}()
    for ln in lines[2:end]
        isempty(strip(ln)) && continue
        f = split_csv_line(ln)
        push!(keys, String(f[1]))
        vals = Vector{Union{Missing, Float64}}(undef, p)
        for j in 1:p
            s = strip(f[j + 1])
            vals[j] = (isempty(s) || s == "NA") ? missing : parse(Float64, s)
        end
        push!(rows, vals)
    end
    X = Matrix{Union{Missing, Float64}}(undef, length(rows), p)
    for i in eachindex(rows), j in 1:p
        X[i, j] = rows[i][j]
    end
    return drop_degenerate_cols(X, keys, bench)
end

# Drop columns useless for covariance/factoring (parity with R's prep_matrix):
# fewer than `min_obs` observed values, or zero variance among observed values
# (constant column -> no correlational signal, and would standardize to 0).
function drop_degenerate_cols(X::AbstractMatrix{Union{Missing, Float64}},
                              keys, bench; min_obs::Int = 2)
    keep = trues(size(X, 2))
    for j in 1:size(X, 2)
        col = collect(skipmissing(@view X[:, j]))
        if length(col) < min_obs || (length(col) > 1 && std(col) == 0) ||
           length(col) <= 1
            keep[j] = false
        end
    end
    if !all(keep)
        dropped = bench[.!keep]
        @info "dropping degenerate columns (<$(min_obs) obs or zero-variance)" n = count(!, keep) cols = dropped
        X = X[:, keep]
        bench = bench[keep]
    end
    return X, keys, bench
end

# Per-column mean/sd over observed cells; returns (mean, sd) length-p vectors.
function column_moments(X::AbstractMatrix{Union{Missing, Float64}})
    p = size(X, 2)
    μ = zeros(p); σ = ones(p)
    for j in 1:p
        col = collect(skipmissing(@view X[:, j]))
        μ[j] = mean(col)
        s = length(col) > 1 ? std(col) : 1.0
        σ[j] = s > 0 ? s : 1.0
    end
    return μ, σ
end

# Build ragged observations from standardized values: obs[i] = (cols, zvals).
function build_ragged(X, μ, σ)
    n, p = size(X)
    obs = RaggedObs()
    for i in 1:n
        cols = Int[]; vals = Float64[]
        for j in 1:p
            if !ismissing(X[i, j])
                push!(cols, j)
                push!(vals, (X[i, j] - μ[j]) / σ[j])
            end
        end
        push!(obs, (cols, vals))
    end
    return obs
end

# Build ragged observations with raw (unstandardized) values. Used to create
# the holdout split BEFORE computing train-only column moments, avoiding data
# leakage from held-out cells into the standardization.
function build_ragged_raw(X::AbstractMatrix{Union{Missing, Float64}})
    n, p = size(X)
    obs = RaggedObs()
    for i in 1:n
        cols = Int[]; vals = Float64[]
        for j in 1:p
            if !ismissing(X[i, j])
                push!(cols, j)
                push!(vals, X[i, j])
            end
        end
        push!(obs, (cols, vals))
    end
    return obs
end

# Compute per-column mean and sd from raw ragged observations (train split
# only). Uses the unbiased (n-1) variance, matching Julia's std() and R's sd().
function train_column_moments(obs::RaggedObs, p::Int)
    sums   = zeros(p)
    sumsq  = zeros(p)
    counts = zeros(Int, p)
    for (cols, vals) in obs
        for (c, v) in zip(cols, vals)
            sums[c]   += v
            sumsq[c]  += v * v
            counts[c] += 1
        end
    end
    μ = zeros(p); σ = ones(p)
    for j in 1:p
        if counts[j] > 0
            μ[j] = sums[j] / counts[j]
        end
        if counts[j] > 1
            var_j = (sumsq[j] - counts[j] * μ[j]^2) / (counts[j] - 1)
            σ[j] = var_j > 0 ? sqrt(var_j) : 1.0
        end
    end
    return μ, σ
end

# Convert raw ragged obs to standardized with the given moments.
function standardize_ragged(obs::RaggedObs, μ, σ)
    result = RaggedObs()
    for (cols, vals) in obs
        new_vals = Float64[(v - μ[c]) / σ[c] for (c, v) in zip(cols, vals)]
        push!(result, (copy(cols), new_vals))
    end
    return result
end

# Convert raw test cells to standardized with the given moments.
function standardize_test_cells(test_cells, μ, σ)
    result = similar(test_cells)
    for (idx, (kc, kv, hc, hv)) in enumerate(test_cells)
        new_kv = Float64[(v - μ[c]) / σ[c] for (c, v) in zip(kc, kv)]
        new_hv = (hv - μ[hc]) / σ[hc]
        result[idx] = (kc, new_kv, hc, new_hv)
    end
    return result
end

# Hold out a fraction of observed cells (only from rows that keep >= 2 obs), for
# pairwise-product RMSE scoring. Returns (train_obs, test_pairs) where each test
# pair is (a, b, za*zb) for two held-out-or-observed cols in the same row.
function holdout_pairs(obs::RaggedObs; frac::Float64 = 0.2,
                       rng::AbstractRNG = Random.default_rng())
    train = RaggedObs()
    test_pairs = Tuple{Int, Int, Float64}[]
    for (cols, vals) in obs
        k = length(cols)
        n_hold = min(floor(Int, frac * k), k - 2)   # keep >= 2 in training
        if n_hold <= 0
            push!(train, (copy(cols), copy(vals)))
            continue
        end
        perm = randperm(rng, k)
        held = perm[1:n_hold]; kept = perm[(n_hold + 1):end]
        push!(train, (cols[kept], vals[kept]))
        # test pairs: held cols against ALL of this row's cols (their products
        # are the entries of Theta* this row would have informed).
        for h in held, q in 1:k
            q == h && continue
            push!(test_pairs, (cols[h], cols[q], vals[h] * vals[q]))
        end
    end
    return train, test_pairs
end

# ── Cell-level scoring (the metric we actually use) ──────────────────────────
# Column-stratified holdout (matching R's make_holdout), but with an ADDITIONAL
# row constraint: a cell is only held out if its row would still have >= 2 cells
# remaining in training (OSMC needs them to condition the prediction).
# Returns (train_obs, test_cells) where each test cell is
#   (kept_cols::Vector{Int}, kept_vals::Vector{Float64}, held_col::Int, held_val).
function holdout_cells(obs::RaggedObs; frac::Float64 = 0.2,
                       rng::AbstractRNG = Random.default_rng())
    n_rows = length(obs)

    # column -> [(row_idx, pos_in_row, value), ...]
    col_map = Dict{Int, Vector{Tuple{Int, Int, Float64}}}()
    for (row_idx, (cols, vals)) in enumerate(obs)
        for (pos, (c, v)) in enumerate(zip(cols, vals))
            push!(get!(col_map, c, Vector{Tuple{Int, Int, Float64}}()),
                  (row_idx, pos, v))
        end
    end

    row_left = [length(cols) for (cols, _) in obs]
    held_out = Set{Tuple{Int, Int}}()

    # Shuffle column order so early columns don't starve later ones of row capacity.
    col_order = shuffle(rng, collect(keys(col_map)))
    for col in col_order
        entries = col_map[col]
        total_col_obs = length(entries)
        eligible = [(r, p, v) for (r, p, v) in entries if row_left[r] > 2]
        n_eligible = length(eligible)

        n_hold = min(floor(Int, frac * total_col_obs), total_col_obs - 2)
        if OSMC_ALLCOLHOLDOUT && total_col_obs > 2
            n_hold = max(n_hold, 1)
        end
        n_hold = min(n_hold, n_eligible)

        if n_hold > 0
            candidates = shuffle(rng, eligible)
            for (row_idx, _, _) in candidates[1:n_hold]
                push!(held_out, (row_idx, col))
                row_left[row_idx] -= 1
            end
        end
    end

    # Build train RaggedObs and test_cells from the held-out mask.
    train = RaggedObs()
    test_cells = Tuple{Vector{Int}, Vector{Float64}, Int, Float64}[]
    for (row_idx, (cols, vals)) in enumerate(obs)
        kept_cols = Int[]; kept_vals = Float64[]
        held_info = Tuple{Int, Float64}[]
        for (c, v) in zip(cols, vals)
            if (row_idx, c) in held_out
                push!(held_info, (c, v))
            else
                push!(kept_cols, c)
                push!(kept_vals, v)
            end
        end
        push!(train, (kept_cols, kept_vals))
        for (hc, hv) in held_info
            push!(test_cells, (kept_cols, kept_vals, hc, hv))
        end
    end

    return train, test_cells
end

# Conditional-Gaussian (best linear) prediction of a held-out cell from a row's
# observed cells, using Theta-hat = V V'. The textbook form is
#   ẑ_j = Θ̂[j,S] · pinv(Θ̂[S,S]) · z_S
# but Θ̂[S,S] = Vs Vs' is |S| x |S| and rank <= r, so for richly-observed rows
# (|S| >> r, e.g. the raw data) it is severely rank-deficient and pinv's
# near-zero singular values blow the prediction up to ~1e4. Since Θ̂ = V V', the
# formula collapses to a STABLE r-dimensional solve:
#   Θ̂[j,S]·pinv(Θ̂[S,S]) = Vj' Vs' pinv(Vs Vs') = Vj' pinv(Vs)
#   => ẑ_j = Vj' · pinv(Vs) · z_S
# pinv(Vs) is the least-squares pseudoinverse of the TALL, full-column-rank
# |S| x r factor matrix — well-conditioned, no giant near-singular matrix.
function predict_cell(V::AbstractMatrix, kept_cols::Vector{Int},
                      kept_vals::Vector{Float64}, j::Int)
    Vs = @view V[kept_cols, :]            # |S| x r  (tall, full column rank)
    Vj = @view V[j, :]                    # r
    dot(Vj, pinv(Vs) * kept_vals)         # = Vj' · pinv(Vs) · z_S
end

# Column-balance flag, mirroring R's BALANCE_HOLDOUT (set via OSMC_BALANCE env).
# Balanced (default): average per-column RMSE/R^2 so high-frequency columns don't
# dominate. Cell-weighted: one global mean over all held-out cells.
const OSMC_BALANCE = get(ENV, "OSMC_BALANCE", "1") != "0"

# When true, every column with >= 3 observations gets at least one held-out cell,
# mirroring R's ALLCOLHOLDOUT. Ensures all benchmarks are represented in scoring.
const OSMC_ALLCOLHOLDOUT = get(ENV, "OSMC_ALLCOLHOLDOUT", "1") != "0"

# Per-column accumulation of (residual^2, baseline^2 = z^2) keyed by column j.
function _percol_ss(V::AbstractMatrix, test_cells)
    ssr = Dict{Int, Float64}(); ssb = Dict{Int, Float64}(); n = Dict{Int, Int}()
    for (kc, kv, j, zj) in test_cells
        r2 = (predict_cell(V, kc, kv, j) - zj)^2
        ssr[j] = get(ssr, j, 0.0) + r2
        ssb[j] = get(ssb, j, 0.0) + zj^2
        n[j]   = get(n, j, 0) + 1
    end
    ssr, ssb, n
end

# Cell-level held-out RMSE (column-balanced unless balance=false).
function cell_rmse(V::AbstractMatrix, test_cells; balance::Bool = OSMC_BALANCE)
    isempty(test_cells) && return NaN
    if !balance
        s = sum((predict_cell(V, kc, kv, j) - zj)^2 for (kc, kv, j, zj) in test_cells)
        return sqrt(s / length(test_cells))
    end
    ssr, _, n = _percol_ss(V, test_cells)
    mean(sqrt(ssr[j] / n[j]) for j in keys(ssr))   # mean of per-column RMSE
end

# Cell-level held-out R^2 (column-balanced unless balance=false). Baseline =
# train column mean (= 0 in z-space), so per-cell baseline = z_j^2.
# R^2 = single pooled ratio of column-balanced MSE / baseline-MSE — NOT a mean of
# per-column R^2 (which blows up to -10..-50 on thin columns).
function cell_r2(V::AbstractMatrix, test_cells; balance::Bool = OSMC_BALANCE)
    isempty(test_cells) && return NaN
    ssr, ssb, n = _percol_ss(V, test_cells)
    if !balance
        tot_r = sum(values(ssr)); tot_b = sum(values(ssb))
        return tot_b == 0 ? NaN : 1 - tot_r / tot_b
    end
    mse  = mean(ssr[j] / n[j] for j in keys(ssr))   # mean of per-column MSE
    base = mean(ssb[j] / n[j] for j in keys(ssb))   # mean of per-column baseline MSE
    base == 0 ? NaN : 1 - mse / base
end

# ── DEAD BRANCH: product-level scoring (OSMC's native estimand) ──────────────
# Superseded by the cell-level metrics above for cross-method comparability with
# softimpute/iterativepca (which score cell reconstruction). Kept, not deleted:
# this is what OSMC *natively* optimises (reconstruction of Theta* entries =
# pairwise products), so it remains the right diagnostic for "is Theta-hat itself
# good" as opposed to "does Theta-hat predict cells". Reachable only by flipping
# OSMC_CELL_METRIC to false.
const OSMC_CELL_METRIC = true

# RMSE of reconstructed pairwise products <V_a, V_b> vs held-out za*zb.
function pairwise_rmse(V::AbstractMatrix, test_pairs)
    isempty(test_pairs) && return NaN
    s = 0.0
    for (a, b, prod) in test_pairs
        pred = dot(@view(V[a, :]), @view(V[b, :]))
        s += (pred - prod)^2
    end
    return sqrt(s / length(test_pairs))
end

# Baseline SS for product-R^2: predict every held-out product with the mean of
# the TRAINING products (built from train obs only, so the baseline is honest).
function train_product_mean(train::RaggedObs)
    s = 0.0; cnt = 0
    @inbounds for (cols, vals) in train
        k = length(cols)
        for j1 in 1:k, j2 in (j1 + 1):k
            s += vals[j1] * vals[j2]; cnt += 1
        end
    end
    cnt == 0 ? 0.0 : s / cnt
end

# Product-R^2 of fitted V: 1 - SS_resid / SS_baseline over held-out products.
function pairwise_r2(V::AbstractMatrix, test_pairs, base_pred::Float64)
    isempty(test_pairs) && return NaN
    ss_resid = 0.0; ss_base = 0.0
    for (a, b, prod) in test_pairs
        pred = dot(@view(V[a, :]), @view(V[b, :]))
        ss_resid += (pred - prod)^2
        ss_base  += (base_pred - prod)^2
    end
    ss_base == 0 ? NaN : 1 - ss_resid / ss_base
end

"""
    select_rank(obs, d; ...) -> (best_r, rs, rmse_by_r, r2_by_r)

Fit Theta-hat at each r on a held-out split and score the held-out cells with
CELL-level RMSE + R^2 (conditional-Gaussian prediction via Theta-hat; comparable
to softimpute). Returns the r minimising RMSE plus both full curves. The
product-level scoring is retained as a dead branch (OSMC_CELL_METRIC = false).

When `train` and `test_cells` are provided (pre-split, pre-standardized with
train-only moments), those are used directly and the `obs` argument is ignored
for holdout creation.
"""
function select_rank(obs::RaggedObs, d::Int; r_grid = 1:10, seed::Int = 1,
                     iters::Int = 1500, lr::Float64 = 0.05, frac::Float64 = 0.2,
                     train::Union{Nothing, RaggedObs} = nothing,
                     test_cells::Union{Nothing, Vector{Tuple{Vector{Int}, Vector{Float64}, Int, Float64}}} = nothing)
    rng = MersenneTwister(seed)
    rs = collect(r_grid)

    if OSMC_CELL_METRIC
        if train === nothing || test_cells === nothing
            train, test_cells = holdout_cells(obs; frac = frac, rng = rng)
        end
        if isempty(test_cells)
            @warn "no held-out cells for rank selection; defaulting to r=1"
            return 1, rs, fill(NaN, length(rs)), fill(NaN, length(rs))
        end
        rmse = Float64[]; r2 = Float64[]
        for r in rs
            V, _ = fit_onesided_ragged(train, d, r; iters = iters, lr = lr,
                                       rng = MersenneTwister(seed + r))
            push!(rmse, cell_rmse(V, test_cells))
            push!(r2,   cell_r2(V, test_cells))
        end
        best_r = all(isnan, rmse) ? 1 : rs[argmin(replace(rmse, NaN => Inf))]
        return best_r, rs, rmse, r2
    end

    # DEAD BRANCH: product-level scoring (OSMC's native estimand).
    train, test_pairs = holdout_pairs(obs; frac = frac, rng = rng)
    if isempty(test_pairs)
        @warn "no held-out pairs for rank selection; defaulting to r=1"
        return 1, rs, fill(NaN, length(rs)), fill(NaN, length(rs))
    end
    base_pred = train_product_mean(train)
    rmse = Float64[]; r2 = Float64[]
    for r in rs
        V, _ = fit_onesided_ragged(train, d, r; iters = iters, lr = lr,
                                   rng = MersenneTwister(seed + r))
        push!(rmse, pairwise_rmse(V, test_pairs))
        push!(r2,   pairwise_r2(V, test_pairs, base_pred))
    end
    best_r = all(isnan, rmse) ? 1 : rs[argmin(replace(rmse, NaN => Inf))]
    return best_r, rs, rmse, r2
end

# Synthesize an n x p surrogate with covariance ~ Theta-hat, then un-standardize
# back to original column scale (× σ, + μ). Z ~ N(0, I_p) so cov(Z W') = W W' =
# Theta-hat where W = Q sqrt(Λ).
function synthesize_surrogate(Θ::AbstractMatrix, n::Int, μ, σ;
                              rng::AbstractRNG = Random.default_rng())
    Θm = Matrix(Θ)
    if !all(isfinite, Θm)
        @warn "Theta-hat has non-finite entries; zeroing them before eigen"
        @. Θm = ifelse(isfinite(Θm), Θm, 0.0)
    end
    F = eigen(Symmetric(Θm))
    λ = max.(F.values, 0.0)              # clamp tiny negatives from fp
    W = F.vectors * Diagonal(sqrt.(λ))   # p x p
    Z = randn(rng, n, size(W, 1))
    Xz = Z * W'                          # standardized surrogate, cov ~ Θ
    p = length(μ)
    Xorig = similar(Xz)
    for j in 1:p
        @. Xorig[:, j] = Xz[:, j] * σ[j] + μ[j]
    end
    return Xorig
end

# RFC4180-quote a field if it contains a comma, quote, or newline (mirrors
# split_csv_line above). collapse_key values like
# `RL-IP/TT (Choubey et al., 2021)` contain a comma and must be quoted on
# write, or the row silently shifts columns on read-back.
function csv_escape(s::AbstractString)
    if occursin(',', s) || occursin('"', s) || occursin('\n', s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

# Write surrogate matrix in the imputed-table format consumed by R factoring.
function write_surrogate(path, keys, bench, X)
    open(path, "w") do io
        println(io, "collapse_key," * join(csv_escape.(bench), ","))
        for i in 1:size(X, 1)
            println(io, csv_escape(keys[i]) * "," * join(string.(@view X[i, :]), ","))
        end
    end
    @info "wrote surrogate" path
end

# Fit Theta-hat at rank r and synthesize a surrogate (n rows). Helper so the
# rank sweep and the best-r run share one code path.
function surrogate_at(obs::RaggedObs, p::Int, r::Int, n::Int, μ, σ;
                      seed::Int = 1, iters::Int = 2000, lr::Float64 = 0.05)
    V, _ = fit_onesided_ragged(obs, p, r; iters = iters, lr = lr,
                               rng = MersenneTwister(seed))
    synthesize_surrogate(theta_hat(V), n, μ, σ; rng = MersenneTwister(seed + 999))
end

"""
    run_onesided(in_path, out_dir, sweep_dir; r_grid, seed, ...)

Full pipeline for one densified table. Selects rank, then synthesizes a surrogate
at EVERY r in the grid (so the orchestrator can factor each for the dashboard's
per-r panels) plus the best-r surrogate. The best-r surrogate is the imputed
output (out_dir); per-r surrogates + the rank curve go to sweep_dir. The
orchestrator owns all factoring; this module only emits matrices.
"""
function run_onesided(in_path::AbstractString, out_dir::AbstractString,
                      sweep_dir::AbstractString;
                      r_grid = 1:10, seed::Int = 1,
                      iters::Int = 2000, lr::Float64 = 0.05)
    X, keys, bench = read_table(in_path)
    n, p = size(X)
    nobs = count(!ismissing, X)
    @info "onesided input" in_path n p density=round(100nobs/(n*p), digits=2)

    # Build raw ragged obs, create holdout FIRST, then compute train-only
    # column moments — avoids leaking held-out cell values into the
    # standardization (which would inflate R² and distort the correlation
    # matrix that the imputer sees).
    raw_obs = build_ragged_raw(X)
    rng_holdout = MersenneTwister(seed)
    raw_train, raw_test = holdout_cells(raw_obs; frac = 0.2, rng = rng_holdout)
    μ, σ = train_column_moments(raw_train, p)
    obs_train = standardize_ragged(raw_train, μ, σ)
    obs_full  = standardize_ragged(raw_obs, μ, σ)
    test_cells = standardize_test_cells(raw_test, μ, σ)

    best_r, rgrid, rmse, r2 = select_rank(obs_train, p; r_grid = r_grid, seed = seed,
                                           iters = max(1000, iters ÷ 2), lr = lr,
                                           train = obs_train, test_cells = test_cells)
    @info "rank selection" best_r rmse_by_r=collect(zip(rgrid, round.(rmse, digits=4)))

    mkpath(sweep_dir)
    # per-r surrogates for the dashboard's per-param factoring panels.
    for r in rgrid
        Xr = surrogate_at(obs_full, p, r, n, μ, σ; seed = seed, iters = iters, lr = lr)
        write_surrogate(joinpath(sweep_dir, "surrogate_r$(r).csv"), keys, bench, Xr)
    end
    # rank-selection curve (cell-level held-out RMSE + R^2 by default).
    open(joinpath(sweep_dir, "rank_sweep.csv"), "w") do io
        println(io, "r,rmse,r2")
        for (r, e, q) in zip(rgrid, rmse, r2); println(io, "$(r),$(e),$(q)"); end
    end

    # best-r surrogate = the imputed output.
    mkpath(out_dir)
    out_csv = joinpath(out_dir, "imputed_model_benchmark_table.csv")
    Xbest = surrogate_at(obs_full, p, best_r, n, μ, σ; seed = seed, iters = iters, lr = lr)
    write_surrogate(out_csv, keys, bench, Xbest)

    return (; best_r, rgrid, rmse, out_csv, sweep_dir, n, p)
end

"""
    sensitivity_onesided(in_path, out_csv; r_grid, n_seeds, ...)

Seed-sweep sensitivity. Because the data is MNAR, each seed draws a DIFFERENT
held-out split, so the best-r and the pairwise-RMSE curve vary seed to seed —
that spread is the sensitivity. Writes a CSV with one row per seed:
    seed, chosen_r, rmse_r1, rmse_r2, ..., rmse_rR
which R reads into the same sensitivity grid as the other methods.

Seeds are independent and the heavy part (refit at every r), so the sweep runs
over Threads.@threads — start Julia with `--threads` to use multiple cores.
`obs` is read-only and shared; each seed has its own RNG and its own output slot,
so there is no shared mutable state.
"""
function sensitivity_onesided(in_path::AbstractString, out_csv::AbstractString;
                              r_grid = 1:10, n_seeds::Int = 50,
                              iters::Int = 1000, lr::Float64 = 0.05)
    X, _, _ = read_table(in_path)
    n, p = size(X)
    raw_obs = build_ragged_raw(X)
    rs = collect(r_grid)

    @info "OSMC sensitivity seed-sweep" n_seeds nthreads = Threads.nthreads()
    chosen = Vector{Int}(undef, n_seeds)
    rmse_rows = Vector{Vector{Float64}}(undef, n_seeds)
    r2_rows   = Vector{Vector{Float64}}(undef, n_seeds)
    Threads.@threads for s in 1:n_seeds
        # Each seed gets its own holdout split with train-only moments.
        rng = MersenneTwister(s)
        raw_train, raw_test = holdout_cells(raw_obs; frac = 0.2, rng = rng)
        μ_s, σ_s = train_column_moments(raw_train, p)
        obs_train = standardize_ragged(raw_train, μ_s, σ_s)
        test_cells = standardize_test_cells(raw_test, μ_s, σ_s)
        best_r, _, rmse, r2 = select_rank(obs_train, p; r_grid = rs, seed = s,
                                           iters = iters, lr = lr,
                                           train = obs_train, test_cells = test_cells)
        chosen[s] = best_r
        rmse_rows[s] = rmse
        r2_rows[s]   = r2
    end

    mkpath(dirname(out_csv))
    open(out_csv, "w") do io
        header = "seed,chosen_r," *
                 join(["rmse_r$(r)" for r in rs], ",") * "," *
                 join(["r2_r$(r)" for r in rs], ",")
        println(io, header)
        for s in 1:n_seeds
            println(io, "$(s),$(chosen[s])," *
                        join(string.(rmse_rows[s]), ",") * "," *
                        join(string.(r2_rows[s]), ","))
        end
    end
    @info "wrote OSMC sensitivity" out_csv n_seeds
    return out_csv
end
