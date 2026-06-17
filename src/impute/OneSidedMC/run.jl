# ─────────────────────────────────────────────────────────────────────────────
# OneSidedMC real-data driver.
#
# For each densified table, recover Theta-hat and write a covariance-matched
# surrogate to data/imputed/onesidedmc/<densifier>/<strategy>/. Also writes a
# small rank-selection sweep CSV per table for the orchestrator's plots.
#
# Usage (from repo root):
#   julia --project=impute/OneSidedMC impute/OneSidedMC/run.jl
# Optional args: densifiers and strategies can be overridden via ENV
#   OSMC_DENSIFIERS=C,R,S  OSMC_STRATEGIES=all_standard,all_aggressive
# ─────────────────────────────────────────────────────────────────────────────

import Pkg
Pkg.activate(joinpath(@__DIR__))

using OneSidedMC
include(joinpath(@__DIR__, "src", "pipeline.jl"))

# @__DIR__ = src/impute/OneSidedMC -> repo root is three levels up.
const REPO = abspath(joinpath(@__DIR__, "..", "..", ".."))

getlist(env, default) = haskey(ENV, env) ? split(ENV[env], ',') : default

function main()
    densifiers = getlist("OSMC_DENSIFIERS", ["raw", "C", "S", "R"])
    strategies = getlist("OSMC_STRATEGIES", ["all_standard", "all_aggressive"])
    # DATA_ROOT lets the smoke fixture reuse this driver (absolute or repo-rel).
    data_root = get(ENV, "OSMC_DATA_ROOT", joinpath(REPO, "data"))
    isabspath(data_root) || (data_root = joinpath(REPO, data_root))

    for dz in densifiers, st in strategies
        # "raw" = undensified table under combinations/; C/S/R = combinations_<dz>/
        subdir = dz == "raw" ? "combinations" : "combinations_$(dz)"
        in_path = joinpath(data_root, subdir, st, "model_benchmark_table.csv")
        if !isfile(in_path)
            @warn "missing input, skipping" in_path
            continue
        end
        out_dir = joinpath(data_root, "imputed", "onesidedmc", dz, st)
        # per-r surrogates + curve are intermediate plotting inputs, staged under
        # results/_osmc_sweep so they don't pollute data/imputed (CSVs only there).
        results_root = get(ENV, "OSMC_RESULTS_ROOT", joinpath(REPO, "results"))
        sweep_dir = joinpath(results_root, "_osmc_sweep", "$(dz)_$(st)")
        @info "===== onesidedmc =====" densifier=dz strategy=st

        res = run_onesided(in_path, out_dir, sweep_dir; r_grid = 1:10, seed = 1)
        @info "done" best_r=res.best_r out=res.out_csv sweep=res.sweep_dir

        # opt-in seed-sweep sensitivity (slow); R reads the CSV for plotting.
        if get(ENV, "OSMC_SENSITIVITY", "") != ""
            @info "  sensitivity seed-sweep..."
            sensitivity_onesided(in_path,
                joinpath(sweep_dir, "sensitivity.csv"); r_grid = 1:10, n_seeds = 50)
        end
    end
end

main()
