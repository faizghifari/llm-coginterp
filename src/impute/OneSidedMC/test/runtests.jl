using Test
using OneSidedMC
using LinearAlgebra
using Random
using Statistics

# Gating: TEST env var selects which test groups to run.
#   "all"  (default) — core + MNAR
#   "core"           — everything except MNAR
#   "mnar"           — only MNAR (plus the lightweight unit tests)
# NSEEDS controls the seeds-per-parameter-set count for MNAR (default 10).
const TEST_MODE = lowercase(get(ENV, "TEST", "all"))
@assert TEST_MODE in ("all", "core", "mnar") "TEST must be all/core/mnar"

@testset "OneSidedMC ($TEST_MODE)" begin
    # Always-on lightweight unit tests.
    include("test_metrics.jl")
    include("test_data.jl")
    include("test_loss.jl")

    if TEST_MODE in ("all", "core")
        include("test_recovery.jl")
    end
    if TEST_MODE in ("all", "mnar")
        include("test_mnar.jl")
    end
end
