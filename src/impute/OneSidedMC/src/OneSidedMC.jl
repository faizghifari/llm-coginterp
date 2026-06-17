module OneSidedMC

using LinearAlgebra
using Random
using Statistics
using Printf

include("data.jl")
include("metrics.jl")
include("loss.jl")
include("algorithm.jl")
include("baselines.jl")
include("realdata.jl")

export gaussian_factors, sample_observations, sample_observations_weighted,
       loss_and_grad,
       fit_onesided, right_singular_vectors,
       direct_factorization, factorization_without_diagonal,
       full_matrix_completion,
       rowspace_error, mean_principal_angle_deg,
       # real-data extensions (ragged observations from CSV tables)
       RaggedObs, ragged_loss_and_grad, fit_onesided_ragged, theta_hat

end # module
