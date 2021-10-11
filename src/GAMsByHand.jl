module GAMsByHand

using Random, Distributions, GLM, Plots, StatsBase, DataFrames

include("PolyOperations.jl")
include("FitPolynomialBasisSpline.jl")
include("FitColouredLinearBasis.jl")

export FitPolynomialBasisSpline
export FitColouredLinearBasis

end
