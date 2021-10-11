using GAMsByHand
using Test
using Random, Distributions, Plots

@testset "GAMsByHand.jl" begin
    
    #--------------
    # Simulate data
    #--------------

    Random.seed!(123)
    x = rand(Uniform(0, 1), 500)
    μ = zeros(500)
    y = zeros(500)

    # Generate a nonlinear process as per https://m-clark.github.io/generalized-additive-models/technical.html

    for i in 1:size(x, 1)
        μ[i] = sin(2 * (4 * x[i] - 2)) + 2 * exp(-(16 ^ 2) * ((x[i] - .5) ^ 2))
        y[i] = rand(Normal(μ[i][1], .3))
    end

    #------------------
    # Run the functions
    #------------------

    # Cubic basis

    p = FitPolynomialBasisSpline(x, y, 10, 3)
    @test p isa Plots.Plot

    # Linear basis

    p1 = FitPolynomialBasisSpline(x, y, 10, 1)
    @test p1 isa Plots.Plot

    # Coloured linear basis

    p2 = FitColouredLinearBasis(x, y, 10)
    @test p2 isa Plots.Plot
end
