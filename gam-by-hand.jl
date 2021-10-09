#----------------------------------------
# This script sets out to replicate some
# of the code found at:  
# https://m-clark.github.io/generalized-additive-models/technical.html
#----------------------------------------

#----------------------------------------
# Author: Trent Henderson, 7 October 2021
#----------------------------------------

using Random, Distributions, GLM, Plots, StatsBase

# Simulate data

Random.seed!(123)
x = rand(Uniform(0, 1), 500)
μ = zeros(500)
y = zeros(500)

for i in 1:size(x, 1)
    μ[i] = sin(2 * (4 * x[i] - 2)) + 2 * exp(-(16 ^ 2) * ((x[i] - .5) ^ 2))
    y[i] = rand(Normal(μ[i][1], .3))
end

# Plot data

gr()
myPlot = plot(x, y, seriestype = :scatter, legend = false, markeralpha = 0.5, markercolor = :black)
display(myPlot)

#--------------- GAM routines ---------------

#---------------
# Generate knots
#---------------

knots = collect(0.0:0.1:0.9)
l = 1

# Generate polynomial splines

SplineMatrix = zeros(size(x, 1), size(knots, 1) + 1)

# Add a column of ones for intercept

for i in 1:size(SplineMatrix, 1)
    SplineMatrix[i, size(SplineMatrix, 2)] = 1
end

# Fill remainder of the matrix

for i in 1:size(SplineMatrix, 1)
    for j in 1:(size(SplineMatrix, 2) - 1)
        if x[i] >= knots[j]
            SplineMatrix[i, j] = (x[i] - knots[j]) ^ l 
        else
            SplineMatrix[i, j] = 0.0
        end
    end
end

# Reorder columns to get intercept first

ordering = vcat(size(SplineMatrix, 2), 1:(size(SplineMatrix, 2) - 1))
SplineMatrix = SplineMatrix[:, ordering]

# Plot

for i in 1:size(SplineMatrix, 2)
    plot!(x, SplineMatrix[:, i], color = palette(:default)[i], seriestype = :line, legend = false)
end

display(myPlot)

#-------------------
# Fit a linear model
# for each basis
# function
#-------------------

# Fit model and extract coefficients

m₁ = lm(SplineMatrix, y)
coefs = zeros(size(SplineMatrix, 2))

for i in 1:size(SplineMatrix, 2)
    coefs[i] = GLM.coef(m₁)[i]
end

# Get the knot each data point corresponds to

knotGroup = round.(Int, zeros(size(x)))

for i in 1:size(x, 1)
    for j in 1:size(knots, 1)
        if j == size(knots, 1)
            if x[i] > knots[j]
                knotGroup[i] = j
            else
            end
        else
            if x[i] > knots[j] && x[i] < (knots[j] + 1)
                knotGroup[i] = j
            else
            end
        end
    end
end

# Progress update plot

myPlot2 = plot(x, y, group = knotGroup, seriestype = :scatter, markeralpha = 0.7, legend = :topright, legendfontsize = 6, legendtitle = "Knot", legendtitlefont = (7))

display(myPlot2)

# Multiply each basis function by its coefficient

ScaledMatrix = SplineMatrix

for i in 1:size(ScaledMatrix, 1)
    for j in 1:size(ScaledMatrix, 2)
        ScaledMatrix[i, j] = ScaledMatrix[i, j] * coefs[j]
    end
end

# Create array with knot groupings to filter by for range restricted lines

ScaledMatrix = hcat(ScaledMatrix, x, y, knotGroup)

# Re-plot with lines

myPlot3 = plot(ScaledMatrix[:, (size(coefs, 1) + 1)], ScaledMatrix[:, (size(coefs, 1) + 2)], group = knotGroup, seriestype = :scatter, markeralpha = 0.3, legend = false)

for i in 2:(size(ScaledMatrix, 2) - 3)
    plot!(ScaledMatrix[(ScaledMatrix[:, size(ScaledMatrix, 2)] .== convert(Float64, (i - 1))), (size(ScaledMatrix, 2) - 2)], ScaledMatrix[(ScaledMatrix[:, size(ScaledMatrix, 2)] .== convert(Float64, (i - 1))), i], color = palette(:default)[i - 1], seriestype = :line, legend = false)
end

display(myPlot3)

#-------------------------------
# Sum the basis functions to get 
# better smoothed approximation
#-------------------------------

# Get fitted (summed) values from the linear model that form the basic spline

fittedValues = predict(m₁)
ScaledMatrix2 = hcat(ScaledMatrix, fittedValues)

# Re-plot

myPlot4 = plot(ScaledMatrix2[:, (size(coefs, 1) + 1)], ScaledMatrix2[:, (size(coefs, 1) + 2)], group = knotGroup, seriestype = :scatter, markeralpha = 0.2, legend = false)

for i in 1:size(knots, 1)
    plot!(ScaledMatrix2[(ScaledMatrix2[:, (size(ScaledMatrix2, 2) - 1)] .== convert(Float64, i)), (size(ScaledMatrix2, 2) - 3)], ScaledMatrix2[(ScaledMatrix2[:, (size(ScaledMatrix2, 2) - 1)] .== convert(Float64, i)), size(ScaledMatrix2, 2)], color = palette(:default)[i], seriestype = :line, legend = false)
end

display(myPlot4)

#------------------ Polynomial approach -----------------

#----------------------
# Define operations
# that can be used with
# GLM.jl
#
# NOTE: This is taken from:
# https://juliastats.org/StatsModels.jl/stable/internals/#An-example-of-custom-syntax:-poly-1
#----------------------

# Syntax: best practice to define a _new_ function

poly(x, n) = x^n

# Type of model where syntax applies: here this applies to any model type

const POLY_CONTEXT = Any

# struct for behavior

struct PolyTerm{T,D} <: AbstractTerm
    term::T
    deg::D
end

Base.show(io::IO, p::PolyTerm) = print(io, "poly($(p.term), $(p.deg))")

# For `poly` use at run-time (outside @formula), return a schema-less PolyTerm

poly(t::Symbol, d::Int) = PolyTerm(term(t), term(d))

# For `poly` use inside @formula: create a schemaless PolyTerm and apply_schema

function StatsModels.apply_schema(t::FunctionTerm{typeof(poly)},
                                  sch::StatsModels.Schema,
                                  Mod::Type{<:POLY_CONTEXT})
    apply_schema(PolyTerm(t.args_parsed...), sch, Mod)
end

# apply_schema to internal Terms and check for proper types

function StatsModels.apply_schema(t::PolyTerm,
                                  sch::StatsModels.Schema,
                                  Mod::Type{<:POLY_CONTEXT})
    term = apply_schema(t.term, sch, Mod)
    isa(term, ContinuousTerm) ||
        throw(ArgumentError("PolyTerm only works with continuous terms (got $term)"))
    isa(t.deg, ConstantTerm) ||
        throw(ArgumentError("PolyTerm degree must be a number (got $t.deg)"))
    PolyTerm(term, t.deg.n)
end

function StatsModels.modelcols(p::PolyTerm, d::NamedTuple)
    col = modelcols(p.term, d)
    reduce(hcat, [col.^n for n in 1:p.deg])
end

# The basic terms contained within a PolyTerm (for schema extraction)

StatsModels.terms(p::PolyTerm) = terms(p.term)

# Names variables from the data that a PolyTerm relies on

StatsModels.termvars(p::PolyTerm) = StatsModels.termvars(p.term)

# Number of columns in the matrix this term produces

StatsModels.width(p::PolyTerm) = p.deg

StatsBase.coefnames(p::PolyTerm) = coefnames(p.term) .* "^" .* string.(1:p.deg)

#----------------------
# Functionalise all of
# the above for easy
# extensibility
#----------------------

"""
    FitPolynomialSpline(x, y, k, l)

    Compute a basic univariate additive polynomial basis function spline fit to mimic the machinery of a generalised additive model (GAM). Uses basis functions composed of linear models over a given specification of knots and polynomial order. This is a functionalised generalisation to infinite knot and polynomial order space of the original post by Michael Clark at https://m-clark.github.io/generalized-additive-models/technical.html.

Usage:
```julia-repl
FitPolynomialSpline(x, y, k, l)
```

Arguments:
- `x` : Predictor variable.
- `y` : Response variable.
- `k` : Number of knots to use.
- `l` : Order of the polynomial.
"""
function FitPolynomialSpline(x::Array, y::Array, k::Int64 = 5, l::Int64 = 1)

    # Check arguments

    k < size(x, 1) || error("Number of knots `k` should be less than length of input `x`.")
    size(x, 1) == size(y, 1) || error("`x` and `y` should be the same length.")

    #---------- Basis function knot operations ----------

    # Genrate sequence of knots
    
    knots = collect(range(minimum(x), stop = maximum(x), length = k))

    # Instantiate matrix and add a column of ones for intercept

    SplineMatrix = zeros(size(x, 1), size(knots, 1) + 1)

    for i in 1:size(SplineMatrix, 1)
        SplineMatrix[i, size(SplineMatrix, 2)] = 1
    end

    # Fill remainder of the matrix

    for i in 1:size(SplineMatrix, 1)
        for j in 1:(size(SplineMatrix, 2) - 1)
            if x[i] >= knots[j]
                SplineMatrix[i, j] = (x[i] - knots[j]) ^ l 
            else
                SplineMatrix[i, j] = 0.0
            end
        end
    end

    # Reorder columns to get intercept first

    ordering = vcat(size(SplineMatrix, 2), 1:(size(SplineMatrix, 2) - 1))
    SplineMatrix = SplineMatrix[:, ordering]

    #---------- Linear model fit and coefficients ----------

    # Fit model

    m₁ = lm(SplineMatrix, y)
    coefs = zeros(size(SplineMatrix, 2))

    for i in 1:size(SplineMatrix, 2)
        coefs[i] = GLM.coef(m₁)[i]
    end

    # Get the knot each data point corresponds to

    knotGroup = round.(Int, zeros(size(x)))

    for i in 1:size(x, 1)
        for j in 1:size(knots, 1)
            if j == size(knots, 1)
                if x[i] > knots[j]
                    knotGroup[i] = j
                else
                end
            else
                if x[i] > knots[j] && x[i] < (knots[j] + 1)
                    knotGroup[i] = j
                else
                end
            end
        end
    end

    # Multiply each basis function by its coefficient

    ScaledMatrix = SplineMatrix

    for i in 1:size(ScaledMatrix, 1)
        for j in 1:size(ScaledMatrix, 2)
            ScaledMatrix[i, j] = ScaledMatrix[i, j] * coefs[j]
        end
    end

    # Create array with knot groupings to filter by for range restricted lines

    ScaledMatrix = hcat(ScaledMatrix, x, y, knotGroup)

    #---------- Sum fitted model values for spline -----------

    # Get fitted (summed) values from the linear model that form the basic spline

    fittedValues = predict(m₁)
    ScaledMatrix2 = hcat(ScaledMatrix, fittedValues)

    #---------- Plot ----------

    gr()
    
    myPlot = plot(ScaledMatrix2[:, (size(coefs, 1) + 1)], ScaledMatrix2[:, (size(coefs, 1) + 2)], group = knotGroup, seriestype = :scatter, markeralpha = 0.2, legend = false)

    for i in 1:size(knots, 1)
        plot!(ScaledMatrix2[(ScaledMatrix2[:, (size(ScaledMatrix2, 2) - 1)] .== convert(Float64, i)), (size(ScaledMatrix2, 2) - 3)], ScaledMatrix2[(ScaledMatrix2[:, (size(ScaledMatrix2, 2) - 1)] .== convert(Float64, i)), size(ScaledMatrix2, 2)], color = palette(:default)[i], seriestype = :line, legend = false)
    end
    
    return myPlot
end

# Run the function

FitPolynomialSpline(x, y, 10, 1)
