#----------------------------------------
# This script sets out to replicate some
# of the code found at:  
# https://m-clark.github.io/generalized-additive-models/technical.html
#----------------------------------------

#----------------------------------------
# Author: Trent Henderson, 7 October 2021
#----------------------------------------

using Random, Distributions, GLM, Plots

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
myPlot = plot(x, y, seriestype = :scatter, legend = false, markeralpha = 0.3, markercolor = :black)

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

myPlot2 = plot(x, y, group = knotGroup, seriestype = :scatter, markeralpha = 0.2, legend = false)

display(myPlot2)

# Multiply each basis function by its coefficient

knotIndex = collect(1:1:10)
numRepeats = round(Int, size(SplineMatrix, 1) / size(knotIndex, 1))
knotVector = repeat(knotIndex, numRepeats)
ScaledSplineMatrix = ScaledSplineMatrix .* coefs[knotVector - 1]
coefs2 = coefs[2:size(coefs, 1)]
ScaledSplineMatrix = SplineMatrix

for i in 1:size(ScaledSplineMatrix, 1)
    for j in 1:size(ScaledSplineMatrix, 2)
        ScaledSplineMatrix[i, j] = ScaledSplineMatrix[i, j] * coefs2[knotVector[i]]
    end
end

# Re-plot

myPlot2 = plot(x, y, group = knotGroup, seriestype = :scatter, markeralpha = 0.2, legend = false)

display(myPlot2)

# Need to filter x to just each basis for appropriate plot length

for i in 2:size(ScaledSplineMatrix, 2)
    if (i - 1) == size(knots, 1)
        plot!(x[x .> knots[i - 1]], ScaledSplineMatrix[:, i], color = palette(:default)[i], seriestype = :line, legend = false)
    else
        plot!(x[(x .> knots[i - 1]) .& (x .< knots[i])], ScaledSplineMatrix[:, i], color = palette(:default)[i], seriestype = :line, legend = false)
    end
end

display(myPlot2)

#-------------------------------------------
# Switch to polynomial degree of 3 for cubic
# basis spline-esque fit
#-------------------------------------------

# NOTE: Try https://stackoverflow.com/questions/58265223/polynomial-regression-in-julia-glm

"""
    FitPolynomialSpline(x, y, k, l, doPlot)

    Compute a basic univariate additive polynomial basis function spline fit to mimic the machinery of a generalised additive model (GAM). Uses basis functions composed of linear models over a given specification of knots and polynomial order. This is a functionalised generalisation to infinite knot and polynomial order space of the original post by Michael Clark at https://m-clark.github.io/generalized-additive-models/technical.html.

Usage:
```julia-repl
FitPolynomialSpline(x, y, k, l, doPlot)
```

Arguments:
- `x` : Predictor variable.
- `y` : Response variable.
- `k` : Number of knots to use.
- `l` : Order of the polynomial.
- `doPlot` : Whether to plot the data or return the data matrix.
"""
function FitPolynomialSpline(x::Array, y::Array, k::Int64 = 5, l::Int64 = 3, doPlot = false)

    # Check arguments

    k < size(x, 1) || error("Number of knots `k` should be less than length of input `x`.")

    size(x, 1) == size(y, 1) || error("`x` and `y` should be the same length.")

    #---------- Basis function knot operations ----------

    # Genrate sequence of knots
    
    knots = collect(range(minimum(x), stop = maximum(x), length = k))

    # Instantiate matrix

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

    #---------- Linear model fit and coefficients ----------

    # Fit model

    m₁ = lm(SplineMatrix, y)
    coefs = zeros(size(SplineMatrix, 2))

    for i in 1:size(SplineMatrix, 2)
        coefs[i] = GLM.coef(m₁)[i]
    end

    # Multiply each basis function by its coefficient

    ScaledSplineMatrix

    #---------- Final returns ----------

    if doPlot == true
        gr()
        myPlot = plot(x, y, seriestype = :scatter, legend = false, markeralpha = 0.3, markercolor = :black, title = string("Polynomial spline fit with ", k, " knots and polynomial order ", l))
        return myPlot
    else
        return ScaledSplineMatrix
    end
end

FitPolynomialSpline(x, y, 9, 3, true)
