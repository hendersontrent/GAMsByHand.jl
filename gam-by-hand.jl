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

# Re-plot

myPlot3 = plot(ScaledMatrix[:, (size(coefs, 1) + 1)], ScaledMatrix[:, (size(coefs, 1) + 2)], group = knotGroup, seriestype = :scatter, markeralpha = 0.2, legend = false)

display(myPlot3)

for i in 2:(size(ScaledMatrix, 2) - 3)
    plot!(ScaledMatrix[(ScaledMatrix[:, size(ScaledMatrix, 2)] .== convert(Float64, (i - 1))), (size(ScaledMatrix, 2) - 2)], ScaledMatrix[(ScaledMatrix[:, size(ScaledMatrix, 2)] .== convert(Float64, (i - 1))), i], color = palette(:default)[i - 1], seriestype = :line, legend = false)
end

display(myPlot3)

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
