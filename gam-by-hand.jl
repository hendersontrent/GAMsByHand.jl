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

# Multiply each basis function by the coefficient

ScaledSplineMatrix = SplineMatrix

for i in 1:size(ScaledSplineMatrix, 2)
    ScaledSplineMatrix[:, i] = ScaledSplineMatrix[:, i] * coefs[i]
end

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

knotGroup = string.("Knot ", knotGroup)

# Plot

myPlot2 = plot(x, y, group = knotGroup, seriestype = :scatter, markeralpha = 0.3, legend = false)

display(myPlot2)



#----------------------
# Switch to polynomial
# degree of 3 for cubic
# basis spline-esque fit
#----------------------

# Functionalise all of the above and redo

"""
    FitPolynomialSpline(x, y, k, l, doPlot)

    Compute a basic univariate polynomial spline fit to mimic the machinery of a generalised additive model (GAM) using basis functions composed of linear models over a given specification of knots and polynomial order.

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

    # Check knot argument

    k < size(x, 1) || error("Number of knots `k` should be less than length of input `x`.")

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

    # Multiply each basis function by the coefficient

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
