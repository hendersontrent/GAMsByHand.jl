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
plot(x, y, seriestype = :scatter, legend = false, markeralpha = 0.4, markercolor = :black)

#--------------- GAM routines ---------------

#---------------
# Generate knots
#---------------

knots = collect(0.0:0.1:0.9)
l = 1

# Generate polynomial splines

SplineMatrix = zeros(size(x, 1), size(knots, 1))

for i in 1:size(SplineMatrix, 1)
    for j in 1:size(SplineMatrix, 2)
        if x[i] >= knots[j]
            SplineMatrix[i, j] = (x[i] - knots[j]) ^ l 
        else
            SplineMatrix[i, j] = 0.0
        end
    end
end



# Plot



#-------------------
# Fit a linear model
# for each basis
# function
#-------------------

lmMod = lm(@formula(y ~ 0 + x), SplineMatrix)

# Pull out coefficients



# Multiply each basis function by the coefficient



# Plot



#----------------------
# Switch to polynomial
# degree of 3 for cubic
# basis spline-esque fit
#----------------------

# Functionalise all of the above and redo

function PolynomialSplineFit(x::Array, y::Array, k::Int64 = 5, l::Int64 = 3, doPlot = false)

    # Check knot argument

    k < size(x, 1) || error("Number of knots `k` should be less than length of input `x`.")

    # Do knot operations for basis functions
    
    knots = collect(range(minimum(x), stop = maximum(x), length = k))

    SplineMatrix = zeros(size(x, 1), size(k, 1))

    for i in 1:size(SplineMatrix, 1)
        for j in 1:size(SplineMatrix, 2)
            if x[i] >= knots[j]
                SplineMatrix[i, j] = (x[i] - knots[j]) ^ l 
            else
                SplineMatrix[i, j] = 0.0
            end
        end
    end

    # Fit linear model



    # Pull out coefficients



    # Multiply each basis function by the coefficient

    

    # Final returns

    if doPlot == true
        gr()
        myPlot = plot(x, y, seriestype = :scatter, legend = false, markeralpha = 0.4, markercolor = :black)
        return myPlot
    else
        return BasisMatix
    end
end

PolynomialSplineFit(x, y, 9, 3, true)
