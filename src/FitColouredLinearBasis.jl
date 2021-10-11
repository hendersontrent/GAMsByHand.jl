"""
    FitColouredLinearBasis(x, y, k)

Compute and plot a basic, coloured univariate additive linear basis function spline fit to mimic the machinery of a generalised additive model (GAM) and help with the underlying intuition. This is an adaptation of the original post by Michael Clark at https://m-clark.github.io/generalized-additive-models/technical.html.

Usage:
```julia-repl
FitColouredLinearBasis(x, y, k)
```

Arguments:
- `x` : Predictor variable.
- `y` : Response variable.
- `k` : Number of knots to use.
"""
function FitColouredLinearBasis(x::Array, y::Array, k::Int64 = 5)
    
    #---------------
    # Generate knots
    #---------------

    knots = collect(range(minimum(x), stop = maximum(x), length = k))
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

    #-------------------------------------------
    # Fit a linear model for each basis function
    #-------------------------------------------

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

    # Plot

    myPlot = plot(ScaledMatrix2[:, (size(coefs, 1) + 1)], ScaledMatrix2[:, (size(coefs, 1) + 2)], group = knotGroup, seriestype = :scatter, markeralpha = 0.2, legend = false)

    for i in 1:size(knots, 1)
        plot!(ScaledMatrix2[(ScaledMatrix2[:, (size(ScaledMatrix2, 2) - 1)] .== convert(Float64, i)), (size(ScaledMatrix2, 2) - 3)], ScaledMatrix2[(ScaledMatrix2[:, (size(ScaledMatrix2, 2) - 1)] .== convert(Float64, i)), size(ScaledMatrix2, 2)], color = palette(:default)[i], seriestype = :line, legend = false)
    end

    return myPlot
end