"""
    FitPolynomialBasisSpline(x, y, k, l)

Compute a basic univariate additive polynomial basis function spline fit to mimic the machinery of a generalised additive model (GAM). Uses basis functions composed of linear models over a given specification of knots and polynomial order. This is a functionalised generalisation to infinite knot and polynomial order space of the original post by Michael Clark at https://m-clark.github.io/generalized-additive-models/technical.html.

Usage:
```julia-repl
FitPolynomialBasisSpline(x, y, k, l)
```

Arguments:
- `x` : Predictor variable.
- `y` : Response variable.
- `k` : Number of knots to use.
- `l` : Order of the polynomial.
"""
function FitPolynomialBasisSpline(x::Array, y::Array, k::Int64 = 5, l::Int64 = 3)

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

    #---------- Modelling and coefficient extraction -------

    if l >= 2

        # Set up dynamic modelling for polynomial regression

        SplineMatrixDF = hcat(y, SplineMatrix)
        SplineMatrixDF = DataFrame(SplineMatrixDF, :auto)
        SplineMatrixDF = rename!(SplineMatrixDF, :x1 => :y)
        MyNames = names(SplineMatrixDF)
        MySymbols = Symbol.(MyNames[2:size(MyNames, 1)])
        poly_vars = tuple(MySymbols...,)
        poly_deg = l
        poly_formula = term(:y) ~ term(0) + poly.(poly_vars, poly_deg)
        m₁ = fit(LinearModel, poly_formula, SplineMatrixDF)

        # Extract coefficients 

        coefs = DataFrame(coeftable(m₁))
        coefs = Array(coefs[:, 2])
        orderCoefs = collect(poly_deg:poly_deg:((size(SplineMatrixDF, 2) - 1) * 3))
        orderCoefsFiltered = coefs[filter(x -> (x in orderCoefs), eachindex(coefs))]

        # Get knot groups to help plot
        
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

        # Multiply each basis by its coefficient
        
        ScaledMatrix = SplineMatrix
        
        for i in 1:size(ScaledMatrix, 1)
            for j in 1:size(ScaledMatrix, 2)
                ScaledMatrix[i, j] = ScaledMatrix[i, j] * orderCoefsFiltered[j]
            end
        end

        # Create a single matrix for plotting
        
        ScaledMatrix = hcat(ScaledMatrix, x, y, knotGroup)
        fittedValues = predict(m₁)
        ScaledMatrix2 = hcat(ScaledMatrix, fittedValues)
        
        #--------- Plot -----------

        gr()
        
        myPlot = plot(ScaledMatrix2[:, (size(orderCoefs, 1) + 1)], ScaledMatrix2[:, (size(orderCoefs, 1) + 2)], seriestype = :scatter, markercolor = :black, markeralpha = 0.3, legend = false)
        
        for i in 1:size(knots, 1)
            plot!(ScaledMatrix2[(ScaledMatrix2[:, (size(ScaledMatrix2, 2) - 1)] .== convert(Float64, i)), (size(ScaledMatrix2, 2) - 3)], ScaledMatrix2[(ScaledMatrix2[:, (size(ScaledMatrix2, 2) - 1)] .== convert(Float64, i)), size(ScaledMatrix2, 2)], color = palette(:default)[1], seriestype = :line, legend = false, linewidth = 2.0)
        end
    
    else

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

        gr()

        #--------- Plot -----------

        myPlot = plot(ScaledMatrix2[:, (size(coefs, 1) + 1)], ScaledMatrix2[:, (size(coefs, 1) + 2)], group = knotGroup, seriestype = :scatter, markercolor = :black, markeralpha = 0.2, legend = false)

        for i in 1:size(knots, 1)
            plot!(ScaledMatrix2[(ScaledMatrix2[:, (size(ScaledMatrix2, 2) - 1)] .== convert(Float64, i)), (size(ScaledMatrix2, 2) - 3)], ScaledMatrix2[(ScaledMatrix2[:, (size(ScaledMatrix2, 2) - 1)] .== convert(Float64, i)), size(ScaledMatrix2, 2)], color = palette(:default)[1], seriestype = :line, legend = false, linewidth = 2.0)
        end
    end
    
    return myPlot
end
