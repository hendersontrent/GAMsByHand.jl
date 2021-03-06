#----------------------------------------------------
# Define poly operations that can be used with GLM.jl
#
# NOTE: This is taken from:
# https://juliastats.org/StatsModels.jl/stable/internals/#An-example-of-custom-syntax:-poly-1
#----------------------------------------------------

#-------------------------
# Overall target function:
#     poly(x, n) = x^n
#-------------------------

#----------------------------
# Struct and type definitions
#----------------------------

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

#---------------------
# Function definitions
#---------------------

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

#--------------------------
# Created terms definitions
#--------------------------

# The basic terms contained within a PolyTerm (for schema extraction)

StatsModels.terms(p::PolyTerm) = terms(p.term)

# Names variables from the data that a PolyTerm relies on

StatsModels.termvars(p::PolyTerm) = StatsModels.termvars(p.term)

# Number of columns in the matrix this term produces

StatsModels.width(p::PolyTerm) = p.deg
StatsBase.coefnames(p::PolyTerm) = coefnames(p.term) .* "^" .* string.(1:p.deg)
