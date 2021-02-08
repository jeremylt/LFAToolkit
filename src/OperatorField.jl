# ------------------------------------------------------------------------------
# operator fields
# ------------------------------------------------------------------------------

"""
```julia
OperatorField(
    basis,
    evaluationmodes
)
```

Finite Element operator input or output, with a basis and evaluation mode

# Arguments:
- `basis`:           finite element basis for the field
- `evaluationmodes`: array of basis evaluation modes,
                         note that quadrature weights must be listed in a
                         separate operator field

# Returns:
- Finite element operator field object

# Example:
```jldoctest
# basis
basis = TensorH1LagrangeBasis(4, 3, 2);

# quadrature weights field
weightsfield = OperatorField(basis, [EvaluationMode.quadratureweights]);

# verify
println(weightsfield)

# input or output field
inputfield = OperatorField(basis, [
    EvaluationMode.interpolation,
    EvaluationMode.gradient,
]);

# verify
println(inputfield)

# output
operator field:
tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 3
    dimension: 2
  evaluation mode:
    quadratureweights
operator field:
tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 3
    dimension: 2
  evaluation modes:
    interpolation
    gradient
```
"""
struct OperatorField
    # data
    basis::AbstractBasis
    evaluationmodes::AbstractArray{EvaluationMode.EvalMode}
    name::String

    # inner constructor
    OperatorField(
        basis::AbstractBasis,
        evaluationmodes::AbstractArray{EvaluationMode.EvalMode},
    ) = (
        # validity checking
        if length(evaluationmodes) > 1 &&
           EvaluationMode.quadratureweights in evaluationmodes
            error("quadrature weights must be a separate operator field") # COV_EXCL_LINE
        end;

        # constructor
        new(basis, evaluationmodes)
    )

    # inner constructor
    OperatorField(
        basis::AbstractBasis,
        evaluationmodes::AbstractArray{EvaluationMode.EvalMode},
        name::String,
    ) = (
        # validity checking
        if length(evaluationmodes) > 1 &&
           EvaluationMode.quadratureweights in evaluationmodes
            error("quadrature weights must be a separate operator field") # COV_EXCL_LINE
        end;

        # constructor
        new(basis, evaluationmodes, name)
    )
end

# printing
# COV_EXCL_START
function Base.show(io::IO, field::OperatorField)
    print(io, "operator field:\n")

    # name
    if isdefined(field, :name)
        print(io, "  name:", field.name)
    end

    # basis
    print(io, field.basis)

    # evaluation modes
    if length(field.evaluationmodes) == 1
        print(io, "\n  evaluation mode:")
    else
        print(io, "\n  evaluation modes:")
    end
    for mode in field.evaluationmodes
        print(io, "\n    ", mode)
    end
end
# COV_EXCL_STOP

function Base.setproperty!(field::OperatorField, f::Symbol, value)
    if f == :basis || f == :evaluationmodes || f == :name
        throw(ReadOnlyMemoryError()) # COV_EXCL_LINE
    else
        return setfield!(operator, f, value)
    end
end

# ------------------------------------------------------------------------------
