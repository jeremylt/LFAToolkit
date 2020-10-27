# ------------------------------------------------------------------------------
# Jacobi preconditioner
# ------------------------------------------------------------------------------

"""
```julia
Jacobi(operator)
```

Jacobi diagonal preconditioner for finite element operators

# Arguments:
- `operator`: finite element operator to precondition

# Returns:
- Jacobi preconditioner object

# Example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
basis = TensorH1LagrangeBasis(4, 4, 2);
    
function massweakform(u::Array{Float64}, w::Array{Float64})
    v = u*w[1]
    return [v]
end
    
# mass operator
inputs = [
    OperatorField(basis, [EvaluationMode.interpolation]),
    OperatorField(basis, [EvaluationMode.quadratureweights]),
];
outputs = [OperatorField(basis, [EvaluationMode.interpolation])];
mass = Operator(massweakform, mesh, inputs, outputs);

# preconditioner
jacobi = Jacobi(mass);

# verify
println(jacobi)
println(jacobi.operator)

# output
jacobi preconditioner
finite element operator:
2d mesh:
    dx: 1.0
    dy: 1.0

2 inputs:
operator field:
tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 4
    dimension: 2
  evaluation mode:
    interpolation
operator field:
tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 4
    dimension: 2
  evaluation mode:
    quadratureweights

1 output:
operator field:
tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 4
    dimension: 2
  evaluation mode:
    interpolation
```
"""
mutable struct Jacobi <: AbstractPreconditioner
    # data never changed
    operator::Operator

    # data empty until assembled
    operatordiagonalinverse::AbstractArray{Float64}

    # inner constructor
    Jacobi(operator) = new(operator)
end

# printing
# COV_EXCL_START
Base.show(io::IO, preconditioner::Jacobi) = print(io, "jacobi preconditioner")
# COV_EXCL_STOP

# ------------------------------------------------------------------------------
# data for computing symbols
# ------------------------------------------------------------------------------

"""
```julia
getoperatordiagonalinverse(preconditioner)
```

Compute or retrieve the inverse of the symbol matrix diagonal for a Jacobi
    preconditioner

# Returns:
- Symbol matrix diagonal inverse for the operator

# Example:
```jldoctest
# setup
mesh = Mesh1D(1.0);
basis = TensorH1LagrangeBasis(3, 4, 1);
    
function diffusionweakform(du::Array{Float64}, w::Array{Float64})
    dv = du*w[1]
    return [dv]
end
    
# mass operator
inputs = [
    OperatorField(basis, [EvaluationMode.gradient]),
    OperatorField(basis, [EvaluationMode.quadratureweights]),
];
outputs = [OperatorField(basis, [EvaluationMode.gradient])];
diffusion = Operator(diffusionweakform, mesh, inputs, outputs);

# preconditioner
jacobi = Jacobi(diffusion)

# note: either syntax works
diagonalinverse = LFAToolkit.getoperatordiagonalinverse(jacobi);
diagonalinverse = jacobi.operatordiagonalinverse;

# verify
@assert diagonalinverse ≈ [3/14 0; 0 3/16]
 
# output

```
"""
function getoperatordiagonalinverse(preconditioner::Jacobi)
    # assemble if needed
    if !isdefined(preconditioner, :operatordiagonalinverse)
        # retrieve diagonal and invert
        diagonalinverse = preconditioner.operator.diagonal^-1

        # store
        preconditioner.operatordiagonalinverse = diagonalinverse
    end

    # return
    return getfield(preconditioner, :operatordiagonalinverse)
end

# ------------------------------------------------------------------------------
# get/set property
# ------------------------------------------------------------------------------

function Base.getproperty(preconditioner::Jacobi, f::Symbol)
    if f == :operatordiagonalinverse
        return getoperatordiagonalinverse(preconditioner)
    else
        return getfield(preconditioner, f)
    end
end

function Base.setproperty!(preconditioner::Jacobi, f::Symbol, value)
    if f == :operator
        throw(ReadOnlyMemoryError()) # COV_EXCL_LINE
    else
        return setfield!(preconditioner, f, value)
    end
end

# ------------------------------------------------------------------------------
# compute symbols
# ------------------------------------------------------------------------------

"""
```julia
computesymbols(preconditioner, ω, θ)
```

Compute or retrieve the symbol matrix for a Jacobi preconditioned operator

# Arguments:
- `preconditioner`: Jacobi preconditioner to compute symbol matrix for
- `ω`:              Smoothing weighting factor array
- `θ`:              Fourier mode frequency array (one frequency per dimension)

# Returns:
- Symbol matrix for the Jacobi preconditioned operator

# Example:
```jldoctest
using LinearAlgebra

for dimension in 1:3
    # setup
    mesh = []
    if dimension == 1
        mesh = Mesh1D(1.0);
    elseif dimension == 2
        mesh = Mesh2D(1.0, 1.0);
    elseif dimension == 3
        mesh = Mesh3D(1.0, 1.0, 1.0);
    end
    basis = TensorH1LagrangeBasis(3, 4, dimension);
    
    function diffusionweakform(du::Array{Float64}, w::Array{Float64})
        dv = du*w[1]
        return [dv]
    end
    
    # diffusion operator
    inputs = [
        OperatorField(basis, [EvaluationMode.gradient]),
        OperatorField(basis, [EvaluationMode.quadratureweights]),
    ];
    outputs = [OperatorField(basis, [EvaluationMode.gradient])];
    diffusion = Operator(diffusionweakform, mesh, inputs, outputs);

    # preconditioner
    jacobi = Jacobi(diffusion);

    # compute symbols
    A = computesymbols(jacobi, [1.0], π*ones(dimension));

    # verify
    using LinearAlgebra;
    eigenvalues = real(eigvals(A));
    if dimension == 1
        @assert max(eigenvalues...) ≈ 1/7
    elseif dimension == 2
        @assert min(eigenvalues...) ≈ -1/14
    elseif dimension == 3
        @assert min(eigenvalues...) ≈ -0.33928571428571486
    end
end

# output

```
"""
function computesymbols(preconditioner::Jacobi, ω::Array, θ::Array)
    # validate number of parameters
    if length(ω) != 1
        Throw(error("only one parameter for Jacobi smoothing")) # COV_EXCL_LINE
    end

    # return
    return I -
           ω[1]*
           preconditioner.operatordiagonalinverse*
           computesymbols(preconditioner.operator, θ)
end

# ------------------------------------------------------------------------------