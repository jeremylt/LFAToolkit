# ------------------------------------------------------------------------------
# p-multigrid example
# ------------------------------------------------------------------------------

# setup
mesh = Mesh2D(1.0, 1.0)
finep = 3
coarsep = 2
dimension = 2
finebasis = TensorH1LagrangeBasis(finep, finep, dimension)
coarsebasis = TensorH1LagrangeBasis(coarsep, finep, dimension)
ctofbasis = TensorH1LagrangeBasis(coarsep, finep, dimension, lagrangequadrature = true)

function diffusionweakform(du::Array{Float64}, w::Array{Float64})
    dv = du*w[1]
    return [dv]
end

# diffusion operators
finediffusion = GalleryOperator("diffusion", finep, finep, mesh)
coarsediffusion = GalleryOperator("diffusion", coarsep, finep, mesh)

# Jacobi smoother
jacobi = Jacobi(finediffusion)

# p-multigrid preconditioner
multigrid = PMultigrid(finediffusion, coarsediffusion, jacobi, [ctofbasis])

# compute operator symbols
A = computesymbols(multigrid, [0.7], [1, 1], [π, π])
eigenvalues = real(eigvals(A))

# ------------------------------------------------------------------------------
