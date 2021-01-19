# ------------------------------------------------------------------------------
# p-multigrid multilevel example
# ------------------------------------------------------------------------------

# setup
mesh = Mesh2D(1.0, 1.0)
finep = 5
midp = 3
coarsep = 2
dimension = 2
ctombasis = TensorH1LagrangeBasis(coarsep, midp, dimension, lagrangequadrature = true)
mtofbasis = TensorH1LagrangeBasis(midp, finep, dimension, lagrangequadrature = true)

# diffusion operators
finediffusion = GalleryOperator("diffusion", finep, finep, mesh)
middiffusion = GalleryOperator("diffusion", midp, finep, mesh)
coarsediffusion = GalleryOperator("diffusion", coarsep, finep, mesh)

# Jacobi smoothers
finejacobi = Jacobi(finediffusion)
midjacobi = Jacobi(middiffusion)

# p-multigrid preconditioner
midmultigrid = PMultigrid(middiffusion, coarsediffusion, midjacobi, [ctombasis])
multigrid = PMultigrid(finediffusion, midmultigrid, finejacobi, [mtofbasis])

# compute operator symbols
A = computesymbols(multigrid, [0.7], [1, 1], [π, π])
eigenvalues = real(eigvals(A))

# ------------------------------------------------------------------------------
