# ------------------------------------------------------------------------------
# Chebyshev smoother example
# ------------------------------------------------------------------------------

using LinearAlgebra

# setup
mesh = Mesh2D(1.0, 1.0)
p = 3

# diffusion operator
diffusion = GalleryOperator("diffusion", p + 1, p + 1, mesh)

# Chebyshev smoother
chebyshev = Chebyshev(diffusion)

# compute operator symbols
A = computesymbols(chebyshev, [3], [π, π])
eigenvalues = real(eigvals(A))

# ------------------------------------------------------------------------------
