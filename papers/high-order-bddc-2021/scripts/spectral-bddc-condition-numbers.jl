# dependencies
using LFAToolkit
using LinearAlgebra
using CSV
using DataFrames

# setup
numberelements1d = 2
dimension = 2
mesh = Mesh2D(1.0, 1.0)
conditionnumbers = DataFrame()

# test range
for p in [2, 3, 4, 8, 16, 32]
    println("p = ", p)
    # setup
    # -- diffusion operators
    diffusion = GalleryMacroElementOperator("diffusion", p + 1, p + 1, numberelements1d, mesh)

    # -- smoothers
    bddc = DirichletBDDC(diffusion)

    # compute smoothing factor
    # -- setup
    numbersteps = 32
    θ_min = -π / 2
    θ_max = 3π / 2
    θ_step = 2π / numbersteps
    θ_range = θ_min:θ_step:(θ_max-θ_step)

    # -- compute
    mineigenvalue = 1
    maxeigenvalue = 0
    θ_mineigenvalue = -1 * ones(dimension)
    θ_maxeigenvalue = -1 * ones(dimension)
    for i = 1:numbersteps, j = 1:numbersteps
        θ = [θ_range[i], θ_range[j]]
        if sqrt(abs(θ[1])^2 + abs(θ[2])^2) > π / 128
            M = computesymbols(bddc, [1.0], θ)
            eigenvalues = [abs(val) for val in eigvals(I - M)]
            currentmin = min(eigenvalues...)
            currentmax = max(eigenvalues...)
            if (currentmin < mineigenvalue)
                mineigenvalue = currentmin
                θ_mineigenvalue = θ / π
            end
            if (currentmax > maxeigenvalue)
                maxeigenvalue = currentmax
                θ_maxeigenvalue = θ / π
            end
        end
    end
    append!(
        conditionnumbers,
        DataFrame(
            p = p,
            m = numberfineelements1d,
            kind = "Dirichlet",
            θ = θ_maxeigenvalue,
            λ_min = mineigenvalue,
            λ_max = maxeigenvalue,
            κ = maxeigenvalue / mineigenvalue,
        ),
    )
end

CSV.write("spectral-bddc-condition-numbers.csv", conditionnumbers)
