# dependencies
using LFAToolkit
using LinearAlgebra
using CSV
using DataFrames

# setup
p = 1
dimension = 2
mesh = Mesh2D(1.0, 1.0)
conditionnumbers = DataFrame()

# test range
for numberfineelements1d in [4, 6, 8, 10, 12, 14, 16, 32]
    println("m = ", numberfineelements1d)
    # setup
    # -- diffusion operators
    diffusion = GalleryMacroElementOperator("diffusion", p + 1, p + 1, numberfineelements1d, mesh)

    # -- smoothers
    lumpedbddc = LumpedBDDC(diffusion)
    dirichletbddc = DirichletBDDC(diffusion)
    bddc = [lumpedbddc, dirichletbddc]
    names = ["lumped", "dirichlet"]

    # compute smoothing factor
    # -- setup
    numbersteps = 32
    θ_min = -π / 2
    θ_max = 3π / 2
    θ_step = 2π / numbersteps
    θ_range = θ_min:θ_step:(θ_max-θ_step)

    # -- compute
    mineigenvalue = ones(2)
    maxeigenvalue = zeros(2)
    θ_mineigenvalue = -1 * ones(2, dimension)
    θ_maxeigenvalue = -1 * ones(2, dimension)
    for i = 1:numbersteps, j = 1:numbersteps
        θ = [θ_range[i], θ_range[j]]
        if sqrt(abs(θ[1])^2 + abs(θ[2])^2) > π / 128
            for b = 1:2
                M = computesymbols(bddc[b], [1.0], θ)
                eigenvalues = [abs(val) for val in eigvals(I - M)]
                currentmin = min(eigenvalues...)
                currentmax = max(eigenvalues...)
                if (currentmin < mineigenvalue[b])
                    mineigenvalue[b] = currentmin
                    θ_mineigenvalue[b, :] = θ / π
                end
                if (currentmax > maxeigenvalue[b])
                    maxeigenvalue[b] = currentmax
                    θ_maxeigenvalue[b, :] = θ / π
                end
            end
        end
    end
    for b = 1:2
        append!(
            conditionnumbers,
            DataFrame(
                p = p,
                m = numberfineelements1d,
                kind = names[b],
                θ = θ_maxeigenvalue[b, :],
                λ_min = mineigenvalue[b],
                λ_max = maxeigenvalue[b],
                κ = maxeigenvalue[b] / mineigenvalue[b],
            ),
        )
    end
end

CSV.write("bddc-condition-numbers.csv", conditionnumbers)
