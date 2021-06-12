# dependencies
using LFAToolkit
using LinearAlgebra
using CSV
using DataFrames

# setup
dimension = 1
numbercomponents = 1
mesh = []
if dimension == 1
    mesh = Mesh1D(1.0)
elseif dimension == 2
    mesh = Mesh2D(1.0, 1.0)
end
smoothingparameters = DataFrame()

# test range
for P = 1:4
    println("p = ", 2^P)
    # setup
    # -- bases
    p = 2^P

    # -- diffusion operator
    diffusion = GalleryOperator("diffusion", p + 1, p + 1, mesh)

    # -- Jacobi smoother
    jacobi = Jacobi(diffusion)

    # compute smoothing factor
    # -- setup
    numberruns = 100
    maxeigenvalue = 0
    θ_min = -π / 2
    θ_min_high = π / 2
    θ_max = 3π / 2
    ω = 1.0

    # -- compute
    λ_minhigh = 100
    λ_maxhigh = -100
    λ_max = -100
    # -- 1D --
    if dimension == 1
        for i = 1:numberruns
            θ = [θ_min + (θ_max - θ_min) * i / numberruns]
            if abs(θ[1] % 2π) > π / 128
                A = computesymbols(jacobi, [ω], θ)
                eigenvalues = [abs(val) for val in eigvals(I - A)]
                if (θ[1] > θ_min_high)
                    λ_minhigh = min(λ_minhigh, eigenvalues...)
                    λ_maxhigh = max(λ_maxhigh, eigenvalues...)
                end
                λ_max = max(λ_max, eigenvalues...)
            end
        end
        # -- 2D --
    elseif dimension == 2
        for i = 1:numberruns, j = 1:numberruns
            θ = [
                θ_min + (θ_max - θ_min) * i / numberruns,
                θ_min + (θ_max - θ_min) * j / numberruns,
            ]
            if sqrt(abs(θ[1] % 2π)^2 + abs(θ[2] % 2π)^2) > π / 128
                A = computesymbols(jacobi, [ω], θ)
                eigenvalues = [abs(val) for val in eigvals(I - A)]
                if (θ[1] > θ_min_high || θ[2] > θ_min_high)
                    λ_minhigh = min(λ_minhigh, eigenvalues...)
                    λ_maxhigh = max(λ_maxhigh, eigenvalues...)
                end
                λ_max = max(λ_max, eigenvalues...)
            end
        end
    end
    ω_classical = 2 / (λ_minhigh + λ_maxhigh)
    ω_highorder = 2 / (λ_minhigh + λ_max)
    append!(
        smoothingparameters,
        DataFrame(p = p, ω_classical = ω_classical, ω_highorder = ω_highorder),
    )
end

CSV.write("jacobi-smoothing-factor-estimate.csv", smoothingparameters)