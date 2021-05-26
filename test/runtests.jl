# ------------------------------------------------------------------------------
# test suite
# ------------------------------------------------------------------------------

using Test, Documenter, LFAToolkit, LinearAlgebra
DocMeta.setdocmeta!(LFAToolkit, :DocTestSetup, :(using LFAToolkit); recursive = true)

# ------------------------------------------------------------------------------
# documentation tests
# ------------------------------------------------------------------------------

@testset "  documentation tests                        " begin
    doctest(LFAToolkit; manual = false)
end

# ------------------------------------------------------------------------------
# mass matrix example
# ------------------------------------------------------------------------------

@testset "  ex001: mass example                        " begin
    include("../examples/ex001_mass.jl")

    @test min(eigenvalues...) ≈ 0.008379422444571976
    @test max(eigenvalues...) ≈ 0.17361111111111088
end

# ------------------------------------------------------------------------------
# diffusion operator example
# ------------------------------------------------------------------------------

@testset "  ex002: diffusion example                   " begin
    include("../examples/ex002_diffusion.jl")

    @test min(eigenvalues...) ≈ 1.3284547608834352
    @test max(eigenvalues...) ≈ 8.524474960788456
end

# ------------------------------------------------------------------------------
# Jacobi smoother example
# ------------------------------------------------------------------------------

@testset "  ex101: Jacobi example                      " begin
    include("../examples/ex101_jacobi.jl")

    @test min(eigenvalues...) ≈ -0.6289239142744161
    @test max(eigenvalues...) ≈ 0.6405931989084651
end

# ------------------------------------------------------------------------------
# Chebyshev smoother example
# ------------------------------------------------------------------------------

@testset "  ex111: Chebyshev example                   " begin
    include("../examples/ex111_chebyshev.jl")

    @test min(eigenvalues...) ≈ -0.4154429715651581
    @test max(eigenvalues...) ≈ 0.1854866859631604
end

# ------------------------------------------------------------------------------
# p-multigrid example
# ------------------------------------------------------------------------------

@testset "  ex201: p-multigrid example                 " begin
    include("../examples/ex201_pmultigrid.jl")

    @test max(eigenvalues...) ≈ 0.023313871226588227
end

# ------------------------------------------------------------------------------
# p-multigrid multilevel example
# ------------------------------------------------------------------------------

@testset "  ex202: p-multigrid multilevel example      " begin
    include("../examples/ex202_pmultigrid_multilevel.jl")

    @test max(eigenvalues...) ≈ 0.5292890335817175
end

# ------------------------------------------------------------------------------
# h-multigrid example
# ------------------------------------------------------------------------------

@testset "  ex211: h-multigrid example                 " begin
    include("../examples/ex211_hmultigrid.jl")

    @test max(eigenvalues...) ≈ 0.029660493827160444
end

# ------------------------------------------------------------------------------
# h-multigrid multilevel example
# ------------------------------------------------------------------------------

@testset "  ex212: h-multigrid multilevel example      " begin
    include("../examples/ex212_hmultigrid_multilevel.jl")

    @test max(eigenvalues...) ≈ 0.07185378086419703
end

# ------------------------------------------------------------------------------
# lumped BDDC example
# ------------------------------------------------------------------------------

@testset "  ex221: lumped BDDC example                 " begin
    include("../examples/ex221_lumped_bddc.jl")

    @test min(eigenvalues...) ≈ 0.9999999999999992
    @test max(eigenvalues...) ≈ 5.999999999999997
end

# ------------------------------------------------------------------------------
# lumped BDDC on macro elements example
# ------------------------------------------------------------------------------

@testset "  ex222: lumped BDDC macro element example   " begin
    include("../examples/ex222_lumped_bddc.jl")

    @test min(eigenvalues...) ≈ 0.9999999999999989
    @test max(eigenvalues...) ≈ 4.443546705148796
end

# ------------------------------------------------------------------------------
# Dirichlet BDDC example
# ------------------------------------------------------------------------------

@testset "  ex223: Dirichlet BDDC example              " begin
    include("../examples/ex223_dirichlet_bddc.jl")

    @test min(eigenvalues...) ≈ 0.9999999997897949
    @test max(eigenvalues...) ≈ 5.999999999999993
end

# ------------------------------------------------------------------------------
# Dirichlet BDDC on macro elements example
# ------------------------------------------------------------------------------

@testset "  ex224: Dirichlet BDDC macro element example" begin
    include("../examples/ex224_dirichlet_bddc.jl")

    @test min(eigenvalues...) ≈ 0.9999999911836086
    @test max(eigenvalues...) ≈ 4.443546705148796
end

# ------------------------------------------------------------------------------
# solid mechanics example
# ------------------------------------------------------------------------------

@testset "  ex301: solid mechanics example             " begin
    include("../examples/ex301_solid_mechanics.jl")

    @test max(eigenvalues...) ≈ 0.2851697038981916
end

# ------------------------------------------------------------------------------
