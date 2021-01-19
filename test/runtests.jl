# ------------------------------------------------------------------------------
# test suite
# ------------------------------------------------------------------------------

using Test, Documenter, LFAToolkit, LinearAlgebra
DocMeta.setdocmeta!(LFAToolkit, :DocTestSetup, :(using LFAToolkit); recursive = true)

# ------------------------------------------------------------------------------
# documentation tests
# ------------------------------------------------------------------------------

@testset "  documentation tests                " begin
    doctest(LFAToolkit; manual = false)
end

# ------------------------------------------------------------------------------
# mass matrix example
# ------------------------------------------------------------------------------

@testset "  ex1: mass example                  " begin
    include("../examples/ex1_mass.jl")

    @test min(eigenvalues...) ≈ 0.008379422444571976
    @test max(eigenvalues...) ≈ 0.17361111111111088
end

# ------------------------------------------------------------------------------
# diffusion operator example
# ------------------------------------------------------------------------------

@testset "  ex2: diffusion example             " begin
    include("../examples/ex2_diffusion.jl")

    @test min(eigenvalues...) ≈ 1.3284547608834352
    @test max(eigenvalues...) ≈ 8.524474960788456
end

# ------------------------------------------------------------------------------
# Jacobi smoother example
# ------------------------------------------------------------------------------

@testset "  ex3: Jacobi example                " begin
    include("../examples/ex3_jacobi.jl")

    @test min(eigenvalues...) ≈ -0.6289239142744161
    @test max(eigenvalues...) ≈ 0.6405931989084651
end

# ------------------------------------------------------------------------------
# p-multigrid example
# ------------------------------------------------------------------------------

@testset "  ex4: p-multigrid example           " begin
    include("../examples/ex4_pmultigrid.jl")

    @test max(eigenvalues...) ≈ 0.09000000000000002
end

# ------------------------------------------------------------------------------
# p-multigrid multilevel example
# ------------------------------------------------------------------------------

@testset "  ex5: p-multigrid multilevel example" begin
    include("../examples/ex5_pmultigrid_multilevel.jl")

    @test max(eigenvalues...) ≈ 0.2463055119550956
end

# ------------------------------------------------------------------------------
