# Allocation regression tests for PreallocationTools.jl
# These tests ensure that key functions remain zero-allocation at runtime.

using Test
using PreallocationTools
using ForwardDiff

@testset "Zero Allocation Tests" begin
    # Setup test data
    u_vec = ones(100)
    u_mat = ones(10, 10)
    chunk_size = ForwardDiff.pickchunksize(length(u_vec))
    DualType = ForwardDiff.Dual{Nothing, Float64, chunk_size}
    dual_vec = zeros(DualType, 100)
    dual_mat = zeros(DualType, 10, 10)

    @testset "DiffCache - Vector" begin
        cache = DiffCache(u_vec, chunk_size)

        # Warm up to populate caches
        get_tmp(cache, u_vec)
        get_tmp(cache, dual_vec)
        get_tmp(cache, first(u_vec))
        get_tmp(cache, first(dual_vec))

        # Test zero allocations
        @test (@allocated get_tmp(cache, u_vec)) == 0
        @test (@allocated get_tmp(cache, dual_vec)) == 0
        @test (@allocated get_tmp(cache, first(u_vec))) == 0
        @test (@allocated get_tmp(cache, first(dual_vec))) == 0
    end

    @testset "DiffCache - Matrix" begin
        cache = DiffCache(u_mat, chunk_size)

        # Warm up
        get_tmp(cache, u_mat)
        get_tmp(cache, dual_mat)

        # Test zero allocations
        @test (@allocated get_tmp(cache, u_mat)) == 0
        @test (@allocated get_tmp(cache, dual_mat)) == 0
    end

    @testset "FixedSizeDiffCache - Vector" begin
        cache = FixedSizeDiffCache(u_vec, chunk_size)

        # Warm up
        get_tmp(cache, u_vec)
        get_tmp(cache, dual_vec)
        get_tmp(cache, first(u_vec))
        get_tmp(cache, first(dual_vec))

        # Test zero allocations
        @test (@allocated get_tmp(cache, u_vec)) == 0
        @test (@allocated get_tmp(cache, dual_vec)) == 0
        @test (@allocated get_tmp(cache, first(u_vec))) == 0
        @test (@allocated get_tmp(cache, first(dual_vec))) == 0
    end

    @testset "FixedSizeDiffCache - Matrix" begin
        cache = FixedSizeDiffCache(u_mat, chunk_size)

        # Warm up
        get_tmp(cache, u_mat)
        get_tmp(cache, dual_mat)

        # Test zero allocations
        @test (@allocated get_tmp(cache, u_mat)) == 0
        @test (@allocated get_tmp(cache, dual_mat)) == 0
    end

    @testset "LazyBufferCache" begin
        lbc = LazyBufferCache()

        # Warm up (first call allocates the buffer, subsequent calls should not)
        get_tmp(lbc, u_vec)
        get_tmp(lbc, u_mat)

        # Test zero allocations on subsequent calls
        @test (@allocated get_tmp(lbc, u_vec)) == 0
        @test (@allocated get_tmp(lbc, u_mat)) == 0

        # Test with getindex syntax
        @test (@allocated lbc[u_vec]) == 0
        @test (@allocated lbc[u_mat]) == 0
    end

    @testset "LazyBufferCache with size mapping" begin
        lbc = LazyBufferCache(s -> 2 .* s)

        # Warm up
        get_tmp(lbc, u_vec)

        # Test zero allocations on subsequent calls
        @test (@allocated get_tmp(lbc, u_vec)) == 0
    end

    @testset "GeneralLazyBufferCache" begin
        glbc = GeneralLazyBufferCache(u -> zeros(eltype(u), size(u)))

        # Warm up (first call generates the buffer)
        get_tmp(glbc, u_vec)
        get_tmp(glbc, u_mat)

        # Additional warm-up to ensure all code paths are compiled
        for _ in 1:3
            get_tmp(glbc, u_vec)
            get_tmp(glbc, u_mat)
            glbc[u_vec]
            glbc[u_mat]
        end

        # Test zero allocations on subsequent calls
        @test (@allocated get_tmp(glbc, u_vec)) == 0
        @test (@allocated get_tmp(glbc, u_mat)) == 0

        # Test with getindex syntax
        @test (@allocated glbc[u_vec]) == 0
        @test (@allocated glbc[u_mat]) == 0
    end
end

@testset "Type Inference Tests" begin
    u = ones(10)
    chunk_size = ForwardDiff.pickchunksize(length(u))

    @testset "LazyBufferCache type inference" begin
        lbc = LazyBufferCache()
        get_tmp(lbc, u)  # warm up

        # Verify type inference
        @test @inferred(get_tmp(lbc, u)) isa Vector{Float64}
        @test @inferred(lbc[u]) isa Vector{Float64}
    end

    @testset "DiffCache type inference with normal arrays" begin
        cache = DiffCache(u, chunk_size)

        # Type inference for normal array input
        @test @inferred(get_tmp(cache, u)) isa Vector{Float64}
        @test @inferred(get_tmp(cache, 1.0)) isa Vector{Float64}
    end

    @testset "FixedSizeDiffCache type inference with normal arrays" begin
        cache = FixedSizeDiffCache(u, chunk_size)

        # Type inference for normal array input
        @test @inferred(get_tmp(cache, u)) isa Vector{Float64}
        @test @inferred(get_tmp(cache, 1.0)) isa Vector{Float64}
    end
end
