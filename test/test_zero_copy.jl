using Test, PreallocationTools, ForwardDiff

@testset "zero and copy dispatches" begin
    @testset "DiffCache" begin
        u = rand(10)
        cache = DiffCache(u, 5)

        # Test zero
        zero_cache = zero(cache)
        @test isa(zero_cache, DiffCache)
        @test all(zero_cache.du .== 0)
        @test all(zero_cache.dual_du .== 0)
        @test isempty(zero_cache.any_du)

        # Test copy
        copy_cache = copy(cache)
        @test isa(copy_cache, DiffCache)
        @test copy_cache.du == cache.du
        @test copy_cache.dual_du == cache.dual_du
        @test copy_cache.any_du == cache.any_du
        # Ensure it's a copy, not a reference
        copy_cache.du[1] = -999
        @test cache.du[1] != -999
    end

    @testset "FixedSizeDiffCache" begin
        u = rand(10)
        cache = FixedSizeDiffCache(u, Val{5})

        # Test zero
        zero_cache = zero(cache)
        @test isa(zero_cache, FixedSizeDiffCache)
        @test all(zero_cache.du .== 0)
        @test all(zero_cache.dual_du .== 0)
        @test isempty(zero_cache.any_du)

        # Test copy
        copy_cache = copy(cache)
        @test isa(copy_cache, FixedSizeDiffCache)
        @test copy_cache.du == cache.du
        @test copy_cache.dual_du == cache.dual_du
        @test copy_cache.any_du == cache.any_du
        # Ensure it's a copy, not a reference
        copy_cache.du[1] = -999
        @test cache.du[1] != -999
    end

    @testset "LazyBufferCache" begin
        lbc = LazyBufferCache(identity; initializer! = buf -> fill!(buf, 0.0))
        u = rand(10)
        buf = lbc[u]  # Create a buffer in the cache

        # Test zero - creates a new empty cache with same configuration
        zero_lbc = zero(lbc)
        @test isa(zero_lbc, LazyBufferCache)
        @test isempty(zero_lbc.bufs)
        @test zero_lbc.sizemap === lbc.sizemap
        @test zero_lbc.initializer! === lbc.initializer!

        # Test copy
        copy_lbc = copy(lbc)
        @test isa(copy_lbc, LazyBufferCache)
        @test copy_lbc.sizemap === lbc.sizemap
        @test copy_lbc.initializer! === lbc.initializer!
        # Check that buffers were copied
        @test !isempty(copy_lbc.bufs)
        # Modify the copy to ensure it's independent
        buf_copy = copy_lbc[u]
        buf_copy[1] = -999
        @test buf[1] != -999
    end

    @testset "GeneralLazyBufferCache" begin
        glbc = GeneralLazyBufferCache(u -> similar(u))
        u = rand(10)
        buf = glbc[u]  # Create a buffer in the cache

        # Test zero - creates a new empty cache with same function
        zero_glbc = zero(glbc)
        @test isa(zero_glbc, GeneralLazyBufferCache)
        @test isempty(zero_glbc.bufs)
        @test zero_glbc.f === glbc.f

        # Test copy
        copy_glbc = copy(glbc)
        @test isa(copy_glbc, GeneralLazyBufferCache)
        @test copy_glbc.f === glbc.f
        # Check that buffers were copied
        @test !isempty(copy_glbc.bufs)
        # Modify the copy to ensure it's independent
        buf_copy = copy_glbc[u]
        buf_copy[1] = -999
        @test buf[1] != -999
    end

    @testset "DiffCache with matrix" begin
        u = rand(5, 5)
        cache = DiffCache(u, 3)

        # Test zero
        zero_cache = zero(cache)
        @test isa(zero_cache, DiffCache)
        @test size(zero_cache.du) == size(u)
        @test all(zero_cache.du .== 0)

        # Test copy
        copy_cache = copy(cache)
        @test isa(copy_cache, DiffCache)
        @test size(copy_cache.du) == size(u)
        @test copy_cache.du == cache.du
        # Ensure it's a copy, not a reference
        copy_cache.du[1, 1] = -999
        @test cache.du[1, 1] != -999
    end
end

@testset "fill! dispatches" begin
    @testset "DiffCache fill!" begin
        u = rand(10)
        cache = DiffCache(u, 5)

        # Fill with non-zero values initially
        fill!(cache.du, 1.0)
        fill!(cache.dual_du, 2.0)
        push!(cache.any_du, 3.0)

        # Test fill! with 0
        fill!(cache, 0.0)
        @test all(cache.du .== 0)
        @test all(cache.dual_du .== 0)
        @test all(cache.any_du .=== nothing)

        # Test fill! with other values
        fill!(cache, 5.0)
        @test all(cache.du .== 5.0)
        @test all(cache.dual_du .== 5.0)
    end

    @testset "FixedSizeDiffCache fill!" begin
        u = rand(10)
        cache = FixedSizeDiffCache(u, Val{5})

        # Fill with non-zero values initially
        fill!(cache.du, 1.0)
        fill!(cache.dual_du, 2.0)
        push!(cache.any_du, 3.0)

        # Test fill! with 0
        fill!(cache, 0.0)
        @test all(cache.du .== 0)
        @test all(cache.dual_du .== 0)
        @test all(cache.any_du .=== nothing)

        # Test fill! with other values
        fill!(cache, 3.0)
        @test all(cache.du .== 3.0)
        @test all(cache.dual_du .== 3.0)
    end

    @testset "LazyBufferCache fill!" begin
        lbc = LazyBufferCache(identity)
        u = rand(10)
        v = rand(5, 5)

        # Create and fill buffers
        buf1 = lbc[u]
        fill!(buf1, 1.0)
        buf2 = lbc[v]
        fill!(buf2, 2.0)

        # Test fill! with 0
        fill!(lbc, 0.0)
        @test all(buf1 .== 0)
        @test all(buf2 .== 0)
        # Check that the buffers are still in the cache
        @test lbc[u] === buf1
        @test lbc[v] === buf2

        # Test fill! with other values
        fill!(lbc, 7.0)
        @test all(buf1 .== 7.0)
        @test all(buf2 .== 7.0)
    end

    @testset "GeneralLazyBufferCache fill!" begin
        glbc = GeneralLazyBufferCache(u -> similar(u))
        u = rand(10)

        # Create and fill buffer
        buf = glbc[u]
        fill!(buf, 1.0)

        # Test fill! with 0
        fill!(glbc, 0.0)
        @test all(buf .== 0)
        # Check that the buffer is still in the cache
        @test glbc[u] === buf

        # Test fill! with other values
        fill!(glbc, -2.5)
        @test all(buf .== -2.5)
    end

    @testset "LazyBufferCache fill! with mixed types" begin
        lbc = LazyBufferCache(identity)
        u_float = rand(Float64, 10)
        u_int = rand(Int, 5)

        # Create and fill buffers
        buf_float = lbc[u_float]
        fill!(buf_float, 1.5)
        buf_int = lbc[u_int]
        fill!(buf_int, 7)

        # Test fill! with 0
        fill!(lbc, 0)
        @test all(buf_float .== 0.0)
        @test all(buf_int .== 0)
        @test eltype(buf_float) == Float64
        @test eltype(buf_int) == Int
    end
end
