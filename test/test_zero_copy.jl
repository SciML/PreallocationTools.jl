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