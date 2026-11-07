using PreallocationTools, ForwardDiff, SparseArrays, SparseConnectivityTracer, Test

const CACHE_CONSTRUCTORS = (
    "DiffCache" => DiffCache,
    "FixedSizeDiffCache" => u -> FixedSizeDiffCache(u, 12),
)

# Bidiagonal pattern with a stored zero diagonal.
function bidiagonal_prototype(n)
    return spdiagm(-1 => ones(n - 1), 0 => zeros(n))
end

# Writes only the subdiagonal; the stored diagonal is left untouched.
function rhs!(du, u, cache)
    P = get_tmp(cache, u)
    for i in firstindex(u):(lastindex(u) - 1)
        P[i + 1, i] = u[i]^2
    end
    du .= vec(sum(P; dims = 2)) .- u
    return du
end

construction_allocs(make_cache, u) = (make_cache(u); @allocated make_cache(u))
lookup_allocs(cache, ::Type{T}) where {T} = (get_tmp(cache, T); @allocated get_tmp(cache, T))
lookup_allocs(cache, u) = (get_tmp(cache, u); @allocated get_tmp(cache, u))

@testset "$name dual storage scales with stored entries" for (name, make_cache) in CACHE_CONSTRUCTORS
    n = 1000
    proto = bidiagonal_prototype(n)
    @test construction_allocs(make_cache, proto) < 20 * Base.summarysize(proto)
    cache = make_cache(proto)
    @test isempty(cache.dual_du)

    D = ForwardDiff.Dual{Nothing, Float64, 12}
    workspace = @inferred get_tmp(cache, D)
    @test workspace isa SparseMatrixCSC{D, Int}
    @test size(workspace) == size(proto)
    @test workspace.colptr == proto.colptr
    @test workspace.rowval == proto.rowval
    @test all(iszero, nonzeros(workspace))
    @test Base.summarysize(cache) < 20 * Base.summarysize(proto)

    @test get_tmp(cache, D) === workspace
    @test get_tmp(cache, zeros(D, n)) === workspace
    @test get_tmp(cache, proto) === cache.du
    @test lookup_allocs(cache, D) == 0

    # Inserting entries into the dual workspace leaves the primal pattern intact.
    workspace[1, n] = one(D)
    @test nnz(cache.du) == nnz(proto)
    @test get_tmp(cache, D) === workspace
end

@testset "Jacobian through sparse $name" for (name, make_cache) in CACHE_CONSTRUCTORS
    n = 50
    u = rand(n)
    sparse_cache = make_cache(bidiagonal_prototype(n))
    dense_cache = DiffCache(Matrix(bidiagonal_prototype(n)))
    J_sparse = ForwardDiff.jacobian((du, u) -> rhs!(du, u, sparse_cache), zeros(n), u)
    J_dense = ForwardDiff.jacobian((du, u) -> rhs!(du, u, dense_cache), zeros(n), u)
    @test J_sparse == J_dense
    @test rhs!(zeros(n), u, sparse_cache) == rhs!(zeros(n), u, dense_cache)
end

@testset "SparseVector $name" for (name, make_cache) in CACHE_CONSTRUCTORS
    v = sparsevec([2, 7], [1.0, 2.0], 10)
    cache = make_cache(v)
    @test isempty(cache.dual_du)
    D = ForwardDiff.Dual{Nothing, Float64, 3}
    workspace = @inferred get_tmp(cache, D)
    @test workspace isa SparseVector{D, Int}
    @test findnz(workspace)[1] == findnz(v)[1]
    @test all(iszero, nonzeros(workspace))
    @test get_tmp(cache, D) === workspace
end

@testset "Wrapped sparse $name" for (name, make_cache) in CACHE_CONSTRUCTORS
    A = bidiagonal_prototype(6)
    D = ForwardDiff.Dual{Nothing, Float64, 2}
    for u in (view(A, :, 2:4), transpose(A), reshape(DiffCache(A), (4, 9)).du)
        cache = make_cache(u)
        @test isempty(cache.dual_du)
        for T in (D, BigFloat)
            workspace = @inferred get_tmp(cache, T)
            @test workspace isa SparseMatrixCSC{T}
            @test size(workspace) == size(u)
            workspace[1, 1] = one(T)
            @test get_tmp(cache, T) === workspace
            @test get_tmp(cache, T)[1, 1] == one(T)
            @test lookup_allocs(cache, T) == 0
        end
    end
end

@testset "reshape of a sparse DiffCache" begin
    cache = DiffCache(spzeros(2, 6))
    D = ForwardDiff.Dual{Nothing, Float64, 2}
    for T in (D, BigFloat)
        workspace = get_tmp(cache, T)
        reshaped = reshape(cache, 3, 4)
        reshaped_workspace = @inferred get_tmp(reshaped, T)
        @test size(reshaped_workspace) == (3, 4)
        @test get_tmp(reshaped, T) === reshaped_workspace
        @test get_tmp(cache, T) === workspace
    end
end

@testset "zero of a sparse $name keeps the pattern" for (name, make_cache) in CACHE_CONSTRUCTORS
    proto = bidiagonal_prototype(5)
    cache = make_cache(proto)
    cache.du[2, 1] = 3.0
    zeroed = zero(cache)
    @test zeroed.du isa SparseMatrixCSC{Float64, Int}
    @test findnz(zeroed.du)[1:2] == findnz(proto)[1:2]
    @test all(iszero, nonzeros(zeroed.du))
    @test cache.du[2, 1] == 3.0
    D = ForwardDiff.Dual{Nothing, Float64, 2}
    @test findnz(get_tmp(zeroed, D))[1:2] == findnz(proto)[1:2]
end

@testset "Non-bits workspaces of sparse $name are zeroed" for (name, make_cache) in CACHE_CONSTRUCTORS
    proto = bidiagonal_prototype(5)
    workspace = get_tmp(make_cache(proto), BigFloat)
    @test nonzeros(workspace) == zeros(BigFloat, nnz(proto))
end

@testset "fill! of a sparse $name" for (name, make_cache) in CACHE_CONSTRUCTORS
    n = 6
    proto = bidiagonal_prototype(n)
    D = ForwardDiff.Dual{Nothing, Float64, 2}

    # Zero keeps the stored pattern, including for sparsity tracers, whose
    # `iszero` does not return a `Bool`.
    cache = make_cache(proto)
    workspace = get_tmp(cache, D)
    jacobian_sparsity(
        (du, u) -> rhs!(du, u, cache), zeros(n), rand(n), TracerSparsityDetector()
    )
    tracer_workspace = only(buf for (T, buf) in cache.typed_du if T !== D)
    nonzeros(workspace) .= one(D)
    @test fill!(cache, 0.0) === cache
    @test nnz(workspace) == nnz(proto)
    @test all(iszero, nonzeros(workspace))
    @test nnz(tracer_workspace) == nnz(proto)

    # A nonzero value stores every entry.
    cache = make_cache(proto)
    workspace = get_tmp(cache, D)
    fill!(cache, 1.0)
    @test nnz(workspace) == n^2
    @test all(isone, workspace)
end

@testset "resize! of a sparse $name" for (name, make_cache) in CACHE_CONSTRUCTORS
    cache = make_cache(sparsevec([2, 7], [1.0, 2.0], 10))
    @test_throws "only supported for $name with resizable vector arrays" resize!(cache, 12)
end

@testset "Sparsity detection through sparse $name" for (name, make_cache) in CACHE_CONSTRUCTORS
    n = 20
    cache = make_cache(bidiagonal_prototype(n))
    pattern = jacobian_sparsity(
        (du, u) -> rhs!(du, u, cache), zeros(n), rand(n),
        TracerSparsityDetector()
    )
    expected = spdiagm(-1 => trues(n - 1), 0 => trues(n))
    @test pattern == expected
end

@testset "LazyBufferCache with sparse prototypes" begin
    A = bidiagonal_prototype(10)
    v = sparsevec([2, 7], [1.0, 2.0], 10)
    for (u, sizemap, B) in (
            (A, identity, SparseMatrixCSC{Float64, Int}),
            (A, s -> 2 .* s, SparseMatrixCSC{Float64, Int}),
            (A, s -> (prod(s),), SparseVector{Float64, Int}),
            (v, identity, SparseVector{Float64, Int}),
            (v, s -> (s[1], 3), SparseMatrixCSC{Float64, Int}),
        )
        cache = LazyBufferCache(sizemap)
        buf = @inferred get_tmp(cache, u)
        @test buf isa B
        @test size(buf) == sizemap(size(u))
        @test get_tmp(cache, u) === buf
        @test lookup_allocs(cache, u) == 0
    end
end

@testset "Matching stored patterns" begin
    match! = PreallocationTools._match_stored_pattern!
    src = sparse([1, 3], [1, 2], [1.0, 2.0], 3, 3)

    dest = sparse([1], [1], [5.0], 3, 3)
    @test match!(dest, src) === dest
    @test findnz(dest) == ([1, 3], [1, 2], [0.0, 0.0])
    # Values are kept while the pattern is unchanged.
    nonzeros(dest) .= 7
    match!(dest, src)
    @test nonzeros(dest) == [7.0, 7.0]

    dest = sparse([1], [1], [5.0], 3, 3)
    @test parent(match!(transpose(dest), transpose(src))) === dest
    @test findnz(dest) == ([1, 3], [1, 2], [0.0, 0.0])

    v = sparsevec([2], [5.0], 4)
    @test match!(v, sparsevec([1, 4], [1.0, 2.0], 4)) === v
    @test findnz(v) == ([1, 4], [0.0, 0.0])

    @test_throws "cannot match the stored pattern of a (3, 3) sparse matrix in one of size (2, 2)" match!(
        spzeros(2, 2), src
    )
    @test_throws "cannot match the stored pattern of a sparse vector of length 4 in one of length 3" match!(
        spzeros(3), sparsevec([1], [1.0], 4)
    )
end
