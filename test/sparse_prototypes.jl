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
    proto_pattern = findnz(proto)[1:2]
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
    @test (1, n) in zip(findnz(workspace)[1:2]...)
    @test findnz(cache.du)[1:2] == proto_pattern
    @test findnz(proto)[1:2] == proto_pattern
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
            # A reshape gets the reshaped workspace of its parent.
            root = u isa Base.ReshapedArray ? parent(workspace) : workspace
            @test root isa SparseMatrixCSC{T}
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
        @test parent(reshaped_workspace) === workspace
        @test lookup_allocs(reshaped, T) == 0
    end
end

@testset "Reshapes of a sparse DiffCache share workspaces" begin
    D = ForwardDiff.Dual{Nothing, Float64, 2}
    v = sparsevec([2, 7], [1.0, 2.0], 12)
    for u in (bidiagonal_prototype(6), view(bidiagonal_prototype(6), 1:4, :), v, transpose(bidiagonal_prototype(6)))
        cache = DiffCache(u)
        # Allocated first through a reshape, then fetched by the parent cache.
        reshaped = reshape(cache, 2, 3, length(u) ÷ 6)
        workspace = @inferred get_tmp(reshaped, D)
        @test size(workspace) == size(reshaped.du)
        @test get_tmp(reshape(cache, size(reshaped.du)), D) == workspace
        root = get_tmp(cache, D)
        @test size(root) == size(u)
        root[firstindex(root)] = one(D)
        @test workspace[firstindex(workspace)] == one(D)
        @test lookup_allocs(reshape(cache, length(u)), D) == 0
    end
    @test findnz(parent(parent(get_tmp(reshape(DiffCache(v), 3, 4), D))))[1] == [2, 7]
end

@testset "Three-dimensional reshape of a sparse DiffCache" begin
    cache = DiffCache(sparse([1, 2], [1, 6], [1.0, 2.0], 2, 6))
    D = ForwardDiff.Dual{Nothing, Float64, 2}
    reshaped = reshape(cache, 2, 3, 2)
    workspace = @inferred get_tmp(reshaped, D)
    @test axes(workspace) == axes(reshaped.du)
    @test eltype(workspace) == D
    # Sparse storage with the stored pattern of the parent.
    @test parent(workspace) isa SparseMatrixCSC{D, Int}
    @test findnz(parent(workspace))[1:2] == ([1, 2], [1, 6])
    @test get_tmp(reshaped, D) === workspace
    @test lookup_allocs(reshaped, D) == 0
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

    # Wrappers keep their type and axes and get a zeroed copy of their root.
    A = sparse([1, 2, 3], [1, 2, 3], [1.0, 2.0, 3.0])
    for u in (view(A, 1:2, 1:2), transpose(A))
        cache = make_cache(u)
        zeroed = zero(cache)
        @test zeroed.du isa typeof(cache.du)
        @test axes(zeroed.du) == axes(u)
        @test findnz(parent(zeroed.du))[1:2] == findnz(A)[1:2]
        @test all(iszero, nonzeros(parent(zeroed.du)))
        @test parent(zeroed.du) !== A
        @test nonzeros(A) == [1.0, 2.0, 3.0]
    end
end

@testset "copy of a sparse $name keeps the wrapper and pattern" for (name, make_cache) in CACHE_CONSTRUCTORS
    A = sparse([1, 2, 3], [1, 2, 3], [1.0, 0.0, 3.0])
    D = ForwardDiff.Dual{Nothing, Float64, 2}
    for u in (A, view(A, 1:2, 1:2), transpose(A), PermutedDimsArray(A, (2, 1)))
        cache = make_cache(u)
        workspace = get_tmp(cache, D)
        copied = copy(cache)
        @test copied.du isa typeof(cache.du)
        @test copied.du == cache.du
        root = u === A ? copied.du : parent(copied.du)
        @test findnz(root) == findnz(A)
        @test root !== A
        @test get_tmp(copied, D) isa typeof(workspace)
        @test get_tmp(copied, D) !== workspace
    end
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

    # Also for sparsity tracers, without testing the converted value for zero.
    cache = make_cache(proto)
    jacobian_sparsity(
        (du, u) -> rhs!(du, u, cache), zeros(n), rand(n), TracerSparsityDetector()
    )
    tracer_workspace = only(values(cache.typed_du))
    @test fill!(cache, 1.0) === cache
    @test nnz(tracer_workspace) == n^2
    @test findnz(tracer_workspace)[1:2] == findnz(sparse(ones(n, n)))[1:2]

    v = sparsevec([2, 5], [1.0, 2.0], n)
    cache = make_cache(v)
    tracer_workspace = get_tmp(cache, eltype(tracer_workspace))
    fill!(cache, 1.0)
    @test findnz(tracer_workspace)[1] == 1:n
    @test all(isone, cache.du)
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

    # A view matches the whole root.
    dest = sparse([1, 2], [1, 3], [5.0, 6.0], 3, 3)
    @test parent(match!(view(dest, 1:2, 1:2), view(src, 1:2, 1:2))) === dest
    @test findnz(dest) == ([1, 3], [1, 2], [0.0, 0.0])

    # Same row indices, but an entry moves to another column.
    dest = sparse([1, 2], [1, 1], [5.0, 6.0], 2, 2)
    @test match!(dest, sparse([1, 2], [1, 2], [1.0, 2.0], 2, 2)) === dest
    @test findnz(dest) == ([1, 2], [1, 2], [0.0, 0.0])

    v = sparsevec([2], [5.0], 4)
    @test match!(v, sparsevec([1, 4], [1.0, 2.0], 4)) === v
    @test findnz(v) == ([1, 4], [0.0, 0.0])
    # A shrinking pattern removes the entries not stored in the source.
    @test match!(v, sparsevec([4], [3.0], 4)) === v
    @test findnz(v) == ([4], [0.0])
    nonzeros(v) .= 7
    match!(v, sparsevec([4], [3.0], 4))
    @test nonzeros(v) == [7.0]

    @test_throws "cannot match the stored pattern of a (3, 3) sparse matrix in one of size (2, 2)" match!(
        spzeros(2, 2), src
    )
    @test_throws "cannot match the stored pattern of a sparse vector of length 4 in one of length 3" match!(
        spzeros(3), sparsevec([1], [1.0], 4)
    )

    # Stored values may be unassigned until the caller writes them.
    unassigned = similar(sparsevec([1, 3], [0.0, 0.0], 4), Real)
    @test !isassigned(nonzeros(unassigned), 1)
    dest = similar(unassigned)
    @test match!(dest, unassigned) === dest
    @test !isassigned(nonzeros(dest), 1)
    dest = sparsevec([2], Real[5.0], 4)
    @test match!(dest, unassigned) === dest
    @test findnz(dest) == ([1, 3], [0, 0])

    # Index arrays shared with the source (as from `Enzyme.make_zero`) see the
    # entries stored in the source later, but the stored values do not.
    msg = "the shadow of a sparse workspace has 2 stored values, but the workspace has 3 stored entries"
    A = sparse([1, 2], [1, 2], [1.0, 2.0], 3, 3)
    dest = SparseMatrixCSC(3, 3, A.colptr, A.rowval, [5.0, 6.0])
    @test match!(dest, A) === dest
    @test nonzeros(dest) == [5.0, 6.0]
    A[3, 3] = 3.0
    @test_throws msg match!(dest, A)
    @test nonzeros(dest) == [5.0, 6.0]
    v = sparsevec([1, 2], [1.0, 2.0], 3)
    dest = SparseVector(3, v.nzind, [5.0, 6.0])
    v[3] = 3.0
    @test_throws msg match!(dest, v)

    match_allocs(dest, src) = (match!(dest, src); @allocated match!(dest, src))
    @testset "Unchanged patterns of $n stored entries match without allocating" for n in (10, 10_000)
        v = sparsevec(1:2:(2n), ones(n), 2n)
        A = sparse(1:n, 1:n, ones(n))
        @test match_allocs(copy(v), v) == 0
        @test match_allocs(copy(A), A) == 0
        @test match_allocs(transpose(copy(A)), transpose(A)) == 0
    end
end
