using LinearAlgebra,
    Test, PreallocationTools, ForwardDiff, LabelledArrays,
    RecursiveArrayTools

function test(u0, dual, chunk_size)
    cache = PreallocationTools.DiffCache(u0, chunk_size)
    allocs_normal1 = @allocated get_tmp(cache, u0)
    allocs_normal2 = @allocated get_tmp(cache, first(u0))
    allocs_dual1 = @allocated get_tmp(cache, dual)
    allocs_dual2 = @allocated get_tmp(cache, first(dual))
    result_normal1 = get_tmp(cache, u0)
    result_normal2 = get_tmp(cache, first(u0))
    result_dual1 = get_tmp(cache, dual)
    result_dual2 = get_tmp(cache, first(dual))
    return allocs_normal1, allocs_normal2, allocs_dual1, allocs_dual2, result_normal1,
        result_normal2, result_dual1,
        result_dual2
end

function structequal(struct1, struct2)
    if typeof(struct1) == typeof(struct2)
        fn = fieldnames(typeof(struct1))
        all(getfield(a, fn[i]) == getfield(b, fn[i]) for i in eachindex(fn))
    else
        return false
    end
end

struct IsbitsPair{T}
    x::T
    y::T
end

Base.zero(::Type{IsbitsPair{T}}) where {T} = IsbitsPair(zero(T), zero(T))

@testset "DiffCache accepts isbits elements without zero" begin
    u = [(; a = 1.0, b = (2.0, 3.0)), (; a = 4.0, b = (5.0, 6.0))]
    cache = DiffCache(u, 3)
    DualT = ForwardDiff.Dual{Nothing, Float64, 3}
    tmp = get_tmp(cache, DualT)
    expected_eltype = NamedTuple{(:a, :b), Tuple{DualT, Tuple{DualT, DualT}}}

    @test eltype(cache.dual_du) == eltype(u)
    @test length(cache.dual_du) == 8
    @test size(tmp) == size(u)
    @test eltype(tmp) == expected_eltype

    tmp[1] = (; a = zero(DualT), b = (zero(DualT), zero(DualT)))
    @test tmp[1] == (; a = zero(DualT), b = (zero(DualT), zero(DualT)))
end

#Setup Base Array tests
chunk_size = 5
u0 = ones(5, 5)
dual = zeros(
    ForwardDiff.Dual{
        ForwardDiff.Tag{nothing, Float64}, Float64,
        chunk_size,
    }, 5, 5
)
results = test(u0, dual, chunk_size)
#allocation tests
@test results[1] == 0
@test results[2] == 0
@test results[3] == 0
@test results[4] == 0
#size tests
@test size(results[5]) == size(u0)
@test size(results[6]) == size(u0)
@test size(results[7]) == size(u0)
@test size(results[8]) == size(u0)
#type tests
@test typeof(results[5]) == typeof(u0)
@test typeof(results[6]) == typeof(u0)
@test_broken typeof(results[7]) == typeof(dual)
@test_broken typeof(results[8]) == typeof(dual)
#eltype tests
@test eltype(results[5]) == eltype(u0)
@test eltype(results[7]) == eltype(dual)

chunk_size = 5
u0_B = ones(5, 5)
dual_B = zeros(
    ForwardDiff.Dual{
        ForwardDiff.Tag{typeof(something), Float64}, Float64,
        chunk_size,
    },
    2, 2
)
cache_B = FixedSizeDiffCache(u0_B, chunk_size)
tmp_du_BA = get_tmp(cache_B, u0_B)
tmp_dual_du_BA = get_tmp(cache_B, dual_B)
tmp_du_BN = get_tmp(cache_B, u0_B[1])
tmp_dual_du_BN = get_tmp(cache_B, dual_B[1])
@test size(tmp_du_BA) == size(u0_B)
@test typeof(tmp_du_BA) == typeof(u0_B)
@test eltype(tmp_du_BA) == eltype(u0_B)
@test size(tmp_dual_du_BA) == size(u0_B)
@test_broken typeof(tmp_dual_du_BA) == typeof(dual_B)
@test eltype(tmp_dual_du_BA) == eltype(dual_B)
@test size(tmp_du_BN) == size(u0_B)
@test typeof(tmp_du_BN) == typeof(u0_B)
@test eltype(tmp_du_BN) == eltype(u0_B)
@test size(tmp_dual_du_BN) == size(u0_B)
@test_broken typeof(tmp_dual_du_BN) == typeof(dual_B)
@test eltype(tmp_dual_du_BN) == eltype(dual_B)

#LArray tests
chunk_size = 4
u0 = LArray((2, 2); a = 1.0, b = 1.0, c = 1.0, d = 1.0)
zerodual = zero(
    ForwardDiff.Dual{
        ForwardDiff.Tag{nothing, Float64}, Float64,
        chunk_size,
    }
)
dual = LArray((2, 2); a = zerodual, b = zerodual, c = zerodual, d = zerodual)
results = test(u0, dual, chunk_size)
#allocation tests
@test results[1] == 0
@test results[2] == 0
@test_broken results[3] == 0
@test_broken results[4] == 0
#size tests
@test size(results[5]) == size(u0)
@test size(results[6]) == size(u0)
@test size(results[7]) == size(u0)
@test size(results[8]) == size(u0)
#type tests
@test typeof(results[5]) == typeof(u0)
@test typeof(results[6]) == typeof(u0)
@test typeof(results[7]) == typeof(dual)
@test typeof(results[8]) == typeof(dual)
#eltype tests
@test eltype(results[5]) == eltype(u0)
@test eltype(results[7]) == eltype(dual)

#ArrayPartition tests
chunk_size = 2
u0 = ArrayPartition(ones(2, 2), ones(3, 3))
dual_a = zeros(
    ForwardDiff.Dual{
        ForwardDiff.Tag{nothing, Float64}, Float64,
        chunk_size,
    }, 2, 2
)
dual_b = zeros(
    ForwardDiff.Dual{
        ForwardDiff.Tag{nothing, Float64}, Float64,
        chunk_size,
    }, 3, 3
)
dual = ArrayPartition(dual_a, dual_b)
results = test(u0, dual, chunk_size)
#allocation tests
@test results[1] == 0
@test results[2] == 0
@test_broken results[3] == 0
@test_broken results[4] == 0
#size tests
@test size(results[5]) == size(u0)
@test size(results[6]) == size(u0)
@test size(results[7]) == size(u0)
@test size(results[8]) == size(u0)
#type tests
@test typeof(results[5]) == typeof(u0)
@test typeof(results[6]) == typeof(u0)
@test typeof(results[7]) == typeof(dual)
@test typeof(results[8]) == typeof(dual)
#eltype tests
@test eltype(results[5]) == eltype(u0)
@test eltype(results[7]) == eltype(dual)

u0_AP = ArrayPartition(ones(2, 2), ones(3, 3))
dual_a = zeros(
    ForwardDiff.Dual{
        ForwardDiff.Tag{typeof(something), Float64}, Float64,
        chunk_size,
    },
    2, 2
)
dual_b = zeros(
    ForwardDiff.Dual{
        ForwardDiff.Tag{typeof(something), Float64}, Float64,
        chunk_size,
    },
    3, 3
)
dual_AP = ArrayPartition(dual_a, dual_b)
cache_AP = FixedSizeDiffCache(u0_AP, chunk_size)
tmp_du_APA = get_tmp(cache_AP, u0_AP)
tmp_dual_du_APA = get_tmp(cache_AP, dual_AP)
tmp_du_APN = get_tmp(cache_AP, u0_AP[1])
tmp_dual_du_APN = get_tmp(cache_AP, dual_AP[1])
@test size(tmp_du_APA) == size(u0_AP)
@test typeof(tmp_du_APA) == typeof(u0_AP)
@test eltype(tmp_du_APA) == eltype(u0_AP)
@test size(tmp_dual_du_APA) == size(u0_AP)
@test_broken typeof(tmp_dual_du_APA) == typeof(dual_AP)
@test eltype(tmp_dual_du_APA) == eltype(dual_AP)
@test size(tmp_du_APN) == size(u0_AP)
@test typeof(tmp_du_APN) == typeof(u0_AP)
@test eltype(tmp_du_APN) == eltype(u0_AP)
@test size(tmp_dual_du_APN) == size(u0_AP)
@test_broken typeof(tmp_dual_du_APN) == typeof(dual_AP)
@test eltype(tmp_dual_du_APN) == eltype(dual_AP)

a = PreallocationTools.DiffCache(zeros(4), 4)
b = PreallocationTools.DiffCache(zeros(4), Val{4}())
c = PreallocationTools.DiffCache(zeros(4), Val{4})
@test structequal(a, b)
@test structequal(a, b)

# FixedSizeDiffCache get_tmp ReinterpretArray test
# Ensures that get_tmp doesn't produce a Reinterpret array in some cases

dual_cache = FixedSizeDiffCache([0.0, 0.0, 0.0], 3)
dl = zeros(ForwardDiff.Dual{Nothing, Float64, 3}, 3)

@test !(get_tmp(dual_cache, dl) isa Base.ReinterpretArray)

@testset "DiffCache preserves isbits wrappers around duals" begin
    z = zeros(ComplexF64, 20)
    zd = DiffCache(z)

    function sum_cis(θ)
        ztmp = get_tmp(zd, θ)
        @test eltype(ztmp) <: Complex
        for i in eachindex(ztmp)
            ztmp[i] = cis(i * θ)
        end
        abs(sum(ztmp))
    end

    @test ForwardDiff.derivative(sum_cis, 1.1) isa Float64

    DualT = ForwardDiff.Dual{Nothing, Float64, 2}
    pair_cache = DiffCache(zeros(IsbitsPair{Float64}, 3), 2)
    pair_tmp = get_tmp(pair_cache, DualT)

    @test size(pair_tmp) == (3,)
    @test eltype(pair_tmp) == IsbitsPair{DualT}

    pair_tmp[1] = IsbitsPair(zero(DualT), zero(DualT))
    @test pair_tmp[1] == IsbitsPair(zero(DualT), zero(DualT))

    complex32_cache = DiffCache(zeros(ComplexF32, 3), 5)
    complex32_tmp = get_tmp(complex32_cache, DualT)
    @test eltype(complex32_tmp) == Complex{DualT}
end

@testset "DiffCache with Dual base eltype does not nest tags" begin
    # AD-over-AD: the base buffer is allocated at an outer dual level (e.g.
    # `DiffCache(similar(u))` while ForwardDiff-ing through a solver whose
    # state was promoted to Dual). Requesting a buffer for the same dual type
    # must return exactly that eltype, not a Dual re-tagged over the base.
    OuterDual = ForwardDiff.Dual{ForwardDiff.Tag{:outer, Float64}, Float64, 3}
    base = zeros(OuterDual, 5)
    dc = DiffCache(base, 3)

    same_level = get_tmp(dc, base)
    @test eltype(same_level) === OuterDual

    same_level_scalar = get_tmp(dc, first(base))
    @test eltype(same_level_scalar) === OuterDual

    # A genuinely nested request (inner AD over the outer-dual state) must
    # still return the nested dual type unchanged.
    NestedDual = ForwardDiff.Dual{ForwardDiff.Tag{:inner, OuterDual}, OuterDual, 2}
    nested = get_tmp(dc, zeros(NestedDual, 5))
    @test eltype(nested) === NestedDual
end

@testset "wrapper-of-Dual cache eltypes do not nest or corrupt tags" begin
    # An isbits wrapper containing a dual as the cache base eltype (e.g. a
    # complex-valued buffer allocated under an outer AD pass). A same-level
    # dual request must return the cache's own eltype — descending into the
    # stored dual would rewrite its Tag's value type (breaking ForwardDiff tag
    # matching) and fabricate nesting one wrapper deeper.
    D1 = ForwardDiff.Dual{ForwardDiff.Tag{:L1, Float64}, Float64, 3}
    D2 = ForwardDiff.Dual{ForwardDiff.Tag{:L2, D1}, D1, 2}
    D3 = ForwardDiff.Dual{ForwardDiff.Tag{:L3, D2}, D2, 4}

    dc1 = DiffCache(zeros(Complex{D1}, 4), 3)
    @test eltype(get_tmp(dc1, zeros(D1, 4))) === Complex{D1}       # same level
    @test eltype(get_tmp(dc1, zeros(D2, 4))) === Complex{D2}       # genuinely nested
    @test eltype(get_tmp(dc1, first(zeros(D1, 4)))) === Complex{D1}

    dc2 = DiffCache(zeros(Complex{D2}, 4), 2)
    @test eltype(get_tmp(dc2, zeros(D2, 4))) === Complex{D2}       # same level, depth 2
    @test eltype(get_tmp(dc2, zeros(D3, 4))) === Complex{D3}       # deeper still

    # A same-level request with a DIFFERENT tag (sibling AD) cannot be
    # composed meaningfully with the stored dual; it must fall back to the
    # bare requested dual with the stored dual left untouched — never a
    # rewritten-tag hybrid.
    D1b = ForwardDiff.Dual{ForwardDiff.Tag{:sibling, Float64}, Float64, 3}
    T_sib = eltype(get_tmp(dc1, zeros(D1b, 4)))
    @test T_sib === D1b

    # plain-wrapper composition still works at every depth
    dc0 = DiffCache(zeros(ComplexF64, 4), 3)
    @test eltype(get_tmp(dc0, zeros(D1, 4))) === Complex{D1}
    @test eltype(get_tmp(dc0, zeros(D2, 4))) === Complex{D2}
    @test eltype(get_tmp(dc0, zeros(D3, 4))) === Complex{D3}
end
