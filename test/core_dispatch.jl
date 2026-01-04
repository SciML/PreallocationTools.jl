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
