using PreallocationTools, Test

module MockADExtension

using PreallocationTools

struct MockDual{N} <: Number
    value::Float64
end

Base.zero(::Type{MockDual{N}}) where {N} = MockDual{N}(0.0)

mutable struct MockVector{T} <: AbstractVector{T}
    data::Vector{T}
end

Base.IndexStyle(::Type{<:MockVector}) = IndexLinear()
Base.size(v::MockVector) = size(v.data)
Base.getindex(v::MockVector, i::Int) = v.data[i]
Base.setindex!(v::MockVector, x, i::Int) = (v.data[i] = x)
Base.resize!(v::MockVector, n::Integer) = (resize!(v.data, n); v)

function PreallocationTools.dualarraycreator(
        u::MockVector{T}, size, ::Type{Val{N}}
    ) where {T, N}
    return MockVector(fill(zero(MockDual{N}), prod(size)))
end

PreallocationTools.chunksize(::Type{MockDual{N}}) where {N} = N

function PreallocationTools._restructure(normal_cache::MockVector, duals)
    return MockVector(duals)
end

function PreallocationTools.get_tmp(
        dc::PreallocationTools.DiffCache, ::Type{MockDual{N}}
    ) where {N}
    needed = N * length(dc.du)
    if needed > length(dc.dual_du)
        PreallocationTools.enlargediffcache!(dc, needed)
    end
    duals = reinterpret(MockDual{N}, view(dc.dual_du, 1:length(dc.du)))
    return PreallocationTools._restructure(dc.du, duals)
end

end

@testset "Independent AD extension contract" begin
    prototype = MockADExtension.MockVector([1.0, 2.0, 3.0])
    fixed = FixedSizeDiffCache(prototype, 3)

    @test fixed.du !== prototype
    @test fixed.dual_du isa MockADExtension.MockVector{MockADExtension.MockDual{3}}
    @test size(fixed.dual_du) == size(prototype)
    @test PreallocationTools.chunksize(MockADExtension.MockDual{3}) == 3
    @test PreallocationTools.forwarddiff_compat_chunk_size(3) >= 0

    resize!(fixed, 5)
    @test length(fixed.du) == 5
    @test length(fixed.dual_du) == 5

    dual_storage = MockADExtension.MockVector(
        [
            MockADExtension.MockDual{3}(1.0),
            MockADExtension.MockDual{3}(2.0),
            MockADExtension.MockDual{3}(3.0),
        ]
    )
    restructured = PreallocationTools._restructure(prototype, dual_storage.data)
    @test restructured isa MockADExtension.MockVector
    @test axes(restructured) == axes(prototype)
    restructured[1] = MockADExtension.MockDual{3}(4.0)
    @test dual_storage[1] == MockADExtension.MockDual{3}(4.0)

    dynamic = DiffCache(zeros(3), 0; warn_on_resize = false)
    tmp = get_tmp(dynamic, MockADExtension.MockDual{3})
    @test axes(tmp) == axes(dynamic.du)
    @test eltype(tmp) == MockADExtension.MockDual{3}
    @test length(dynamic.dual_du) == 9
    tmp[1] = MockADExtension.MockDual{3}(5.0)
    @test reinterpret(MockADExtension.MockDual{3}, dynamic.dual_du)[1] == tmp[1]

    resize!(dynamic, 5)
    @test length(dynamic.du) == 5
    @test length(dynamic.dual_du) == 15
end
