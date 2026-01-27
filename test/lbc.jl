using PreallocationTools: LazyBufferCache
using Test

b = LazyBufferCache(Returns(10); initializer! = buf -> fill!(buf, 0))

@test b[Float64[]] == zeros(10)

# Test that LazyBufferCache preserves wrapper array types when size matches
# (regression test for ComponentArray-like types where similar(x, size(x)) strips the wrapper)
struct WrapperArray{T, N, A <: AbstractArray{T, N}} <: AbstractArray{T, N}
    data::A
end
Base.size(w::WrapperArray) = size(w.data)
Base.getindex(w::WrapperArray, i...) = getindex(w.data, i...)
Base.setindex!(w::WrapperArray, v, i...) = setindex!(w.data, v, i...)
Base.similar(w::WrapperArray) = WrapperArray(similar(w.data))
Base.similar(w::WrapperArray, ::Type{T}) where {T} = WrapperArray(similar(w.data, T))
# similar with dims intentionally returns plain Array (mimics ComponentArray behavior)
Base.similar(::WrapperArray, ::Type{T}, dims::Dims) where {T} = Array{T}(undef, dims)

lbc = LazyBufferCache()
w = WrapperArray(ones(3))
result = lbc[w]
@test result isa WrapperArray
