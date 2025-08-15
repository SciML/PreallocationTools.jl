module PreallocationTools

using ArrayInterface, Adapt
using PrecompileTools

struct FixedSizeDiffCache{T <: AbstractArray, S <: AbstractArray}
    du::T
    dual_du::S
    any_du::Vector{Any}
end

# Mutable container to hold dual array creator that can be updated by extension
dualarraycreator(args...) = nothing

function FixedSizeDiffCache(u::AbstractArray{T}, siz,
        ::Type{Val{chunk_size}}) where {T, chunk_size}
    x = dualarraycreator(u, siz, Val{chunk_size})
    xany = Any[]
    FixedSizeDiffCache(deepcopy(u), x, xany)
end

forwarddiff_compat_chunk_size(n) = 0

"""
`FixedSizeDiffCache(u::AbstractArray, N = Val{default_cache_size(length(u))})`

Builds a `FixedSizeDiffCache` object that stores both a version of the cache for `u`
and for the `Dual` version of `u`, allowing use of pre-cached vectors with
forward-mode automatic differentiation.
"""
function FixedSizeDiffCache(u::AbstractArray,
        ::Type{Val{N}} = Val{forwarddiff_compat_chunk_size(length(u))}) where {
        N,
}
    FixedSizeDiffCache(u, size(u), Val{N})
end

function FixedSizeDiffCache(u::AbstractArray, N::Integer)
    FixedSizeDiffCache(u, size(u), Val{N})
end

# Generic fallback for chunksize
chunksize(::Type{T}) where {T} = 0

# ForwardDiff-specific methods moved to extension

function get_tmp(dc::FixedSizeDiffCache, u::Union{Number, AbstractArray})
    if promote_type(eltype(dc.du), eltype(u)) <: eltype(dc.du)
        dc.du
    else
        if length(dc.du) > length(dc.any_du)
            resize!(dc.any_du, length(dc.du))
        end
        _restructure(dc.du, dc.any_du)
    end
end

function get_tmp(dc::FixedSizeDiffCache, ::Type{T}) where {T <: Number}
    if promote_type(eltype(dc.du), T) <: eltype(dc.du)
        dc.du
    else
        if length(dc.du) > length(dc.any_du)
            resize!(dc.any_du, length(dc.du))
        end
        _restructure(dc.du, dc.any_du)
    end
end

# DiffCache

struct DiffCache{T <: AbstractArray, S <: AbstractArray}
    du::T
    dual_du::S
    any_du::Vector{Any}
end

function DiffCache(u::AbstractArray{T}, siz, chunk_sizes) where {T}
    x = adapt(ArrayInterface.parameterless_type(u),
        zeros(T, prod(chunk_sizes .+ 1) * prod(siz)))
    xany = Any[]
    DiffCache(u, x, xany)
end

"""
`DiffCache(u::AbstractArray, N::Int = forwarddiff_compat_chunk_size(length(u)); levels::Int = 1)`
`DiffCache(u::AbstractArray; N::AbstractArray{<:Int})`

Builds a `DiffCache` object that stores both a version of the cache for `u`
and for the `Dual` version of `u`, allowing use of pre-cached vectors with
forward-mode automatic differentiation via
[ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) (when available).
Supports nested AD via keyword `levels` or specifying an array of chunk sizes.

The `DiffCache` also supports sparsity detection via
[SparseConnectivityTracer.jl](https://github.com/adrhill/SparseConnectivityTracer.jl/).
"""
function DiffCache(u::AbstractArray, N::Int = forwarddiff_compat_chunk_size(length(u));
        levels::Int = 1)
    DiffCache(u, size(u), N * ones(Int, levels))
end
DiffCache(u::AbstractArray, N::AbstractArray{<:Int}) = DiffCache(u, size(u), N)
function DiffCache(u::AbstractArray, ::Type{Val{N}}; levels::Int = 1) where {N}
    DiffCache(u, N; levels)
end
DiffCache(u::AbstractArray, ::Val{N}; levels::Int = 1) where {N} = DiffCache(u, N; levels)

# Legacy deprecate later
const dualcache = DiffCache

"""
`get_tmp(dc::DiffCache, u)`

Returns the `Dual` or normal cache array stored in `dc` based on the type of `u`.
"""
# ForwardDiff-specific methods moved to extension

function get_tmp(dc::DiffCache, u::Union{Number, AbstractArray})
    if promote_type(eltype(dc.du), eltype(u)) <: eltype(dc.du)
        dc.du
    else
        if length(dc.du) > length(dc.any_du)
            resize!(dc.any_du, length(dc.du))
        end

        _restructure(dc.du, dc.any_du)
    end
end

function get_tmp(dc::DiffCache, ::Type{T}) where {T <: Number}
    if promote_type(eltype(dc.du), T) <: eltype(dc.du)
        dc.du
    else
        if length(dc.du) > length(dc.any_du)
            resize!(dc.any_du, length(dc.du))
        end

        _restructure(dc.du, dc.any_du)
    end
end

get_tmp(dc, u) = dc

function _restructure(normal_cache::Array, duals)
    reshape(duals, size(normal_cache)...)
end

function _restructure(normal_cache::AbstractArray, duals)
    ArrayInterface.restructure(normal_cache, duals)
end

function enlargediffcache!(dc, nelem) #warning comes only once per DiffCache.
    chunksize = div(nelem, length(dc.du)) - 1
    @warn "The supplied DiffCache was too small and was enlarged. This incurs allocations
    on the first call to `get_tmp`. If few calls to `get_tmp` occur and optimal performance is essential,
    consider changing 'N'/chunk size of this DiffCache to $chunksize." maxlog=1
    resize!(dc.dual_du, nelem)
end

# LazyBufferCache

"""
    b = LazyBufferCache(f = identity; initializer! = identity)

A lazily allocated buffer object.  Given an array `u`, `b[u]` returns an array of the
same type and size `f(size(u))` (defaulting to the same size), which is allocated as
needed and then cached within `b` for subsequent usage.

By default the created buffers are not initialized, but a function `initializer!`
can be supplied which is applied to the buffer when it is created, for instance `buf -> fill!(buf, 0.0)`.

Optionally, the size can be explicitly given at calltime using `b[u,s]`, which will
return a cache of size `s`.
"""
struct LazyBufferCache{F <: Function, I <: Function}
    bufs::Dict{Any, Any} # a dictionary mapping (type, size) pairs to buffers
    sizemap::F
    initializer!::I
    function LazyBufferCache(
            f::F = identity; initializer!::I = identity) where {
            F <: Function, I <: Function}
        new{F, I}(Dict(), f, initializer!)
    end # start with empty dict
end

similar_type(x::AbstractArray, s::Integer) = similar_type(x, (s,))
function similar_type(x::AbstractArray{T}, s::NTuple{N, Integer}) where {T, N}
    # The compiler is smart enough to not allocate
    # here for simple types like Array and SubArray
    typeof(similar(x, ntuple(Returns(1), N)))
end

function get_tmp(
        b::LazyBufferCache, u::T, s = b.sizemap(size(u))) where {T <: AbstractArray}
    get!(b.bufs, (T, s)) do
        buffer = similar(u, s) # buffer to allocate if it was not found in b.bufs
        b.initializer!(buffer)
        buffer
    end::similar_type(u, s) # declare type since b.bufs dictionary is untyped
end

# override the [] method
function Base.getindex(
        b::LazyBufferCache, u::T, s = b.sizemap(size(u))) where {T <: AbstractArray}
    get_tmp(b, u, s)
end

# GeneralLazyBufferCache

"""
    b = GeneralLazyBufferCache(f=identity)

A lazily allocated buffer object.  Given an array `u`, `b[u]` returns a cache object
generated by `f(u)`, but the generator is only run the first time (and all subsequent
times it reuses the same cache)

## Limitation

The main limitation of this method is that its return is not type-inferred, and thus
it can be slower than some other preallocation techniques. However, if used
correct using things like function barriers, then this is a general technique that
is sufficiently fast.
"""
struct GeneralLazyBufferCache{F <: Function}
    bufs::Dict{Any, Any} # a dictionary mapping types to buffers
    f::F
    GeneralLazyBufferCache(f::F = identity) where {F <: Function} = new{F}(Dict(), f) # start with empty dict
end

function get_tmp(b::GeneralLazyBufferCache, u::T) where {T}
    get!(b.bufs, T) do
        b.f(u)
    end
end
Base.getindex(b::GeneralLazyBufferCache, u::T) where {T} = get_tmp(b, u)

# resize! methods for PreallocationTools types
# Note: resize! only works for 1D arrays (vectors)
function Base.resize!(dc::DiffCache, n::Integer)
    # Only resize if the array is a vector
    if dc.du isa AbstractVector
        resize!(dc.du, n)
    else
        throw(ArgumentError("resize! is only supported for DiffCache with vector arrays, got $(typeof(dc.du))"))
    end
    # dual_du is often pre-allocated for ForwardDiff dual numbers, 
    # and may need special handling based on chunk size
    # Only resize if it's a vector
    if dc.dual_du isa AbstractVector
        resize!(dc.dual_du, n)
    end
    # Always resize the any_du cache
    resize!(dc.any_du, n)
    return dc
end

function Base.resize!(dc::FixedSizeDiffCache, n::Integer)
    # Only resize if the array is a vector
    if dc.du isa AbstractVector
        resize!(dc.du, n)
    else
        throw(ArgumentError("resize! is only supported for FixedSizeDiffCache with vector arrays, got $(typeof(dc.du))"))
    end
    # dual_du is often pre-allocated for ForwardDiff dual numbers,
    # and may need special handling based on chunk size
    # Only resize if it's a vector
    if dc.dual_du isa AbstractVector
        resize!(dc.dual_du, n)
    end
    # Always resize the any_du cache
    resize!(dc.any_du, n)
    return dc
end

export GeneralLazyBufferCache, FixedSizeDiffCache, DiffCache, LazyBufferCache, dualcache
export get_tmp

# Export internal functions for extension use (but not public API)
# These are needed by the ForwardDiff extension

@setup_workload begin
    @compile_workload begin
        # Basic precompilation
        u = rand(10)
        cache = DiffCache(u)
        get_tmp(cache, u)
    end
end

end
