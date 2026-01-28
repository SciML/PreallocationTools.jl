module PreallocationTools

using Adapt: Adapt, adapt
using ArrayInterface: ArrayInterface
using PrecompileTools: @compile_workload, @setup_workload

struct FixedSizeDiffCache{T <: AbstractArray, S <: AbstractArray}
    du::T
    dual_du::S
    any_du::Vector{Any}
end

# Mutable container to hold dual array creator that can be updated by extension
dualarraycreator(args...) = nothing

function FixedSizeDiffCache(
        u::AbstractArray{T}, siz,
        ::Type{Val{chunk_size}}
    ) where {T, chunk_size}
    x = dualarraycreator(u, siz, Val{chunk_size})
    xany = Any[]
    return FixedSizeDiffCache(deepcopy(u), x, xany)
end

forwarddiff_compat_chunk_size(n) = 0

"""
`FixedSizeDiffCache(u::AbstractArray, N = Val{default_cache_size(length(u))})`

Builds a `FixedSizeDiffCache` object that stores both a version of the cache for `u`
and for the `Dual` version of `u`, allowing use of pre-cached vectors with
forward-mode automatic differentiation.
"""
function FixedSizeDiffCache(
        u::AbstractArray,
        ::Type{Val{N}} = Val{forwarddiff_compat_chunk_size(length(u))}
    ) where {
        N,
    }
    return FixedSizeDiffCache(u, size(u), Val{N})
end

function FixedSizeDiffCache(u::AbstractArray, N::Integer)
    return FixedSizeDiffCache(u, size(u), Val{N})
end

# Generic fallback for chunksize
chunksize(::Type{T}) where {T} = 0

# ForwardDiff-specific methods moved to extension

"""
    get_tmp(dc::FixedSizeDiffCache, u::Union{Number, AbstractArray})

Returns the appropriate cache array from the `FixedSizeDiffCache` based on the type of `u`.

If `u` is a regular array or number, returns the standard cache `dc.du`. If `u` contains
dual numbers (e.g., from ForwardDiff.jl), returns the dual cache array. The function
automatically handles type promotion and resizing of internal caches as needed.

This function enables seamless switching between regular and automatic differentiation
computations without manual cache management.
"""
function get_tmp(dc::FixedSizeDiffCache, u::Union{Number, AbstractArray})
    return get_tmp(dc, eltype(u))
end

function get_tmp(dc::FixedSizeDiffCache, ::Type{T}) where {T <: Number}
    return if promote_type(eltype(dc.du), T) <: eltype(dc.du)
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
    x = adapt(
        ArrayInterface.parameterless_type(u),
        zeros(T, prod(chunk_sizes .+ 1) * prod(siz))
    )
    xany = Any[]
    return DiffCache(u, x, xany)
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
function DiffCache(
        u::AbstractArray, N::Int = forwarddiff_compat_chunk_size(length(u));
        levels::Int = 1
    )
    return DiffCache(u, size(u), N * ones(Int, levels))
end
DiffCache(u::AbstractArray, N::AbstractArray{<:Int}) = DiffCache(u, size(u), N)
function DiffCache(u::AbstractArray, ::Type{Val{N}}; levels::Int = 1) where {N}
    return DiffCache(u, N; levels)
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
    return if promote_type(eltype(dc.du), eltype(u)) <: eltype(dc.du)
        dc.du
    else
        if length(dc.du) > length(dc.any_du)
            resize!(dc.any_du, length(dc.du))
        end

        _restructure(dc.du, dc.any_du)
    end
end

function get_tmp(dc::DiffCache, ::Type{T}) where {T <: Number}
    return if promote_type(eltype(dc.du), T) <: eltype(dc.du)
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
    return reshape(duals, size(normal_cache)...)
end

function _restructure(normal_cache::AbstractArray, duals)
    return ArrayInterface.restructure(normal_cache, duals)
end

"""
    enlargediffcache!(dc::DiffCache, nelem::Integer)

Enlarges the dual cache array in a `DiffCache` when it's found to be too small.

This function is called internally when automatic differentiation requires a larger
dual cache than initially allocated. It resizes `dc.dual_du` to accommodate `nelem`
elements and issues a one-time warning suggesting an appropriate chunk size for
optimal performance.

## Arguments

  - `dc`: The `DiffCache` object to enlarge
  - `nelem`: The new required number of elements

## Notes

The warning is shown only once per `DiffCache` instance to avoid spam. For optimal
performance in production code, pre-allocate with the suggested chunk size to avoid
runtime allocations.
"""
function enlargediffcache!(dc, nelem) #warning comes only once per DiffCache.
    chunksize = div(nelem, length(dc.du)) - 1
    @warn "The supplied DiffCache was too small and was enlarged. This incurs allocations
    on the first call to `get_tmp`. If few calls to `get_tmp` occur and optimal performance is essential,
    consider changing 'N'/chunk size of this DiffCache to $chunksize." maxlog = 1
    return resize!(dc.dual_du, nelem)
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
            f::F = identity; initializer!::I = identity
        ) where {
            F <: Function, I <: Function,
        }
        return new{F, I}(Dict(), f, initializer!)
    end # start with empty dict
end

similar_type(x::AbstractArray, s::Integer) = similar_type(x, (s,))
function similar_type(x::AbstractArray{T}, s::NTuple{N, Integer}) where {T, N}
    # The compiler is smart enough to not allocate
    # here for simple types like Array and SubArray
    return typeof(similar(x, ntuple(Returns(1), N)))
end

# Compute the type that `_make_buffer` would return for a given array and size.
# When size matches the original, preserve the original type
_buffer_type(x::AbstractArray, s) = s == size(x) ? typeof(x) : similar_type(x, s)

function get_tmp(
        b::LazyBufferCache, u::T, s = b.sizemap(size(u))
    ) where {T <: AbstractArray}
    return get!(b.bufs, (T, s)) do
        # Use similar(u) when size matches to preserve wrapper types like
        # ComponentArrays. similar(u, s) can strip wrapper types since the
        # dims-based dispatch may not preserve them.
        buffer = s == size(u) ? similar(u) : similar(u, s)
        b.initializer!(buffer)
        buffer
    end::_buffer_type(u, s) # declare type since b.bufs dictionary is untyped
end

# override the [] method
function Base.getindex(
        b::LazyBufferCache, u::T, s = b.sizemap(size(u))
    ) where {T <: AbstractArray}
    return get_tmp(b, u, s)
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
    return get!(b.bufs, T) do
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

# zero dispatches for PreallocationTools types
function Base.zero(dc::DiffCache)
    return DiffCache(zero(dc.du), zero(dc.dual_du), Any[])
end

function Base.zero(dc::FixedSizeDiffCache)
    return FixedSizeDiffCache(zero(dc.du), zero(dc.dual_du), Any[])
end

function Base.zero(lbc::LazyBufferCache)
    return LazyBufferCache(lbc.sizemap; initializer! = lbc.initializer!)
end

function Base.zero(glbc::GeneralLazyBufferCache)
    return GeneralLazyBufferCache(glbc.f)
end

# copy dispatches for PreallocationTools types
function Base.copy(dc::DiffCache)
    return DiffCache(copy(dc.du), copy(dc.dual_du), copy(dc.any_du))
end

function Base.copy(dc::FixedSizeDiffCache)
    return FixedSizeDiffCache(copy(dc.du), copy(dc.dual_du), copy(dc.any_du))
end

function Base.copy(lbc::LazyBufferCache)
    new_lbc = LazyBufferCache(lbc.sizemap; initializer! = lbc.initializer!)
    # Copy the internal buffer dictionary
    for (key, val) in lbc.bufs
        new_lbc.bufs[key] = copy(val)
    end
    return new_lbc
end

function Base.copy(glbc::GeneralLazyBufferCache)
    new_glbc = GeneralLazyBufferCache(glbc.f)
    # Copy the internal buffer dictionary
    for (key, val) in glbc.bufs
        new_glbc.bufs[key] = copy(val)
    end
    return new_glbc
end

# fill! dispatches for PreallocationTools types
"""
    fill!(dc::DiffCache, val)

Fill all allocated buffers in the DiffCache with the given value.
"""
function Base.fill!(dc::DiffCache, val)
    fill!(dc.du, val)
    fill!(dc.dual_du, val)
    fill!(dc.any_du, nothing)
    return dc
end

"""
    fill!(dc::FixedSizeDiffCache, val)

Fill all allocated buffers in the FixedSizeDiffCache with the given value.
"""
function Base.fill!(dc::FixedSizeDiffCache, val)
    fill!(dc.du, val)
    fill!(dc.dual_du, val)
    fill!(dc.any_du, nothing)
    return dc
end

"""
    fill!(lbc::LazyBufferCache, val)

Fill all allocated buffers in the LazyBufferCache with the given value.
"""
function Base.fill!(lbc::LazyBufferCache, val)
    for (_, buffer) in lbc.bufs
        if buffer isa AbstractArray
            fill!(buffer, val)
        end
    end
    return lbc
end

"""
    fill!(glbc::GeneralLazyBufferCache, val)

Fill all allocated buffers in the GeneralLazyBufferCache with the given value.
"""
function Base.fill!(glbc::GeneralLazyBufferCache, val)
    for (_, buffer) in glbc.bufs
        if buffer isa AbstractArray
            fill!(buffer, val)
        elseif applicable(fill!, buffer, val)
            fill!(buffer, val)
        end
    end
    return glbc
end

export GeneralLazyBufferCache, FixedSizeDiffCache, DiffCache, LazyBufferCache, dualcache
export get_tmp

# Export internal functions for extension use (but not public API)
# These are needed by the ForwardDiff extension

@setup_workload begin
    @compile_workload begin
        # Precompile DiffCache with vectors and matrices
        u = ones(10)
        cache = DiffCache(u)
        get_tmp(cache, u)

        m = ones(3, 3)
        cache_m = DiffCache(m)
        get_tmp(cache_m, m)

        # Precompile LazyBufferCache
        lbc = LazyBufferCache()
        get_tmp(lbc, u)
        get_tmp(lbc, m)

        # Precompile GeneralLazyBufferCache
        glbc = GeneralLazyBufferCache()
        get_tmp(glbc, u)
    end
end

end
