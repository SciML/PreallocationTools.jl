module PreallocationTools

using Adapt: Adapt, adapt
using ArrayInterface: ArrayInterface
using PrecompileTools: @compile_workload, @setup_workload

"""
    FixedSizeDiffCache(u::AbstractArray, N = Val{forwarddiff_compat_chunk_size(length(u))})
    FixedSizeDiffCache(u::AbstractArray, N::Integer)

Build a fixed-size cache with storage for both the element type of `u` and the
corresponding forward-mode automatic differentiation dual type.

Use `get_tmp(cache, u)` to retrieve the cache matching the element type of `u`.
`FixedSizeDiffCache` is most useful when the dual chunk size is known in
advance and the cache size does not need to grow during differentiation.

# Arguments

  - `u`: prototype array whose shape and primal element type determine the cache.
  - `N`: ForwardDiff chunk size. Pass `Val{N}` to encode it in the cache type, or an
    integer as a convenience constructor.

# Fields

  - `du`: primal workspace with the shape and storage type of `u`.
  - `dual_du`: workspace for dual-number evaluations.
  - `any_du`: reusable temporary storage for nested dual reconstruction.

# Examples

```julia
cache = FixedSizeDiffCache(zeros(3), 2)
workspace = get_tmp(cache, zeros(3))
```
"""
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
    get_tmp(cache, u)
    get_tmp(cache, u, size)

Return cache storage appropriate for `u`.

For `DiffCache` and `FixedSizeDiffCache`, this returns normal storage when `u`
has the cached primal element type and dual-compatible storage when `u` carries
automatic differentiation element types. For `LazyBufferCache` and
`GeneralLazyBufferCache`, this lazily creates and reuses storage keyed by the
type and size requested.
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

"""
    DiffCache(u::AbstractArray, N::Int = forwarddiff_compat_chunk_size(length(u)); levels::Int = 1, warn_on_resize::Bool = true)
    DiffCache(u::AbstractArray, N::AbstractArray{<:Int}; warn_on_resize::Bool = true)

Build a cache with storage for both the element type of `u` and the
corresponding forward-mode automatic differentiation dual type.

Use `get_tmp(cache, u)` to retrieve storage matching the element type of `u`.
The `levels` keyword or vector-valued `N` supports nested automatic
differentiation. Set `warn_on_resize = false` to suppress the warning emitted
when `get_tmp` enlarges the dual cache, which can be useful when adaptive
algorithms are expected to resize the cache.

`DiffCache` also supports sparsity detection via
[SparseConnectivityTracer.jl](https://github.com/adrhill/SparseConnectivityTracer.jl/).

# Arguments

  - `u`: prototype array whose primal storage is cached.
  - `N`: ForwardDiff chunk size, or one chunk size per nested AD level.

# Keyword Arguments

  - `levels`: number of nested forward-mode AD levels when `N` is scalar.
  - `warn_on_resize`: whether to emit a one-time warning if dual storage grows.

# Fields

  - `du`: primal workspace.
  - `dual_du`: dual-number workspace, enlarged on demand when necessary.
  - `any_du`: reusable temporary storage for nested dual reconstruction.
  - `warn_on_resize`: controls the resize warning policy.

# Examples

```julia
cache = DiffCache(zeros(3), 2)
workspace = get_tmp(cache, zeros(3))
```
"""
struct DiffCache{T <: AbstractArray, S <: AbstractArray}
    du::T
    dual_du::S
    any_du::Vector{Any}
    warn_on_resize::Bool
end

function DiffCache(u::AbstractArray{T}, siz, chunk_sizes; warn_on_resize::Bool = true) where {T}
    x = adapt(
        ArrayInterface.parameterless_type(u),
        _zeroed_or_uninitialized(T, prod(chunk_sizes .+ 1) * prod(siz))
    )
    xany = Any[]
    return DiffCache(u, x, xany, warn_on_resize)
end

function _zeroed_or_uninitialized(::Type{T}, dims...) where {T}
    return hasmethod(zero, Tuple{Type{T}}) ? zeros(T, dims...) : Array{T}(undef, dims...)
end

function DiffCache(
        u::AbstractArray, N::Int = forwarddiff_compat_chunk_size(length(u));
        levels::Int = 1, warn_on_resize::Bool = true
    )
    return DiffCache(u, size(u), N * ones(Int, levels); warn_on_resize)
end
DiffCache(u::AbstractArray, N::AbstractArray{<:Int}; warn_on_resize::Bool = true) = DiffCache(u, size(u), N; warn_on_resize)
function DiffCache(u::AbstractArray, ::Type{Val{N}}; levels::Int = 1, warn_on_resize::Bool = true) where {N}
    return DiffCache(u, N; levels, warn_on_resize)
end
DiffCache(u::AbstractArray, ::Val{N}; levels::Int = 1, warn_on_resize::Bool = true) where {N} = DiffCache(u, N; levels, warn_on_resize)

# Deprecated: use DiffCache instead
"""
    dualcache(args...; warn_on_resize::Bool = true, kwargs...)

Deprecated alias for `DiffCache(args...; warn_on_resize, kwargs...)`.
Use `DiffCache` in new code.
"""
function dualcache(args...; warn_on_resize::Bool = true, kwargs...)
    Base.depwarn("`dualcache` is deprecated, use `DiffCache` instead.", :dualcache)
    return DiffCache(args...; warn_on_resize, kwargs...)
end

"""
    get_tmp(dc::DiffCache, u)

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

"""
    reshape(dc::DiffCache, dims...)
    reshape(dc::DiffCache, dims)

Return a `DiffCache` whose normal cache has shape `dims`.

This is useful for vector-backed caches that need to be resized. Resize the
backing `DiffCache` with `resize!`, then call `reshape` again with the updated
dimensions. The returned cache shares the normal cache storage with `dc`; the
raw dual cache storage remains vector-backed so `get_tmp` can reinterpret it for
the requested automatic differentiation element type.
"""
function Base.reshape(dc::DiffCache, dims::Tuple{Vararg{Integer}})
    shape = map(Int, dims)
    return DiffCache(_resizeable_reshape(dc.du, shape), dc.dual_du, dc.any_du, dc.warn_on_resize)
end

Base.reshape(dc::DiffCache, dims::Integer...) = reshape(dc, dims)

_resizeable_reshape(a::AbstractVector, shape) = reshape(view(a, :), shape)
_resizeable_reshape(a::AbstractArray, shape) = reshape(a, shape)

get_tmp(dc, u) = dc

"""
    _restructure(normal_cache::AbstractArray, duals)

Internal function that reshapes a flat array of dual numbers to match the shape of the
normal cache array. For standard `Array` types, uses `reshape`. For other `AbstractArray`
types, delegates to `ArrayInterface.restructure` to handle custom array types properly.
"""
function _restructure(normal_cache::Array, duals)
    return reshape(duals, size(normal_cache)...)
end

function _restructure(normal_cache::AbstractArray, duals)
    if _has_vector_view_parent(normal_cache)
        return reshape(duals, size(normal_cache)...)
    end
    return ArrayInterface.restructure(normal_cache, duals)
end

function _has_vector_view_parent(a)
    a isa SubArray{<:Any, 1, <:AbstractVector} && return true
    p = applicable(parent, a) ? parent(a) : nothing
    return p isa SubArray{<:Any, 1, <:AbstractVector}
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
    if dc.warn_on_resize
        chunksize = div(nelem, length(dc.du)) - 1
        @warn "The supplied DiffCache was too small and was enlarged. This incurs allocations
        on the first call to `get_tmp`. If few calls to `get_tmp` occur and optimal performance is essential,
        consider changing 'N'/chunk size of this DiffCache to $chunksize." maxlog = 1
    end
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

# `similar(x)` preserves wrapper types (e.g. ComponentArrays) that dims-based
# `similar(x, dims)` can strip, so its type is what the size-match branch of the
# `get!` in `get_tmp` returns. The allocation is used only for its type: inference
# resolves the result to a constant, and `:removable` lets the compiler delete the
# full-size runtime allocation, which Julia 1.10/1.11 otherwise perform on every
# `get_tmp` lookup (1.12+ elides it without the annotation).
Base.@assume_effects :removable _preserved_similar_type(x::AbstractArray) = typeof(similar(x))

# Compute the type that the buffer creation in `get_tmp` would return for a given
# array and size. When size matches the original, preserve the original type.
function _buffer_type(x::AbstractArray, s)
    return s == size(x) ? _preserved_similar_type(x) : similar_type(x, s)
end

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
        dual_length = length(dc.du) == 0 ? n : cld(length(dc.dual_du), length(dc.du)) * n
        resize!(dc.du, n)
    else
        throw(ArgumentError("resize! is only supported for DiffCache with vector arrays, got $(typeof(dc.du))"))
    end
    if dc.dual_du isa AbstractVector
        resize!(dc.dual_du, dual_length)
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
    return DiffCache(zero(dc.du), zero(dc.dual_du), Any[], dc.warn_on_resize)
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
    return DiffCache(copy(dc.du), copy(dc.dual_du), copy(dc.any_du), dc.warn_on_resize)
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
