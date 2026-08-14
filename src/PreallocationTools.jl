module PreallocationTools

using Adapt: Adapt, adapt
using ArrayInterface: ArrayInterface
using PrecompileTools: @compile_workload, @setup_workload
using SciMLPublic: @public

"""
    FixedSizeDiffCache(u::AbstractArray, N = Val{forwarddiff_compat_chunk_size(length(u))})
    FixedSizeDiffCache(u::AbstractArray, N::Integer)

Build a fixed-size cache with storage for both the element type of `u` and the
corresponding forward-mode automatic differentiation dual type.

Use `get_tmp(cache, u)` to retrieve the cache matching the element type of `u`.
`FixedSizeDiffCache` is most useful when the dual chunk size is known in
advance and the cache size does not need to grow during differentiation.

For vector-backed caches, `resize!(cache, n)` resizes the primal, dual, and
nested-dual scratch storage and returns `cache`. Resizing a cache with a
non-vector primal workspace throws `ArgumentError`; an extension-provided dual
workspace must support `resize!` when callers need this operation.

# Arguments

  - `u`: prototype array whose shape and primal element type determine the cache.
  - `N`: ForwardDiff chunk size. Pass `Val{N}` to encode it in the cache type, or an
    integer as a convenience constructor.

# Fields

  - `du`: primal workspace with the shape and storage type of `u`.
  - `dual_du`: workspace for dual-number evaluations.
  - `typed_du`: lazily allocated persistent workspaces for element types that can
    reuse neither `du` nor `dual_du`, keyed by element type.

# Examples

```julia
cache = FixedSizeDiffCache(zeros(3), 2)
workspace = get_tmp(cache, zeros(3))
```
"""
struct FixedSizeDiffCache{T <: AbstractArray, S <: AbstractArray}
    du::T
    dual_du::S
    typed_du::Dict{DataType, Any}
end

"""
    dualarraycreator(u::AbstractArray, size, ::Type{Val{N}})

Construct the automatic-differentiation workspace of a
[`FixedSizeDiffCache`](@ref).

# Interface

This is a versioned developer interface for an AD package or array package,
not an end-user customization point. An extension may add a method only when
it owns the concrete type of `u` or the AD scalar stored by the result. Define
a method with the following shape:

```julia
PreallocationTools.dualarraycreator(
    u::MyArray{T}, size, ::Type{Val{N}}
) where {T, N}
```

`FixedSizeDiffCache` calls this hook while constructing its dual workspace.
The method must not extend an array representation or scalar type owned by an
unrelated package.

# Arguments

  - `u`: primal cache prototype. Its representation and axes define the
    representation expected from the returned workspace.
  - `size`: dimensions of the workspace to allocate. It is a tuple of
    nonnegative integer dimensions supplied by the constructor.
  - `N`: nonnegative AD chunk size encoded by `Val`.

# Return

Return a newly allocated `AbstractArray` with dimensions `size`, whose element
type can represent the extension's AD values with chunk size `N`. The result
must preserve the array representation and axes required by `u`; it must not
alias `u`. For vector-backed caches, provide a resizable result when callers
need to use [`resize!`](@ref) on the cache.

# Failure Behavior

Do not define a method for unsupported representations. A constructor for
which no extension provides a workspace fails normally rather than silently
creating an incompatible cache.

# Example

```julia
PreallocationTools.dualarraycreator(
    u::MyAD.Array{T}, size, ::Type{Val{N}}
) where {T, N} = MyAD.Array{MyAD.Dual{T, N}}(undef, size)
```
"""
dualarraycreator(args...) = nothing
@public dualarraycreator

function FixedSizeDiffCache(
        u::AbstractArray{T}, siz,
        ::Type{Val{chunk_size}}
    ) where {T, chunk_size}
    x = dualarraycreator(u, siz, Val{chunk_size})
    return FixedSizeDiffCache(deepcopy(u), x, Dict{DataType, Any}())
end

"""
    forwarddiff_compat_chunk_size(n::Integer)

Return the default AD chunk size for a cache with `n` primal elements.

# Interface

This is a versioned developer interface used by [`DiffCache`](@ref) and
[`FixedSizeDiffCache`](@ref) when callers omit `N`. An AD backend that provides
the process-wide default may specialize
`forwarddiff_compat_chunk_size(n::Int)`. Because the argument is a built-in
integer, only one active backend should provide that default; packages that
need a different chunk size should pass `N` explicitly to the constructor.

# Arguments

  - `n`: number of primal elements in the requested cache. It is nonnegative.

# Return

Return a nonnegative `Int` accepted by the backend. The fallback returns `0`,
which represents no preallocated dual partials.

# Failure Behavior

Returning a negative value or a value unsupported by the backend violates the
interface and may make cache construction fail.

# Example

```julia
PreallocationTools.forwarddiff_compat_chunk_size(n::Int) = MyAD.default_chunk_size(n)
```
"""
forwarddiff_compat_chunk_size(n::Integer) = 0
@public forwarddiff_compat_chunk_size

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

"""
    chunksize(::Type{T})

Return the AD chunk size encoded by scalar type `T`.

# Interface

This is a versioned developer interface for AD scalar types. An extension may
specialize `chunksize(::Type{MyADScalar})` only for scalar types it owns. The
result lets [`FixedSizeDiffCache`](@ref) decide whether its dual workspace can
be reused for a requested scalar type.

# Arguments

  - `T`: a scalar type. `T` may encode an AD chunk size in its type parameters.

# Return

Return the nonnegative `Int` chunk size encoded by `T`. Return `0` when `T`
does not encode chunk-size information; this is the fallback behavior.

# Failure Behavior

Do not return a chunk size different from the representation encoded by `T`.
An incorrect result can select a workspace with incompatible capacity.

# Example

```julia
PreallocationTools.chunksize(::Type{MyAD.Dual{T, N}}) where {T, N} = N
```
"""
chunksize(::Type{T}) where {T} = 0
@public chunksize

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

# Arguments

  - `cache`: a cache created by this package, or another object for which the
    generic fallback should return `cache` itself.
  - `u`: a value, array, or scalar type describing the element type and, for
    lazy buffers, the requested shape.
  - `size`: optional lazy-buffer shape; it is accepted only by
    `LazyBufferCache`.

# Return

The returned workspace is owned by `cache` and is reused by later matching
lookups. Callers must fully overwrite scratch storage before reading it and
must not retain it across calls that can request the same cache entry.

# Developer Interface

An AD extension may specialize `get_tmp` for `DiffCache` or
`FixedSizeDiffCache` and a scalar type that it owns. The method must return
scratch storage with the cache's logical axes and an element representation
compatible with the requested scalar type. It must use only the public cache
fields and developer hooks documented on the Developer API page, and it must
not return storage that aliases the primal workspace unless that representation
explicitly permits it.

# Examples

```jldoctest
julia> using PreallocationTools

julia> cache = DiffCache(zeros(2), 1);

julia> get_tmp(cache, zeros(2)) === cache.du
true
```
"""
function get_tmp(dc::FixedSizeDiffCache, u::Union{Number, AbstractArray})
    return get_tmp(dc, eltype(u))
end

function _promotes_to_primal(::Type{P}, ::Type{T}) where {P, T}
    # Query the requested type first so its custom rule can avoid an ambiguous reverse rule.
    promoted = Base.promote_rule(T, P)
    return promoted === Union{} ? promote_type(P, T) <: P : promoted <: P
end

function get_tmp(dc::FixedSizeDiffCache, ::Type{T}) where {T <: Number}
    return if _promotes_to_primal(eltype(dc.du), T)
        dc.du
    else
        _typed_tmp(dc, T)
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

For vector-backed caches, `resize!(cache, n)` resizes the primal and
nested-dual scratch storage while preserving the current dual-storage capacity
per primal element, then returns `cache`. Resizing a cache with a non-vector
primal workspace throws `ArgumentError`.

# Arguments

  - `u`: prototype array whose primal storage is cached.
  - `N`: ForwardDiff chunk size, or one chunk size per nested AD level.

# Keyword Arguments

  - `levels`: number of nested forward-mode AD levels when `N` is scalar.
  - `warn_on_resize`: whether to emit a one-time warning if dual storage grows.

# Fields

  - `du`: primal workspace.
  - `dual_du`: dual-number workspace, enlarged on demand when necessary.
  - `typed_du`: lazily allocated persistent workspaces for element types that can
    reuse neither `du` nor `dual_du` (e.g. sparsity tracers), keyed by element type.
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
    typed_du::Dict{DataType, Any}
    warn_on_resize::Bool
end

function DiffCache(u::AbstractArray{T}, siz, chunk_sizes; warn_on_resize::Bool = true) where {T}
    x = adapt(
        _parameterless_type(u),
        _zeroed_or_uninitialized(T, prod(chunk_sizes .+ 1) * prod(siz))
    )
    return DiffCache(u, x, Dict{DataType, Any}(), warn_on_resize)
end

_parameterless_type(x) = typeof(x).name.wrapper

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

# Arguments

  - `args`: positional arguments accepted by [`DiffCache`](@ref).

# Keyword Arguments

  - `warn_on_resize`: forwarded to [`DiffCache`](@ref); controls whether an
    undersized dual workspace emits its one-time resize warning.
  - `kwargs`: additional keywords accepted by [`DiffCache`](@ref).
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

# The compiler resolves this to a constant, and `:removable` lets it delete the
# runtime allocation (same pattern as `_preserved_similar_type` for `LazyBufferCache`).
Base.@assume_effects :removable function _typed_tmp_type(x::AbstractArray, ::Type{T}) where {T}
    return typeof(_restructure(x, similar(x, T, length(x))))
end

# Persistent workspace for element types that can reuse neither `du` nor `dual_du`
# (sparsity tracers, exotic number types, nested-dual reconstruction). One flat,
# concretely typed buffer per element type, allocated on first use and shared by
# every fetch: `get_tmp`'s contract is that repeated fetches from the same cache
# see the same storage, and callers like `one(eltype(c))` need a concrete eltype —
# a shared `Vector{Any}` can satisfy only the first requirement. The `get!` lookup
# is type-asserted like `LazyBufferCache.get_tmp`, since the table is untyped.
function _typed_tmp(dc, ::Type{T}) where {T}
    buf = get!(dc.typed_du, T) do
        similar(dc.du, T, length(dc.du))
    end
    if length(buf) != length(dc.du)
        # `du` was resized behind the cache's back (`Base.resize!` keeps the typed
        # workspaces in sync). Contents are scratch, so recreate.
        buf = dc.typed_du[T] = similar(dc.du, T, length(dc.du))
    end
    return _restructure(dc.du, buf)::_typed_tmp_type(dc.du, T)
end

function get_tmp(dc::DiffCache, u::Union{Number, AbstractArray})
    return if _promotes_to_primal(eltype(dc.du), eltype(u))
        dc.du
    else
        _typed_tmp(dc, eltype(u))
    end
end

function get_tmp(dc::DiffCache, ::Type{T}) where {T <: Number}
    return if _promotes_to_primal(eltype(dc.du), T)
        dc.du
    else
        _typed_tmp(dc, T)
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
    return DiffCache(_resizeable_reshape(dc.du, shape), dc.dual_du, dc.typed_du, dc.warn_on_resize)
end

Base.reshape(dc::DiffCache, dims::Integer...) = reshape(dc, dims)

_resizeable_reshape(a::AbstractVector, shape) = reshape(view(a, :), shape)
_resizeable_reshape(a::AbstractArray, shape) = reshape(a, shape)

get_tmp(dc, u) = dc

"""
    _restructure(normal_cache::AbstractArray, duals)

Give AD workspace storage the representation and shape of `normal_cache`.

# Interface

This is a versioned developer interface for AD or array extensions, not an
end-user customization point. An extension may specialize
`_restructure(normal_cache::MyArray, duals)` only when it owns the concrete
array representation of `normal_cache`. The default uses `reshape` for
ordinary arrays and `ArrayInterface.restructure` for custom representations.

# Arguments

  - `normal_cache`: primal workspace whose representation and axes must be
    preserved.
  - `duals`: AD storage containing at least `length(normal_cache)` logical
    values in linear-index order.

# Return

Return an `AbstractArray` with `axes(result) == axes(normal_cache)`. Its values
must correspond to `duals` in linear-index order, and mutations through the
result must update the supplied `duals`; implementations must not copy the
workspace merely to change its representation.

# Failure Behavior

Throw a clear error when `duals` cannot represent the requested axes or when
the extension cannot preserve the required array representation.

# Example

```julia
PreallocationTools._restructure(cache::MyAD.Array, duals) = MyAD.Array(duals, axes(cache))
```
"""
function _restructure(normal_cache::Array, duals)
    return reshape(duals, size(normal_cache)...)
end
@public _restructure

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

Resize the dual workspace owned by a [`DiffCache`](@ref).

# Interface

This is a versioned developer interface for AD extensions. Call it from a
specialized [`get_tmp`](@ref) method only after establishing that the requested
AD representation needs additional storage. The extension must own the scalar
type used to select that representation. It must not resize `dc.dual_du`
directly, because this helper applies the cache's warning policy.

# Arguments

  - `dc`: `DiffCache` whose dual workspace is vector-backed and resizable.
  - `nelem`: required number of elements in `dc.dual_du`. It must be at least
    the current capacity and nonnegative.

# Return

Return the resized `dc.dual_du` workspace. The returned storage remains owned
by `dc`; callers must use [`_restructure`](@ref) before exposing it with the
primal cache's representation.

# Failure Behavior

Calling this hook for a non-resizable dual workspace, or with an invalid
capacity, is unsupported and may throw from `resize!`.

# Example

```julia
needed = chunksize(ADScalar) * length(cache.du)
needed > length(cache.dual_du) && enlargediffcache!(cache, needed)
```
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
@public enlargediffcache!

# LazyBufferCache

"""
    b = LazyBufferCache(f = identity; initializer! = identity)

A lazily allocated buffer cache. Given an array `u`, `b[u]` or `get_tmp(b, u)`
returns an array with a shape chosen from `size(u)`. The cache allocates that
buffer on its first matching lookup and returns the same object on later
lookups.

# Arguments

  - `f`: maps `size(u)` to the requested buffer dimensions. The default,
    `identity`, preserves the input shape.

# Keyword Arguments

  - `initializer!`: function applied once to each newly allocated buffer. The
    default leaves the buffer uninitialized; use, for example,
    `buf -> fill!(buf, 0.0)` when callers require initialized scratch storage.

# Fields

  - `bufs`: cache from prototype array type and requested dimensions to a
    reusable buffer.
  - `sizemap`: shape-mapping function supplied as `f`.
  - `initializer!`: initialization function applied to newly allocated buffers.

Pass an explicit shape as `b[u, s]` or `get_tmp(b, u, s)` to override `f` for
one cache entry. Scratch buffers are shared for matching keys, so callers must
overwrite them before reading.

# Examples

```jldoctest
julia> using PreallocationTools

julia> cache = LazyBufferCache();

julia> size(get_tmp(cache, zeros(2), (3,)))
(3,)
```
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

A lazily allocated cache keyed by the concrete type of its input. Given `u`,
`b[u]` or `get_tmp(b, u)` calls `f(u)` on the first lookup for `typeof(u)` and
returns that same cached object for later lookups of the type.

# Arguments

  - `f`: function used to construct a cache object from the first input of each
    concrete type. The default is `identity`.

# Fields

  - `bufs`: map from concrete input type to the reusable object produced by `f`.
  - `f`: cache-construction function.

# Limitation

The result is not type-inferred because the backing map has `Any` values. Use a
function barrier around the lookup when inference matters.

# Examples

```jldoctest
julia> using PreallocationTools

julia> cache = GeneralLazyBufferCache(T -> Vector{T}(undef, 2));

julia> get_tmp(cache, Float64) === get_tmp(cache, Float64)
true
```
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

"""
    resize!(cache::DiffCache, n::Integer)
    resize!(cache::FixedSizeDiffCache, n::Integer)

Resize a vector-backed cache to `n` primal elements.

# Arguments

  - `cache`: a `DiffCache` or `FixedSizeDiffCache` whose primal workspace is a
    vector.
  - `n`: nonnegative target length for the primal workspace.

# Return

Return the same cache object. `DiffCache` preserves its current dual-storage
capacity per primal element; `FixedSizeDiffCache` resizes vector-backed dual
storage to `n`. Both methods resize the nested-dual scratch vector to `n`.

# Developer Interface

An extension that supplies vector-backed dual storage through
[`dualarraycreator`](@ref) must implement `resize!` for that storage when it
expects callers to resize the enclosing cache. It must preserve the storage's
element representation after resizing.

# Failure Behavior

Resizing is unsupported for caches with non-vector primal storage and throws
`ArgumentError`.
"""
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
    # Typed workspaces mirror the length of `du`, so they follow its resizing;
    # non-resizable buffers are dropped and lazily recreated at the next fetch.
    _resize_typed!(dc.typed_du, n)
    return dc
end

function _resize_typed!(typed_du::Dict{DataType, Any}, n::Integer)
    for k in collect(keys(typed_du))
        buf = typed_du[k]
        if buf isa Vector
            resize!(buf, n)
        else
            delete!(typed_du, k)
        end
    end
    return typed_du
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
    _resize_typed!(dc.typed_du, n)
    return dc
end

# zero dispatches for PreallocationTools types
function Base.zero(dc::DiffCache)
    return DiffCache(zero(dc.du), zero(dc.dual_du), Dict{DataType, Any}(), dc.warn_on_resize)
end

function Base.zero(dc::FixedSizeDiffCache)
    return FixedSizeDiffCache(zero(dc.du), zero(dc.dual_du), Dict{DataType, Any}())
end

function Base.zero(lbc::LazyBufferCache)
    return LazyBufferCache(lbc.sizemap; initializer! = lbc.initializer!)
end

function Base.zero(glbc::GeneralLazyBufferCache)
    return GeneralLazyBufferCache(glbc.f)
end

# copy dispatches for PreallocationTools types
function Base.copy(dc::DiffCache)
    return DiffCache(
        copy(dc.du), copy(dc.dual_du),
        Dict{DataType, Any}(k => copy(v) for (k, v) in dc.typed_du), dc.warn_on_resize
    )
end

function Base.copy(dc::FixedSizeDiffCache)
    return FixedSizeDiffCache(
        copy(dc.du), copy(dc.dual_du),
        Dict{DataType, Any}(k => copy(v) for (k, v) in dc.typed_du)
    )
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
    _fill_typed!(dc.typed_du, val)
    return dc
end

# Fill each typed workspace through its own eltype: `convert(T, val)` defines what
# `val` means for that element type (for a sparsity tracer it is the empty tracer,
# i.e. "constant with no dependencies"). Incompatible eltypes throw, matching the
# `fill!(::AbstractArray{T}, val)` contract.
function _fill_typed!(typed_du::Dict{DataType, Any}, val)
    for buf in values(typed_du)
        fill!(buf, val)
    end
    return typed_du
end

"""
    fill!(dc::FixedSizeDiffCache, val)

Fill all allocated buffers in the FixedSizeDiffCache with the given value.
"""
function Base.fill!(dc::FixedSizeDiffCache, val)
    fill!(dc.du, val)
    fill!(dc.dual_du, val)
    _fill_typed!(dc.typed_du, val)
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
