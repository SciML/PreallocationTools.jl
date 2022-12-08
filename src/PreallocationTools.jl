module PreallocationTools

using ForwardDiff, ArrayInterfaceCore, Adapt

struct FixedSizeDiffCache{T <: AbstractArray, S <: AbstractArray}
    du::T
    dual_du::S
    any_du::Vector{Any}
end

function FixedSizeDiffCache(u::AbstractArray{T}, siz,
                            ::Type{Val{chunk_size}}) where {T, chunk_size}
    x = ArrayInterfaceCore.restructure(u,
                                       zeros(ForwardDiff.Dual{nothing, T, chunk_size},
                                             siz...))
    xany = Any[]
    FixedSizeDiffCache(deepcopy(u), x, xany)
end

"""

`FixedSizeDiffCache(u::AbstractArray, N = Val{default_cache_size(length(u))})`

Builds a `FixedSizeDiffCache` object that stores both a version of the cache for `u`
and for the `Dual` version of `u`, allowing use of pre-cached vectors with
forward-mode automatic differentiation.
"""
function FixedSizeDiffCache(u::AbstractArray,
                            ::Type{Val{N}} = Val{ForwardDiff.pickchunksize(length(u))}) where {
                                                                                               N
                                                                                               }
    FixedSizeDiffCache(u, size(u), Val{N})
end

function FixedSizeDiffCache(u::AbstractArray, N::Integer)
    FixedSizeDiffCache(u, size(u), Val{N})
end

chunksize(::Type{ForwardDiff.Dual{T, V, N}}) where {T, V, N} = N

function get_tmp(dc::FixedSizeDiffCache, u::T) where {T <: ForwardDiff.Dual}
    x = reinterpret(T, dc.dual_du)
    if chunksize(T) === chunksize(eltype(dc.dual_du))
        x
    else
        @view x[axes(dc.du)...]
    end
end

function get_tmp(dc::FixedSizeDiffCache, u::AbstractArray{T}) where {T <: ForwardDiff.Dual}
    x = reinterpret(T, dc.dual_du)
    if chunksize(T) === chunksize(eltype(dc.dual_du))
        x
    else
        @view x[axes(dc.du)...]
    end
end

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

# DiffCache

struct DiffCache{T <: AbstractArray, S <: AbstractArray}
    du::T
    dual_du::S
    any_du::Vector{Any}
end

function DiffCache(u::AbstractArray{T}, siz, chunk_sizes) where {T}
    x = adapt(ArrayInterfaceCore.parameterless_type(u),
              zeros(T, prod(chunk_sizes .+ 1) * prod(siz)))
    xany = Any[]
    DiffCache(u, x, xany)
end

"""
`DiffCache(u::AbstractArray, N::Int = ForwardDiff.pickchunksize(length(u)); levels::Int = 1)`
`DiffCache(u::AbstractArray; N::AbstractArray{<:Int})`

Builds a `DiffCache` object that stores both a version of the cache for `u`
and for the `Dual` version of `u`, allowing use of pre-cached vectors with
forward-mode automatic differentiation. Supports nested AD via keyword `levels`
or specifying an array of chunk_sizes.

"""
function DiffCache(u::AbstractArray, N::Int = ForwardDiff.pickchunksize(length(u));
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
function get_tmp(dc::DiffCache, u::T) where {T <: ForwardDiff.Dual}
    nelem = div(sizeof(T), sizeof(eltype(dc.dual_du))) * length(dc.du)
    if nelem > length(dc.dual_du)
        enlargediffcache!(dc, nelem)
    end
    _restructure(dc.du, reinterpret(T, view(dc.dual_du, 1:nelem)))
end

function get_tmp(dc::DiffCache, u::AbstractArray{T}) where {T <: ForwardDiff.Dual}
    nelem = div(sizeof(T), sizeof(eltype(dc.dual_du))) * length(dc.du)
    if nelem > length(dc.dual_du)
        enlargediffcache!(dc, nelem)
    end
    _restructure(dc.du, reinterpret(T, view(dc.dual_du, 1:nelem)))
end

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

get_tmp(dc, u) = dc

function _restructure(normal_cache::Array, duals)
    reshape(duals, size(normal_cache)...)
end

function _restructure(normal_cache::AbstractArray, duals)
    ArrayInterfaceCore.restructure(normal_cache, duals)
end

function enlargediffcache!(dc, nelem) #warning comes only once per DiffCache.
    chunksize = div(nelem, length(dc.du)) - 1
    @warn "The supplied DiffCache was too small and was enlarged. This incurs allocations
    on the first call to `get_tmp`. If few calls to `get_tmp` occur and optimal performance is essential,
    consider changing 'N'/chunk size of this DiffCache to $chunksize."
    resize!(dc.dual_du, nelem)
end

# LazyBufferCache

"""
    b = LazyBufferCache(f=identity)

A lazily allocated buffer object.  Given an array `u`, `b[u]` returns an array of the
same type and size `f(size(u))` (defaulting to the same size), which is allocated as
needed and then cached within `b` for subsequent usage.

"""
struct LazyBufferCache{F <: Function}
    bufs::Dict # a dictionary mapping types to buffers
    sizemap::F
    LazyBufferCache(f::F = identity) where {F <: Function} = new{F}(Dict(), f) # start with empty dict
end

# override the [] method
function Base.getindex(b::LazyBufferCache, u::T) where {T <: AbstractArray}
    s = b.sizemap(size(u)) # required buffer size
    buf = get!(b.bufs, (T, s)) do
        similar(u, s) # buffer to allocate if it was not found in b.bufs
    end::T  # declare type since b.bufs dictionary is untyped
    return buf
end

export FixedSizeDiffCache, DiffCache, LazyBufferCache, dualcache
export get_tmp

end
