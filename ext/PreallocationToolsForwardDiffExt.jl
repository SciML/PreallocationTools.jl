module PreallocationToolsForwardDiffExt

using PreallocationTools
using ForwardDiff
using ArrayInterface
using Adapt
using PrecompileTools

function PreallocationTools.dualarraycreator(
        u::AbstractArray{T}, siz,
        ::Type{Val{chunk_size}}
    ) where {T, chunk_size}
    return ArrayInterface.restructure(
        u,
        zeros(
            ForwardDiff.Dual{Nothing, T, chunk_size},
            siz...
        )
    )
end

PreallocationTools.forwarddiff_compat_chunk_size(x::Int) = ForwardDiff.pickchunksize(x)

# Define chunksize for ForwardDiff.Dual types
PreallocationTools.chunksize(::Type{ForwardDiff.Dual{T, V, N}}) where {T, V, N} = N

function dual_eltype(::Type{CacheT}, ::Type{DualT}) where {
        CacheT, Tag, V, N, DualT <: ForwardDiff.Dual{Tag, V, N},
    }
    # A `Dual`-eltype cache means the base buffer was allocated at an outer
    # dual level (AD-over-AD, e.g. `DiffCache(similar(u))` while
    # ForwardDiff-ing through a solver). The requested dual type must then be
    # returned as-is: recursing into `CacheT`'s type parameters would re-tag
    # its value type and fabricate a nested dual the caller never asked for.
    CacheT <: ForwardDiff.Dual && return DualT
    # The wrapper analog of the same situation: a cache eltype like
    # `Complex{Dual{Tag, V, N}}` that already contains the requested dual is
    # already at the requested level — replacing inside it would rewrite the
    # dual's tag/value parameters and fabricate nesting one wrapper deeper.
    type_contains(CacheT, DualT) && return CacheT
    dual_cache_t = replace_type_parameter(CacheT, V, DualT)
    return dual_cache_t === CacheT ? DualT : dual_cache_t
end

function _type_contains(T, X)
    T === X && return true
    T isa DataType || return false
    return any(p -> p isa Type && _type_contains(p, X), T.parameters)
end
# @generated so the recursive parameter walk runs at compile time and folds to
# a constant — a runtime svec iteration here would allocate in the `get_tmp`
# hot path. (`_type_contains` must be defined above: generated function bodies
# may only call functions from an older world age.)
@generated function type_contains(::Type{T}, ::Type{X}) where {T, X}
    return _type_contains(T, X)
end

function replace_type_parameter(::Type{T}, ::Type{From}, ::Type{To}) where {T, From, To}
    T === From && return To
    T <: Number && From <: Number && promote_type(T, From) === From && return To
    # Duals are atomic: replace them whole (the matches above) or leave them
    # alone. Recursing into a Dual's parameters would rewrite its Tag's value
    # type, silently breaking ForwardDiff's tag matching.
    T <: ForwardDiff.Dual && return T

    typename = Base.typename(T)
    wrapper = typename.wrapper
    parameters = T.parameters
    isempty(parameters) && return T

    new_parameters = map(parameters) do parameter
        parameter isa Type ? replace_type_parameter(parameter, From, To) : parameter
    end

    return new_parameters == parameters ? T : Core.apply_type(wrapper, new_parameters...)
end

function diffcache_dual_tmp(dc::PreallocationTools.DiffCache, ::Type{T}) where {T <: ForwardDiff.Dual}
    cache_eltype = dual_eltype(eltype(dc.du), T)
    return if isbitstype(cache_eltype)
        # Branch on a type-level size condition so it constant-folds per
        # specialization (each call site sees exactly one branch — keeping the
        # common path type-stable and allocation-free, bit-identical to the
        # original code).
        if sizeof(cache_eltype) % sizeof(eltype(dc.dual_du)) == 0
            nelem = div(sizeof(cache_eltype), sizeof(eltype(dc.dual_du))) *
                length(dc.du)
            if nelem > length(dc.dual_du)
                PreallocationTools.enlargediffcache!(dc, nelem)
            end
            PreallocationTools._restructure(
                dc.du, reinterpret(cache_eltype, view(dc.dual_du, 1:nelem))
            )
        else
            # A resolved eltype SMALLER than (or not a multiple of) the storage
            # eltype — e.g. a bare-dual fallback against a wrapper-of-dual
            # buffer. The per-element `div` ratio floors to zero there, so use
            # ceiling byte arithmetic and take exactly `length(du)` elements
            # (the ceiling can over-cover by a fraction of an element).
            nelem = cld(
                sizeof(cache_eltype) * length(dc.du), sizeof(eltype(dc.dual_du))
            )
            if nelem > length(dc.dual_du)
                PreallocationTools.enlargediffcache!(dc, nelem)
            end
            reint = reinterpret(cache_eltype, view(dc.dual_du, 1:nelem))
            PreallocationTools._restructure(dc.du, view(reint, 1:length(dc.du)))
        end
    else
        PreallocationTools._restructure(dc.du, zeros(cache_eltype, size(dc.du)))
    end
end

# Define get_tmp methods for ForwardDiff.Dual types
function PreallocationTools.get_tmp(dc::PreallocationTools.FixedSizeDiffCache, u::T) where {T <: ForwardDiff.Dual}
    x = reinterpret(T, dc.dual_du)
    return if PreallocationTools.chunksize(T) === PreallocationTools.chunksize(eltype(dc.dual_du))
        x
    else
        @view x[axes(dc.du)...]
    end
end

function PreallocationTools.get_tmp(dc::PreallocationTools.FixedSizeDiffCache, u::Type{T}) where {T <: ForwardDiff.Dual}
    x = reinterpret(T, dc.dual_du)
    return if PreallocationTools.chunksize(T) === PreallocationTools.chunksize(eltype(dc.dual_du))
        x
    else
        @view x[axes(dc.du)...]
    end
end

function PreallocationTools.get_tmp(dc::PreallocationTools.FixedSizeDiffCache, u::AbstractArray{T}) where {T <: ForwardDiff.Dual}
    x = reinterpret(T, dc.dual_du)
    return if PreallocationTools.chunksize(T) === PreallocationTools.chunksize(eltype(dc.dual_du))
        x
    else
        @view x[axes(dc.du)...]
    end
end

function PreallocationTools.get_tmp(dc::PreallocationTools.DiffCache, u::T) where {T <: ForwardDiff.Dual}
    return diffcache_dual_tmp(dc, T)
end

function PreallocationTools.get_tmp(dc::PreallocationTools.DiffCache, ::Type{T}) where {T <: ForwardDiff.Dual}
    return diffcache_dual_tmp(dc, T)
end

function PreallocationTools.get_tmp(dc::PreallocationTools.DiffCache, u::AbstractArray{T}) where {T <: ForwardDiff.Dual}
    return diffcache_dual_tmp(dc, T)
end

@setup_workload begin
    @compile_workload begin
        # Precompile ForwardDiff-specific code paths
        u = ones(10)

        # DiffCache with Dual numbers
        cache = PreallocationTools.DiffCache(u)
        dual_u = ForwardDiff.Dual.(1:10, 1.0)
        PreallocationTools.get_tmp(cache, dual_u)

        # FixedSizeDiffCache creation and usage with Dual
        fcache = PreallocationTools.FixedSizeDiffCache(u)
        PreallocationTools.get_tmp(fcache, u)
        PreallocationTools.get_tmp(fcache, dual_u)
    end
end

end
