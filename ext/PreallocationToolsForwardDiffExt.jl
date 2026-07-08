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
    dual_cache_t = replace_type_parameter(CacheT, V, DualT)
    return dual_cache_t === CacheT ? DualT : dual_cache_t
end

function replace_type_parameter(::Type{T}, ::Type{From}, ::Type{To}) where {T, From, To}
    T === From && return To
    T <: Number && From <: Number && promote_type(T, From) === From && return To

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
        nelem = div(sizeof(cache_eltype), sizeof(eltype(dc.dual_du))) * length(dc.du)
        if nelem > length(dc.dual_du)
            PreallocationTools.enlargediffcache!(dc, nelem)
        end
        PreallocationTools._restructure(
            dc.du, reinterpret(cache_eltype, view(dc.dual_du, 1:nelem))
        )
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
