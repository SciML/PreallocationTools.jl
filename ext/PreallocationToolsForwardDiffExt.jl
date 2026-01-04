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
    return if isbitstype(T)
        nelem = div(sizeof(T), sizeof(eltype(dc.dual_du))) * length(dc.du)
        if nelem > length(dc.dual_du)
            PreallocationTools.enlargediffcache!(dc, nelem)
        end
        PreallocationTools._restructure(dc.du, reinterpret(T, view(dc.dual_du, 1:nelem)))
    else
        PreallocationTools._restructure(dc.du, zeros(T, size(dc.du)))
    end
end

function PreallocationTools.get_tmp(dc::PreallocationTools.DiffCache, ::Type{T}) where {T <: ForwardDiff.Dual}
    return if isbitstype(T)
        nelem = div(sizeof(T), sizeof(eltype(dc.dual_du))) * length(dc.du)
        if nelem > length(dc.dual_du)
            PreallocationTools.enlargediffcache!(dc, nelem)
        end
        PreallocationTools._restructure(dc.du, reinterpret(T, view(dc.dual_du, 1:nelem)))
    else
        PreallocationTools._restructure(dc.du, zeros(T, size(dc.du)))
    end
end

function PreallocationTools.get_tmp(dc::PreallocationTools.DiffCache, u::AbstractArray{T}) where {T <: ForwardDiff.Dual}
    return if isbitstype(T)
        nelem = div(sizeof(T), sizeof(eltype(dc.dual_du))) * length(dc.du)
        if nelem > length(dc.dual_du)
            PreallocationTools.enlargediffcache!(dc, nelem)
        end
        PreallocationTools._restructure(dc.du, reinterpret(T, view(dc.dual_du, 1:nelem)))
    else
        PreallocationTools._restructure(dc.du, zeros(T, size(dc.du)))
    end
end

@setup_workload begin
    @compile_workload begin
        # Precompile ForwardDiff-specific code paths
        u = rand(10)

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
