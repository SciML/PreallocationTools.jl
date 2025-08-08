module PreallocationToolsForwardDiffExt

using PreallocationTools
using ForwardDiff
using ArrayInterface
using Adapt

# Initialize on module load
function __init__()
    # Set the dual array creator function
    PreallocationTools.DUAL_ARRAY_CREATOR[] = function(u::AbstractArray{T}, siz,
            ::Type{Val{chunk_size}}) where {T, chunk_size}
        ArrayInterface.restructure(u,
            zeros(ForwardDiff.Dual{Nothing, T, chunk_size},
                siz...))
    end
    
    # Set the chunk size function to use ForwardDiff's pickchunksize
    PreallocationTools.CHUNK_SIZE_FUNC[] = ForwardDiff.pickchunksize
end

# Define chunksize for ForwardDiff.Dual types
PreallocationTools.chunksize(::Type{ForwardDiff.Dual{T, V, N}}) where {T, V, N} = N

# Define get_tmp methods for ForwardDiff.Dual types
function PreallocationTools.get_tmp(dc::PreallocationTools.FixedSizeDiffCache, u::T) where {T <: ForwardDiff.Dual}
    x = reinterpret(T, dc.dual_du)
    if PreallocationTools.chunksize(T) === PreallocationTools.chunksize(eltype(dc.dual_du))
        x
    else
        @view x[axes(dc.du)...]
    end
end

function PreallocationTools.get_tmp(dc::PreallocationTools.FixedSizeDiffCache, u::Type{T}) where {T <: ForwardDiff.Dual}
    x = reinterpret(T, dc.dual_du)
    if PreallocationTools.chunksize(T) === PreallocationTools.chunksize(eltype(dc.dual_du))
        x
    else
        @view x[axes(dc.du)...]
    end
end

function PreallocationTools.get_tmp(dc::PreallocationTools.FixedSizeDiffCache, u::AbstractArray{T}) where {T <: ForwardDiff.Dual}
    x = reinterpret(T, dc.dual_du)
    if PreallocationTools.chunksize(T) === PreallocationTools.chunksize(eltype(dc.dual_du))
        x
    else
        @view x[axes(dc.du)...]
    end
end

function PreallocationTools.get_tmp(dc::PreallocationTools.DiffCache, u::T) where {T <: ForwardDiff.Dual}
    if isbitstype(T)
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
    if isbitstype(T)
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
    if isbitstype(T)
        nelem = div(sizeof(T), sizeof(eltype(dc.dual_du))) * length(dc.du)
        if nelem > length(dc.dual_du)
            PreallocationTools.enlargediffcache!(dc, nelem)
        end
        PreallocationTools._restructure(dc.du, reinterpret(T, view(dc.dual_du, 1:nelem)))
    else
        PreallocationTools._restructure(dc.du, zeros(T, size(dc.du)))
    end
end

end