module PreallocationToolsSparseConnectivityTracerExt

using PreallocationTools: PreallocationTools, DiffCache, get_tmp
using SparseConnectivityTracer: AbstractTracer, Dual

function PreallocationTools.get_tmp(dc::DiffCache, u::T) where {
        T <:
        Union{AbstractTracer, Dual},
    }
    return get_tmp(dc, typeof(u))
end

function PreallocationTools.get_tmp(
        dc::DiffCache, u::AbstractArray{<:T}
    ) where {T <: Union{AbstractTracer, Dual}}
    return get_tmp(dc, eltype(u))
end

function PreallocationTools.get_tmp(dc::DiffCache, ::Type{T}) where {
        T <: Union{
            AbstractTracer, Dual,
        },
    }
    # We allocate memory here since we assume that sparsity connection happens only
    # once (or maybe a few times). This simplifies the implementation and allows us
    # to save memory in the long run since we do not need to store an additional
    # cache for the sparsity detection that would be used only once but carried
    # around forever.
    return similar(dc.du, T)
end

end
