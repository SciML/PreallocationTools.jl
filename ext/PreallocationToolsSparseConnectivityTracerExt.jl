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

# The compiler resolves this to a constant, and `:removable` lets it delete the
# runtime allocation (same pattern as `_preserved_similar_type` for `LazyBufferCache`).
Base.@assume_effects :removable function _tracer_buffer_type(
        x::AbstractArray, ::Type{T}
    ) where {T}
    return typeof(similar(x, T, ntuple(Returns(1), ndims(x))))
end

function PreallocationTools.get_tmp(dc::DiffCache, ::Type{T}) where {
        T <: Union{
            AbstractTracer, Dual,
        },
    }
    # `get_tmp`'s contract is that repeated fetches from the same `DiffCache` share
    # storage: callers commonly write through one fetch and read through another
    # (e.g. the collocation loops in BoundaryValueDiffEq), so a fresh `similar` per
    # call would hand the reader unwritten memory. The workspace must nevertheless
    # keep a concrete eltype — `one(eltype(c))`-style caller code cannot work with
    # the `Vector{Any}` fallback storage — so each tracer type gets its own lazily
    # allocated persistent buffer, like `LazyBufferCache`.
    buf = get!(dc.typed_du, T) do
        similar(dc.du, T)
    end::_tracer_buffer_type(dc.du, T)
    if size(buf) != size(dc.du)
        # `du` was resized behind the cache's back (`Base.resize!(dc, n)` keeps the
        # typed workspaces in sync). Contents are scratch, so recreate.
        buf = similar(dc.du, T)
        dc.typed_du[T] = buf
    end
    return buf
end

end
