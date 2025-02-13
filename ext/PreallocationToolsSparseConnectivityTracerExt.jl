module PreallocationToolsSparseConnectivityTracerExt

using PreallocationTools
isdefined(Base, :get_extension) ? (import SparseConnectivityTracer) :
(import ..SparseConnectivityTracer)

function PreallocationTools.get_tmp(
        dc::DiffCache, u::T) where {T <: SparseConnectivityTracer.Dual}
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

function PreallocationTools.get_tmp(
        dc::DiffCache, ::Type{T}) where {T <: SparseConnectivityTracer.Dual}
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

function PreallocationTools.get_tmp(
        dc::DiffCache, u::AbstractArray{T}) where {T <: SparseConnectivityTracer.Dual}
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
