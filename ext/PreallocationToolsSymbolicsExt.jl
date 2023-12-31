module PreallocationToolsSymbolicsExt

using PreallocationTools
import PreallocationTools: _restructure, get_tmp
using Symbolics, ForwardDiff

function get_tmp(dc::Union{FixedSizeDiffCache,DiffCache}, u::Type{X}) where {T,N, X<: ForwardDiff.Dual{T, Num, N}}
    if length(dc.du) > length(dc.any_du)
        resize!(dc.any_du, length(dc.du))
    end
    _restructure(dc.du, dc.any_du)
end

function get_tmp(dc::Union{FixedSizeDiffCache,DiffCache}, u::X) where {T,N, X<: ForwardDiff.Dual{T, Num, N}}
    if length(dc.du) > length(dc.any_du)
        resize!(dc.any_du, length(dc.du))
    end
    _restructure(dc.du, dc.any_du)
end

function get_tmp(dc::Union{FixedSizeDiffCache,DiffCache}, u::AbstractArray{X}) where {T,N, X<: ForwardDiff.Dual{T, Num, N}}
    if length(dc.du) > length(dc.any_du)
        resize!(dc.any_du, length(dc.du))
    end
    _restructure(dc.du, dc.any_du)
end

end
