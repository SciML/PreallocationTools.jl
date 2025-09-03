module PreallocationToolsEnzymeCoreExt

using PreallocationTools
import EnzymeCore: EnzymeRules, Const, Duplicated

# TODO: Support Batched mode, on 1.11
# if VERSION >= v"1.11.0"
# function tuple_of_vectors(M::Matrix{T}, shape) where {T}
#     n, m = size(M)
#     return ntuple(m) do i
#         vec = Base.wrap(Array, memoryref(M.ref, (i - 1) * n + 1), (n,))
#         reshape(vec, shape)
#     end
# end
# end

# TODO: Support reverse mode?

function EnzymeRules.forward(config, func::Const{typeof(PreallocationTools.get_tmp)}, ::Type{<:Duplicated},
                             dc::Duplicated{<:PreallocationTools.DiffCache}, u::Union{Const{T}, Duplicated{T}}) where {T}
    du = PreallocationTools.get_tmp(dc.val, u.val)
    ddu = PreallocationTools.get_tmp(dc.dval, u.val)
    Duplicated(du, ddu)
end

function EnzymeRules.forward(config, func::Const{typeof(PreallocationTools.get_tmp)}, ::Type{<:Duplicated}, 
                             dc::Const{<:PreallocationTools.DiffCache}, u::Union{Const{T}, Duplicated{T}}) where {T}
    dc = dc.val
    du = PreallocationTools.get_tmp(dc, u.val)

    # ddu = if isbitstype(T)
    #     nelem = div(sizeof(T), sizeof(eltype(dc.dual_du))) * length(dc.du)
    #     if nelem > length(dc.dual_du)
    #         PreallocationTools.enlargediffcache!(dc, nelem)
    #     end
    #     PreallocationTools._restructure(dc.du, reinterpret(T, view(dc.dual_du, 1:nelem)))
    # else
    #     PreallocationTools._restructure(dc.du, zeros(T, size(dc.du)))
    # end

    # Enzyme requires that Duplicated types have the same type and structure
    # the above code fails since it creates something like a `Base.ReshapedArray{Float64, 2, SubArray{…}, Tuple{}})`

    # TODO: How does this interact with Enzyme over ForwardDiff?
    ddu = dc.dual_du
    resize!(ddu, length(du))

    Duplicated(du, reshape(ddu, size(du)))
end

end
