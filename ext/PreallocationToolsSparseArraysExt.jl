module PreallocationToolsSparseArraysExt

using PreallocationTools: PreallocationTools
using SparseArrays: AbstractSparseArray, AbstractSparseMatrixCSC, AbstractSparseVector,
    SparseMatrixCSC, SparseVector, getcolptr, nonzeroinds, nonzeros, rowvals

const SparseCSC = Union{SparseMatrixCSC, SparseVector}

# Sparse arrays keep their stored values in a `Vector`, which cannot wrap a
# reinterpreted view of `dual_du`. `PreallocationTools._sparse_backed` extends
# this to wrappers of sparse arrays (views, reshapes, transposes).
PreallocationTools._sparse_backed(::Type{<:AbstractSparseArray}) = true

# Workspaces for other element types are persistent arrays from `similar(du, T)`,
# so memory scales with the number of stored entries rather than with
# `length(du)`. For `SparseMatrixCSC` and `SparseVector`, each has the stored
# pattern of `du` at its allocation; entries stored in `du` later are not added.
function PreallocationTools._shaped_typed_tmp(dc, ::Type{T}) where {T}
    buf = get(dc.typed_du, T, nothing)
    if buf === nothing
        buf = dc.typed_du[T] = _zero_stored!(similar(dc.du, T))
    end
    return buf::_workspace_type(dc.du, T)
end

_workspace_type(::SparseMatrixCSC{<:Any, Ti}, ::Type{T}) where {T, Ti} = SparseMatrixCSC{T, Ti}
_workspace_type(::SparseVector{<:Any, Ti}, ::Type{T}) where {T, Ti} = SparseVector{T, Ti}
# The compiler resolves this to a constant, and `:removable` lets it delete the
# runtime allocation (same pattern as `PreallocationTools._typed_buffer_type`).
Base.@assume_effects :removable function _workspace_type(du::AbstractArray, ::Type{T}) where {T}
    return typeof(similar(du, T))
end

# Setting the stored values of the sparse root keeps the pattern and costs
# O(stored entries); `fill!` on a wrapper would visit every index, and `fill!` on
# a sparse array requires `iszero` to return a `Bool`, which sparsity tracers
# do not. For a wrapper, this fills every stored entry of the root, including
# those outside the wrapper's region, so `buf` must own its root, as the typed
# workspaces of a cache do.
function PreallocationTools._fill_stored!(
        buf::Union{AbstractSparseMatrixCSC, AbstractSparseVector}, x
    )
    fill!(nonzeros(buf), x)
    return buf
end
function PreallocationTools._fill_stored!(buf, x)
    p = parent(buf)
    p === buf ? fill!(buf, x) : PreallocationTools._fill_stored!(p, x)
    return buf
end

# Gives `dest` the stored pattern and storage lengths of `src`, zeroing the stored
# values of `dest` if they change and leaving them untouched otherwise. For
# wrappers, this matches the sparse roots, which must have equal sizes.
function PreallocationTools._match_stored_pattern!(dest::SparseMatrixCSC, src::SparseMatrixCSC)
    size(dest) == size(src) ||
        throw(DimensionMismatch("cannot match the stored pattern of a $(size(src)) sparse matrix in one of size $(size(dest))"))
    if getcolptr(dest) != getcolptr(src) || rowvals(dest) != rowvals(src) ||
            length(nonzeros(dest)) != length(nonzeros(src))
        copyto!(getcolptr(dest), getcolptr(src))
        copyto!(resize!(rowvals(dest), length(rowvals(src))), rowvals(src))
        fill!(resize!(nonzeros(dest), length(nonzeros(src))), zero(eltype(dest)))
    end
    return dest
end
function PreallocationTools._match_stored_pattern!(dest::SparseVector, src::SparseVector)
    length(dest) == length(src) ||
        throw(DimensionMismatch("cannot match the stored pattern of a sparse vector of length $(length(src)) in one of length $(length(dest))"))
    if nonzeroinds(dest) != nonzeroinds(src) || length(nonzeros(dest)) != length(nonzeros(src))
        copyto!(resize!(nonzeroinds(dest), length(nonzeroinds(src))), nonzeroinds(src))
        fill!(resize!(nonzeros(dest), length(nonzeros(src))), zero(eltype(dest)))
    end
    return dest
end
function PreallocationTools._match_stored_pattern!(dest, src)
    p = parent(dest)
    p === dest &&
        throw(ArgumentError("matching the stored pattern of $(typeof(dest)) is not supported"))
    PreallocationTools._match_stored_pattern!(p, parent(src))
    return dest
end

# Stored values from `similar` are uninitialized, and sparse operations read them.
_zero_stored!(buf) = PreallocationTools._fill_stored!(buf, zero(eltype(buf)))

# `zero` of a sparse array drops its stored entries.
PreallocationTools._zero_primal(du::SparseCSC) = _zero_stored!(copy(du))

# `LazyBufferCache` buffer types, computed without allocating a sparse array per
# lookup. These match `similar(x, dims)` for one and two dimensions.
function PreallocationTools.similar_type(
        ::Union{SparseMatrixCSC{Tv, Ti}, SparseVector{Tv, Ti}}, ::Tuple{Integer}
    ) where {Tv, Ti}
    return SparseVector{Tv, Ti}
end
function PreallocationTools.similar_type(
        ::Union{SparseMatrixCSC{Tv, Ti}, SparseVector{Tv, Ti}}, ::Tuple{Integer, Integer}
    ) where {Tv, Ti}
    return SparseMatrixCSC{Tv, Ti}
end

end
