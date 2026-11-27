module PreallocationToolsSparseArraysExt

using PreallocationTools: PreallocationTools
using SparseArrays: SparseArrays, SparseMatrixCSC, SparseVector, getcolptr, nnz,
    nonzeros, rowvals

const SparseCSC = Union{SparseMatrixCSC, SparseVector}

# Sparse arrays keep their stored values in a `Vector`, which cannot wrap a
# reinterpreted view of `dual_du`. `PreallocationTools._sparse_backed` extends
# this to wrappers of sparse arrays (views, reshapes, transposes).
PreallocationTools._sparse_backed(::Type{<:SparseCSC}) = true

# `similar` of these keeps the stored pattern; the type is computed without
# allocating a sparse array per lookup.
function PreallocationTools._workspace_type(::SparseMatrixCSC{<:Any, Ti}, ::Type{T}) where {T, Ti}
    return SparseMatrixCSC{T, Ti}
end
function PreallocationTools._workspace_type(::SparseVector{<:Any, Ti}, ::Type{T}) where {T, Ti}
    return SparseVector{T, Ti}
end

# Setting the stored values of the sparse root keeps the pattern and costs
# O(stored entries); `fill!` on a wrapper would visit every index, and `fill!` on
# a sparse array requires `iszero` to return a `Bool`, which sparsity tracers
# do not. For a wrapper, this fills every stored entry of the root, including
# those outside the wrapper's region, so `buf` must own its root, as the typed
# workspaces of a cache and the result of `deepcopy` do.
function PreallocationTools._fill_stored!(buf::SparseCSC, x)
    fill!(nonzeros(buf), x)
    return buf
end
function PreallocationTools._fill_stored!(buf, x)
    p = parent(buf)
    p === buf ? fill!(buf, x) : PreallocationTools._fill_stored!(p, x)
    return buf
end

# Stores every entry of `buf` with value `x`, which the caller has found to be
# nonzero. Unlike `fill!`, this does not test `convert(eltype(buf), x)` for zero,
# which fails for sparsity tracers. Wrappers fall back to `fill!`.
function PreallocationTools._fill_entries!(A::SparseMatrixCSC, x)
    xT = convert(eltype(A), x)
    m, n = size(A)
    colptr, rowval, nzval = getcolptr(A), rowvals(A), nonzeros(A)
    m * n < typemax(eltype(colptr)) ||
        throw(ArgumentError("cannot store all $m×$n entries of a sparse matrix with $(eltype(colptr)) indices"))
    resize!(rowval, m * n)
    resize!(nzval, m * n)
    k = 0
    for j in axes(A, 2)
        colptr[j] = k + 1
        for i in axes(A, 1)
            k += 1
            rowval[k] = i
        end
    end
    colptr[end] = k + 1
    fill!(nzval, xT)
    return A
end
function PreallocationTools._fill_entries!(v::SparseVector, x)
    xT = convert(eltype(v), x)
    nzind, nzval = SparseArrays.nonzeroinds(v), nonzeros(v)
    resize!(nzind, length(v))
    resize!(nzval, length(v))
    copyto!(nzind, eachindex(v))
    fill!(nzval, xT)
    return v
end
PreallocationTools._fill_entries!(buf, x) = fill!(buf, x)

# Gives `dest` the stored pattern of `src`, zeroing the stored values of `dest`
# if it changes and leaving them untouched otherwise. For wrappers, this matches
# the sparse roots, which must have equal sizes; like `_fill_stored!`, it changes
# the whole root, so `dest` must own its root.
function PreallocationTools._match_stored_pattern!(dest::SparseMatrixCSC, src::SparseMatrixCSC)
    size(dest) == size(src) ||
        throw(DimensionMismatch("cannot match the stored pattern of a $(size(src)) sparse matrix in one of size $(size(dest))"))
    _same_stored_pattern(dest, src) && return dest
    _check_unshared_indices(getcolptr(dest) === getcolptr(src) || rowvals(dest) === rowvals(src), dest, src)
    return _copy_stored_pattern!(dest, src)
end
function PreallocationTools._match_stored_pattern!(dest::SparseVector, src::SparseVector)
    length(dest) == length(src) ||
        throw(DimensionMismatch("cannot match the stored pattern of a sparse vector of length $(length(src)) in one of length $(length(dest))"))
    _same_stored_pattern(dest, src) && return dest
    _check_unshared_indices(SparseArrays.nonzeroinds(dest) === SparseArrays.nonzeroinds(src), dest, src)
    return _copy_stored_pattern!(dest, src)
end
function PreallocationTools._match_stored_pattern!(dest, src)
    p = parent(dest)
    p === dest &&
        throw(ArgumentError("matching the stored pattern of $(typeof(dest)) is not supported"))
    PreallocationTools._match_stored_pattern!(p, parent(src))
    return dest
end

# Compares only the stored indices, since the stored values of a workspace may
# be unassigned references until the caller writes them. When `dest` has the
# index arrays of `src` (as Enzyme passes user-supplied shadows to rules, and as
# `Enzyme.make_zero` creates them), only the length of its stored values shows
# whether entries were stored in `src` after `dest` was created.
function _same_stored_pattern(dest::SparseMatrixCSC, src::SparseMatrixCSC)
    n = nnz(src)
    getcolptr(dest) == getcolptr(src) || return false
    length(nonzeros(dest)) >= n && length(rowvals(dest)) >= n || return false
    return view(rowvals(dest), 1:n) == view(rowvals(src), 1:n)
end
function _same_stored_pattern(dest::SparseVector, src::SparseVector)
    return length(nonzeros(dest)) >= nnz(src) && rowvals(dest) == rowvals(src)
end

# A `dest` with the index arrays of `src` cannot get the pattern of `src`: the
# positions of its stored values in its own pattern are unknown, and resizing
# them would leave the array that `dest` came from with inconsistent buffers.
function _check_unshared_indices(shared, dest, src)
    shared || return nothing
    throw(ArgumentError(
        "the shadow of a sparse workspace has $(length(nonzeros(dest))) stored values, \
        but the workspace has $(nnz(src)) stored entries; entries were stored in \
        the workspace after the shadow was created. Create the shadow (e.g. with \
        `Enzyme.make_zero`) after the stored pattern of the workspace changes."
    ))
end

# `copyto!` between sparse arrays of equal size copies the stored pattern,
# including stored zeros, and removes entries stored only in `dest`.
function _copy_stored_pattern!(dest, src)
    copyto!(dest, src)
    fill!(nonzeros(dest), zero(eltype(dest)))
    return dest
end

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
