module PreallocationToolsReverseDiffExt

using PreallocationTools
isdefined(Base, :get_extension) ? (import ReverseDiff) : (import ..ReverseDiff)

# PreallocationTools https://github.com/SciML/PreallocationTools.jl/issues/39
function Base.getindex(b::PreallocationTools.LazyBufferCache, u::ReverseDiff.TrackedArray)
    s = b.sizemap(size(u)) # required buffer size
    T = ReverseDiff.TrackedArray
    buf = get!(b.bufs, (T, s)) do
        # declare type since b.bufs dictionary is untyped
        similar(u, s)
    end
    return buf
end

end
