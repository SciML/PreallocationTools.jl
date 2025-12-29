module PreallocationToolsForwardDiffExt

using PreallocationTools
using ForwardDiff
using ArrayInterface
using Adapt
using PrecompileTools

function PreallocationTools.dualarraycreator(u::AbstractArray{T}, siz,
            ::Type{Val{chunk_size}}) where {T, chunk_size}
    # For complex arrays, create duals with the real part type
    # and allocate twice the space for real and imaginary parts
    if T <: Complex
        RealT = real(T)
        # Create a flat array of duals with real element type
        # with enough space for 2 * prod(siz) elements (real and imaginary parts)
        dual_arr = Adapt.adapt(ArrayInterface.parameterless_type(u),
            zeros(ForwardDiff.Dual{Nothing, RealT, chunk_size}, 2 * prod(siz)))
        dual_arr
    else
        ArrayInterface.restructure(u,
            zeros(ForwardDiff.Dual{Nothing, T, chunk_size},
                siz...))
    end
end

PreallocationTools.forwarddiff_compat_chunk_size(x::Int) = ForwardDiff.pickchunksize(x)

# Define chunksize for ForwardDiff.Dual types
PreallocationTools.chunksize(::Type{ForwardDiff.Dual{T, V, N}}) where {T, V, N} = N

# Helper function to compute the target element type for get_tmp
# When the cache stores Complex{V} and input dual has real value type V,
# return Complex{Dual{Tag, V, N}} so that ForwardDiff operations work correctly
function _target_eltype(::Type{ForwardDiff.Dual{Tag, V, N}}, ::Type{CacheT}) where {Tag, V, N, CacheT}
    if CacheT <: Complex && !(V <: Complex)
        RealT = real(CacheT)
        # Check if input dual's value type is compatible with the cache's real type
        if promote_type(V, RealT) <: RealT
            # Return Complex{Dual} since ForwardDiff operations on complex produce Complex{Dual}
            return Complex{ForwardDiff.Dual{Tag, RealT, N}}
        end
    end
    # For real caches or already-complex duals, use the input dual type as-is
    return ForwardDiff.Dual{Tag, V, N}
end

# Check if cache stores complex numbers
_is_complex_cache(::Type{<:Complex}) = true
_is_complex_cache(::Type) = false

# Helper for FixedSizeDiffCache with complex arrays
function _get_complex_dual_array_fixed(dc::PreallocationTools.FixedSizeDiffCache, ::Type{DualT}, ::Type{ComplexDualT}) where {DualT, ComplexDualT}
    # For complex caches, dual_du is a flat array of real duals
    # We need to reinterpret as the target dual type and reshape
    x = reinterpret(DualT, dc.dual_du)
    # Take a view of the appropriate number of elements (2 * length(du) for real+imag parts)
    n_complex = length(dc.du)
    x_view = @view x[1:(2 * n_complex)]
    # Reshape to (2, size(dc.du)...) for complex reinterpret
    reshaped = reshape(x_view, 2, size(dc.du)...)
    reinterpret(reshape, ComplexDualT, reshaped)
end

# Define get_tmp methods for ForwardDiff.Dual types
function PreallocationTools.get_tmp(dc::PreallocationTools.FixedSizeDiffCache, u::T) where {T <: ForwardDiff.Dual}
    TargetElType = _target_eltype(T, eltype(dc.du))
    if TargetElType <: Complex
        DualT = TargetElType.parameters[1]
        _get_complex_dual_array_fixed(dc, DualT, TargetElType)
    else
        x = reinterpret(TargetElType, dc.dual_du)
        if PreallocationTools.chunksize(TargetElType) === PreallocationTools.chunksize(eltype(dc.dual_du))
            x
        else
            @view x[axes(dc.du)...]
        end
    end
end

function PreallocationTools.get_tmp(dc::PreallocationTools.FixedSizeDiffCache, u::Type{T}) where {T <: ForwardDiff.Dual}
    TargetElType = _target_eltype(T, eltype(dc.du))
    if TargetElType <: Complex
        DualT = TargetElType.parameters[1]
        _get_complex_dual_array_fixed(dc, DualT, TargetElType)
    else
        x = reinterpret(TargetElType, dc.dual_du)
        if PreallocationTools.chunksize(TargetElType) === PreallocationTools.chunksize(eltype(dc.dual_du))
            x
        else
            @view x[axes(dc.du)...]
        end
    end
end

function PreallocationTools.get_tmp(dc::PreallocationTools.FixedSizeDiffCache, u::AbstractArray{T}) where {T <: ForwardDiff.Dual}
    TargetElType = _target_eltype(T, eltype(dc.du))
    if TargetElType <: Complex
        DualT = TargetElType.parameters[1]
        _get_complex_dual_array_fixed(dc, DualT, TargetElType)
    else
        x = reinterpret(TargetElType, dc.dual_du)
        if PreallocationTools.chunksize(TargetElType) === PreallocationTools.chunksize(eltype(dc.dual_du))
            x
        else
            @view x[axes(dc.du)...]
        end
    end
end

# Helper to create a complex dual array from the dual buffer
function _get_complex_dual_array(dc::PreallocationTools.DiffCache, ::Type{DualT}, ::Type{ComplexDualT}) where {DualT, ComplexDualT}
    if isbitstype(DualT)
        # For complex, we need twice as many dual elements (real and imaginary parts)
        nelem = div(sizeof(DualT), sizeof(eltype(dc.dual_du))) * 2 * length(dc.du)
        if nelem > length(dc.dual_du)
            PreallocationTools.enlargediffcache!(dc, nelem)
        end
        # Reinterpret as duals first
        dual_view = reinterpret(DualT, view(dc.dual_du, 1:nelem))
        # Reshape to (2, size(dc.du)...) to prepare for complex reinterpret
        reshaped = reshape(dual_view, 2, size(dc.du)...)
        # Reinterpret to complex, which removes the first dimension
        reinterpret(reshape, ComplexDualT, reshaped)
    else
        zeros(ComplexDualT, size(dc.du))
    end
end

function PreallocationTools.get_tmp(dc::PreallocationTools.DiffCache, u::T) where {T <: ForwardDiff.Dual}
    TargetElType = _target_eltype(T, eltype(dc.du))
    if TargetElType <: Complex
        DualT = TargetElType.parameters[1]
        _get_complex_dual_array(dc, DualT, TargetElType)
    else
        if isbitstype(TargetElType)
            nelem = div(sizeof(TargetElType), sizeof(eltype(dc.dual_du))) * length(dc.du)
            if nelem > length(dc.dual_du)
                PreallocationTools.enlargediffcache!(dc, nelem)
            end
            PreallocationTools._restructure(dc.du, reinterpret(TargetElType, view(dc.dual_du, 1:nelem)))
        else
            PreallocationTools._restructure(dc.du, zeros(TargetElType, size(dc.du)))
        end
    end
end

function PreallocationTools.get_tmp(dc::PreallocationTools.DiffCache, ::Type{T}) where {T <: ForwardDiff.Dual}
    TargetElType = _target_eltype(T, eltype(dc.du))
    if TargetElType <: Complex
        DualT = TargetElType.parameters[1]
        _get_complex_dual_array(dc, DualT, TargetElType)
    else
        if isbitstype(TargetElType)
            nelem = div(sizeof(TargetElType), sizeof(eltype(dc.dual_du))) * length(dc.du)
            if nelem > length(dc.dual_du)
                PreallocationTools.enlargediffcache!(dc, nelem)
            end
            PreallocationTools._restructure(dc.du, reinterpret(TargetElType, view(dc.dual_du, 1:nelem)))
        else
            PreallocationTools._restructure(dc.du, zeros(TargetElType, size(dc.du)))
        end
    end
end

function PreallocationTools.get_tmp(dc::PreallocationTools.DiffCache, u::AbstractArray{T}) where {T <: ForwardDiff.Dual}
    TargetElType = _target_eltype(T, eltype(dc.du))
    if TargetElType <: Complex
        DualT = TargetElType.parameters[1]
        _get_complex_dual_array(dc, DualT, TargetElType)
    else
        if isbitstype(TargetElType)
            nelem = div(sizeof(TargetElType), sizeof(eltype(dc.dual_du))) * length(dc.du)
            if nelem > length(dc.dual_du)
                PreallocationTools.enlargediffcache!(dc, nelem)
            end
            PreallocationTools._restructure(dc.du, reinterpret(TargetElType, view(dc.dual_du, 1:nelem)))
        else
            PreallocationTools._restructure(dc.du, zeros(TargetElType, size(dc.du)))
        end
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
