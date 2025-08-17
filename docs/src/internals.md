# Internal Functions

This page documents internal functions that are not part of the public API but may be encountered during debugging or when extending PreallocationTools.jl functionality.

!!! warning
    These are internal implementation details and may change without notice in any release. They should not be relied upon for user code.

## Cache Management Functions

```@docs
PreallocationTools.get_tmp(::FixedSizeDiffCache, ::Union{Number, AbstractArray})
PreallocationTools.get_tmp(::FixedSizeDiffCache, ::Type{T}) where {T <: Number}
PreallocationTools.get_tmp(::DiffCache, ::Union{Number, AbstractArray})
PreallocationTools.get_tmp(::DiffCache, ::Type{T}) where {T <: Number}
PreallocationTools.enlargediffcache!
```

## Internal Helper Functions

```@docs
PreallocationTools._restructure
```