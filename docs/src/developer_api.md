# Developer API

The names on this page are versioned extension points for packages that
implement automatic-differentiation backends or PreallocationTools extensions.
They are not intended for ordinary user code. Use the documented cache
constructors and [`get_tmp`](@ref) unless you own such an extension.

```@docs
PreallocationTools.dualarraycreator
PreallocationTools.forwarddiff_compat_chunk_size
PreallocationTools.chunksize
PreallocationTools._restructure
PreallocationTools.enlargediffcache!
```
