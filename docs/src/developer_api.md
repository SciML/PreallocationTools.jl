# Developer API

The names on this page are versioned extension points for packages that
implement automatic-differentiation backends or PreallocationTools extensions.
They are not intended for ordinary user code. Use the documented cache
constructors and [`get_tmp`](@ref) unless you own such an extension.

An extension must only add methods whose dispatch includes a type it owns. It
must preserve the cache representation, axes, and scratch-storage aliasing
rules stated on each definition. The package tests these contracts using an
independent test-only AD extension rather than relying only on the bundled
ForwardDiff extension.

```@docs
PreallocationTools.dualarraycreator
PreallocationTools.forwarddiff_compat_chunk_size
PreallocationTools.chunksize
PreallocationTools._restructure
PreallocationTools.enlargediffcache!
```
