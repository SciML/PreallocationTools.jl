# API

```@autodocs
Modules = [PreallocationTools]
Filter = t -> !(t in (
    PreallocationTools.dualarraycreator,
    PreallocationTools.forwarddiff_compat_chunk_size,
    PreallocationTools.chunksize,
    PreallocationTools._restructure,
    PreallocationTools.enlargediffcache!,
))
```
