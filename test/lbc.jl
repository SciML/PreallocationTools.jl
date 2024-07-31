using PreallocationTools: LazyBufferCache
using Test

b = LazyBufferCache(Returns(10); initializer! = buf -> fill!(buf, 0))

@test b[Float64[]] == zeros(10)
