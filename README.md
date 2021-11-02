# PreallocationTools.jl

[![Join the chat at https://gitter.im/JuliaDiffEq/Lobby](https://badges.gitter.im/JuliaDiffEq/Lobby.svg)](https://gitter.im/JuliaDiffEq/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Build Status](https://github.com/SciML/PreallocationTools.jl/workflows/CI/badge.svg)](https://github.com/SciML/PreallocationTools.jl/actions?query=workflow%3ACI)
[![Build status](https://badge.buildkite.com/8e62ff2622721bf7a82aa5effb466d311d53fe63dc89bf2f34.svg)](https://buildkite.com/julialang/preallocationtools-dot-jl)

PreallocationTools.jl is a set of tools for helping build non-allocating
pre-cached functions for high-performance computing in Julia. Its tools handle
edge cases of automatic differentiation to make it easier for users to get
high performance even in the cases where code generation may change the
function that is being called.

## dualcache

`dualcache` is a method for generating doubly-preallocated vectors which are
compatible with non-allocating forward-mode automatic differentiation by
ForwardDiff.jl. Since ForwardDiff uses chunked duals in its forward pass, two
vector sizes are required in order for the arrays to be properly defined.
`dualcache` creates a dispatching type to solve this, so that by passing a
qualifier it can automatically switch between the required cache. This method
is fully type-stable and non-dynamic, made for when the highest performance is
needed.

### Using dualcache

```julia
dualcache(u::AbstractArray, N = Val{default_cache_size(length(u))})
```

The `dualcache` function builds a `DualCache` object that stores both a version
of the cache for `u` and for the `Dual` version of `u`, allowing use of
pre-cached vectors with forward-mode automatic differentiation. To access the
caches, one uses:

```julia
get_tmp(tmp::DualCache, u)
```

When `u` has an element subtype of `Dual` numbers, then it returns the `Dual`
version of the cache. Otherwise it returns the standard cache (for use in the
calls without automatic differentiation).

In order to preallocate to the right size, the `dualcache` needs to be specified
to have the corrent `N` matching the chunk size of the dual numbers or larger.
In a differential equation, optimization, etc., the default chunk size is computed
from the state vector `u`, and thus if one creates the `dualcache` via
`dualcache(u)` it will match the default chunking of the solver libraries.

### dualcache Example 1: Direct Usage

```julia
randmat = rand(10, 2)
sto = similar(randmat)
stod = dualcache(sto)

function claytonsample!(sto, τ; randmat=randmat)
    sto = get_tmp(sto, τ)
    sto .= randmat
    τ == 0 && return sto

    n = size(sto, 1)
    for i in 1:n
        v = sto[i, 2]
        u = sto[i, 1]
        sto[i, 2] = (1 - u^(-τ) + u^(-τ)*v^(-(τ/(1 + τ))))^(-1/τ)
    end
    return sto
end

ForwardDiff.derivative(τ -> claytonsample!(stod, τ), 0.3)
```

### dualcache Example 2: ODEs

```julia
using LinearAlgebra, OrdinaryDiffEq
function foo(du, u, (A, tmp), t)
    mul!(tmp, A, u)
    @. du = u + tmp
    nothing
end
prob = ODEProblem(foo, ones(5, 5), (0., 1.0), (ones(5,5), zeros(5,5)))
solve(prob, Rosenbrock23())
```

fails because `tmp` is only real numbers, but during automatic differentiation
we need `tmp` to be a cache of dual numbers. Since `u` is the value that will
have the dual numbers, we dispatch based on that:

```julia
using LinearAlgebra, OrdinaryDiffEq, PreallocationTools
function foo(du, u, (A, tmp), t)
    tmp = get_tmp(tmp, u)
    mul!(tmp, A, u)
    @. du = u + tmp
    nothing
end
chunk_size = 5
prob = ODEProblem(foo, ones(5, 5), (0., 1.0), (ones(5,5), dualcache(zeros(5,5), Val{chunk_size})))
solve(prob, TRBDF2(chunk_size=chunk_size))
```

or just using the default chunking:

```julia
using LinearAlgebra, OrdinaryDiffEq, PreallocationTools
function foo(du, u, (A, tmp), t)
    tmp = get_tmp(tmp, u)
    mul!(tmp, A, u)
    @. du = u + tmp
    nothing
end
chunk_size = 5
prob = ODEProblem(foo, ones(5, 5), (0., 1.0), (ones(5,5), dualcache(zeros(5,5))))
solve(prob, TRBDF2())
```

## LazyBufferCache

```julia
LazyBufferCache(f::F=identity)
```

A `LazyBufferCache` is a `Dict`-like type for the caches which automatically defines
new cache vectors on demand when they are required. The function `f` is a length
map which maps `length_of_cache = f(length(u))`, which by default creates cache
vectors of the same length.

Note that `LazyBufferCache` does cause a dynamic dispatch, though it is type-stable.
This gives it a ~100ns overhead, and thus on very small problems it can reduce
performance, but for any sufficiently sized calculation (e.g. >20 ODEs) this
may not be even measurable. The upside of `LazyBufferCache` is that the user does
not have to worry about potential issues with chunk sizes and such: `LazyBufferCache`
is much easier!

### Example

```julia
using LinearAlgebra, OrdinaryDiffEq, PreallocationTools
function foo(du, u, (A, lbc), t)
    tmp = lbc[u]
    mul!(tmp, A, u)
    @. du = u + tmp
    nothing
end
prob = ODEProblem(foo, ones(5, 5), (0., 1.0), (ones(5,5), LazyBufferCache()))
solve(prob, TRBDF2())
```

## Similar Projects

[AutoPreallocation.jl](https://github.com/oxinabox/AutoPreallocation.jl) tries
to do this automatically at the compiler level. [Alloc.jl](https://github.com/FluxML/Alloc.jl)
tries to do this with a bump allocator.
