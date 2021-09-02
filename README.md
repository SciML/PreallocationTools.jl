# PreallocationTools.jl

[![Join the chat at https://gitter.im/JuliaDiffEq/Lobby](https://badges.gitter.im/JuliaDiffEq/Lobby.svg)](https://gitter.im/JuliaDiffEq/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Build Status](https://github.com/SciML/PreallocationTools.jl/workflows/CI/badge.svg)](https://github.com/SciML/PreallocationTools.jl/actions?query=workflow%3ACI)

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

### Example

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

### Handling Chunk Sizes

It is important that the `DiffCache` matches the chunk sizes used in the actual differentiation. Let's
understand this by looking at a real problem:

```julia
using ForwardDiff
using PreallocationTools

randmat = rand(10, 2)
sto = similar(randmat)
stod = dualcache(sto)

function claytonsample!(sto, τ; randmat=randmat)
    sto = get_tmp(sto, τ)
    @show typeof(τ)
    @show size(sto), size(randmat)
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

Notice that by default the `dualcache` builds a cache that is compatible with differentiation w.r.t the
a variable sized as the cache variable, i.e. it's naturally made for Jacobians.

```julia
julia> typeof(dualcache(sto))
PreallocationTools.DiffCache{Matrix{Float64}, Matrix{ForwardDiff.Dual{nothing, Float64, 10}}}
```

Notice that it choose a chunk size of 10, matching the chunk size that would be used if `ForwardDiff.jacobian` is used
to calculate the derivative w.r.t. an input matching the size of `sto` (i.e., the Jacobian of `claytonsample!` w.r.t. `randmat`).
However, in our actual differentiation we have:

```julia
typeof(τ) = ForwardDiff.Dual{ForwardDiff.Tag{var"#49#50", Float64}, Float64, 1}
```

a single chunk size because it's differentiating w.r.t. a single dimension. This messes with the sizes in the reinterpretation:

```julia
(size(sto), size(randmat)) = ((55, 2), (10, 2))
```

The fix is to ensure that the cache is generated with the correct chunk size, i.e.:

```julia
stod = dualcache(sto,Val{1})
```

Thus the following code successfully computes the derivative in a non-allocating way:

```julia
using ForwardDiff
using PreallocationTools

randmat = rand(10, 2)
sto = similar(randmat)
stod = dualcache(sto,Val{1})

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

10×2 Matrix{Float64}:
 0.0   0.171602
 0.0  -0.412736
 0.0   0.149273
 0.0   0.18172
 0.0   0.144151
 0.0  -0.110773
 0.0   0.221714
 0.0  -0.111034
 0.0  -0.0723283
 0.0   0.251095
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
