# PreallocationTools.jl

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/PreallocationTools/stable/)

[![codecov](https://codecov.io/gh/SciML/PreallocationTools.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/SciML/PreallocationTools.jl)
[![Build Status](https://github.com/SciML/PreallocationTools.jl/workflows/CI/badge.svg)](https://github.com/SciML/PreallocationTools.jl/actions?query=workflow%3ACI)
[![Build status](https://badge.buildkite.com/8e62ff2622721bf7a82aa5effb466d311d53fe63dc89bf2f34.svg)](https://buildkite.com/julialang/preallocationtools-dot-jl)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

PreallocationTools.jl is a set of tools for helping build non-allocating
pre-cached functions for high-performance computing in Julia. Its tools handle
edge cases of automatic differentiation to make it easier for users to get
high performance even in the cases where code generation may change the
function that is being called.

## DiffCache

`DiffCache` is a type for doubly-preallocated vectors which are
compatible with non-allocating forward-mode automatic differentiation by
ForwardDiff.jl. Since ForwardDiff uses chunked duals in its forward pass, two
vector sizes are required in order for the arrays to be properly defined.
`DiffCache` creates a dispatching type to solve this, so that by passing a
qualifier it can automatically switch between the required cache. This method
is fully type-stable and non-dynamic, made for when the highest performance is
needed.

### Using DiffCache

```julia
DiffCache(u::AbstractArray, N::Int = ForwardDiff.pickchunksize(length(u)); levels::Int = 1)
DiffCache(u::AbstractArray, N::AbstractArray{<:Int})
```

The `DiffCache` function builds a `DiffCache` object that stores both a version
of the cache for `u` and for the `Dual` version of `u`, allowing use of
pre-cached vectors with forward-mode automatic differentiation. Note that
`DiffCache`, due to its design, is only compatible with arrays that contain concretely
typed elements.

To access the caches, one uses:

```julia
get_tmp(tmp::DiffCache, u)
```

When `u` has an element subtype of `Dual` numbers, then it returns the `Dual`
version of the cache. Otherwise it returns the standard cache (for use in the
calls without automatic differentiation).

In order to preallocate to the right size, the `DiffCache` needs to be specified
to have the correct `N` matching the chunk size of the dual numbers or larger.
If the chunk size `N` specified is too large, `get_tmp` will automatically resize
when dispatching; this remains type-stable and non-allocating, but comes at the
expense of additional memory.

In a differential equation, optimization, etc., the default chunk size is computed
from the state vector `u`, and thus if one creates the `DiffCache` via
`DiffCache(u)` it will match the default chunking of the solver libraries.

`DiffCache` is also compatible with nested automatic differentiation calls through
the `levels` keyword (`N` for each level computed using based on the size of the
state vector) or by specifying `N` as an array of integers of chunk sizes, which
enables full control of chunk sizes on all differentation levels.

### DiffCache Example 1: Direct Usage

```julia
using ForwardDiff, PreallocationTools
randmat = rand(5, 3)
sto = similar(randmat)
stod = DiffCache(sto)

function claytonsample!(sto, τ, α; randmat = randmat)
    sto = get_tmp(sto, τ)
    sto .= randmat
    τ == 0 && return sto

    n = size(sto, 1)
    for i in 1:n
        v = sto[i, 2]
        u = sto[i, 1]
        sto[i, 1] = (1 - u^(-τ) + u^(-τ) * v^(-(τ / (1 + τ))))^(-1 / τ) * α
        sto[i, 2] = (1 - u^(-τ) + u^(-τ) * v^(-(τ / (1 + τ))))^(-1 / τ)
    end
    return sto
end

ForwardDiff.derivative(τ -> claytonsample!(stod, τ, 0.0), 0.3)
ForwardDiff.jacobian(x -> claytonsample!(stod, x[1], x[2]), [0.3; 0.0])
```

In the above, the chunk size of the dual numbers has been selected based on the size
of `randmat`, resulting in a chunk size of 8 in this case. However, since the derivative
is calculated with respect to τ and the Jacobian is calculated with respect to τ and α,
specifying the `DiffCache` with `stod = DiffCache(sto, 1)` or `stod = DiffCache(sto, 2)`,
respectively, would have been the most memory efficient way of performing these calculations
(only really relevant for much larger problems).

### DiffCache Example 2: ODEs

```julia
using LinearAlgebra, OrdinaryDiffEq
function foo(du, u, (A, tmp), t)
    mul!(tmp, A, u)
    @. du = u + tmp
    nothing
end
prob = ODEProblem(foo, ones(5, 5), (0.0, 1.0), (ones(5, 5), zeros(5, 5)))
solve(prob, TRBDF2())
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
prob = ODEProblem(foo,
    ones(5, 5),
    (0.0, 1.0),
    (ones(5, 5), DiffCache(zeros(5, 5), chunk_size)))
solve(prob, TRBDF2(chunk_size = chunk_size))
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
prob = ODEProblem(foo, ones(5, 5), (0.0, 1.0), (ones(5, 5), DiffCache(zeros(5, 5))))
solve(prob, TRBDF2())
```

### DiffCache Example 3: Nested AD calls in an optimization problem involving a Hessian matrix

```julia
using LinearAlgebra, OrdinaryDiffEq, PreallocationTools, Optimization, OptimizationOptimJL
function foo(du, u, p, t)
    tmp = p[2]
    A = reshape(p[1], size(tmp.du))
    tmp = get_tmp(tmp, u)
    mul!(tmp, A, u)
    @. du = u + tmp
    nothing
end

coeffs = -collect(0.1:0.1:0.4)
cache = DiffCache(zeros(2, 2), levels = 3)
prob = ODEProblem(foo, ones(2, 2), (0.0, 1.0), (coeffs, cache))
realsol = solve(prob, TRBDF2(), saveat = 0.0:0.1:10.0, reltol = 1e-8)

function objfun(x, prob, realsol, cache)
    prob = remake(prob, u0 = eltype(x).(prob.u0), p = (x, cache))
    sol = solve(prob, TRBDF2(), saveat = 0.0:0.1:10.0, reltol = 1e-8)

    ofv = 0.0
    if any((s.retcode != :Success for s in sol))
        ofv = 1e12
    else
        ofv = sum((sol .- realsol) .^ 2)
    end
    return ofv
end
fn(x, p) = objfun(x, p[1], p[2], p[3])
optfun = OptimizationFunction(fn, Optimization.AutoForwardDiff())
optprob = OptimizationProblem(optfun, zeros(length(coeffs)), (prob, realsol, cache))
solve(optprob, Newton())
```

Solves an optimization problem for the coefficients, `coeffs`, appearing in a differential equation.
The optimization is done with [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl)'s `Newton()`
algorithm. Since this involves automatic differentiation in the ODE solver and the calculation
of Hessians, three automatic differentiations are nested within each other. Therefore, the `DiffCache`
is specified with `levels = 3`.

## FixedSizeDiffCache

`FixedSizeDiffCache` is a lot like `DiffCache`, but it stores dual numbers in its caches
instead of a flat array. Because of this, it can avoid a view, making it a little bit
more performant for generating caches of non-`Array` types. However, it is a lot less
flexible than `DiffCache`, and is thus only recommended for cases where the chunk size
is known in advance (for example, ODE solvers) and where `u` is not an `Array`.

The interface is almost exactly the same, except with the constructor:

```julia
FixedSizeDiffCache(u::AbstractArray, chunk_size = Val{ForwardDiff.pickchunksize(length(u))})
FixedSizeDiffCache(u::AbstractArray, chunk_size::Integer)
```

Note that the `FixedSizeDiffCache` can support duals that are of a smaller chunk size than
the preallocated ones, but not a larger size. Nested duals are not supported with this
construct.

## LazyBufferCache

```julia
LazyBufferCache(f::F = identity)
```

A `LazyBufferCache` is a `Dict`-like type for the caches which automatically defines
new cache arrays on demand when they are required. The function `f` maps
`size_of_cache = f(size(u))`, which by default creates cache arrays of the same size.

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
prob = ODEProblem(foo, ones(5, 5), (0.0, 1.0), (ones(5, 5), LazyBufferCache()))
solve(prob, TRBDF2())
```

## GeneralLazyBufferCache

```julia
GeneralLazyBufferCache(f = identity)
```

A `GeneralLazyBufferCache` is a `Dict`-like type for the caches which automatically defines
new caches on demand when they are required. The function `f` generates the cache matching
for the type of `u`, and subsequent indexing reuses that cache if that type of `u` has
already ben seen.

Note that `LazyBufferCache` does cause a dynamic dispatch and its return is not type-inferred.
This means it's the slowest of the preallocation methods, but it's the most general.

### Example

In all of the previous cases our cache was an array. However, in this case we want to preallocate
a DifferentialEquations `ODEIntegrator` object. This object is the one created via
`DifferentialEquations.init(ODEProblem(ode_fnc, y₀, (0.0, T), p), Tsit5(); saveat = t)`, and we
want to optimize `p` in a way that changes its type to ForwardDiff. Thus what we can do is make a
GeneralLazyBufferCache which holds these integrator objects, defined by `p`, and indexing it with
`p` in order to retrieve the cache. The first time it's called it will build the integrator, and
in subsequent calls it will reuse the cache.

Defining the cache as a function of `p` to build an integrator thus looks like:

```julia
lbc = GeneralLazyBufferCache(function (p)
    DifferentialEquations.init(ODEProblem(ode_fnc, y₀, (0.0, T), p), Tsit5(); saveat = t)
end)
```

then `lbc[p]` will be smart and reuse the caches. A full example looks like the following:

```julia
using Random, DifferentialEquations, LinearAlgebra, Optimization, OptimizationNLopt,
    OptimizationOptimJL, PreallocationTools

lbc = GeneralLazyBufferCache(function (p)
    DifferentialEquations.init(ODEProblem(ode_fnc, y₀, (0.0, T), p), Tsit5(); saveat = t)
end)

Random.seed!(2992999)
λ, y₀, σ = -0.5, 15.0, 0.1
T, n = 5.0, 200
Δt = T / n
t = [j * Δt for j in 0:n]
y = y₀ * exp.(λ * t)
yᵒ = y .+ [0.0, σ * randn(n)...]
ode_fnc(u, p, t) = p * u
function loglik(θ, data, integrator)
    yᵒ, n, ε = data
    λ, σ, u0 = θ
    integrator.p = λ
    reinit!(integrator, u0)
    solve!(integrator)
    ε = yᵒ .- integrator.sol.u
    ℓ = -0.5n * log(2π * σ^2) - 0.5 / σ^2 * sum(ε .^ 2)
end
θ₀ = [-1.0, 0.5, 19.73]
negloglik = (θ, p) -> -loglik(θ, p, lbc[θ[1]])
fnc = OptimizationFunction(negloglik, Optimization.AutoForwardDiff())
ε = zeros(n)
prob = OptimizationProblem(fnc,
    θ₀,
    (yᵒ, n, ε),
    lb = [-10.0, 1e-6, 0.5],
    ub = [10.0, 10.0, 25.0])
solve(prob, LBFGS())
```

## Similar Projects

[AutoPreallocation.jl](https://github.com/oxinabox/AutoPreallocation.jl) tries
to do this automatically at the compiler level. [Alloc.jl](https://github.com/FluxML/Alloc.jl)
tries to do this with a bump allocator.
