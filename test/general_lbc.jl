using Random,
    OrdinaryDiffEq, LinearAlgebra, Optimization, OptimizationOptimJL,
    PreallocationTools

lbc = GeneralLazyBufferCache(function (p)
    init(ODEProblem(ode_fnc, y₀,
            (0.0, T), p),
        Tsit5(); saveat = t)
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
prob = OptimizationProblem(fnc, θ₀, (yᵒ, n, ε), lb = [-10.0, 1e-6, 0.5],
    ub = [10.0, 10.0, 25.0])
solve(prob, LBFGS())

cache = LazyBufferCache()
x = rand(1000)
@inferred cache[x]
@test 0 == @allocated cache[x]

cache = GeneralLazyBufferCache(T -> Vector{T}(undef, 1000))
# GeneralLazyBufferCache is documented not to infer.
# @inferred cache[Float64]
cache[Float64] # generate the buffer
@test 0 == @allocated cache[Float64]
