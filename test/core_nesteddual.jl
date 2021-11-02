using LinearAlgebra, OrdinaryDiffEq, Test, PreallocationTools, ForwardDiff, GalacticOptim, Optim

randmat = rand(5, 3)
sto = similar(randmat)
function claytonsample!(sto, τ, α; randmat=randmat)
    sto = get_tmp(sto, τ)
    sto .= randmat
    τ == 0 && return sto

    n = size(sto, 1)
    for i in 1:n
        v = sto[i, 2]
        u = sto[i, 1]
        sto[i, 1] = (1 - u^(-τ) + u^(-τ)*v^(-(τ/(1 + τ))))^(-1/τ)*α
        sto[i, 2] = (1 - u^(-τ) + u^(-τ)*v^(-(τ/(1 + τ))))^(-1/τ)
    end
    return sto
end

#= taking the second derivative of claytonsample! with respect to τ with manual chunk_sizes. In setting up the dualcache, 
we are setting chunk_size to [1, 1], because we differentiate only with respect to τ.
This initializes the cache with the minimum memory needed. =#
stod = dualcache(sto, [1, 1]) 
df3 = ForwardDiff.derivative(τ -> ForwardDiff.derivative(ξ -> claytonsample!(stod, ξ, 0.0), τ), 0.3)

#= taking the second derivative of claytonsample! with respect to τ, auto-detect. For the given size of sto, ForwardDiff's heuristic
chooses chunk_size = 8. Since this is greater than what's needed (1+1), the auto-allocated cache is big enough to handle the nested
dual numbers. This should in general not be relied on to work, especially if more levels of nesting occurs (as below). =#
stod = dualcache(sto) 
df4 = ForwardDiff.derivative(τ -> ForwardDiff.derivative(ξ -> claytonsample!(stod, ξ, 0.0), τ), 0.3)

@test df3 ≈ df4

## Checking nested dual numbers: Checking an optimization problem inspired by the above tests 
## (using Optim.jl's Newton() (involving Hessians) and BFGS() (involving gradients))
function foo(du, u, p, t)
    tmp = p[2]
    A = reshape(p[1], size(tmp.du))
    tmp = get_tmp(tmp, u)
    mul!(tmp, A, u)
    @. du = u + tmp
    nothing
end

ps = 2 #use to specify problem size (ps ∈ {1,2})
coeffs = rand(ps^2)
cache = dualcache(zeros(ps,ps), [4, 4, 4])
prob = ODEProblem(foo, ones(ps, ps), (0., 1.0), (coeffs, cache))
realsol = solve(prob, TRBDF2(), saveat = 0.0:0.01:1.0, reltol = 1e-8)

function objfun(x, prob, realsol, cache)
    prob = remake(prob, u0 = eltype(x).(ones(ps, ps)), p = (x, cache))
    sol = solve(prob, TRBDF2(), saveat = 0.0:0.01:1.0, reltol = 1e-8)

    ofv = 0.0
    if any((s.retcode != :Success for s in sol))
      ofv = 1e12
    else
      ofv = sum((sol.-realsol).^2)
    end    
    return ofv
end

fn(x,p) = objfun(x, p[1], p[2], p[3])

optfun = OptimizationFunction(fn, GalacticOptim.AutoForwardDiff())
optprob = OptimizationProblem(optfun, rand(size(coeffs)...), (prob, realsol, cache))
newtonsol = solve(optprob, Newton())
bfgssol = solve(optprob, BFGS()) #since only gradients are used here, we could go with a slim dualcache(zeros(ps,ps), [4,4]) as well.

@test all(abs.(coeffs .- newtonsol.u) .< 1e-3)
@test all(abs.(coeffs .- bfgssol.u) .< 1e-3)

#an example where chunk_sizes are not the same on all differentiation levels:
function foo(du, u, p, t)
    tmp = p[2]
    A = ones(size(tmp.du)).*p[1]
    tmp = get_tmp(tmp, u)
    mul!(tmp, A, u)
    @. du = u + tmp
    nothing
end

ps = 2 #use to specify problem size (ps ∈ {1,2})
coeffs = rand(1)
cache = dualcache(zeros(ps,ps), [1, 1, 4])
prob = ODEProblem(foo, ones(ps, ps), (0., 1.0), (coeffs, cache))
realsol = solve(prob, TRBDF2(), saveat = 0.0:0.01:1.0, reltol = 1e-8)

function objfun(x, prob, realsol, cache)
    prob = remake(prob, u0 = eltype(x).(ones(ps, ps)), p = (x, cache))
    sol = solve(prob, TRBDF2(), saveat = 0.0:0.01:1.0, reltol = 1e-8)

    ofv = 0.0
    if any((s.retcode != :Success for s in sol))
      ofv = 1e12
    else
      ofv = sum((sol.-realsol).^2)
    end    
    return ofv
end

fn(x,p) = objfun(x, p[1], p[2], p[3])

optfun = OptimizationFunction(fn, GalacticOptim.AutoForwardDiff())
optprob = OptimizationProblem(optfun, rand(size(coeffs)...), (prob, realsol, cache))
newtonsol2 = solve(optprob, Newton())

@test all(abs.(coeffs .- newtonsol2.u) .< 1e-3)