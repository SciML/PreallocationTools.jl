using LinearAlgebra, OrdinaryDiffEq, Test, PreallocationTools, ForwardDiff, Optimization,
      OptimizationOptimJL

randmat = rand(5, 3)
sto = similar(randmat)
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

#= taking the second derivative of claytonsample! with respect to τ with manual chunk_sizes.
In setting up the dualcache, we are setting chunk_size to [1, 1], because we differentiate 
only with respect to τ. This initializes the cache with the minimum memory needed. =#
stod = dualcache(sto, [1, 1])
df3 = ForwardDiff.derivative(τ -> ForwardDiff.derivative(ξ -> claytonsample!(stod, ξ, 0.0),
                                                         τ), 0.3)

#= taking the second derivative of claytonsample! with respect to τ with auto-detected chunk-size. 
For the given size of sto, ForwardDiff's heuristic chooses chunk_size = 8. Since this is greater 
than what's needed (1+1), the auto-allocated cache is big enough to handle the nested dual numbers, even
if we don't specify the keyword argument levels = 2. This should in general not be relied on to work, 
especially if more levels of nesting occur (see optimization example below). =#
stod = dualcache(sto)
df4 = ForwardDiff.derivative(τ -> ForwardDiff.derivative(ξ -> claytonsample!(stod, ξ, 0.0),
                                                         τ), 0.3)

@test df3 ≈ df4

#= taking the second derivative of claytonsample! with respect to τ with auto-detected chunk-size. 
For the given size of sto, ForwardDiff's heuristic chooses chunk_size = 8 and with keyword arg levels = 2,
the created cache size is larger than what's needed (even more so than the last example). =#
stod = dualcache(sto, levels = 2)
df5 = ForwardDiff.derivative(τ -> ForwardDiff.derivative(ξ -> claytonsample!(stod, ξ, 0.0),
                                                         τ), 0.3)

@test df3 ≈ df5

#= Checking nested dual numbers using optimization problem involving Optim.jl's Newton() (involving Hessians);
so, we will need one level of AD for the ODE solver (TRBDF2) and two more to calculate the Hessian =#
function foo(du, u, p, t)
    tmp = p[2]
    A = reshape(p[1], size(tmp.du))
    tmp = get_tmp(tmp, u)
    mul!(tmp, A, u)
    @. du = u + tmp
    nothing
end

ps = 2 #use to specify problem size; don't go crazy on this, because of the compilation time...
coeffs = -collect(0.1:0.1:(ps^2 / 10))
cache = dualcache(zeros(ps, ps), levels = 3)
prob = ODEProblem{true, SciMLBase.FullSpecialize}(foo, ones(ps, ps), (0.0, 1.0),
                                                  (coeffs, cache))
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
newtonsol = solve(optprob, Newton())

@test all(abs.(coeffs .- newtonsol.u) .< 1e-3)

#an example where chunk_sizes are not the same on all differentiation levels:
cache = dualcache(zeros(ps, ps), [4, 4, 2])
prob = ODEProblem{true, SciMLBase.FullSpecialize}(foo, ones(ps, ps), (0.0, 1.0),
                                                  (coeffs, cache))
realsol = solve(prob, TRBDF2(chunk_size = 2), saveat = 0.0:0.1:10.0, reltol = 1e-8)

function objfun(x, prob, realsol, cache)
    prob = remake(prob, u0 = eltype(x).(prob.u0), p = (x, cache))
    sol = solve(prob, TRBDF2(chunk_size = 2), saveat = 0.0:0.1:10.0, reltol = 1e-8)

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
newtonsol2 = solve(optprob, Newton())

@test all(abs.(coeffs .- newtonsol2.u) .< 1e-3)
