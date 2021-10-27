using LinearAlgebra, OrdinaryDiffEq, Test, PreallocationTools, ForwardDiff, LabelledArrays, GalacticOptim, Optim

## Check ODE problem with specified chunk_size
function foo(du, u, (A, tmp), t)
    tmp = get_tmp(tmp, u)
    mul!(tmp, A, u)
    @. du = u + tmp
    nothing
end
chunk_size = 5
prob = ODEProblem(foo, ones(5, 5), (0., 1.0), (ones(5,5), dualcache(zeros(5,5), chunk_size)))
solve(prob, TRBDF2(chunk_size=chunk_size))

## Check ODE problem with auto-detected chunk_size
function foo(du, u, (A, tmp), t)
    tmp = get_tmp(tmp, u)
    mul!(tmp, A, u)
    @. du = u + tmp
    nothing
end
prob = ODEProblem(foo, ones(5, 5), (0., 1.0), (ones(5,5), dualcache(zeros(5,5))))
solve(prob, TRBDF2())

## Check ODE problem with a lazy buffer cache
function foo(du, u, (A, lbc), t)
    tmp = lbc[u]
    mul!(tmp, A, u)
    @. du = u + tmp
    nothing
end
prob = ODEProblem(foo, ones(5, 5), (0., 1.0), (ones(5,5), LazyBufferCache()))
solve(prob, TRBDF2())

## Check ODE problem with auto-detected chunk_size and LArray 
A = LArray((2,2); a=1.0, b=1.0, c=1.0, d=1.0)
u0 = LArray((2,2); a=1.0, b=1.0, c=1.0, d=1.0)
function foo(du, u, (A, tmp), t)
    tmp = get_tmp(tmp, u)
    mul!(tmp, A, u)
    @. du = u + tmp
    nothing
end
prob = ODEProblem(foo, u0, (0., 1.0), (A, dualcache(A)))
solve(prob, TRBDF2())

## Check resizing
randmat = rand(5, 3)
sto = similar(randmat)
stod = dualcache(sto)

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

#taking the derivative of claytonsample! with respect to τ only
df1 = ForwardDiff.derivative(τ -> claytonsample!(stod, τ, 0.0), 0.3)

#calculating the jacobian of claytonsample! with respect to τ and α
df2 = ForwardDiff.jacobian(x -> claytonsample!(stod, x[1], x[2]), [0.3; 0.0]) #should give a 15x2 array,
#because ForwardDiff flattens the output of jacobian, see: https://juliadiff.org/ForwardDiff.jl/stable/user/api/#ForwardDiff.jacobian

@test df1[1:5,2] ≈ df2[6:10,1]


## Checking nested dual numbers: second derivatives

#= taking the second derivative of claytonsample! with respect to τ with manual chunk_sizes. In setting up the dualcache, 
we are setting chunk_size to [1, 1], because we differentiate only twice with respect to τ.
This initializes the cache with the minimum memory needed. =#
stod = dualcache(sto, [1, 1]) 
df3 = ForwardDiff.derivative(τ -> ForwardDiff.derivative(ξ -> claytonsample!(stod, ξ, 0.0), τ), 0.3)

#= taking the second derivative of claytonsample! with respect to τ, auto-detect. For the given size of sto, ForwardDiff's heuristic
chooses chunk_size = 8. Since this is greater than (1+1)^2 = 4, the auto-allocated cache is big enough to handle the nested
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
