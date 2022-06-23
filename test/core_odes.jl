using LinearAlgebra, OrdinaryDiffEq, Test, PreallocationTools, LabelledArrays, RecursiveArrayTools

#Base array
function foo(du, u, (A, tmp), t)
    tmp = get_tmp(tmp, u)
    mul!(tmp, A, u)
    @. du = u + tmp
    return nothing
end
#with defined chunk_size
chunk_size = 5
u0 = ones(5, 5)
A = ones(5, 5)
cache = dualcache(zeros(5, 5), chunk_size)
prob = ODEProblem(foo, u0, (0.0, 1.0), (A, cache))
sol = solve(prob, TRBDF2(; chunk_size = chunk_size))
@test sol.retcode == :Success

#with auto-detected chunk_size
prob = ODEProblem(foo, ones(5, 5), (0.0, 1.0), (ones(5, 5), dualcache(zeros(5, 5))))
sol = solve(prob, TRBDF2())
@test sol.retcode == :Success

#Base array with LBC
function foo(du, u, (A, lbc), t)
    tmp = lbc[u]
    mul!(tmp, A, u)
    @. du = u + tmp
    return nothing
end
prob = ODEProblem(foo, ones(5, 5), (0.0, 1.0), (ones(5, 5), LazyBufferCache()))
sol = solve(prob, TRBDF2())
@test sol.retcode == :Success

#LArray
A = LArray((2, 2); a = 1.0, b = 1.0, c = 1.0, d = 1.0)
c = LArray((2, 2); a = 0.0, b = 0.0, c = 0.0, d = 0.0)
u0 = LArray((2, 2); a = 1.0, b = 1.0, c = 1.0, d = 1.0)
function foo(du, u, (A, tmp), t)
    tmp = get_tmp(tmp, u)
    mul!(tmp, A, u)
    @. du = u + tmp
    return nothing
end
#with specified chunk_size
chunk_size = 4
prob = ODEProblem(foo, u0, (0.0, 1.0), (A, dualcache(c, chunk_size)))
sol = solve(prob, TRBDF2(; chunk_size = chunk_size))
@test sol.retcode == :Success
#with auto-detected chunk_size
prob = ODEProblem(foo, u0, (0.0, 1.0), (A, dualcache(c)))
sol = solve(prob, TRBDF2())
@test sol.retcode == :Success
