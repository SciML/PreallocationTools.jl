using LinearAlgebra, OrdinaryDiffEq, Test, PreallocationTools, LabelledArrays,
      RecursiveArrayTools

# upstream
OrdinaryDiffEq.DiffEqBase.anyeltypedual(x::FixedSizeDiffCache, counter = 0) = Any

#Base array
function foo(du, u, (A, tmp), t)
    tmp = get_tmp(tmp, u)
    mul!(tmp, A, u)
    @. du = u + tmp
    nothing
end
#with defined chunk_size
chunk_size = 5
u0 = ones(5, 5)
A = ones(5, 5)
cache = FixedSizeDiffCache(zeros(5, 5), chunk_size)
prob = ODEProblem{true, SciMLBase.FullSpecialize}(foo, u0, (0.0, 1.0), (A, cache))
sol = solve(prob, TRBDF2(chunk_size = chunk_size))
@test sol.retcode == ReturnCode.Success

#with auto-detected chunk_size
prob = ODEProblem(foo, ones(5, 5), (0.0, 1.0),
                  (ones(5, 5), FixedSizeDiffCache(zeros(5, 5))))
sol = solve(prob, TRBDF2())
@test sol.retcode == ReturnCode.Success

#Base array with LBC
function foo(du, u, (A, lbc), t)
    tmp = lbc[u]
    mul!(tmp, A, u)
    @. du = u + tmp
    nothing
end
prob = ODEProblem(foo, ones(5, 5), (0.0, 1.0), (ones(5, 5), LazyBufferCache()))
sol = solve(prob, TRBDF2())
@test sol.retcode == ReturnCode.Success
