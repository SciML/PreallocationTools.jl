using LinearAlgebra,
      OrdinaryDiffEq, Test, PreallocationTools, LabelledArrays,
      RecursiveArrayTools, ADTypes

#Base array
function foo(du, u, (A, tmp), t)
    tmp = get_tmp(tmp, promote_type(eltype(u), typeof(t)))
    mul!(tmp, A, u)
    @. du = u + tmp
    nothing
end
#with defined chunk_size
chunk_size = 5
u0 = ones(5, 5)
A = ones(5, 5)
cache = DiffCache(zeros(5, 5), chunk_size)
prob = ODEProblem{true, SciMLBase.FullSpecialize}(foo, u0, (0.0, 1.0), (A, cache))
sol = solve(prob, Rodas5P(autodiff = AutoForwardDiff(chunksize = chunk_size)))
@test sol.retcode == ReturnCode.Success

cache = FixedSizeDiffCache(zeros(5, 5), chunk_size)
prob = ODEProblem{true, SciMLBase.FullSpecialize}(foo, u0, (0.0, 1.0), (A, cache))
sol = solve(prob, Rodas5P(autodiff = AutoForwardDiff(chunksize = chunk_size)))
@test sol.retcode == ReturnCode.Success

#with auto-detected chunk_size
cache = DiffCache(zeros(5, 5))
prob = ODEProblem{true, SciMLBase.FullSpecialize}(foo, ones(5, 5), (0.0, 1.0), (A, cache))
sol = solve(prob, Rodas5P())
@test sol.retcode == ReturnCode.Success

prob = ODEProblem(foo, ones(5, 5), (0.0, 1.0),
    (ones(5, 5), FixedSizeDiffCache(zeros(5, 5))))
sol = solve(prob, Rodas5P())
@test sol.retcode == ReturnCode.Success

#Base array with LBC
function foo(du, u, (A, lbc), t)
    tmp = lbc[u]
    mul!(tmp, A, u)
    @. du = u + tmp
    nothing
end
prob = ODEProblem{true, SciMLBase.FullSpecialize}(foo, ones(5, 5), (0.0, 1.0),
    (ones(5, 5), LazyBufferCache()))
sol = solve(prob, Rodas5P())
@test sol.retcode == ReturnCode.Success

#LArray
A = LArray((2, 2); a = 1.0, b = 1.0, c = 1.0, d = 1.0)
c = LArray((2, 2); a = 0.0, b = 0.0, c = 0.0, d = 0.0)
u0 = LArray((2, 2); a = 1.0, b = 1.0, c = 1.0, d = 1.0)
function foo(du, u, (A, tmp), t)
    tmp = get_tmp(tmp, u)
    mul!(tmp, A, u)
    @. du = u + tmp
    nothing
end
#with specified chunk_size
chunk_size = 4
prob = ODEProblem{true, SciMLBase.FullSpecialize}(foo, u0, (0.0, 1.0),
    (A, DiffCache(c, chunk_size)))
sol = solve(prob, Rodas5P(autodiff = AutoForwardDiff(chunksize = chunk_size)))
@test sol.retcode == ReturnCode.Success
#with auto-detected chunk_size
prob = ODEProblem{true, SciMLBase.FullSpecialize}(foo, u0, (0.0, 1.0), (A, DiffCache(c)))
sol = solve(prob, Rodas5P())
@test sol.retcode == ReturnCode.Success
