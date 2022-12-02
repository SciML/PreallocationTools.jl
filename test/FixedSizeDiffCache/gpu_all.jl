using LinearAlgebra, OrdinaryDiffEq, Test, PreallocationTools, CUDA, ForwardDiff

chunk_size = 5

#Dispatch tests
chunk_size = 5
u0_B = cu(ones(5, 5))
dual_B = cu(zeros(ForwardDiff.Dual{ForwardDiff.Tag{typeof(something), Float32}, Float32,
                                   chunk_size}, 2, 2))
cache_B = FixedSizeDiffCache(u0_B, chunk_size)
tmp_du_BA = get_tmp(cache_B, u0_B)
tmp_dual_du_BA = get_tmp(cache_B, dual_B)
@test size(tmp_du_BA) == size(u0_B)
@test typeof(tmp_du_BA) == typeof(u0_B)
@test eltype(tmp_du_BA) == eltype(u0_B)
@test size(tmp_dual_du_BA) == size(u0_B)
@test typeof(tmp_dual_du_BA) == typeof(dual_B)
@test eltype(tmp_dual_du_BA) == eltype(dual_B)

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
u0 = cu(ones(5, 5))
A = cu(ones(5, 5))
cache = FixedSizeDiffCache(cu(zeros(5, 5)), chunk_size)
prob = ODEProblem{true, SciMLBase.FullSpecialize}(foo, u0, (0.0f0, 1.0f0), (A, cache))
sol = solve(prob, TRBDF2(chunk_size = chunk_size))
@test sol.retcode == ReturnCode.Success

#with auto-detected chunk_size
u0 = cu(rand(10, 10)) #example kept small for test purposes.
A = cu(-randn(10, 10))
cache = FixedSizeDiffCache(A)
prob = ODEProblem(foo, u0, (0.0f0, 1.0f0), (A, cache))
sol = solve(prob, TRBDF2())
@test sol.retcode == ReturnCode.Success
