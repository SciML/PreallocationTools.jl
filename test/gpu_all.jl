using LinearAlgebra, OrdinaryDiffEq, Test, PreallocationTools, CUDA, ForwardDiff, ArrayInterface

#Dispatch tests
chunk_size = 5
u0_CU = cu(ones(5,5))
dual_CU = cu(zeros(ForwardDiff.Dual{ForwardDiff.Tag{typeof(something), Float32}, Float32, chunk_size}, 2, 2))
dual_N = ForwardDiff.Dual{ForwardDiff.Tag{typeof(something), Float32}, Float32, 5}(0)
cache_CU = dualcache(u0_CU, chunk_size)
tmp_du_CUA = get_tmp(cache_CU, u0_CU)
tmp_dual_du_CUA = get_tmp(cache_CU, dual_CU)
tmp_du_CUN = get_tmp(cache_CU, 0.0)
tmp_dual_du_CUN = get_tmp(cache_CU, dual_N)
@test ArrayInterface.parameterless_type(typeof(cache_CU.dual_du)) == ArrayInterface.parameterless_type(typeof(u0_CU)) #check that dual cache array is a GPU array for performance reasons.
@test size(tmp_du_CUA) == size(u0_CU)                
@test typeof(tmp_du_CUA) == typeof(u0_CU)
@test eltype(tmp_du_CUA) == eltype(u0_CU)
@test size(tmp_dual_du_CUA) == size(u0_CU)
@test typeof(tmp_dual_du_CUA) == typeof(dual_CU)
@test eltype(tmp_dual_du_CUA) == eltype(dual_CU) 
@test size(tmp_du_CUN) == size(u0_CU) 
@test typeof(tmp_du_CUN) == typeof(u0_CU)
@test eltype(tmp_du_CUN) == eltype(u0_CU)
@test size(tmp_dual_du_CUN) == size(u0_CU)
@test typeof(tmp_dual_du_CUN) == typeof(dual_CU)
@test eltype(tmp_dual_du_CUN) == eltype(dual_CU) 

#ODE tests
function foo(du, u, (A, tmp), t)
    tmp = get_tmp(tmp, u)
    mul!(tmp, A, u)
    @. du = u + tmp
    nothing
end
#with specified chunk_size
chunk_size = 10
u0 = cu(rand(10,10)) #example kept small for test purposes.
A  = cu(-randn(10,10))                  
cache = dualcache(cu(zeros(10, 10)), chunk_size)
prob = ODEProblem(foo, u0, (0.0f0, 1.0f0), (A, cache))
sol = solve(prob, TRBDF2(chunk_size = chunk_size))
@test sol.retcode == :Success

#with auto-detected chunk_size
u0 = cu(rand(10,10)) #example kept small for test purposes.
A  = cu(-randn(10,10))                  
cache = dualcache(cu(zeros(10, 10)))
prob = ODEProblem(foo, u0, (0.0f0,1.0f0), (A, cache))
sol = solve(prob, TRBDF2())
@test sol.retcode == :Success

randmat = cu(rand(5, 3))
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

#resizing tests
#taking the derivative of claytonsample! with respect to τ only
df1 = ForwardDiff.derivative(τ -> claytonsample!(stod, τ, 0.0), 0.3)
@test size(randmat) == size(df1)

#calculating the jacobian of claytonsample! with respect to τ and α
df2 = ForwardDiff.jacobian(x -> claytonsample!(stod, x[1], x[2]), [0.3; 0.0]) #should give a 15x2 array,
#because ForwardDiff flattens the output of jacobian, see: https://juliadiff.org/ForwardDiff.jl/stable/user/api/#ForwardDiff.jacobian

@test (length(randmat), 2) == size(df2)
@test df1[1:5,2] ≈ df2[6:10,1]