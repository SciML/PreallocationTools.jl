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
using LinearAlgebra, OrdinaryDiffEq, Test, PreallocationTools, ForwardDiff, LabelledArrays, CUDA, RecursiveArrayTools

@testset verbose = true "PreallocationTools tests" begin
    @testset "Dispatch" verbose = true begin #tests dispatching without changing chunk_size
        chunk_size = 5
        #base array tests
        @testset "Base Arrays" begin
            u0_B = ones(5, 5)
            dual_B = zeros(ForwardDiff.Dual{ForwardDiff.Tag{typeof(something), Float64}, Float64, chunk_size}, 2, 2)
            cache_B = dualcache(u0_B, Val{chunk_size})
            tmp_du_BA = get_tmp(cache_B, u0_B)
            tmp_dual_du_BA = get_tmp(cache_B, dual_B)
            tmp_du_BN = get_tmp(cache_B, u0_B[1])
            tmp_dual_du_BN = get_tmp(cache_B, dual_B[1])
            @test size(tmp_du_BA) == size(u0_B)
            @test typeof(tmp_du_BA) == typeof(u0_B)
            @test eltype(tmp_du_BA) == eltype(u0_B)
            @test size(tmp_dual_du_BA) == size(u0_B)
            @test typeof(tmp_dual_du_BA) == typeof(dual_B)
            @test eltype(tmp_dual_du_BA) == eltype(dual_B) 
            @test size(tmp_du_BN) == size(u0_B) 
            @test typeof(tmp_du_BN) == typeof(u0_B)
            @test eltype(tmp_du_BN) == eltype(u0_B)
            @test size(tmp_dual_du_BN) == size(u0_B)
            @test typeof(tmp_dual_du_BN) == typeof(dual_B)
            @test eltype(tmp_dual_du_BN) == eltype(dual_B) 
        end
        @testset "Labelled Arrays" begin
            chunk_size = 4
            u0_L = LArray((2,2); a=1.0, b=1.0, c=1.0, d=1.0)
            zerodual = zero(ForwardDiff.Dual{ForwardDiff.Tag{typeof(something), Float64}, Float64, chunk_size})
            dual_L = LArray((2,2); a=zerodual, b=zerodual, c=zerodual, d=zerodual) 
            cache_L = dualcache(u0_L, Val{chunk_size})
            tmp_du_LA = get_tmp(cache_L, u0_L)
            tmp_dual_du_LA = get_tmp(cache_L, dual_L)
            tmp_du_LN = get_tmp(cache_L, u0_L[1])
            tmp_dual_du_LN = get_tmp(cache_L, dual_L[1])
            @test size(tmp_du_LA) == size(u0_L)
            @test typeof(tmp_du_LA) == typeof(u0_L)
            @test eltype(tmp_du_LA) == eltype(u0_L)
            @test size(tmp_dual_du_LA) == size(u0_L)
            @test typeof(tmp_dual_du_LA) == typeof(dual_L)
            @test eltype(tmp_dual_du_LA) == eltype(dual_L) 
            @test size(tmp_du_LN) == size(u0_L) 
            @test typeof(tmp_du_LN) == typeof(u0_L)
            @test eltype(tmp_du_LN) == eltype(u0_L)
            @test size(tmp_dual_du_LN) == size(u0_L)
            @test typeof(tmp_dual_du_LN) == typeof(dual_L)
            @test eltype(tmp_dual_du_LN) == eltype(dual_L) 
        end
        @testset "Array Partitions" begin
            u0_AP = ArrayPartition(ones(2,2), ones(3,3))
            dual_a = zeros(ForwardDiff.Dual{ForwardDiff.Tag{typeof(something), Float64}, Float64, chunk_size}, 2, 2)
            dual_b = zeros(ForwardDiff.Dual{ForwardDiff.Tag{typeof(something), Float64}, Float64, chunk_size}, 3, 3)
            dual_AP = ArrayPartition(dual_a, dual_b) 
            cache_AP = dualcache(u0_AP, Val{chunk_size})
            tmp_du_APA = get_tmp(cache_AP, u0_AP)
            tmp_dual_du_APA = get_tmp(cache_AP, dual_AP)
            tmp_du_APN = get_tmp(cache_AP, u0_AP[1])
            tmp_dual_du_APN = get_tmp(cache_AP, dual_AP[1])
            @test size(tmp_du_APA) == size(u0_AP)
            @test typeof(tmp_du_APA) == typeof(u0_AP)
            @test eltype(tmp_du_APA) == eltype(u0_AP)
            @test size(tmp_dual_du_APA) == size(u0_AP)
            @test typeof(tmp_dual_du_APA) == typeof(dual_AP)
            @test eltype(tmp_dual_du_APA) == eltype(dual_AP) 
            @test size(tmp_du_APN) == size(u0_AP) 
            @test typeof(tmp_du_APN) == typeof(u0_AP)
            @test eltype(tmp_du_APN) == eltype(u0_AP)
            @test size(tmp_dual_du_APN) == size(u0_AP)
            @test typeof(tmp_dual_du_APN) == typeof(dual_AP)
            @test eltype(tmp_dual_du_APN) == eltype(dual_AP)  
        end
        @testset "Cu Arrays" begin
            u0_CU = cu(ones(5,5))
            dual_CU = cu(zeros(ForwardDiff.Dual{ForwardDiff.Tag{typeof(something), Float64}, Float64, chunk_size}, 2, 2))
            cache_CU = dualcache(u0_CU, Val{chunk_size})
            tmp_du_CUA = get_tmp(cache_CU, u0_CU)
            tmp_dual_du_CUA = get_tmp(cache_CU, dual_CU)
            tmp_du_CUN = get_tmp(cache_CU, u0_CU[1])
            tmp_dual_du_CUN = get_tmp(cache_CU, dual_CU[1])
            @test typeof(cache_CU.dual_du) == typeof(u0_CU) #check that dual cache array is a GPU array for performance reasons.
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
        end      
    end
    @testset "ODE tests" verbose = true begin      
        @testset "Base Array" begin
            function foo(du, u, (A, tmp), t)
                tmp = get_tmp(tmp, u)
                mul!(tmp, A, u)
                @. du = u + tmp
                nothing
            end
            #with defined chunk_size
            chunk_size = 5
            u0 = ones(5, 5)
            A = ones(5,5)
            cache = dualcache(zeros(5,5), Val{chunk_size})
            prob = ODEProblem(foo, u0, (0., 1.0), (A, cache))
            sol = solve(prob, TRBDF2(chunk_size=chunk_size))
            @test sol.retcode == :Success
            
            #with auto-detected chunk_size
            prob = ODEProblem(foo, ones(5, 5), (0., 1.0), (ones(5,5), dualcache(zeros(5,5))))
            sol = solve(prob, TRBDF2())
            @test sol.retcode == :Success
        end

        @testset "Base Array and LBC" begin
            function foo(du, u, (A, lbc), t)
            tmp = lbc[u]
            mul!(tmp, A, u)
            @. du = u + tmp
            nothing
            end
            prob = ODEProblem(foo, ones(5, 5), (0., 1.0), (ones(5,5), LazyBufferCache()))
            sol = solve(prob, TRBDF2())
            @test sol.retcode == :Success
        end

        @testset "LArray" begin
            A = LArray((2,2); a=1.0, b=1.0, c=1.0, d=1.0)
            c = LArray((2,2); a=0.0, b=0.0, c=0.0, d=0.0)
            u0 = LArray((2,2); a=1.0, b=1.0, c=1.0, d=1.0)
            function foo(du, u, (A, tmp), t)
                tmp = get_tmp(tmp, u)
                mul!(tmp, A, u)
                @. du = u + tmp
                nothing
            end
            #with specified chunk_size
            chunk_size = 4
            prob = ODEProblem(foo, u0, (0., 1.0), (A, dualcache(c, Val{chunk_size})))
            sol = solve(prob, TRBDF2(chunk_size = chunk_size))
            @test sol.retcode == :Success
            #with auto-detected chunk_size
            prob = ODEProblem(foo, u0, (0., 1.0), (A, dualcache(c)))
            sol = solve(prob, TRBDF2())
            @test sol.retcode == :Success
        end
        
        @testset "cuarray" begin
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
            cache = dualcache(A, Val{chunk_size})
            prob = ODEProblem(foo, u0, (0.0f0,1.0f0), (A, cache))
            sol = solve(prob, TRBDF2(chunk_size = chunk_size))
            @test sol.retcode == :Success

            #with auto-detected chunk_size
            u0 = cu(rand(10,10)) #example kept small for test purposes.
            A  = cu(-randn(10,10))                  
            cache = dualcache(A)
            prob = ODEProblem(foo, u0, (0.0f0,1.0f0), (A, cache))
            sol = solve(prob, TRBDF2())
            @test sol.retcode == :Success
        end
    end

    @testset "Change of chunk_size" verbose = true begin
        @testset "Base array" begin
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
            @test size(randmat) == size(df1)

            #calculating the jacobian of claytonsample! with respect to τ and α
            df2 = ForwardDiff.jacobian(x -> claytonsample!(stod, x[1], x[2]), [0.3; 0.0]) #should give a 15x2 array,
            #because ForwardDiff flattens the output of jacobian, see: https://juliadiff.org/ForwardDiff.jl/stable/user/api/#ForwardDiff.jacobian

            @test (length(randmat), 2) == size(df2)
            @test df1[1:5,2] ≈ df2[6:10,1]
        end

        @testset "cuarray" begin
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

            #taking the derivative of claytonsample! with respect to τ only
            df1 = ForwardDiff.derivative(τ -> claytonsample!(stod, τ, 0.0), 0.3)
            @test size(randmat) == size(df1)

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
            #calculating the jacobian of claytonsample! with respect to τ and α
            df2 = ForwardDiff.jacobian(x -> claytonsample!(stod, x[1], x[2]), [0.3; 0.0]) #should give a 15x2 array,
            #because ForwardDiff flattens the output of jacobian, see: https://juliadiff.org/ForwardDiff.jl/stable/user/api/#ForwardDiff.jacobian

            @test (length(randmat), 2) == size(df2)
            @test df1[1:5,2] ≈ df2[6:10,1]
        end
    end
end
