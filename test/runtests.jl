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

            #calculating the jacobian of claytonsample! with respect to τ and α
            df2 = ForwardDiff.jacobian(x -> claytonsample!(stod, x[1], x[2]), [0.3; 0.0]) #should give a 15x2 array,
            #because ForwardDiff flattens the output of jacobian, see: https://juliadiff.org/ForwardDiff.jl/stable/user/api/#ForwardDiff.jacobian

            @test (length(randmat), 2) == size(df2)
            @test df1[1:5,2] ≈ df2[6:10,1]
        end
    end
end
