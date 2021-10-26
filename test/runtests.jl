using LinearAlgebra, OrdinaryDiffEq, Test, PreallocationTools, ForwardDiff, LabelledArrays

## Check ODE problem with specified chunk_size
function foo(du, u, (A, tmp), t)
    tmp = get_tmp(tmp, u)
    mul!(tmp, A, u)
    @. du = u + tmp
    nothing
end
chunk_size = 5
prob = ODEProblem(foo, ones(5, 5), (0., 1.0), (ones(5,5), dualcache(zeros(5,5), Val{chunk_size})))
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

@test all(df1[1:5,2] ≈ df2[6:10,1])