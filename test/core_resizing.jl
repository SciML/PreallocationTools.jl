using Test, PreallocationTools, ForwardDiff, LinearAlgebra

#test for downsizing cache
randmat = rand(5, 3)
sto = similar(randmat)
stod = DiffCache(sto)

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

#taking the derivative of claytonsample! with respect to τ only
df1 = ForwardDiff.derivative(τ -> claytonsample!(stod, τ, 0.0), 0.3)
@test size(randmat) == size(df1)

stod = FixedSizeDiffCache(sto)
ForwardDiff.derivative(τ -> claytonsample!(stod, τ), 0.3)
@test size(randmat) == size(df1)

#calculating the jacobian of claytonsample! with respect to τ and α
df2 = ForwardDiff.jacobian(x -> claytonsample!(stod, x[1], x[2]), [0.3; 0.0]) #should give a 15x2 array,
#because ForwardDiff flattens the output of jacobian, see: https://juliadiff.org/ForwardDiff.jl/stable/user/api/#ForwardDiff.jacobian

@test (length(randmat), 2) == size(df2)
@test df1[1:5, 2] ≈ df2[6:10, 1]

#test for enlarging cache
function rhs!(du, u, p, t)
    A = p
    mul!(du, A, u)
end

function loss(du, u, p, t)
    _du = get_tmp(du, p)
    rhs!(_du, u, p, t)
    l = _du[1]
    return l
end

u = [3.0, 0.0]
A = ones(2, 2)

du = similar(u)
_du = DiffCache(du)
f = A -> loss(_du, u, A, 0.0)
analyticalsolution = [3.0 0; 0 0]
@test ForwardDiff.gradient(f, A) ≈ analyticalsolution
