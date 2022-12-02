using Test, PreallocationTools, ForwardDiff

randmat = rand(10, 2)
sto = similar(randmat)
stod = dualcache(sto)

function claytonsample!(sto, τ; randmat=randmat)
    sto = get_tmp(sto, τ)
    sto .= randmat
    τ == 0 && return sto

    n = size(sto, 1)
    for i in 1:n
        v = sto[i, 2]
        u = sto[i, 1]
        sto[i, 2] = (1 - u^(-τ) + u^(-τ)*v^(-(τ/(1 + τ))))^(-1/τ)
    end
    return sto
end

ForwardDiff.derivative(τ -> claytonsample!(stod, τ), 0.3)
