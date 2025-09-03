module TestEnzyme
    using Enzyme
    using PreallocationTools
    using ForwardDiff

    const randmat = rand(5, 3)


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

    sto = similar(randmat)
    stod = DiffCache(sto)

    d_sto_fwd = ForwardDiff.derivative(τ -> claytonsample!(stod, τ, 0.0), 0.3)
    d_sto_enz = Enzyme.autodiff(Forward, claytonsample!, Const(stod), Duplicated(0.3, 1.0), Const(0.0)) |> only

    @test d_sto_enz ≈ d_sto_fwd

    d_sto_enz2 = Enzyme.autodiff(Forward, claytonsample!, Duplicated(stod, Enzyme.make_zero(stod)), Duplicated(0.3, 1.0), Const(0.0)) |> only
    @test d_sto_enz2 ≈ d_sto_fwd

    d_sto_enz3 = Enzyme.autodiff(Forward, claytonsample!, Const(stod), Const(0.3), Const(0.0)) |> only
    @test all(d_sto_enz3 .== 0.0)

    d_sto_enz4 = Enzyme.autodiff(Forward, claytonsample!, Const(stod), Const(0.3), Duplicated(1.0, 1.0)) |> only
    d_sto_fwd4 = reshape(ForwardDiff.jacobian(x -> claytonsample!(stod, x[1], x[2]), [0.3; 0.0])[:, 2], size(sto))
    @test d_sto_enz4 ≈ d_sto_fwd4
end # TestEnzyme

