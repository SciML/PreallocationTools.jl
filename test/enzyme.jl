using Enzyme
using PreallocationTools
using ForwardDiff
using Test

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

@testset "Forward mode, direct arguments" begin
    sto = similar(randmat)
    stod = DiffCache(sto)

    d_fwd = ForwardDiff.derivative(τ -> claytonsample!(stod, τ, 0.0), 0.3)

    d_const = only(
        Enzyme.autodiff(
            Forward, claytonsample!, Const(stod), Duplicated(0.3, 1.0), Const(0.0)
        )
    )
    @test d_const ≈ d_fwd

    d_dup = only(
        Enzyme.autodiff(
            Forward, claytonsample!,
            Duplicated(stod, Enzyme.make_zero(stod)), Duplicated(0.3, 1.0), Const(0.0)
        )
    )
    @test d_dup ≈ d_fwd

    d_α = only(
        Enzyme.autodiff(
            Forward, claytonsample!, Const(stod), Const(0.3), Duplicated(0.0, 1.0)
        )
    )
    d_α_fwd = reshape(
        ForwardDiff.jacobian(x -> claytonsample!(stod, x[1], x[2]), [0.3; 0.0])[:, 2],
        size(sto)
    )
    @test d_α ≈ d_α_fwd
end

@testset "Forward mode, cache enclosed in the function" begin
    stod = DiffCache(similar(randmat))
    clo = τ -> sum(claytonsample!(stod, τ, 0.0))
    d_enz = only(Enzyme.autodiff(Forward, Const(clo), Duplicated(0.3, 1.0)))
    @test d_enz ≈ ForwardDiff.derivative(clo, 0.3)
end

function withp(τ, p)
    tmp = get_tmp(p.cache, τ)
    copyto!(tmp, p.mat)
    @. tmp = tmp * τ^2
    return sum(abs2, tmp)
end

# Broadcasting a `Const` array into the cache buffer (`@. tmp = p.mat * τ^2`)
# trips Enzyme's static activity analysis for reasons unrelated to
# PreallocationTools (the same happens with a plain `Duplicated` buffer);
# Enzyme prescribes `set_runtime_activity` for that pattern, which the rules
# support as well.
function withp_bcast(τ, p)
    tmp = get_tmp(p.cache, τ)
    @. tmp = p.mat * τ^2
    return sum(abs2, tmp)
end

@testset "Forward mode, cache stored in p" begin
    p = (cache = DiffCache(similar(randmat)), mat = randmat)
    d_fwd = ForwardDiff.derivative(τ -> withp(τ, p), 0.3)

    d_constp = only(Enzyme.autodiff(Forward, withp, Duplicated(0.3, 1.0), Const(p)))
    @test d_constp ≈ d_fwd

    d_dupp = only(
        Enzyme.autodiff(
            Forward, withp, Duplicated(0.3, 1.0), Duplicated(p, Enzyme.make_zero(p))
        )
    )
    @test d_dupp ≈ d_fwd

    d_rta = only(
        Enzyme.autodiff(
            set_runtime_activity(Forward), withp_bcast, Duplicated(0.3, 1.0), Const(p)
        )
    )
    @test d_rta ≈ d_fwd

    d_rta_dup = only(
        Enzyme.autodiff(
            set_runtime_activity(Forward), withp_bcast, Duplicated(0.3, 1.0),
            Duplicated(p, Enzyme.make_zero(p))
        )
    )
    @test d_rta_dup ≈ d_fwd
end

@testset "Forward mode, batched" begin
    stod = DiffCache(similar(randmat))
    loss(τ, cache) = sum(abs2, claytonsample!(cache, τ, 0.0))
    d_fwd = ForwardDiff.derivative(τ -> loss(τ, stod), 0.3)

    d_batch = only(
        Enzyme.autodiff(
            Forward, loss, BatchDuplicated(0.3, (1.0, 2.0)), Const(stod)
        )
    )
    @test d_batch[1] ≈ d_fwd
    @test d_batch[2] ≈ 2 * d_fwd
end

@testset "Enzyme over ForwardDiff" begin
    stod = DiffCache(similar(randmat))
    inner(τ) = sum(claytonsample!(stod, τ, 0.0))
    fdderiv(τ) = ForwardDiff.derivative(inner, τ)

    d2_fd = ForwardDiff.derivative(fdderiv, 0.3)
    # `set_runtime_activity` is needed for `sto .= randmat` on the dual-valued
    # buffer view, an Enzyme static activity analysis limitation unrelated to
    # PreallocationTools.
    d2_enz = only(
        Enzyme.autodiff(
            set_runtime_activity(Forward), Const(fdderiv), Duplicated(0.3, 1.0)
        )
    )
    @test d2_enz ≈ d2_fd

    # The first derivative computed inside the Enzyme pass must be untouched:
    # the Enzyme shadow must not alias the ForwardDiff dual buffer.
    @test fdderiv(0.3) ≈ ForwardDiff.derivative(inner, 0.3)
end

@testset "Second order: reverse-mode Enzyme over ForwardDiff" begin
    stod = DiffCache(similar(randmat))
    fdderiv(τ, cache) = ForwardDiff.derivative(σ -> sum(claytonsample!(cache, σ, 0.0)), τ)
    d2_fd = ForwardDiff.derivative(τ -> fdderiv(τ, stod), 0.3)

    d2_enz = Enzyme.autodiff(
        set_runtime_activity(Reverse), fdderiv, Active, Active(0.3), Const(stod)
    )[1][1]
    @test d2_enz ≈ d2_fd
end

@testset "Second order: Enzyme over Enzyme" begin
    stod = DiffCache(similar(randmat))
    loss(τ, cache) = sum(abs2, claytonsample!(cache, τ, 0.0))
    d2_fd = ForwardDiff.derivative(
        τ -> ForwardDiff.derivative(σ -> loss(σ, stod), τ), 0.3
    )

    dloss_f(τ, cache) = only(
        Enzyme.autodiff(Forward, loss, Duplicated(τ, 1.0), Const(cache))
    )
    dloss_r(τ, cache) = Enzyme.autodiff(Reverse, loss, Active, Active(τ), Const(cache))[1][1]

    # Forward over forward. Both nesting levels resolve a hidden shadow for
    # the same cache; the tests below rely on the registry keeping the
    # forward-rule and reverse-rule lanes disjoint so the levels never share
    # a shadow buffer in mixed-mode nesting.
    d2_ff = only(
        Enzyme.autodiff(
            set_runtime_activity(Forward), Const(dloss_f), Duplicated(0.3, 1.0), Const(stod)
        )
    )
    @test d2_ff ≈ d2_fd

    # Forward over reverse.
    d2_fr = only(
        Enzyme.autodiff(
            set_runtime_activity(Forward), Const(dloss_r), Duplicated(0.3, 1.0), Const(stod)
        )
    )
    @test d2_fr ≈ d2_fd

    # Reverse over forward.
    d2_rf = Enzyme.autodiff(
        set_runtime_activity(Reverse), dloss_f, Active, Active(0.3), Const(stod)
    )[1][1]
    @test d2_rf ≈ d2_fd
end

function twofetch(τ, cache)
    a = get_tmp(cache, τ)
    fill!(a, τ)
    b = get_tmp(cache, τ)
    return sum(b)
end

@testset "Repeated get_tmp aliases the same shadow" begin
    cache = DiffCache(zeros(4))
    d = only(Enzyme.autodiff(Forward, twofetch, Duplicated(0.5, 1.0), Const(cache)))
    @test d ≈ 4.0

    g = Enzyme.autodiff(Reverse, twofetch, Active, Active(0.5), Const(cache))
    @test g[1][1] ≈ 4.0
end

@testset "Reverse mode, scalar input" begin
    stod = DiffCache(similar(randmat))
    loss(τ, cache) = sum(abs2, claytonsample!(cache, τ, 0.0))
    d_fwd = ForwardDiff.derivative(τ -> loss(τ, stod), 0.3)

    g_const = Enzyme.autodiff(Reverse, loss, Active, Active(0.3), Const(stod))
    @test g_const[1][1] ≈ d_fwd

    g_dup = Enzyme.autodiff(
        Reverse, loss, Active, Active(0.3), Duplicated(stod, Enzyme.make_zero(stod))
    )
    @test g_dup[1][1] ≈ d_fwd
end

function fode!(du, u, p)
    tmp = get_tmp(p.cache, u)
    @. tmp = u * p.a
    @. du = tmp^2
    return nothing
end

@testset "Reverse mode, mutating ODE-style function" begin
    u = rand(4)
    du = zeros(4)
    p = (cache = DiffCache(zeros(4)), a = 2.5)

    bdu = ones(4)
    bu = zeros(4)
    Enzyme.autodiff(
        Reverse, fode!, Const, Duplicated(du, bdu), Duplicated(u, bu), Const(p)
    )
    @test bu ≈ 2 .* p.a^2 .* u
end

@testset "Forward-mode use does not poison later reverse-mode use" begin
    stod = DiffCache(similar(randmat))
    loss(τ, cache) = sum(abs2, claytonsample!(cache, τ, 0.0))
    d_fwd = ForwardDiff.derivative(τ -> loss(τ, stod), 0.3)

    only(Enzyme.autodiff(Forward, loss, Duplicated(0.3, 1.0), Const(stod)))
    g = Enzyme.autodiff(Reverse, loss, Active, Active(0.3), Const(stod))
    @test g[1][1] ≈ d_fwd
end

function lbcfun(u, lbc)
    tmp = lbc[u]
    @. tmp = 2 * u
    return sum(abs2, tmp)
end

function lbcsized(u, lbc)
    tmp = lbc[u, 2]
    tmp[1] = u[1]^2
    tmp[2] = u[2]^3
    return tmp[1] + tmp[2]
end

@testset "LazyBufferCache" begin
    lbc = LazyBufferCache()
    u = rand(4)

    d = only(Enzyme.autodiff(Forward, lbcfun, Duplicated(u, ones(4)), Const(lbc)))
    @test d ≈ sum(8 .* u)

    bu = zeros(4)
    Enzyme.autodiff(Reverse, lbcfun, Active, Duplicated(u, bu), Const(lbc))
    @test bu ≈ 8 .* u

    bu2 = zeros(4)
    Enzyme.autodiff(Reverse, lbcsized, Active, Duplicated(u, bu2), Const(lbc))
    @test bu2 ≈ [2 * u[1], 3 * u[2]^2, 0.0, 0.0]

    lbc_dup = LazyBufferCache()
    bu3 = zeros(4)
    Enzyme.autodiff(
        Reverse, lbcfun, Active, Duplicated(u, bu3),
        Duplicated(lbc_dup, Enzyme.make_zero(lbc_dup))
    )
    @test bu3 ≈ 8 .* u
end

function fscfun(u, cache)
    tmp = get_tmp(cache, u)
    @. tmp = u^2
    return sum(tmp)
end

@testset "FixedSizeDiffCache" begin
    u = rand(4)
    fsc = FixedSizeDiffCache(zeros(4))

    d = only(Enzyme.autodiff(Forward, fscfun, Duplicated(u, ones(4)), Const(fsc)))
    @test d ≈ sum(2 .* u)

    bu = zeros(4)
    Enzyme.autodiff(Reverse, fscfun, Active, Duplicated(u, bu), Const(fsc))
    @test bu ≈ 2 .* u

    d2 = only(
        Enzyme.autodiff(
            Forward, fscfun, Duplicated(u, ones(4)),
            Duplicated(fsc, Enzyme.make_zero(fsc))
        )
    )
    @test d2 ≈ sum(2 .* u)
end
