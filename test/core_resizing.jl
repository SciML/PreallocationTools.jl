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

#calculating the jacobian of claytonsample! with respect to τ and α
df2 = ForwardDiff.jacobian(x -> claytonsample!(stod, x[1], x[2]), [0.3; 0.0]) #should give a 15x2 array,
#because ForwardDiff flattens the output of jacobian, see: https://juliadiff.org/ForwardDiff.jl/stable/user/api/#ForwardDiff.jacobian

@test (length(randmat), 2) == size(df2)
@test df1[1:5, 2] ≈ df2[6:10, 1]

#test for enlarging cache
function rhs!(du, u, p, t)
    A = p
    return mul!(du, A, u)
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

# Test resize! functionality for DiffCache
@testset "resize! for DiffCache" begin
    u = rand(10)
    dc = DiffCache(u, 2)

    # Initial size
    @test length(dc.du) == 10
    @test length(dc.any_du) == 0  # Initially empty

    # Resize to larger
    resize!(dc, 20)
    @test length(dc.du) == 20
    @test length(dc.dual_du) == 20 * 3

    # Resize to smaller
    resize!(dc, 5)
    @test length(dc.du) == 5
    @test length(dc.dual_du) == 5 * 3

    # Test that it returns the cache itself
    @test resize!(dc, 8) === dc
end

@testset "reshape vector-backed DiffCache" begin
    dc = DiffCache(collect(1.0:10.0), 2)
    cache = reshape(dc, 2, 5)

    normal_tmp = get_tmp(cache, 1.0)
    @test size(normal_tmp) == (2, 5)
    @test vec(normal_tmp) == collect(1.0:10.0)

    normal_tmp[2, 3] = 42.0
    @test dc.du[6] == 42.0

    dual = ForwardDiff.Dual{Nothing}(1.0, 1.0, 0.0)
    dual_tmp = get_tmp(cache, dual)
    @test size(dual_tmp) == (2, 5)
    @test eltype(dual_tmp) <: ForwardDiff.Dual
    dual_tmp[1] = ForwardDiff.Dual{Nothing}(2.0, 3.0, 4.0)
    @test dc.dual_du[1:3] == [2.0, 3.0, 4.0]

    resize!(dc, 12)
    @test length(dc.dual_du) == 36

    resized_cache = reshape(dc, (2, 6))
    @test size(get_tmp(resized_cache, 1.0)) == (2, 6)
    @test size(get_tmp(resized_cache, dual)) == (2, 6)

    matrix_dc = DiffCache(zeros(2, 5), 2)
    matrix_cache = reshape(matrix_dc, 5, 2)
    matrix_dual_tmp = get_tmp(matrix_cache, dual)
    @test size(matrix_dual_tmp) == (5, 2)
    matrix_dual_tmp[1] = ForwardDiff.Dual{Nothing}(5.0, 6.0, 7.0)
    @test matrix_dc.dual_du[1:3] == [5.0, 6.0, 7.0]
end

# Test warn_on_resize option
@testset "warn_on_resize option" begin
    # Default: warn_on_resize = true
    dc_warn = DiffCache(zeros(2))
    @test dc_warn.warn_on_resize == true

    # Explicit: warn_on_resize = false
    dc_nowarn = DiffCache(zeros(2); warn_on_resize = false)
    @test dc_nowarn.warn_on_resize == false

    # warn_on_resize = false suppresses warning on cache enlargement
    dc_nowarn2 = DiffCache(zeros(2), 0; warn_on_resize = false)
    @test_nowarn ForwardDiff.gradient(x -> get_tmp(dc_nowarn2, x)[1], ones(4))

    # zero and copy preserve warn_on_resize
    dc = DiffCache(zeros(3); warn_on_resize = false)
    @test zero(dc).warn_on_resize == false
    @test copy(dc).warn_on_resize == false

    dc2 = DiffCache(zeros(3); warn_on_resize = true)
    @test zero(dc2).warn_on_resize == true
    @test copy(dc2).warn_on_resize == true
end

# Test resize! functionality for FixedSizeDiffCache
@testset "resize! for FixedSizeDiffCache" begin
    u = rand(10)
    dc = FixedSizeDiffCache(u)

    # Initial size
    @test length(dc.du) == 10
    @test length(dc.any_du) == 0  # Initially empty

    # Resize to larger
    resize!(dc, 20)
    @test length(dc.du) == 20

    # Resize to smaller
    resize!(dc, 5)
    @test length(dc.du) == 5

    # Test that it returns the cache itself
    @test resize!(dc, 8) === dc
end
