module TestSparseConnectivityTracer

using PreallocationTools, SparseConnectivityTracer, ForwardDiff, SparseArrays, Test

function f1(u, cache)
    c = get_tmp(cache, u)
    # This will throw if a fallback definition is used
    # such that `eltype(c) == Any`
    T = eltype(c)
    @. c = u^2 + one(T)
    return sum(c)
end

@testset "out of place" begin
    u = rand(10)
    cache = DiffCache(u)

    @test_nowarn @inferred f1(u, cache)
    @test_nowarn ForwardDiff.gradient(u) do u
        f1(u, cache)
    end
    @test_nowarn jacobian_sparsity(u, TracerSparsityDetector()) do u
        f1(u, cache)
    end
    @test_nowarn hessian_sparsity(u, TracerSparsityDetector()) do u
        f1(u, cache)
    end
    @test_nowarn jacobian_sparsity(u, TracerLocalSparsityDetector()) do u
        f1(u, cache)
    end
    @test_nowarn hessian_sparsity(u, TracerLocalSparsityDetector()) do u
        f1(u, cache)
    end
end

function f1!(du, u, cache)
    c = get_tmp(cache, u)
    # This will throw if a fallback definition is used
    # such that `eltype(c) == Any`
    T = eltype(c)
    @. c = u^2 + one(T)
    du[1] = sum(c)
    return nothing
end

@testset "in place" begin
    u = rand(10)
    cache = DiffCache(u)
    du = similar(u, (1,))

    @test_nowarn @inferred f1!(du, u, cache)
    @test_nowarn ForwardDiff.jacobian(du, u) do du, u
        f1!(du, u, cache)
    end
    @test_nowarn jacobian_sparsity(du, u, TracerSparsityDetector()) do du, u
        f1!(du, u, cache)
    end
    @test_nowarn jacobian_sparsity(du, u, TracerLocalSparsityDetector()) do du, u
        f1!(du, u, cache)
    end
end

end
