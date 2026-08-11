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

function f2(u, cache)
    writer = get_tmp(cache, u)
    writer .= u
    reader = get_tmp(cache, u)
    # The second fetch must observe what was written through the first one
    return sum(reader)
end

# https://github.com/SciML/PreallocationTools.jl/issues/197
@testset "repeated fetches share storage" begin
    u = rand(4)
    cache = DiffCache(u)

    # Writing through one fetch and reading through another must propagate every
    # input to the output; with per-call allocation the read observes unwritten
    # memory instead and errors (or segfaults) during detection.
    S = jacobian_sparsity(x -> f2(x, cache), u, TracerSparsityDetector())
    @test S == ones(1, length(u))
    S = jacobian_sparsity(x -> f2(x, cache), u, TracerLocalSparsityDetector())
    @test S == ones(1, length(u))

    # Aliasing assertion that fails cleanly instead of segfaulting, on the tracer
    # type detection actually uses.
    tracer_types = DataType[]
    jacobian_sparsity(u, TracerSparsityDetector()) do x
        c = get_tmp(cache, x)
        push!(tracer_types, eltype(c))
        c .= x
        sum(c)
    end
    T = only(unique(tracer_types))
    @test isconcretetype(T)
    @test eltype(get_tmp(cache, T)) === T
    @test Base.mightalias(get_tmp(cache, T), get_tmp(cache, T))

    # Tracer workspaces follow cache resizing
    resize!(cache, 6)
    @test length(get_tmp(cache, T)) == 6
end

# The accumulator pattern: zero the workspace with a plain literal, then accumulate.
# `fill!(c, 0.0)` must convert through the workspace eltype — for a tracer that is the
# empty tracer ("constant, no dependencies"), so the accumulation records exactly the
# inputs and nothing stale survives from the previous call.
function f_fill(u, cache)
    c = get_tmp(cache, u)
    fill!(c, 0.0)
    @. c += u^2 + one(eltype(c))
    return sum(c)
end

@testset "fill! inside the differentiated function" begin
    u = rand(10)
    cache = DiffCache(u)

    @test_nowarn @inferred f_fill(u, cache)
    @test ForwardDiff.gradient(x -> f_fill(x, cache), u) ≈ 2 .* u

    S = jacobian_sparsity(x -> f_fill(x, cache), u, TracerSparsityDetector())
    @test S == ones(1, length(u))
    S = jacobian_sparsity(x -> f_fill(x, cache), u, TracerLocalSparsityDetector())
    @test S == ones(1, length(u))

    # Same, resetting the whole cache (primal, dual, and typed workspaces) instead of
    # the fetched workspace; the typed buffer already exists from the calls above and
    # must come back zeroed with its concrete eltype intact.
    function f_fill_cache(u, cache)
        fill!(cache, 0.0)
        c = get_tmp(cache, u)
        @. c += u^2
        return sum(c)
    end
    S = jacobian_sparsity(x -> f_fill_cache(x, cache), u, TracerSparsityDetector())
    @test S == ones(1, length(u))
    @test ForwardDiff.gradient(x -> f_fill_cache(x, cache), u) ≈ 2 .* u
end

end
