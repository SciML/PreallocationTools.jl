using SafeTestsets
using SciMLTesting

run_tests(;
    core = () -> begin
        @safetestset "DiffCache Dispatch" include("core_dispatch.jl")
        @safetestset "DiffCache ODE tests" include("core_odes.jl")
        @safetestset "DiffCache Resizing" include("core_resizing.jl")
        @safetestset "DiffCache Nested Duals" include("core_nesteddual.jl")
        @safetestset "DiffCache Sparsity Support" include("sparsity_support.jl")
        @safetestset "DiffCache with SparseConnectivityTracer" include("sparse_connectivity_tracer.jl")
        @safetestset "LazyBufferCache" include("lbc.jl")
        @safetestset "GeneralLazyBufferCache" include("general_lbc.jl")
        @safetestset "Zero, Copy, and Fill Dispatches" include("test_zero_copy.jl")
        @safetestset "Allocation Regression Tests" include("alloc_tests.jl")
    end,
    qa = (;
        env = joinpath(@__DIR__, "qa"),
        body = joinpath(@__DIR__, "qa", "qa.jl")
    ),
    groups = Dict(
        # GPU declares its own sub-env, so it runs ONLY for GROUP="GPU" and is
        # excluded from "All" (matches the original `if GROUP == "GPU"` dispatch).
        "GPU" => (;
            env = joinpath(@__DIR__, "GPU"), body = () -> begin
                @safetestset "GPU tests" include(joinpath("GPU", "gpu_all.jl"))
            end
        ),
    ),
)
