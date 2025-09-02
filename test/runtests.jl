using Pkg
using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")

function activate_downstream_env()
    Pkg.activate("GPU")
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    Pkg.instantiate()
end

if GROUP == "All" || GROUP == "Core"
    @safetestset "Quality Assurance" include("qa.jl")
    @safetestset "DiffCache Dispatch" include("core_dispatch.jl")
    @safetestset "DiffCache ODE tests" include("core_odes.jl")
    @safetestset "DiffCache Resizing" include("core_resizing.jl")
    @safetestset "DiffCache Nested Duals" include("core_nesteddual.jl")
    @safetestset "DiffCache Sparsity Support" include("sparsity_support.jl")
    @safetestset "DiffCache with SparseConnectivityTracer" include("sparse_connectivity_tracer.jl")
    @safetestset "LazyBufferCache" include("lbc.jl")
    @safetestset "GeneralLazyBufferCache" include("general_lbc.jl")
    @safetestset "Zero, Copy, and Fill Dispatches" include("test_zero_copy.jl")
end

if GROUP == "GPU"
    activate_downstream_env()
    @safetestset "GPU tests" include("gpu_all.jl")
end
