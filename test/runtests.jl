using Pkg
using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")
const is_APPVEYOR = Sys.iswindows() && haskey(ENV, "APPVEYOR")

function activate_downstream_env()
    Pkg.activate("GPU")
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    Pkg.instantiate()
end

if GROUP == "All" || GROUP == "Core"
    @safetestset "DiffCache Dispatch" begin include("DiffCache/core_dispatch.jl") end
    @safetestset "DiffCache ODE tests" begin include("DiffCache/core_odes.jl") end
    @safetestset "DiffCache Resizing" begin include("DiffCache/core_resizing.jl") end
    @safetestset "DiffCache Nested Duals" begin include("DiffCache/core_nesteddual.jl") end
    @safetestset "DiffCache Sparsity Support" begin include("DiffCache/sparsity_support.jl") end

    @safetestset "FixedSizeDiffCache Dispatch" begin include("FixedSizeDiffCache/core_dispatch.jl") end
    @safetestset "FixedSizeDiffCache ODE tests" begin include("FixedSizeDiffCache/core_odes.jl") end
    @safetestset "FixedSizeDiffCache Base Array Resizing" begin include("FixedSizeDiffCache/core_resizing.jl") end
end

if !is_APPVEYOR && GROUP == "GPU"
    activate_downstream_env()
    @safetestset "GPU tests" begin include("FixedSizeDiffCache/gpu_all.jl") end
    @safetestset "GPU tests" begin include("DiffCache/gpu_all.jl") end
end
