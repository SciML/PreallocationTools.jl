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
    @safetestset "DiffCache Base Array Resizing" begin include("DiffCache/core_resizing.jl") end

    @safetestset "ResizingDiffCache Dispatch" begin include("ResizingDiffCache/core_dispatch.jl") end
    @safetestset "ResizingDiffCache ODE tests" begin include("ResizingDiffCache/core_odes.jl") end
    @safetestset "ResizingDiffCache Resizing" begin include("ResizingDiffCache/core_resizing.jl") end
    @safetestset "ResizingDiffCache Nested Duals" begin include("ResizingDiffCache/core_nesteddual.jl") end
    @safetestset "ResizingDiffCache Sparsity Support" begin include("ResizingDiffCache/sparsity_support.jl") end
end

if !is_APPVEYOR && GROUP == "GPU"
    activate_downstream_env()
    @safetestset "GPU tests" begin include("DiffCache/gpu_all.jl") end
    @safetestset "GPU tests" begin include("ResizingDiffCache/gpu_all.jl") end
end
