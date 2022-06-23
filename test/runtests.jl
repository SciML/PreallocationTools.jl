using Pkg
using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")
const is_APPVEYOR = Sys.iswindows() && haskey(ENV, "APPVEYOR")

function activate_downstream_env()
    Pkg.activate("GPU")
    Pkg.develop(PackageSpec(; path = dirname(@__DIR__)))
    return Pkg.instantiate()
end

if GROUP == "All" || GROUP == "Core"
    @safetestset "Dispatch" begin
        include("core_dispatch.jl")
    end
    @safetestset "ODE tests" begin
        include("core_odes.jl")
    end
    @safetestset "Resizing" begin
        include("core_resizing.jl")
    end
    @safetestset "Nested Duals" begin
        include("core_nesteddual.jl")
    end
end

if !is_APPVEYOR && GROUP == "GPU"
    activate_downstream_env()
    @safetestset "GPU tests" begin
        include("gpu_all.jl")
    end
end
