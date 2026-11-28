import Pkg
import TOML

using SafeTestsets
using SciMLTesting

# Enzyme 0.13.219+ breaks nested forward custom rules on Julia 1.10 (EnzymeAD/Enzyme.jl#4005).
function _pin_enzyme_for_lts!()
    VERSION < v"1.11" || return nothing
    enzyme = Base.locate_package(
        Base.PkgId(Base.UUID("7da242da-08ed-463a-9acd-ee780be4f1d9"), "Enzyme")
    )
    project = TOML.parsefile(joinpath(dirname(dirname(enzyme)), "Project.toml"))
    VersionNumber(project["version"]) >= v"0.13.219" || return nothing
    Pkg.pin(Pkg.PackageSpec(name = "Enzyme", version = v"0.13.218"))
    return nothing
end

run_tests(;
    core = () -> begin
        _pin_enzyme_for_lts!()
        @safetestset "Developer Interface" include("developer_interface.jl")
        @safetestset "DiffCache Dispatch" include("core_dispatch.jl")
        @safetestset "DiffCache ODE tests" include("core_odes.jl")
        @safetestset "DiffCache Resizing" include("core_resizing.jl")
        @safetestset "DiffCache Nested Duals" include("core_nesteddual.jl")
        @safetestset "DiffCache Sparsity Support" include("sparsity_support.jl")
        @safetestset "DiffCache with SparseConnectivityTracer" include("sparse_connectivity_tracer.jl")
        @safetestset "Enzyme Support" include("enzyme.jl")
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
