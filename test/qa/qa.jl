using SciMLTesting, PreallocationTools, Test

run_qa(
    PreallocationTools;
    explicit_imports = true,
    aqua_kwargs = (; ambiguities = (; recursive = false)),
    ei_kwargs = (;
        # The package extensions load (and are analyzed by ExplicitImports) whenever
        # their weakdeps are present in the session, e.g. on the OS lanes where QA runs
        # alongside Core. The entries below cover the extension modules' legitimate uses.
        no_implicit_imports = (;
            ignore = (
                :PreallocationTools,           # extension parent package (used qualified)
                :ForwardDiff,                  # weakdep ForwardDiff (used qualified)
                :Adapt,                        # Adapt (extension dependency)
                :ArrayInterface,               # ArrayInterface (extension dependency)
                :PrecompileTools,              # PrecompileTools (extension dependency)
                Symbol("@compile_workload"),   # PrecompileTools macro
                Symbol("@setup_workload"),     # PrecompileTools macro
            ),
        ),
        # Qualified accesses to non-public names. The package's own internals are
        # extended across the package/extension boundary, and the base libraries'
        # names are de-facto stable; ignore until they are declared public.
        all_qualified_accesses_are_public = (;
            ignore = (
                :depwarn,                       # Base.depwarn
                :parameterless_type,            # ArrayInterface.parameterless_type
                :restructure,                   # ArrayInterface.restructure
                :_restructure,                  # PreallocationTools internal (extended in ext)
                :chunksize,                     # PreallocationTools internal (extended in ext)
                :dualarraycreator,              # PreallocationTools internal (extended in ext)
                :enlargediffcache!,             # PreallocationTools internal (extended in ext)
                :forwarddiff_compat_chunk_size, # PreallocationTools internal (extended in ext)
                :Dual,                          # ForwardDiff.Dual (non-public, stable)
                :pickchunksize,                 # ForwardDiff.pickchunksize (non-public, stable)
            ),
        ),
        # Explicit imports of non-public names from SparseConnectivityTracer.
        all_explicit_imports_are_public = (;
            ignore = (
                :AbstractTracer,  # SparseConnectivityTracer.AbstractTracer (non-public)
                :Dual,            # SparseConnectivityTracer.Dual (non-public)
            ),
        ),
    ),
)

@testset "AllocCheck" begin
    include("allocation_tests.jl")
end
