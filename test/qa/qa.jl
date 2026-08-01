using SciMLTesting, PreallocationTools, Test

# ExplicitImports only checks an extension module once it has been loaded, which
# requires its trigger package to be present. Loading every weakdep here puts all
# four extensions under the QA checks.
using EnzymeCore, ForwardDiff, ReverseDiff, SparseConnectivityTracer

run_qa(
    PreallocationTools;
    ei_kwargs = (;
        all_explicit_imports_are_public = (;
            ignore = (
                # EnzymeCore neither exports nor declares `EnzymeRules` public,
                # yet the submodule is the only entry point for defining Enzyme
                # custom rules.
                :EnzymeRules,
                # SparseConnectivityTracer exports only its detectors and
                # sparsity entry points. `AbstractTracer` and `Dual` are the
                # types a `get_tmp` method has to dispatch on for sparsity
                # detection to reach the cache, and neither has a public name.
                :AbstractTracer, :Dual,
            ),
        ),
        all_qualified_accesses_are_public = (;
            ignore = (
                # `EnzymeCore.EnzymeRules` declares `forward`,
                # `augmented_primal` and `reverse` as bare `function ... end`
                # stubs without exporting them. Adding methods to them is the
                # documented way to write a custom rule.
                :forward, :augmented_primal, :reverse,
                # ForwardDiff exports only `DiffResults`. `Dual` is the type
                # every dual-cache method dispatches on and `pickchunksize` is
                # the chunk heuristic the cache sizing mirrors.
                :Dual, :pickchunksize,
                # ReverseDiff likewise exports only `DiffResults`;
                # `TrackedArray` is the type the LazyBufferCache method keys on.
                :TrackedArray,
                # `Base.typename` is the only way to recover a DataType's
                # UnionAll wrapper so its type parameters can be substituted;
                # Base offers no public equivalent.
                :typename,
            ),
        ),
    )
)

@testset "AllocCheck" begin
    include("allocation_tests.jl")
end
