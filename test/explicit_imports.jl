using ExplicitImports
using PreallocationTools
using Test

@testset "Explicit Imports" begin
    @test check_no_implicit_imports(PreallocationTools) === nothing
    @test check_no_stale_explicit_imports(PreallocationTools) === nothing
end
