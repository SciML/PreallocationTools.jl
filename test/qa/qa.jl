using SciMLTesting, PreallocationTools, Test

run_qa(PreallocationTools)

@testset "AllocCheck" begin
    include("allocation_tests.jl")
end
