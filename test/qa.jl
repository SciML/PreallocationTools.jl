using PreallocationTools, Aqua
@testset "Aqua" begin
    Aqua.find_persistent_tasks_deps(PreallocationTools)
    Aqua.test_ambiguities(PreallocationTools, recursive = false)
    Aqua.test_deps_compat(PreallocationTools)
    Aqua.test_piracies(PreallocationTools)
    Aqua.test_project_extras(PreallocationTools)
    Aqua.test_stale_deps(PreallocationTools)
    Aqua.test_unbound_args(PreallocationTools)
    Aqua.test_undefined_exports(PreallocationTools)
end
