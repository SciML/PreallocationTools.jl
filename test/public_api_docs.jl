using PreallocationTools, Test

function docstring_text(value)
    return sprint(show, MIME"text/plain"(), value)
end

function has_section(text, heading)
    return occursin("# $heading", text) || occursin("\n  $heading\n", text)
end

@testset "Public constructor documentation" begin
    constructors = (
        (@doc PreallocationTools.FixedSizeDiffCache),
        (@doc PreallocationTools.DiffCache),
        (@doc PreallocationTools.LazyBufferCache),
        (@doc PreallocationTools.GeneralLazyBufferCache),
    )

    for doc in constructors
        text = docstring_text(doc)
        @test has_section(text, "Arguments")
        @test has_section(text, "Fields")
        @test has_section(text, "Returns")
        @test has_section(text, "Examples")
    end
end

@testset "Generic get_tmp fallback" begin
    marker = Ref(1)
    @test get_tmp(marker, :unused) === marker
end
