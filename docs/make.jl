using Documenter, PreallocationTools

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

include("pages.jl")

makedocs(sitename = "PreallocationTools.jl",
    authors = "Chris Rackauckas",
    modules = [PreallocationTools],
    clean = true, doctest = false, linkcheck = true,
    format = Documenter.HTML(assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/PreallocationTools/stable/"),
    pages = pages)

deploydocs(repo = "github.com/SciML/PreallocationTools.jl.git"; push_preview = true)
