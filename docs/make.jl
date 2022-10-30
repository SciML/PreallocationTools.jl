using Documenter, PreallocationTools

include("pages.jl")

makedocs(sitename = "PreallocationTools.jl",
         authors = "Chris Rackauckas",
         modules = [PreallocationTools],
         clean = true,
         doctest = false,
         format = Documenter.HTML(assets = ["assets/favicon.ico"],
                                  canonical = "https://docs.sciml.ai/PreallocationTools/stable/"),
         pages = pages)

deploydocs(repo = "github.com/SciML/PreallocationTools.jl.git"; push_preview = true)
