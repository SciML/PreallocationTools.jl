using SciMLTesting, PreallocationTools, Test

run_qa(
    PreallocationTools;
    explicit_imports = true,
    aqua_kwargs = (; ambiguities = (; recursive = false)),
    ei_kwargs = (;
        # Qualified accesses to non-public names of other packages. These go public
        # as the base libraries declare them so; ignore until then.
        all_qualified_accesses_are_public = (;
            ignore = (
                :depwarn,             # Base.depwarn
                :parameterless_type,  # ArrayInterface.parameterless_type
                :restructure,         # ArrayInterface.restructure
            ),
        ),
    ),
)
