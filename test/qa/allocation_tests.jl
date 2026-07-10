using AllocCheck, PreallocationTools, Test

_qa_reshape_varargs(cache, rows, cols) = reshape(cache, rows, cols)
_qa_reshape_tuple(cache, dims) = reshape(cache, dims)
_qa_get_tmp_number(cache, x) = get_tmp(cache, x)
_qa_get_tmp_type(cache, ::Type{T}) where {T} = get_tmp(cache, T)
_qa_first_tmp_value(cache, x) = get_tmp(cache, x)[1]

function _qa_test_no_allocs(f, argtypes)
    allocs = check_allocs(f, argtypes)
    @test isempty(allocs)
    return allocs
end

@testset "reshaped DiffCache" begin
    storage = DiffCache(collect(1.0:10.0), 2)
    reshaped = reshape(storage, 2, 5)

    parent = collect(1.0:10.0)
    view_storage = DiffCache(view(parent, 1:10), 2)
    view_reshaped = reshape(view_storage, 2, 5)

    _qa_test_no_allocs(_qa_reshape_varargs, Tuple{typeof(storage), Int, Int})
    _qa_test_no_allocs(_qa_reshape_tuple, Tuple{typeof(storage), Tuple{Int, Int}})
    _qa_test_no_allocs(_qa_reshape_varargs, Tuple{typeof(view_storage), Int, Int})

    _qa_test_no_allocs(_qa_get_tmp_number, Tuple{typeof(reshaped), Float64})
    _qa_test_no_allocs(_qa_get_tmp_type, Tuple{typeof(reshaped), Type{Float64}})
    _qa_test_no_allocs(_qa_first_tmp_value, Tuple{typeof(reshaped), Float64})
    _qa_test_no_allocs(_qa_get_tmp_number, Tuple{typeof(view_reshaped), Float64})
end
