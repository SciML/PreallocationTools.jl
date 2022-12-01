using Symbolics, PreallocationTools, ForwardDiff, SparseArrays, Test

struct GlobalDAE{T}
    x::T  # array of variables
    f::T  # array of residuals
end

dae_cache = GlobalDAE(ResizingDiffCache([0.0, 0.0]), ResizingDiffCache([0.0, 0.0]));

# multiple residual functions that collectively build the residual array
function res1!(dae::GlobalDAE, u)
    f = get_tmp(dae.f, u)
    x = get_tmp(dae.x, u)
    @views f[1] = 2 * x[1] * x[1]
    nothing
end

function res2!(dae::GlobalDAE, u)
    f = get_tmp(dae.f, u)
    x = get_tmp(dae.x, u)
    @views f[2] = 2 * x[2] * x[2]
    nothing
end

# A dispatch function to set the guess from the solver, call the residual
# functions, and write to the residual array
function res_dae_cache!(dae, output, x0)
    get_tmp(dae.x, x0) .= x0   # <- errors here for x0 = Num[...]

    res1!(dae, x0)
    res2!(dae, x0)

    output .= get_tmp(dae.f, x0)
    nothing
end

wrap_res_dae_cache! = (output, x0) -> res_dae_cache!(dae_cache, output, x0)

x0 = [1.0, 2.0];
output = similar(x0);
res_dae_cache!(dae_cache, output, x0)

ForwardDiff.jacobian(wrap_res_dae_cache!, output, x0)  # works

A = Symbolics.jacobian_sparsity(wrap_res_dae_cache!, output, x0)
@test A isa SparseArrays.SparseMatrixCSC
@test nnz(A) == 2
B = sparse([1 0; 0 1])
@test A == B
@test findnz(A) == findnz(B)
