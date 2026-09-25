using PreallocationTools, BenchmarkTools
using ForwardDiff, StableRNGs

const SUITE = BenchmarkGroup()
const rng = StableRNG(123)

u0 = rand(rng, 500)
duals = ForwardDiff.Dual.(u0, rand(rng, 500))

# =============================================================================
# LazyBufferCache / dualcache
# =============================================================================

SUITE["dualcache"] = BenchmarkGroup()

SUITE["dualcache"]["construct"] = @benchmarkable dualcache($u0)
dc = dualcache(u0)
SUITE["dualcache"]["get_tmp_float"] = @benchmarkable get_tmp($dc, 1.0)
SUITE["dualcache"]["get_tmp_dual"] = @benchmarkable get_tmp($dc, $(duals[1]))

# =============================================================================
# FixedSizeDiffCache
# =============================================================================

SUITE["fixed_size"] = BenchmarkGroup()

fdc = FixedSizeDiffCache(u0)
SUITE["fixed_size"]["construct"] = @benchmarkable FixedSizeDiffCache($u0)
SUITE["fixed_size"]["get_tmp_float"] = @benchmarkable get_tmp($fdc, 1.0)
SUITE["fixed_size"]["get_tmp_dual"] = @benchmarkable get_tmp($fdc, $(duals[1]))

# =============================================================================
# Realistic workload: in-place function over a DiffCache
# =============================================================================

SUITE["workload"] = BenchmarkGroup()

function f_cache!(out, u, cache)
    tmp = get_tmp(cache, u)
    @. tmp = sin(u) * cos(u)
    @. out = tmp + tmp^2
    return nothing
end

out = similar(u0)
SUITE["workload"]["float"] = @benchmarkable f_cache!($out, $u0, $dc)
SUITE["workload"]["dual"] = @benchmarkable f_cache!(
    od, ud, $dc
) setup = (od = similar(u0, eltype(duals)); ud = duals)
