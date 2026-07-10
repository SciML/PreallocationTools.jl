module PreallocationToolsEnzymeCoreExt

using PreallocationTools: DiffCache, FixedSizeDiffCache, LazyBufferCache, get_tmp
using EnzymeCore: EnzymeRules, Annotation, Const, Duplicated, BatchDuplicated,
    MixedDuplicated, BatchMixedDuplicated, Active

# The buffers handed out by `get_tmp` are scratch space: the documented contract
# is that they are fully written within the function that fetches them before
# being read, and no data stored in them is relied upon across separate calls
# into the differentiated program. Under that contract, custom rules only need
# to hand Enzyme a shadow buffer with the same type, shape and aliasing
# structure as the primal buffer:
#
#   * If the cache itself is `Duplicated`/`BatchDuplicated` (e.g. it lives in a
#     `Duplicated` parameter struct), the shadow cache already exists and the
#     shadow buffer is simply `get_tmp(shadow_cache, u)`.
#   * If the cache is `Const` (e.g. it lives in a `Const` parameter struct or a
#     `Const` closure), a shadow buffer still has to come from somewhere. It
#     must NOT alias any memory the primal uses: in particular reusing
#     `dc.dual_du` would collide with ForwardDiff's dual buffer when Enzyme is
#     differentiating over ForwardDiff code. Instead, a persistent shadow cache
#     is kept per primal cache (weakly keyed so it is freed with the cache).
#     Persistence matters: two `get_tmp` calls on the same cache alias the same
#     primal buffer, so their shadows must alias each other as well.
#
# For `LazyBufferCache` the registry shadow is used for every annotation, not
# just `Const` — see the note at `ownshadow`.

const SupportedCache = Union{DiffCache, FixedSizeDiffCache, LazyBufferCache}

# Maps objectid(shadowkey(cache)) => (WeakRef(shadowkey(cache)), shadow caches,
# one per batch lane). Keying must be by identity — the key object is the
# primal buffer, which is mutated between lookups, so anything hash/isequal
# based (e.g. WeakKeyDict) would miss. The WeakRef guards against objectid
# reuse after the original key is garbage collected; dead entries are purged
# whenever a new cache is registered.
# Forward-mode and reverse-mode rule applications use disjoint lane vectors.
# Under nested differentiation, an outer pass applies these rules to the
# `get_tmp` calls inside the (already-transformed) inner rule bodies, so the
# same cache is resolved once per differentiation level; in mixed-mode nesting
# (forward-over-reverse, reverse-over-forward) the levels use different rule
# kinds, and separating the lanes by rule kind keeps the outer level's shadow
# of the primal buffer from aliasing the inner level's shadow. Reverse-mode
# shadows must persist across the augmented/reverse sweep boundary, so that
# aliasing would corrupt derivatives.
const CONST_CACHE_SHADOWS = Dict{UInt, Tuple{WeakRef, Vector{Any}, Vector{Any}}}()
const CONST_CACHE_SHADOWS_LOCK = ReentrantLock()

# Key on a mutable field that is shared by everything aliasing the same
# underlying storage.
shadowkey(dc::Union{DiffCache, FixedSizeDiffCache}) = dc.du
shadowkey(b::LazyBufferCache) = b.bufs

# `warn_on_resize = false`: the primal cache already warns if it is enlarged;
# a second warning from the hidden shadow cache would be confusing.
makeshadow(dc::DiffCache) = DiffCache(zero(dc.du), zero(dc.dual_du), Any[], false)
makeshadow(dc::FixedSizeDiffCache) = zero(dc)
zeroinit!(buf) = fill!(buf, zero(eltype(buf)))
makeshadow(b::LazyBufferCache) = LazyBufferCache(b.sizemap; initializer! = zeroinit!)

shadowfits(s::DiffCache, dc::DiffCache) = size(s.du) == size(dc.du)
shadowfits(s::FixedSizeDiffCache, dc::FixedSizeDiffCache) = size(s.du) == size(dc.du)
shadowfits(s::LazyBufferCache, b::LazyBufferCache) = true
shadowfits(s, dc) = false

function purgedeadshadows!()
    return filter!(CONST_CACHE_SHADOWS) do (_, entry)
        entry[1].value !== nothing
    end
end

# The invokelatest barrier keeps the table operations out of the code Enzyme
# compiles: rule bodies are processed by Enzyme's pipeline, which corrupts the
# statically-visible Dict mutation (observed as broken lookups on the second
# `get_tmp` of a differentiated call).
function constshadow(cache::SupportedCache, i::Int, rev::Bool)
    return Base.invokelatest(_constshadow, cache, i, rev)
end

function _constshadow(cache::SupportedCache, i::Int, rev::Bool)
    key = shadowkey(cache)
    id = objectid(key)
    return lock(CONST_CACHE_SHADOWS_LOCK) do
        entry = get(CONST_CACHE_SHADOWS, id, nothing)
        if entry === nothing || entry[1].value !== key
            purgedeadshadows!()
            entry = (WeakRef(key), Any[], Any[])
            CONST_CACHE_SHADOWS[id] = entry
        end
        lanes = rev ? entry[3] : entry[2]
        while length(lanes) < i
            push!(lanes, nothing)
        end
        s = lanes[i]
        if !shadowfits(s, cache)
            s = makeshadow(cache)
            lanes[i] = s
        end
        return s
    end
end

shadowcache(dc::Const, i::Int, rev::Bool) = constshadow(dc.val, i, rev)
shadowcache(dc::Duplicated, i::Int, rev::Bool) = ownshadow(dc.val, dc.dval, i, rev)
shadowcache(dc::BatchDuplicated, i::Int, rev::Bool) = ownshadow(dc.val, dc.dval[i], i, rev)
shadowcache(dc::MixedDuplicated, i::Int, rev::Bool) = ownshadow(dc.val, dc.dval[], i, rev)
function shadowcache(dc::BatchMixedDuplicated, i::Int, rev::Bool)
    return ownshadow(dc.val, dc.dval[i][], i, rev)
end

ownshadow(cache, dcache, i::Int, rev::Bool) = dcache
# For a duplicated LazyBufferCache the registry shadow is used instead of the
# user-provided one: creating the shadow buffer would have to insert into a
# Dict that Enzyme is tracking, and Dict mutation from inside a rule body
# corrupts the tracked shadow Dict via Enzyme's store mirroring (UndefRefError
# on Julia 1.10). Under the scratch-buffer contract the identity of the shadow
# buffer is unobservable, so this is a pure implementation detail.
ownshadow(cache::LazyBufferCache, dcache, i::Int, rev::Bool) = constshadow(cache, i, rev)

# True when the rule supplies the shadow buffer from the persistent registry
# (rather than from user-managed shadow memory).
ownsshadow(dc::Const) = true
ownsshadow(dc::Annotation) = dc.val isa LazyBufferCache

# Under runtime activity, Enzyme marks dynamically-inactive duplicated data by
# passing a shadow egal to the primal; rules must propagate that marker.
runtimeinactive(dc::Const) = false
runtimeinactive(dc::Duplicated) = dc.dval === dc.val
runtimeinactive(dc::BatchDuplicated) = dc.dval[1] === dc.val
runtimeinactive(dc::MixedDuplicated) = dc.dval[] === dc.val
runtimeinactive(dc::BatchMixedDuplicated) = dc.dval[1][] === dc.val

primaltmp(cache, u) = get_tmp(cache, u)
primaltmp(b::LazyBufferCache, u, s) = get_tmp(b, u, s)

shadowtmp(scache, u) = get_tmp(scache, u)
function shadowtmp(scache::LazyBufferCache, u, s = scache.sizemap(size(u)))
    fresh = !haskey(scache.bufs, (typeof(u), s))
    buf = get_tmp(scache, u, s)
    # A freshly-created shadow buffer must start at zero: the tangent/adjoint
    # of whatever the user's `initializer!` writes is zero, not the
    # initializer values themselves.
    fresh && zeroinit!(buf)
    return buf
end

function fwdrule(config, dc, args...)
    du = primaltmp(dc.val, args...)
    if !EnzymeRules.needs_shadow(config)
        return EnzymeRules.needs_primal(config) ? du : nothing
    end
    W = EnzymeRules.width(config)
    shadows = if EnzymeRules.runtime_activity(config) && runtimeinactive(dc)
        ntuple(Returns(du), Val(W))
    else
        ntuple(i -> shadowtmp(shadowcache(dc, i, false), args...)::typeof(du), Val(W))
    end
    return if EnzymeRules.needs_primal(config)
        W == 1 ? Duplicated(du, shadows[1]) : BatchDuplicated(du, shadows)
    else
        W == 1 ? shadows[1] : shadows
    end
end

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        ::Const{typeof(get_tmp)}, ::Type{<:Annotation},
        dc::Annotation{<:SupportedCache}, u::Annotation
    )
    return fwdrule(config, dc, u.val)
end

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        ::Const{typeof(get_tmp)}, ::Type{<:Annotation},
        b::Annotation{<:LazyBufferCache}, u::Annotation, s::Annotation
    )
    return fwdrule(config, b, u.val, s.val)
end

function augrule(config, dc, args...)
    du = primaltmp(dc.val, args...)
    shadow = if EnzymeRules.needs_shadow(config)
        W = EnzymeRules.width(config)
        shadows = if EnzymeRules.runtime_activity(config) && runtimeinactive(dc)
            ntuple(Returns(du), Val(W))
        else
            ntuple(Val(W)) do i
                sdu = shadowtmp(shadowcache(dc, i, true), args...)::typeof(du)
                # Registry-owned shadows may hold stale tangents from an
                # earlier forward-mode differentiation. Adjoint accumulation
                # only starts in the reverse sweep, after every augmented
                # primal has run, so zeroing here is safe.
                ownsshadow(dc) && zeroinit!(sdu)
                sdu
            end
        end
        W == 1 ? shadows[1] : shadows
    else
        nothing
    end
    return EnzymeRules.AugmentedReturn(
        EnzymeRules.needs_primal(config) ? du : nothing, shadow, nothing
    )
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        ::Const{typeof(get_tmp)}, ::Type{<:Annotation},
        dc::Annotation{<:SupportedCache}, u::Annotation
    )
    return augrule(config, dc, u.val)
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        ::Const{typeof(get_tmp)}, ::Type{<:Annotation},
        b::Annotation{<:LazyBufferCache}, u::Annotation, s::Annotation
    )
    return augrule(config, b, u.val, s.val)
end

# The returned buffer does not depend on the *value* of `u` (only on its type),
# so active arguments receive a zero adjoint.
zeroadjoint(config, ::Annotation) = nothing
function zeroadjoint(config, u::Active)
    W = EnzymeRules.width(config)
    dz = zero(u.val)
    return W == 1 ? dz : ntuple(Returns(dz), Val(W))
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        ::Const{typeof(get_tmp)}, dret, tape,
        dc::Annotation{<:SupportedCache}, u::Annotation
    )
    return (nothing, zeroadjoint(config, u))
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        ::Const{typeof(get_tmp)}, dret, tape,
        b::Annotation{<:LazyBufferCache}, u::Annotation, s::Annotation
    )
    return (nothing, zeroadjoint(config, u), zeroadjoint(config, s))
end

end
