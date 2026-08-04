export pseudo_witness_set, CoordinateNorm

"""
    CoordinateNorm(coordinates, norm = EuclideanNorm())

An [`AbstractNorm`](@ref) that measures only the given `coordinates` of a vector, using
`norm` on that sub-vector. Comparing solutions with a `CoordinateNorm` on the projected
(image) coordinates makes two preimages lying over the same image point have distance zero,
which is how [`pseudo_witness_set`](@ref) keeps one representative per fibre.
"""
struct CoordinateNorm{N<:AbstractNorm} <: AbstractNorm
    coordinates::Vector{Int}
    norm::N
end
CoordinateNorm(coordinates::AbstractVector{<:Integer}) =
    CoordinateNorm(collect(Int, coordinates), EuclideanNorm())

distance(u, v, N::CoordinateNorm) =
    distance(view(u, N.coordinates), view(v, N.coordinates), N.norm)
LinearAlgebra.norm(u, N::CoordinateNorm) = norm(view(u, N.coordinates), N.norm)

# preimage vector of a stored witness (a `PathResult` or a raw solution vector)
_preimage(r::PathResult) = solution(r)
_preimage(r::AbstractVector) = r

# keep one representative per distinct image point, comparing only the image coordinates
function _unique_by_image(R, image_vars; atol::Real = 1e-14, rtol::Real = 1e-8)
    isempty(R) && return R
    imgnorm = CoordinateNorm(image_vars)
    seen = UniquePoints(_preimage(first(R)), 1; distance = imgnorm)
    reps = empty(R)
    for r in R
        _, isnew = add!(seen, _preimage(r), length(reps) + 1; atol = atol, rtol = rtol)
        isnew && push!(reps, r)
    end
    reps
end

# a random slice of codimension `c` in `m`-dimensional space,
# handling the two degenerate cases (no slice, and slice down to a point)
_rand_slice(m::Integer, c::Integer) =
    c == 0 ? LinearSubspace(zeros(ComplexF64, 0, m)) :
    c == m ? LinearSubspace(Matrix{ComplexF64}(LA.I, m, m), randn(ComplexF64, m)) :
    rand_subspace(m; codim = c)

# embed a LinearSubspace living on coordinates `vars` into the full `n`-dimensional space
# (zeros on all other columns), so it constrains only those coordinates
function _embed_subspace(L::LinearSubspace, vars, n::Integer)
    E = extrinsic(L)
    c = size(E.A, 1)
    A = zeros(eltype(E.A), c, n)
    A[:, vars] .= E.A
    LinearSubspace(A, E.b)
end

# track the start solutions `S` from subspace `Ls` to `Lt`; copy to avoid buffer aliasing
_track(F, S, Ls, Lt; options...) = copy.(
    solutions(
        solve(F, S; start_subspace = Ls, target_subspace = Lt, show_progress = false, options...),
    ),
)

# Build a pseudo-witness set by image-monodromy: bake the fibre slice `L₂` into the system
# and loop only the image slice `L₁`, tracking one representative per image point. Because
# `L₂` is fixed and the image slices are zero on the fibre columns, the homotopy moves the
# image constraints only, so each loop tracks just `deg(image)` paths — never the full
# preimage. Returns the representative preimages and the base slices `(L₁, L₂)`.
function _image_monodromy(
    F,
    image_vars,
    fibre_vars,
    e::Integer,
    d::Integer;
    max_no_progress::Int = 10,
    degree1_no_progress::Int = 15,
    atol::Real = 1e-10,
    rtol::Real = 1e-8,
    options...,
)
    n = size(F, 2)
    m₁ = length(image_vars)
    imgnorm = CoordinateNorm(image_vars)

    # fix the fibre slice L₂, baked into the system as F' = X ∩ L₂
    if d - e == 0
        F′ = F
        L₂ = LinearSubspace(zeros(ComplexF64, 0, length(fibre_vars)))
    else
        L₂ = rand_subspace(length(fibre_vars); codim = d - e)
        F′ = slice(F, _embed_subspace(L₂, fibre_vars, n))
    end

    # cheap seed: one point of X' (no full solve)
    sp = find_start_pair(F′)
    isnothing(sp) &&
        error("`pseudo_witness_set`: could not find a start point on the variety.")
    x₀ = copy(sp[1])

    # base image slice through the seed's image
    A₁ = randn(ComplexF64, e, m₁)
    L₁ = LinearSubspace(A₁, A₁ * x₀[image_vars])
    L₁₀ = _embed_subspace(L₁, image_vars, n)
    G = slice(F′, L₁₀)                                   # square system for refinement
    refine(x) = (r = newton(G, x); is_success(r) ? copy(r.x) : copy(x))

    seen = UniquePoints(x₀, 1; distance = imgnorm)
    add!(seen, x₀, 1; atol = atol, rtol = rtol)          # the constructor does NOT insert
    reps = [x₀]
    noprog = 0
    while true
        La = _embed_subspace(_rand_slice(m₁, e), image_vars, n)
        Lb = _embed_subspace(_rand_slice(m₁, e), image_vars, n)
        ends = _track(
            F′,
            _track(F′, _track(F′, reps, L₁₀, La; options...), La, Lb; options...),
            Lb,
            L₁₀;
            options...,
        )
        newc = 0
        for x in ends
            xr = refine(x)
            _, isnew = add!(seen, xr, length(reps) + 1; atol = atol, rtol = rtol)
            if isnew
                push!(reps, xr)
                newc += 1
            end
        end
        noprog = newc == 0 ? noprog + 1 : 0
        # A single image point trivially passes the trace test, so (following the same guard
        # as `numerical_irreducible_decomposition`) don't trust a degree-1 result until many
        # more fruitless loops than for higher degrees.
        stop_at = length(reps) == 1 ? degree1_no_progress : max_no_progress
        noprog ≥ stop_at && break
    end
    reps, L₁, L₂
end

"""
    pseudo_witness_set(F, image_vars; dim_image, dim = nothing, options...)

Compute a pseudo-witness set for the image of the variety `V(F)` under the coordinate
projection onto the variables `image_vars` (a subset of `1:nvariables(F)`).

The result is a [`WitnessSet`](@ref) whose subspace is a [`ProductSubspace`](@ref)
`L₁ × L₂`: `L₁` is a generic slice of codimension `dim_image` on the projected coordinates
`image_vars`, and `L₂` a generic slice of the fibre on the remaining coordinates. Its
[`points`](@ref) are the witness points of the image and its [`degree`](@ref) the degree of
the image variety.

Because the projection need not be injective, several preimages may lie over the same image
point; only one representative preimage per image point is kept (compared with a
[`CoordinateNorm`](@ref) on `image_vars`), so the witness set stores as few solutions as
possible and later homotopies track as few paths as possible.

* `dim_image`: the dimension `e` of the image variety `π(V(F))` — "give the dimension of the
  projected variety".
* `dim`: the dimension `d` of `V(F)`. If omitted it is estimated as `corank(F)`. The fibre
  slice `L₂` gets codimension `d - e`. Pass both to control both numbers explicitly.

Only one representative per fibre is *stored*, so every later operation on `W` (moving the
image slice, trace tests) tracks only `degree(W)` paths.

* `strategy = :monodromy` (default): image-monodromy — loops only the image slice `L₁`,
  never enumerating the full preimage, so the build itself tracks about `degree(W)` paths
  per loop instead of `deg(V(F))`. This is the win when fibres are large. Completeness is
  heuristic (stops after `max_no_progress` loops with no new points) and certified with the
  [`trace_test`](@ref) when `certify = true`.
* `strategy = :solve`: build the flattened product slice and call `solve` once (like
  `witness_set`), then keep one representative per fibre. Complete by construction, but
  tracks all `deg(V(F))` paths.
* `certify = true`: run the pseudo `trace_test` on the result and warn if it does not pass.

Currently only affine (non-homogeneous) systems and coordinate projections are supported.

### Example
```julia-repl
julia> @var x y z;
julia> F = System([x^2 + y^2 + z^2 - 1]);   # the sphere, dim 2
julia> W = pseudo_witness_set(F, [1, 2]; dim_image = 2);   # project onto (x, y)
julia> degree(W)   # image is the plane ℂ², degree 1; the 2-to-1 fibre is deduplicated
1
```
"""
function pseudo_witness_set(
    F::AbstractSystem,
    image_vars::AbstractVector{<:Integer};
    dim_image::Integer,
    dim::Union{Nothing,Integer} = nothing,
    strategy::Symbol = :monodromy,
    certify::Bool = true,
    atol::Real = 1e-10,
    rtol::Real = 1e-8,
    options...,
)
    n = size(F, 2)
    f = System(F)
    is_homogeneous(f) && error(
        "`pseudo_witness_set` currently supports only affine (non-homogeneous) systems.",
    )
    all(v -> 1 ≤ v ≤ n, image_vars) ||
        throw(ArgumentError("`image_vars` must be a subset of 1:$n."))

    d = isnothing(dim) ? corank(F) : dim
    e = dim_image
    0 ≤ e ≤ d ||
        throw(ArgumentError("`dim_image` must satisfy 0 ≤ dim_image ≤ dim (= $d)."))

    image_vars = collect(Int, image_vars)
    fibre_vars = setdiff(1:n, image_vars)

    if strategy == :monodromy
        reps, L₁, L₂ = _image_monodromy(
            F, image_vars, fibre_vars, e, d; atol = atol, rtol = rtol, options...,
        )
        W = WitnessSet(
            F, ProductSubspace(L₁, L₂, image_vars, fibre_vars), reps; projective = false,
        )
    elseif strategy == :solve
        L = ProductSubspace(
            _rand_slice(length(image_vars), e),
            _rand_slice(length(fibre_vars), d - e),
            image_vars,
            fibre_vars,
        )
        Lflat = LinearSubspace(L)
        # `solve` for polynomial systems, otherwise a monodromy start followed by `solve`
        if is_polynomial(f)
            res = solve(F; target_subspace = Lflat, options...)
        else
            mon = monodromy_solve(F; dim = codim(Lflat))
            res = solve(
                F,
                solutions(mon);
                start_subspace = parameters(mon),
                target_subspace = Lflat,
                options...,
            )
        end
        reps = _unique_by_image(
            results(res; only_nonsingular = true), image_vars; atol = atol, rtol = rtol,
        )
        W = WitnessSet(F, L, reps; projective = false)
    else
        throw(ArgumentError("`strategy` must be :monodromy or :solve, got :$strategy."))
    end

    if certify
        tr = trace_test(W)
        if isnothing(tr) || abs(tr) > 1e-6
            @warn "pseudo_witness_set: trace test not satisfied (trace = $tr); the witness " *
                  "set may be incomplete. Try `strategy = :solve` or a larger `max_no_progress`."
        end
    end
    W
end

pseudo_witness_set(F::System, image_vars; compile = COMPILE_DEFAULT[], kwargs...) =
    pseudo_witness_set(fixed(F; compile = compile), image_vars; kwargs...)
pseudo_witness_set(F::Vector{Expression}, image_vars; kwargs...) =
    pseudo_witness_set(System(F), image_vars; kwargs...)

"""
    trace_test(W::WitnessSet{<:Any,<:ProductSubspace,<:Any}; options...)

Trace test for a pseudo-witness set: translate only the image slice `L₁` (keeping the fibre
slice `L₂` fixed), track the stored preimages, and check that the **image** points move
linearly. Returns a value that is ≈ 0 iff the pseudo-witness set is complete (all image
witness points are present), or `nothing` if a move loses points.
"""
function trace_test(W₀::WitnessSet{<:Any,<:ProductSubspace,<:Any}; options...)
    W₀.projective &&
        error("`trace_test` for pseudo-witness sets currently supports only affine systems.")
    P = W₀.L
    F = system(W₀)
    S₀ = solutions(W₀)
    L₀ = LinearSubspace(P)
    image(s) = s[P.vars₁]
    s₀ = sum(points(W₀))

    # translate the image slice L₁ only, keeping the fibre slice L₂ fixed
    v = randn(ComplexF64, codim(P.L₁))
    L₁ = LinearSubspace(ProductSubspace(translate(P.L₁, v), P.L₂, P.vars₁, P.vars₂))
    L₋₁ = LinearSubspace(ProductSubspace(translate(P.L₁, -v), P.L₂, P.vars₁, P.vars₂))

    R₁ = solve(F, S₀; start_subspace = L₀, target_subspace = L₁, options...)
    nsolutions(R₁) == degree(W₀) || return nothing
    R₋₁ = solve(F, S₀; start_subspace = L₀, target_subspace = L₋₁, options...)
    nsolutions(R₋₁) == degree(W₀) || return nothing

    s₁ = sum(image, solutions(R₁))
    s₋₁ = sum(image, solutions(R₋₁))

    # The sum of the image points is affine-linear in the slice translation iff the
    # pseudo-witness set is complete, so the symmetric second difference vanishes. This form
    # stays valid when the image is 1-dimensional (scalar image points).
    LA.norm(s₋₁ - 2 .* s₀ + s₁) / (LA.norm(s₋₁) + LA.norm(s₀) + LA.norm(s₁) + eps())
end
