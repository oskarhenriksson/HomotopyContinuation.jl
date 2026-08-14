export pseudo_witness_set

# the dimension of the image of V(F) under the coordinate projection onto `image_coords`:
# the rank of the projection restricted to the tangent space of V(F) at a generic point
# (the analog of `corank` for ordinary witness sets)
function image_corank(F::AbstractSystem, image_coords)
    sp = find_start_pair(F)
    isnothing(sp) && error(
        "`image_corank`: could not find a point on the variety to estimate the image dimension.",
    )
    u = zeros(ComplexF64, size(F, 1))
    U = zeros(ComplexF64, size(F)...)
    evaluate_and_jacobian!(u, U, F, sp[1])
    LA.rank(LA.nullspace(U)[image_coords, :])
end

"""
    pseudo_witness_set(F, image_variables; dim_image = nothing, dim = nothing, options...)

Compute a pseudo-witness set for the image of the variety `V(F)` under the coordinate
projection onto `image_variables` — a subset of the variables of `F` (their coordinate
indices may be passed instead).

The result is a [`WitnessSet`](@ref) whose subspace is a [`ProductSubspace`](@ref) `L₁ × L₂`:
`L₁` is a generic slice of codimension `dim_image` on the projected coordinates and `L₂` a
generic slice of the fiber on the remaining coordinates. Its [`points`](@ref) are the witness
points of the image and its [`degree`](@ref) the degree of the image variety.

The witness points are all isolated points of `V(F) ∩ (L₁ × L₂)`, so every irreducible
component of the image is covered. Because the projection need not be injective, only one
representative preimage per image point is kept, so the witness set stores — and every later
operation on it (moving the image slice, trace tests) tracks — as few solutions as possible.

* `dim_image`: the dimension `e` of the image variety `π(V(F))`. If omitted it is computed
  as the rank of the projection restricted to the tangent space of `V(F)` at a generic point
  — of a single component, so pass it explicitly for a reducible variety whose components
  have images of different dimensions.
* `dim`: the dimension(s) of the components of `V(F)` to project — an integer or a vector of
  integers, for which the slices are computed with [`solve`](@ref), or `nothing` (default):
  the dimension-`e` part of the image may come from components of any dimension between `e`
  and `e + #fiber coordinates`, and a single [`regeneration`](@ref) pass of `F` together with
  the image slice covers them all at once. The fiber slice `L₂` gets codimension `d - e`.
  One witness set per dimension with witness points is returned — a plain `WitnessSet` if
  that is a single one, otherwise a vector.
* `certify = true`: run the [`trace_test`](@ref) on the result and warn if it does not pass.

A homogeneous `F` is handled as its affine cone: `dim` and `dim_image` are then projective
dimensions, the stored points are cone representatives of the projective witness points, and
`degree` is the degree of the projective image. Only coordinate projections are supported.

### Example
```julia-repl
julia> @var x y z;
julia> F = System([x^2 + y^2 + z^2 - 1]);   # the sphere, dim 2
julia> W = pseudo_witness_set(F, [x, y]; dim_image = 2);   # project onto (x, y)
julia> degree(W)   # image is the plane ℂ², degree 1; the 2-to-1 fiber is deduplicated
1
```
"""
function pseudo_witness_set(
    F::AbstractSystem,
    image_coords::AbstractVector{<:Integer};
    dim_image::Union{Nothing,Integer} = nothing,
    dim::Union{Nothing,Integer,AbstractVector{<:Integer}} = nothing,
    certify::Bool = true,
    atol::Real = 1e-10,
    rtol::Real = 1e-8,
    options...,
)
    n = size(F, 2)
    projective = is_homogeneous(System(F))
    all(v -> 1 ≤ v ≤ n, image_coords) ||
        throw(ArgumentError("`image_coords` must be a subset of 1:$n."))

    image_coords = collect(Int, image_coords)
    fiber_coords = setdiff(1:n, image_coords)
    m₂ = length(fiber_coords)

    # a homogeneous system is handled as its affine cone: the given projective dimensions
    # are shifted by one and everything below stays affine. The dimension-e part of the
    # image may come from components of V(F) of any dimension between e and e + m₂ (the
    # fiber of a coordinate projection lives in the fiber coordinates), so by default a
    # slice for every possible dimension is computed.
    e = isnothing(dim_image) ? image_corank(F, image_coords) : dim_image + projective

    # keep one representative preimage per image point: two preimages agreeing in the image
    # coordinates are the same image point
    image_distance = (u, v) -> LA.norm(view(u, image_coords) - view(v, image_coords))
    keep_one_per_fiber = sols -> begin
        S = empty(sols)
        if !isempty(sols)
            seen = UniquePoints(sols[1], 1; distance = image_distance)
            for (i, s) in enumerate(sols)
                _, new_point = add!(seen, s, i; atol = atol, rtol = rtol)
                new_point && push!(S, s)
            end
        end
        S
    end

    Ws = Vector{WitnessSet}()
    if isnothing(dim)
        # baking the image slice L₁ into the system, the dimension levels of
        # Y = V(F) ∩ π⁻¹(L₁) are exactly the possible fiber dimensions, so a single
        # `regeneration` pass covers components of V(F) of every dimension at once
        L₁ = LinearSubspace(randn(ComplexF64, e, length(image_coords)), randn(ComplexF64, e))
        E₁ = extrinsic(L₁)
        vars = variables(System(F))
        G = System([expressions(System(F)); E₁.A * vars[image_coords] - E₁.b], vars)
        for W in regeneration(G; show_progress = false, options...)
            c = codim(linear_subspace(W))    # the fiber dimension; source dimension e + c
            c ≤ m₂ || continue               # larger fibers belong to higher-dimensional images
            S = keep_one_per_fiber(solutions(W))
            isempty(S) && continue
            # move the representatives onto a fiber-aligned slice — the image slice is part
            # of G, so the start and target slices both have codimension c
            P₂ = rand_subspace(image_coords, fiber_coords; codim₁ = 0, codim₂ = c)
            if c > 0
                res = solve(
                    G,
                    S;
                    start_subspace = linear_subspace(W),
                    target_subspace = LinearSubspace(P₂),
                )
                S = solutions(res)
                isempty(S) && continue
            end
            P = ProductSubspace(L₁, P₂.L₂, image_coords, fiber_coords)
            push!(Ws, WitnessSet(F, P, S; projective = false))
        end
    else
        # every component of V(F) has dimension ≥ n - #equations, so smaller slices are empty
        dim_min = n - size(F, 1)
        dims = dim .+ projective
        all(d -> e ≤ d ≤ e + m₂, dims) || throw(
            ArgumentError(
                "`dim` must satisfy dim_image ≤ dim ≤ dim_image + $(m₂ - projective).",
            ),
        )
        for d in dims
            # a generic product slice: codimension e on the image coordinates, d - e on the fiber
            P = rand_subspace(image_coords, fiber_coords; codim₁ = e, codim₂ = d - e)

            # compute all isolated points of X ∩ (L₁ × L₂) on the flattened product slice —
            # the polyhedral start system reaches every irreducible component
            sols = if d < dim_min
                Vector{Vector{ComplexF64}}()
            else
                res = solve(F; target_subspace = LinearSubspace(P), options...)
                solutions(res; only_nonsingular = true)
            end
            S = keep_one_per_fiber(sols)
            # dimensions without witness points are dropped (unless explicitly requested alone)
            isempty(S) && !(dim isa Integer) && continue
            push!(Ws, WitnessSet(F, P, S; projective = false))
        end
    end

    if certify
        for W in Ws
            degree(W) == 0 && continue
            tr = trace_test(W)
            if isnothing(tr) || abs(tr) > 1e-6
                @warn "pseudo_witness_set: trace test not satisfied (trace = $tr) for " *
                      "dimension $(codim(W.L) - projective); the witness set may be incomplete."
            end
        end
    end
    length(Ws) == 1 ? only(Ws) : Ws
end

pseudo_witness_set(F::System, image; compile = COMPILE_DEFAULT[], kwargs...) =
    pseudo_witness_set(fixed(F; compile = compile), image; kwargs...)
pseudo_witness_set(F::Vector{Expression}, image; kwargs...) =
    pseudo_witness_set(System(F), image; kwargs...)

# the projected variables may be given directly, and are mapped to their coordinate indices
function _image_indices(F::System, image_variables)
    vars = variables(F)
    map(image_variables) do v
        i = findfirst(isequal(v), vars)
        isnothing(i) && throw(ArgumentError("$v is not a variable of the system"))
        i
    end
end
pseudo_witness_set(F::System, image_variables::AbstractVector{<:Variable}; kwargs...) =
    pseudo_witness_set(F, _image_indices(F, image_variables); kwargs...)
pseudo_witness_set(F::AbstractSystem, image_variables::AbstractVector{<:Variable}; kwargs...) =
    pseudo_witness_set(F, _image_indices(System(F), image_variables); kwargs...)
