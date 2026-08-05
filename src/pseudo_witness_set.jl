export pseudo_witness_set

# embed a LinearSubspace living on `coords` into the full `n`-dimensional space
# (zeros on all other columns), so it constrains only those coordinates
function _embed_subspace(L::LinearSubspace, coords, n::Integer)
    E = extrinsic(L)
    c = size(E.A, 1)
    A = zeros(eltype(E.A), c, n)
    A[:, coords] .= E.A
    LinearSubspace(A, E.b)
end

"""
    pseudo_witness_set(F, image_variables; dim_image, dim = nothing, options...)

Compute a pseudo-witness set for the image of the variety `V(F)` under the coordinate
projection onto `image_variables` — a subset of the variables of `F` (their coordinate
indices may be passed instead).

The result is a [`WitnessSet`](@ref) whose subspace is a [`ProductSubspace`](@ref) `L₁ × L₂`:
`L₁` is a generic slice of codimension `dim_image` on the projected coordinates and `L₂` a
generic slice of the fibre on the remaining coordinates. Its [`points`](@ref) are the witness
points of the image and its [`degree`](@ref) the degree of the image variety.

It is built by monodromy: the fibre slice `L₂` is held fixed while [`monodromy_solve`](@ref)
loops the image slice `L₁`. Because the projection need not be injective, only one
representative preimage per image point is kept, so the witness set stores — and every later
operation on it (moving the image slice, trace tests) tracks — as few solutions as possible.

* `dim_image`: the dimension `e` of the image variety `π(V(F))`.
* `dim`: the dimension `d` of `V(F)`; estimated as `corank(F)` if omitted. The fibre slice
  `L₂` gets codimension `d - e`.
* `certify = true`: run the [`trace_test`](@ref) on the result and warn if it does not pass.

Currently only affine (non-homogeneous) systems and coordinate projections are supported.

### Example
```julia-repl
julia> @var x y z;
julia> F = System([x^2 + y^2 + z^2 - 1]);   # the sphere, dim 2
julia> W = pseudo_witness_set(F, [x, y]; dim_image = 2);   # project onto (x, y)
julia> degree(W)   # image is the plane ℂ², degree 1; the 2-to-1 fibre is deduplicated
1
```
"""
function pseudo_witness_set(
    F::AbstractSystem,
    image_coords::AbstractVector{<:Integer};
    dim_image::Integer,
    dim::Union{Nothing,Integer} = nothing,
    certify::Bool = true,
    atol::Real = 1e-10,
    rtol::Real = 1e-8,
    options...,
)
    n = size(F, 2)
    is_homogeneous(System(F)) && error(
        "`pseudo_witness_set` currently supports only affine (non-homogeneous) systems.",
    )
    all(v -> 1 ≤ v ≤ n, image_coords) ||
        throw(ArgumentError("`image_coords` must be a subset of 1:$n."))

    d = isnothing(dim) ? corank(F) : dim
    e = dim_image
    0 ≤ e ≤ d ||
        throw(ArgumentError("`dim_image` must satisfy 0 ≤ dim_image ≤ dim (= $d)."))

    image_coords = collect(Int, image_coords)
    fibre_coords = setdiff(1:n, image_coords)

    # fix the fibre slice L₂ (baked into F′ = X ∩ L₂), leaving only the image slice to move.
    # With no fibre to cut (d == e) the slice is the whole fibre space (no equations).
    m₂ = length(fibre_coords)
    L₂ = d - e == 0 ? LinearSubspace(zeros(ComplexF64, 0, m₂)) : rand_subspace(m₂; codim = d - e)
    F′ = d - e == 0 ? F : slice(F, _embed_subspace(L₂, fibre_coords, n))

    # seed a point of X′ and an image-aligned slice through its image
    sp = find_start_pair(F′)
    isnothing(sp) &&
        error("`pseudo_witness_set`: could not find a start point on the variety.")
    x₀ = sp[1]
    A₁ = randn(ComplexF64, e, length(image_coords))
    L₁ = LinearSubspace(A₁, A₁ * x₀[image_coords])

    # `monodromy_solve` loops the image slice — its loops translate the slice, which keeps it
    # aligned to the image coordinates — and deduplicates by the image coordinates, so it
    # discovers one representative preimage per image point
    image_distance = (u, v) -> LA.norm(view(u, image_coords) - view(v, image_coords))
    mon = monodromy_solve(
        F′,
        [x₀],
        _embed_subspace(L₁, image_coords, n);
        distance = image_distance,
        unique_points_atol = atol,
        unique_points_rtol = rtol,
        options...,
    )
    W = WitnessSet(
        F,
        ProductSubspace(L₁, L₂, image_coords, fibre_coords),
        solutions(mon);
        projective = false,
    )

    if certify
        tr = trace_test(W)
        if isnothing(tr) || abs(tr) > 1e-6
            @warn "pseudo_witness_set: trace test not satisfied (trace = $tr); the witness " *
                  "set may be incomplete."
        end
    end
    W
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
