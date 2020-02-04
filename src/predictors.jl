export Pade21

abstract type AbstractPredictor end
abstract type AbstractPredictorCache end

"""
    Pade21()

This uses a Padé-approximation of type (2,1) for prediction.
"""
struct Pade21 <: AbstractPredictor end

struct Pade21Cache <: AbstractPredictorCache
    x¹::Vector{ComplexF64}
    x²::Vector{ComplexF64}
    x³::Vector{ComplexF64}
    x⁴::Vector{ComplexF64}

    u::Vector{ComplexF64}
    dx¹::NTuple{1,Vector{ComplexF64}}
    dx²::NTuple{2,Vector{ComplexF64}}
    dx³::NTuple{3,Vector{ComplexF64}}

    taylor::Base.RefValue{Bool}
    order::Base.RefValue{Int}
    trust_region::Base.RefValue{Float64}
    err::Vector{ComplexF64}
end

function cache(::Pade21, n::Int)
    x¹ = zeros(ComplexF64, n)
    x² = zeros(ComplexF64, n)
    x³ = zero(x²)
    x⁴ = zero(x²)
    u = zero(x²)
    dx¹ = (x¹,)
    dx² = (x¹,x²)
    dx³ = (x¹,x²,x³)
    err = zero(u)
    Pade21Cache(x¹, x², x³, x⁴, u, dx¹, dx², dx³, Ref(false), Ref(4), Ref(NaN), err)
end

function update!(cache::Pade21Cache, H, x, t, J::Jacobian)
    # unpack stuff to make the rest easier to read
    @unpack u, x¹, x², x³, x⁴, dx¹, dx², dx³ = cache

    diff_t!(u, H, x, t)
    u .= .-u
    LA.ldiv!(x¹, J, u)

    diff_t!(u, H, x, t, dx¹)
    u .= .-u
    LA.ldiv!(x², J, u)

    diff_t!(u, H, x, t, dx²)
    u .= .-u
    LA.ldiv!(x³, J, u)

    trust_region = Inf
    for i in eachindex(x²)
        trust_region = min(trust_region, abs(x²[i]) / abs(x³[i]))
    end

    if trust_region < eps()
        cache.trust_region[] = Inf
        cache.err .= x³ ./ 6
        cache.taylor[] = true
        cache.order[] = 3
    else
        cache.taylor[] = false
        diff_t!(u, H, x, t, dx³)
        u .= .-u
        LA.ldiv!(x⁴, J, u)

        cache.trust_region[] = trust_region
        for i in eachindex(x²)
            cache.err[i] = x⁴[i] - Base.FastMath.div_fast(x³[i]^2, x²[i])
        end
        cache.order[] = 4
    end

    cache
end


function predict!(x̂, cache::Pade21Cache, H, x, t, Δt)
    @unpack x¹, x², x³ = cache

    if cache.taylor[]
        @inbounds for i in eachindex(x)
            x̂[i] = x[i] + Δt * (x¹[i] + Δt * x²[i])
        end
    else
        @inbounds for i in eachindex(x)
            δᵢ = 1 - Δt * x³[i] / x²[i]
            x̂[i] = x[i] + Δt * x¹[i] + Base.FastMath.div_fast(Δt^2 * x²[i], δᵢ)
        end
    end

    nothing
end

order(c::Pade21Cache) = c.order[]
local_error(cache::Pade21Cache) = cache.err
trust_region(cache::Pade21Cache) = cache.trust_region[]
