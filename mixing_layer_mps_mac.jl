#!/usr/bin/env julia

"""
Bosonic-MPS solver for a periodic 2-D mixing layer using a staggered
marker-and-cell (MAC) grid and Chorin's projection method.

The coherent amplitudes are interleaved cell-by-cell along a serpentine MPS:

    [u-face, v-face, cell pressure impulse, conserved scalar] x (nx*ny)

For a coherent state, a generator of the form

    H = sum_i adag_i * F_i(a)

advances its amplitudes according to d(alpha_i)/dt = F_i(alpha).  Three
non-Hermitian MPOs therefore implement the Chorin step:

1. a pressure-free conservative MAC momentum and passive-scalar predictor
   (stored as an exact row-chunked sum of smaller MPOs to avoid a costly
   monolithic conversion);
2. pseudo-time relaxation of laplacian(phi) = divergence(u_star), phi=dt*p;
3. the face-velocity correction u = u_star - gradient(phi).

This is a coarse 32x32 MPS demonstration, not a resolved DNS. The default
Re=100 and strengthened KH seed match the classical visible-mixing reference.
"""

using ITensors
using ITensorMPS
using LinearAlgebra
using Statistics
using Printf
using JLD2
using HDF5
using SHA
using Serialization

const MAX_BOSON = parse(Int, get(ENV, "MPS_MAX_BOSON", "4"))
const OPERATOR_CACHE_VERSION = 2
const OPERATOR_DEFINITION_VERSION = 2
const LEGACY_OPERATOR_SOURCE_DIGESTS = ()
const CHECKPOINT_VERSION = 3
const CHECKPOINT_MANIFEST_FORMAT = "mixing-layer-mps-checkpoint-manifest-v1"
const CHECKPOINT_GENERATIONS_TO_KEEP = 3
const PRESSURE_GAUGE_TARGET = 1.0e-6
const PRESSURE_GAUGE_SCALE_FLOOR = 1.0e-10
const SCALAR_MASS_TOLERANCE = 1.0e-4


# ---------------------------------------------------------------------------
# Truncated bosonic site type (same coherent-state construction as ldc.jl)
# ---------------------------------------------------------------------------

function ITensors.space(::SiteType"MACBoson"; conserve_qns=true)
    if conserve_qns
        return [QN("nb", n) => 1 for n in 0:MAX_BOSON]
    end
    return MAX_BOSON + 1
end

function ITensors.op!(operator::ITensor, ::OpName"Num", ::SiteType"MACBoson", s::Index)
    for n in 1:MAX_BOSON
        operator[s' => n + 1, s => n + 1] = n
    end
end

function ITensors.op!(operator::ITensor, ::OpName"a", ::SiteType"MACBoson", s::Index)
    for n in 1:MAX_BOSON
        operator[s' => n, s => n + 1] = sqrt(n)
    end
end

function ITensors.op!(operator::ITensor, ::OpName"adag", ::SiteType"MACBoson", s::Index)
    for n in 1:MAX_BOSON
        operator[s' => n + 1, s => n] = sqrt(n)
    end
end

function ITensors.op!(operator::ITensor, ::OpName"Ceiling", ::SiteType"MACBoson", s::Index)
    operator[s' => MAX_BOSON + 1, s => MAX_BOSON + 1] = 1.0
end


# ---------------------------------------------------------------------------
# Configuration and indexing
# ---------------------------------------------------------------------------

Base.@kwdef struct MPSMACConfig
    n::Int = 32
    reynolds::Float64 = 100.0
    peclet::Float64 = 100.0
    dt::Float64 = 0.0025
    final_time::Float64 = 3.5
    middle_fraction::Float64 = 0.30
    transition_thickness::Float64 = 0.06
    kh_width::Float64 = 0.10
    kh_mode::Int = 1
    kh_amplitude::Float64 = 0.10
    secondary_mode::Int = 2
    secondary_amplitude::Float64 = 0.025
    kh_phase::Float64 = pi / 5
    velocity_scale::Float64 = 4.0
    pressure_impulse_scale::Float64 = 4.0
    scalar_scale::Float64 = 4.0
    maxdim::Int = 64
    cutoff::Float64 = 1.0e-10
    poisson_pseudo_dt::Float64 = 2.5e-3
    poisson_steps_per_block::Int = 20
    poisson_max_blocks::Int = 12
    poisson_tolerance::Float64 = 1.0e-3
    correction_steps::Int = 4
    predictor_nsite::Int = 1
    pressure_nsite::Int = 1
    correction_nsite::Int = 1
    predictor_chunks::Int = 8
    krylov_tolerance::Float64 = 1.0e-10
    krylov_dimension::Int = 30
    krylov_maxiter::Int = 100
    output_dir::String = "outputs/mps_mac"
end

grid_spacing(config::MPSMACConfig) = 1.0 / config.n
viscosity(config::MPSMACConfig) = 1.0 / config.reynolds
scalar_diffusivity(config::MPSMACConfig) = 1.0 / config.peclet
number_of_cells(config::MPSMACConfig) = config.n^2

function validate_config(config::MPSMACConfig)
    config.n >= 4 || error("n must be at least 4")
    config.reynolds > 0 || error("Re must be positive")
    config.peclet > 0 || error("Pe must be positive")
    config.dt > 0 || error("dt must be positive")
    config.final_time > 0 || error("final_time must be positive")
    0 < config.middle_fraction < 1 || error("middle_fraction must lie in (0,1)")
    config.transition_thickness > 0 || error("transition_thickness must be positive")
    config.velocity_scale > 0 || error("velocity_scale must be positive")
    config.pressure_impulse_scale > 0 || error("pressure_impulse_scale must be positive")
    config.scalar_scale > 0 || error("scalar_scale must be positive")
    config.poisson_steps_per_block > 0 || error("poisson_steps_per_block must be positive")
    config.poisson_max_blocks > 0 || error("poisson_max_blocks must be positive")
    config.correction_steps > 0 || error("correction_steps must be positive")
    config.predictor_nsite in (1, 2) || error("predictor_nsite must be 1 or 2")
    config.pressure_nsite in (1, 2) || error("pressure_nsite must be 1 or 2")
    config.correction_nsite in (1, 2) || error("correction_nsite must be 1 or 2")
    config.predictor_chunks > 0 || error("predictor_chunks must be positive")
    config.krylov_tolerance > 0 || error("krylov_tolerance must be positive")
    config.krylov_dimension >= 2 || error("krylov_dimension must be at least 2")
    config.krylov_maxiter > 0 || error("krylov_maxiter must be positive")

    nsteps = config.final_time / config.dt
    isapprox(nsteps, round(nsteps); atol=1e-12, rtol=1e-12) ||
        error("final_time/dt must be an integer")
    return config
end

"""Map serpentine MPS cell number k to periodic grid indices (j=y, i=x)."""
function grid_from_cell(k::Integer, n::Integer)
    1 <= k <= n^2 || throw(ArgumentError("cell index out of range"))
    j = (k - 1) ÷ n + 1
    position = (k - 1) % n + 1
    i = isodd(j) ? position : n - position + 1
    return j, i
end

"""Inverse serpentine map from grid indices (j=y, i=x) to MPS cell number."""
function cell_from_grid(j::Integer, i::Integer, n::Integer)
    1 <= j <= n || throw(ArgumentError("j out of range"))
    1 <= i <= n || throw(ArgumentError("i out of range"))
    position = isodd(j) ? i : n - i + 1
    return (j - 1) * n + position
end

usite(k::Integer) = 4k - 3
vsite(k::Integer) = 4k - 2
psite(k::Integer) = 4k - 1
csite(k::Integer) = 4k

next_index(i::Integer, n::Integer) = i == n ? 1 : i + 1
previous_index(i::Integer, n::Integer) = i == 1 ? n : i - 1


# ---------------------------------------------------------------------------
# Periodic MAC operators and initial condition
# ---------------------------------------------------------------------------

function mac_divergence(u::AbstractMatrix, v::AbstractMatrix, h::Real)
    n = size(u, 1)
    result = zeros(promote_type(eltype(u), eltype(v)), n, n)
    for j in 1:n, i in 1:n
        ip = next_index(i, n)
        jp = next_index(j, n)
        result[j, i] = (u[j, ip] - u[j, i]) / h + (v[jp, i] - v[j, i]) / h
    end
    return result
end

function mac_pressure_gradient(p::AbstractMatrix, h::Real)
    n = size(p, 1)
    gx = similar(p)
    gy = similar(p)
    for j in 1:n, i in 1:n
        im = previous_index(i, n)
        jm = previous_index(j, n)
        gx[j, i] = (p[j, i] - p[j, im]) / h
        gy[j, i] = (p[j, i] - p[jm, i]) / h
    end
    return gx, gy
end

function periodic_laplacian(field::AbstractMatrix, h::Real)
    n = size(field, 1)
    result = similar(field)
    for j in 1:n, i in 1:n
        ip = next_index(i, n)
        im = previous_index(i, n)
        jp = next_index(j, n)
        jm = previous_index(j, n)
        result[j, i] = (
            field[j, ip] + field[j, im] + field[jp, i] + field[jm, i] -
            4field[j, i]
        ) / h^2
    end
    return result
end

function mac_vorticity(u::AbstractMatrix, v::AbstractMatrix, h::Real)
    n = size(u, 1)
    omega = zeros(promote_type(eltype(u), eltype(v)), n, n)
    for j in 1:n, i in 1:n
        im = previous_index(i, n)
        jm = previous_index(j, n)
        omega[j, i] = (v[j, i] - v[j, im]) / h - (u[j, i] - u[jm, i]) / h
    end
    return omega
end

"""Centered conservative MAC momentum-flux divergence used by the predictor MPO."""
function mac_conservative_advection(u::AbstractMatrix, v::AbstractMatrix, h::Real)
    n = size(u, 1)
    advection_u = zeros(promote_type(eltype(u), eltype(v)), n, n)
    advection_v = similar(advection_u)
    for j in 1:n, i in 1:n
        ip = next_index(i, n)
        im = previous_index(i, n)
        jp = next_index(j, n)
        jm = previous_index(j, n)

        u_e = 0.5 * (u[j, i] + u[j, ip])
        u_w = 0.5 * (u[j, im] + u[j, i])
        u_n = 0.5 * (u[j, i] + u[jp, i])
        u_s = 0.5 * (u[jm, i] + u[j, i])
        v_at_north_u = 0.5 * (v[jp, im] + v[jp, i])
        v_at_south_u = 0.5 * (v[j, im] + v[j, i])
        advection_u[j, i] = (
            u_e^2 - u_w^2 + u_n * v_at_north_u - u_s * v_at_south_u
        ) / h

        v_e = 0.5 * (v[j, i] + v[j, ip])
        v_w = 0.5 * (v[j, im] + v[j, i])
        v_n = 0.5 * (v[j, i] + v[jp, i])
        v_s = 0.5 * (v[jm, i] + v[j, i])
        u_at_east_v = 0.5 * (u[jm, ip] + u[j, ip])
        u_at_west_v = 0.5 * (u[jm, i] + u[j, i])
        advection_v[j, i] = (
            u_at_east_v * v_e - u_at_west_v * v_w + v_n^2 - v_s^2
        ) / h
    end
    return advection_u, advection_v
end

"""Centered conservative flux divergence for a cell-centered passive scalar."""
function mac_conservative_scalar_advection(
    u::AbstractMatrix,
    v::AbstractMatrix,
    scalar::AbstractMatrix,
    h::Real,
)
    n = size(scalar, 1)
    size(scalar) == (n, n) || error("scalar must be square")
    size(u) == size(scalar) == size(v) || error("velocity/scalar size mismatch")
    result = zeros(promote_type(eltype(u), eltype(v), eltype(scalar)), n, n)
    for j in 1:n, i in 1:n
        ip = next_index(i, n)
        im = previous_index(i, n)
        jp = next_index(j, n)
        jm = previous_index(j, n)

        scalar_e = 0.5 * (scalar[j, i] + scalar[j, ip])
        scalar_w = 0.5 * (scalar[j, im] + scalar[j, i])
        scalar_n = 0.5 * (scalar[j, i] + scalar[jp, i])
        scalar_s = 0.5 * (scalar[jm, i] + scalar[j, i])
        result[j, i] = (
            u[j, ip] * scalar_e - u[j, i] * scalar_w +
            v[jp, i] * scalar_n - v[j, i] * scalar_s
        ) / h
    end
    return result
end

function base_velocity_profile(y::Real, config::MPSMACConfig)
    y1 = 0.5 * (1 - config.middle_fraction)
    y2 = 0.5 * (1 + config.middle_fraction)
    delta = config.transition_thickness
    return tanh((y - y1) / delta) - tanh((y - y2) / delta) - 1.0
end

function base_scalar_profile(y::Real, config::MPSMACConfig)
    y1 = 0.5 * (1 - config.middle_fraction)
    y2 = 0.5 * (1 + config.middle_fraction)
    delta = config.transition_thickness
    return 0.5 * (tanh((y - y1) / delta) - tanh((y - y2) / delta))
end

"""Sample and normalize the double-tanh scalar to exact 0/1 grid plateaus."""
function initial_scalar_profile(config::MPSMACConfig)
    h = grid_spacing(config)
    raw = [base_scalar_profile((j - 0.5) * h, config) for j in 1:config.n]
    raw_min, raw_max = extrema(raw)
    raw_max > raw_min || error("degenerate initial scalar profile")
    return @. (raw - raw_min) / (raw_max - raw_min)
end

scalar_reference_mass(config::MPSMACConfig) = mean(initial_scalar_profile(config))

"""Construct a discretely divergence-free MAC double layer plus KH seed."""
function initial_mac_fields(config::MPSMACConfig)
    validate_config(config)
    n = config.n
    h = grid_spacing(config)

    # Base u is sampled directly at u-face heights y=(j-1/2)h. On this very
    # coarse grid the two tanh tails overlap in the thin middle stream, so an
    # affine normalization restores the requested sampled plateaus -1 and +1
    # without changing the tanh transition shape.
    raw_target_u = [base_velocity_profile((j - 0.5) * h, config) for j in 1:n]
    raw_min, raw_max = extrema(raw_target_u)
    raw_max > raw_min || error("degenerate base velocity profile")
    target_u = @. -1.0 + 2.0 * (raw_target_u - raw_min) / (raw_max - raw_min)
    mean_u = mean(target_u)
    fluctuation = target_u .- mean_u

    # Corner streamfunction primitive. Its discrete y difference exactly gives
    # the mean-zero part of target_u, including across the periodic seam.
    base_psi = zeros(Float64, n)
    for j in 1:n-1
        base_psi[j + 1] = base_psi[j] + h * fluctuation[j]
    end

    psi = repeat(reshape(base_psi, n, 1), 1, n)
    y1 = 0.5 * (1 - config.middle_fraction)
    y2 = 0.5 * (1 + config.middle_fraction)
    k1 = 2pi * config.kh_mode
    k2 = 2pi * config.secondary_mode

    for j in 1:n, i in 1:n
        x = (i - 1) * h
        y = (j - 1) * h
        envelope = 0.0
        for center in (y1, y2)
            distance = sin(pi * (y - center))
            width = pi * config.kh_width
            envelope += exp(-(distance / width)^2)
        end
        psi[j, i] += envelope * (
            config.kh_amplitude / k1 * cos(k1 * x) +
            config.secondary_amplitude / k2 * cos(k2 * x + config.kh_phase)
        )
    end
    psi .-= mean(psi)

    u = zeros(Float64, n, n)
    v = zeros(Float64, n, n)
    for j in 1:n, i in 1:n
        jp = next_index(j, n)
        ip = next_index(i, n)
        u[j, i] = mean_u + (psi[jp, i] - psi[j, i]) / h
        v[j, i] = -(psi[j, ip] - psi[j, i]) / h
    end
    phi = zeros(Float64, n, n)
    target_scalar = initial_scalar_profile(config)
    scalar = repeat(reshape(target_scalar, n, 1), 1, n)
    return (;
        u,
        v,
        phi,
        scalar,
        target_u,
        target_scalar,
        mean_u,
        seed_streamfunction=psi,
    )
end

function validate_initial_fields(config::MPSMACConfig; verbose=true)
    fields = initial_mac_fields(config)
    h = grid_spacing(config)
    div = mac_divergence(fields.u, fields.v, h)
    xmean_u = vec(mean(fields.u; dims=2))
    profile_error = maximum(abs.(xmean_u .- fields.target_u))
    max_divergence = maximum(abs.(div))
    scalar_profile_error = maximum(
        abs.(vec(mean(fields.scalar; dims=2)) .- fields.target_scalar),
    )

    verbose && @printf(
        "initial MAC validation: mean(u)=%.8f, profile error=%.3e, max|div|=%.3e, max|v|=%.3e, scalar=[%.3f, %.3f], mean(c)=%.8f\n",
        mean(fields.u), profile_error, max_divergence, maximum(abs.(fields.v)),
        minimum(fields.scalar), maximum(fields.scalar), mean(fields.scalar),
    )
    profile_error < 1e-12 || error("initial mean velocity profile is inconsistent")
    max_divergence < 1e-12 || error("initial MAC velocity is not divergence-free")
    scalar_profile_error < 1e-12 || error("initial scalar is not x-uniform")
    minimum(fields.scalar) ≈ 0.0 || error("initial outer scalar plateau is not zero")
    maximum(fields.scalar) ≈ 1.0 || error("initial middle scalar plateau is not one")
    return fields
end


# ---------------------------------------------------------------------------
# Coherent MPS construction and field extraction
# ---------------------------------------------------------------------------

"""Exact local truncated displacement of the bosonic vacuum."""
function coherent_local_state(site::Index, alpha::Number)
    displacement = exp(alpha * op("adag", site) - conj(alpha) * op("a", site))
    return replaceprime(displacement * onehot(site => 1), 1 => 0)
end

"""
Build the coherent product state directly with dimension-one links.

This is algebraically identical to applying every local displacement gate to
the vacuum MPS, but avoids thousands of general MPS `apply` calls and their
canonicalization/allocation overhead.
"""
function build_initial_mps(fields, sites, config::MPSMACConfig)
    site_count = 4number_of_cells(config)
    length(sites) == site_count || error("site count is inconsistent with the grid")
    links = [Index(1; tags="Link,l=$site") for site in 1:site_count-1]
    tensors = Vector{ITensor}(undef, site_count)

    for k in 1:number_of_cells(config)
        j, i = grid_from_cell(k, config.n)
        amplitudes = (
            fields.u[j, i] / config.velocity_scale,
            fields.v[j, i] / config.velocity_scale,
            fields.phi[j, i] / config.pressure_impulse_scale,
            fields.scalar[j, i] / config.scalar_scale,
        )
        site_numbers = (usite(k), vsite(k), psite(k), csite(k))
        for (alpha, site_number) in zip(amplitudes, site_numbers)
            tensor = coherent_local_state(sites[site_number], alpha)
            site_number > 1 && (tensor *= onehot(dag(links[site_number - 1]) => 1))
            site_number < site_count && (tensor *= onehot(links[site_number] => 1))
            tensors[site_number] = tensor
        end
    end
    return MPS(tensors; ortho_lims=1:1)
end

function field_expectations(
    state::MPS,
    config::MPSMACConfig;
    include_ceiling::Bool=false,
)
    ceiling_probability = NaN
    if include_ceiling
        raw_amplitudes, ceiling = expect(state, "a", "Ceiling")
        amplitudes = ComplexF64.(collect(raw_amplitudes))
        ceiling_probability = maximum(real.(collect(ceiling)))
    else
        amplitudes = ComplexF64.(collect(expect(state, "a")))
    end
    n = config.n
    u = zeros(Float64, n, n)
    v = zeros(Float64, n, n)
    phi = zeros(Float64, n, n)
    scalar = zeros(Float64, n, n)
    max_imaginary = 0.0

    for k in 1:number_of_cells(config)
        j, i = grid_from_cell(k, n)
        au = amplitudes[usite(k)]
        av = amplitudes[vsite(k)]
        ap = amplitudes[psite(k)]
        ac = amplitudes[csite(k)]
        max_imaginary = max(
            max_imaginary,
            abs(imag(au)),
            abs(imag(av)),
            abs(imag(ap)),
            abs(imag(ac)),
        )
        u[j, i] = config.velocity_scale * real(au)
        v[j, i] = config.velocity_scale * real(av)
        phi[j, i] = config.pressure_impulse_scale * real(ap)
        scalar[j, i] = config.scalar_scale * real(ac)
    end
    pressure = phi / config.dt
    return (; u, v, phi, pressure, scalar, max_imaginary, ceiling_probability)
end


# ---------------------------------------------------------------------------
# MPO construction
# ---------------------------------------------------------------------------

"""Mutate `ops` with coefficient*adag(target)*(sum A)*(sum B)."""
function add_quadratic_terms!(
    ops::OpSum,
    coefficient::Real,
    target::Int,
    first_factor,
    second_factor,
)
    for (weight_a, site_a) in first_factor, (weight_b, site_b) in second_factor
        c = coefficient * weight_a * weight_b
        iszero(c) && continue
        add!(ops, c, "adag", target, "a", site_a, "a", site_b)
    end
    return ops
end

function build_predictor_mpo(sites, config::MPSMACConfig)
    n = config.n
    h = grid_spacing(config)
    nu = viscosity(config)
    kappa = scalar_diffusivity(config)
    scale = config.velocity_scale
    # The eight algebraic pieces are also split into contiguous row chunks.
    # Their exact sum is unchanged, while each generic OpSum-to-MPO conversion
    # handles far fewer terms and requires substantially less peak workspace.
    chunks = min(config.predictor_chunks, n)
    ops_uu_x = [OpSum() for _ in 1:chunks]
    ops_uv_y = [OpSum() for _ in 1:chunks]
    ops_uv_x = [OpSum() for _ in 1:chunks]
    ops_vv_y = [OpSum() for _ in 1:chunks]
    ops_diffusion = [OpSum() for _ in 1:chunks]
    ops_scalar_x = [OpSum() for _ in 1:chunks]
    ops_scalar_y = [OpSum() for _ in 1:chunks]
    ops_scalar_diffusion = [OpSum() for _ in 1:chunks]

    for j in 1:n, i in 1:n
        chunk = min(cld(j * chunks, n), chunks)
        ip = next_index(i, n)
        im = previous_index(i, n)
        jp = next_index(j, n)
        jm = previous_index(j, n)

        k = cell_from_grid(j, i, n)
        kip = cell_from_grid(j, ip, n)
        kim = cell_from_grid(j, im, n)
        kjp = cell_from_grid(jp, i, n)
        kjm = cell_from_grid(jm, i, n)
        kjpim = cell_from_grid(jp, im, n)
        kjmip = cell_from_grid(jm, ip, n)

        ut = usite(k)
        vt = vsite(k)
        ct = csite(k)

        # Conservative u-momentum fluxes:
        # d(u^2)/dx + d(uv)/dy at the u face.
        u_e = ((0.5, usite(k)), (0.5, usite(kip)))
        u_w = ((0.5, usite(kim)), (0.5, usite(k)))
        u_n = ((0.5, usite(k)), (0.5, usite(kjp)))
        u_s = ((0.5, usite(kjm)), (0.5, usite(k)))
        v_n_corner = ((0.5, vsite(kjpim)), (0.5, vsite(kjp)))
        v_s_corner = ((0.5, vsite(kim)), (0.5, vsite(k)))

        add_quadratic_terms!(ops_uu_x[chunk], -scale / h, ut, u_e, u_e)
        add_quadratic_terms!(ops_uu_x[chunk], +scale / h, ut, u_w, u_w)
        add_quadratic_terms!(ops_uv_y[chunk], -scale / h, ut, u_n, v_n_corner)
        add_quadratic_terms!(ops_uv_y[chunk], +scale / h, ut, u_s, v_s_corner)

        # Conservative v-momentum fluxes:
        # d(uv)/dx + d(v^2)/dy at the v face.
        v_e = ((0.5, vsite(k)), (0.5, vsite(kip)))
        v_w = ((0.5, vsite(kim)), (0.5, vsite(k)))
        v_n = ((0.5, vsite(k)), (0.5, vsite(kjp)))
        v_s = ((0.5, vsite(kjm)), (0.5, vsite(k)))
        u_e_corner = ((0.5, usite(kjmip)), (0.5, usite(kip)))
        u_w_corner = ((0.5, usite(kjm)), (0.5, usite(k)))

        add_quadratic_terms!(ops_uv_x[chunk], -scale / h, vt, u_e_corner, v_e)
        add_quadratic_terms!(ops_uv_x[chunk], +scale / h, vt, u_w_corner, v_w)
        add_quadratic_terms!(ops_vv_y[chunk], -scale / h, vt, v_n, v_n)
        add_quadratic_terms!(ops_vv_y[chunk], +scale / h, vt, v_s, v_s)

        # Physical viscosity; scaling cancels for linear terms.
        for neighbor in (kip, kim, kjp, kjm)
            add!(ops_diffusion[chunk], nu / h^2, "adag", ut, "a", usite(neighbor))
            add!(ops_diffusion[chunk], nu / h^2, "adag", vt, "a", vsite(neighbor))
        end
        add!(ops_diffusion[chunk], -4nu / h^2, "Num", ut)
        add!(ops_diffusion[chunk], -4nu / h^2, "Num", vt)

        # Conservative passive-scalar fluxes at cell faces:
        # dc/dt = -d(uc)/dx - d(vc)/dy + (1/Pe) laplacian(c).
        # The scalar coherent-amplitude scale cancels between target and
        # scalar factor, leaving only the velocity scale in bilinear terms.
        c_e = ((0.5, csite(k)), (0.5, csite(kip)))
        c_w = ((0.5, csite(kim)), (0.5, csite(k)))
        c_n = ((0.5, csite(k)), (0.5, csite(kjp)))
        c_s = ((0.5, csite(kjm)), (0.5, csite(k)))
        add_quadratic_terms!(
            ops_scalar_x[chunk], -scale / h, ct, ((1.0, usite(kip)),), c_e,
        )
        add_quadratic_terms!(
            ops_scalar_x[chunk], +scale / h, ct, ((1.0, usite(k)),), c_w,
        )
        add_quadratic_terms!(
            ops_scalar_y[chunk], -scale / h, ct, ((1.0, vsite(kjp)),), c_n,
        )
        add_quadratic_terms!(
            ops_scalar_y[chunk], +scale / h, ct, ((1.0, vsite(k)),), c_s,
        )
        for neighbor in (kip, kim, kjp, kjm)
            add!(
                ops_scalar_diffusion[chunk],
                kappa / h^2,
                "adag",
                ct,
                "a",
                csite(neighbor),
            )
        end
        add!(ops_scalar_diffusion[chunk], -4kappa / h^2, "Num", ct)
    end
    components = vcat(
        ops_uu_x,
        ops_uv_y,
        ops_uv_x,
        ops_vv_y,
        ops_diffusion,
        ops_scalar_x,
        ops_scalar_y,
        ops_scalar_diffusion,
    )
    return [MPO(component, sites) for component in components]
end

function build_pressure_relaxation_mpo(sites, config::MPSMACConfig)
    n = config.n
    h = grid_spacing(config)
    source_scale = config.velocity_scale / (config.pressure_impulse_scale * h)
    ops = OpSum()

    for j in 1:n, i in 1:n
        ip = next_index(i, n)
        im = previous_index(i, n)
        jp = next_index(j, n)
        jm = previous_index(j, n)
        k = cell_from_grid(j, i, n)
        kip = cell_from_grid(j, ip, n)
        kim = cell_from_grid(j, im, n)
        kjp = cell_from_grid(jp, i, n)
        kjm = cell_from_grid(jm, i, n)
        pt = psite(k)

        # Chorin pressure impulse phi=dt*p:
        # d(phi)/dtau = laplacian(phi) - divergence(u_star).
        for neighbor in (kip, kim, kjp, kjm)
            add!(ops, 1 / h^2, "adag", pt, "a", psite(neighbor))
        end
        add!(ops, -4 / h^2, "Num", pt)
        add!(ops, -source_scale, "adag", pt, "a", usite(kip))
        add!(ops, +source_scale, "adag", pt, "a", usite(k))
        add!(ops, -source_scale, "adag", pt, "a", vsite(kjp))
        add!(ops, +source_scale, "adag", pt, "a", vsite(k))
    end
    return MPO(ops, sites)
end

function build_pressure_correction_mpo(sites, config::MPSMACConfig)
    n = config.n
    h = grid_spacing(config)
    coefficient = config.pressure_impulse_scale / (config.velocity_scale * h)
    ops = OpSum()

    for j in 1:n, i in 1:n
        im = previous_index(i, n)
        jm = previous_index(j, n)
        k = cell_from_grid(j, i, n)
        kim = cell_from_grid(j, im, n)
        kjm = cell_from_grid(jm, i, n)

        # Unit correction-time evolution integrates exactly to
        # u <- u - Gx(phi), v <- v - Gy(phi), with phi=dt*p.
        add!(ops, -coefficient, "adag", usite(k), "a", psite(k))
        add!(ops, +coefficient, "adag", usite(k), "a", psite(kim))
        add!(ops, -coefficient, "adag", vsite(k), "a", psite(k))
        add!(ops, +coefficient, "adag", vsite(k), "a", psite(kjm))
    end
    return MPO(ops, sites)
end


# ---------------------------------------------------------------------------
# Stable fingerprints, operator cache, and restart checkpoints
# ---------------------------------------------------------------------------

canonical_value(value::Float64) = bitstring(value)
canonical_value(value::Integer) = string(value)
canonical_value(value::Bool) = value ? "true" : "false"
canonical_value(value::AbstractString) = string(ncodeunits(value), ":", value)
canonical_value(value) = canonical_value(string(value))

function stable_fingerprint(entries)
    canonical = join(
        (string(name, "=", canonical_value(value)) for (name, value) in entries),
        '\n',
    )
    return bytes2hex(sha256(canonical))
end

source_digest() = bytes2hex(sha256(read(@__FILE__)))

function manifest_digest()
    path = joinpath(@__DIR__, "Manifest.toml")
    return isfile(path) ? bytes2hex(sha256(read(path))) : "missing"
end

function operator_fingerprint(config::MPSMACConfig)
    return stable_fingerprint((
        (:schema, OPERATOR_CACHE_VERSION),
        (:operator_definition, OPERATOR_DEFINITION_VERSION),
        (:layout, "interleaved-u-v-phi-scalar-serpentine-v2"),
        (:n, config.n),
        (:reynolds, config.reynolds),
        (:peclet, config.peclet),
        (:velocity_scale, config.velocity_scale),
        (:pressure_impulse_scale, config.pressure_impulse_scale),
        (:scalar_scale, config.scalar_scale),
        (:predictor_chunks, config.predictor_chunks),
        (:max_boson, MAX_BOSON),
        (:julia, string(VERSION)),
        (:itensors, string(Base.pkgversion(ITensors))),
        (:itensormps, string(Base.pkgversion(ITensorMPS))),
        (:manifest, manifest_digest()),
    ))
end

function legacy_operator_fingerprint(config::MPSMACConfig, source::AbstractString)
    return stable_fingerprint((
        (:schema, OPERATOR_CACHE_VERSION),
        (:layout, "interleaved-u-v-phi-scalar-serpentine-v2"),
        (:n, config.n),
        (:reynolds, config.reynolds),
        (:peclet, config.peclet),
        (:velocity_scale, config.velocity_scale),
        (:pressure_impulse_scale, config.pressure_impulse_scale),
        (:scalar_scale, config.scalar_scale),
        (:predictor_chunks, config.predictor_chunks),
        (:max_boson, MAX_BOSON),
        (:julia, string(VERSION)),
        (:itensors, string(Base.pkgversion(ITensors))),
        (:itensormps, string(Base.pkgversion(ITensorMPS))),
        (:source, source),
        (:manifest, manifest_digest()),
    ))
end

function evolution_fingerprint(config::MPSMACConfig)
    entries = Pair{Symbol,Any}[
        :schema => CHECKPOINT_VERSION,
        :layout => "interleaved-u-v-phi-scalar-serpentine-v2",
        :max_boson => MAX_BOSON,
        :julia => string(VERSION),
        :itensors => string(Base.pkgversion(ITensors)),
        :itensormps => string(Base.pkgversion(ITensorMPS)),
        :source => source_digest(),
        :manifest => manifest_digest(),
    ]
    for name in fieldnames(MPSMACConfig)
        name == :output_dir && continue
        push!(entries, name => getproperty(config, name))
    end
    return stable_fingerprint(entries)
end

function configuration_metadata(config::MPSMACConfig)
    data = Dict{String,Any}(
        String(name) => getproperty(config, name) for name in fieldnames(MPSMACConfig)
    )
    data["max_boson"] = MAX_BOSON
    data["operator_fingerprint"] = operator_fingerprint(config)
    data["evolution_fingerprint"] = evolution_fingerprint(config)
    return data
end

function fsync_file(path::AbstractString)
    open(path, "r") do io
        result = ccall(:fsync, Cint, (Cint,), fd(io))
        result == 0 || error("fsync failed for $path")
    end
    return path
end

function fsync_directory(directory::AbstractString)
    descriptor = ccall(:open, Cint, (Cstring, Cint), directory, 0)
    descriptor >= 0 || error("could not open directory for fsync: $directory")
    try
        result = ccall(:fsync, Cint, (Cint,), descriptor)
        result == 0 || error("directory fsync failed for $directory")
    finally
        ccall(:close, Cint, (Cint,), descriptor)
    end
    return directory
end

function with_file_lock(worker, path::AbstractString)
    mkpath(dirname(path))
    open(path, "w") do io
        ccall(:flock, Cint, (Cint, Cint), fd(io), 2) == 0 ||
            error("could not acquire file lock: $path")
        try
            return worker()
        finally
            ccall(:flock, Cint, (Cint, Cint), fd(io), 8)
        end
    end
end

function atomic_replace(writer, path::AbstractString)
    directory = dirname(path)
    mkpath(directory)
    temporary = joinpath(
        directory,
        string(".", basename(path), ".tmp.", getpid(), ".", time_ns()),
    )
    try
        writer(temporary)
        fsync_file(temporary)
        mv(temporary, path; force=true)
        fsync_directory(directory)
    finally
        isfile(temporary) && rm(temporary; force=true)
    end
    return path
end

function operator_cache_path(
    cache_directory::AbstractString,
    fingerprint::AbstractString,
)
    return joinpath(
        cache_directory,
        string("operators_", fingerprint[1:20], ".h5"),
    )
end

operator_cache_path(cache_directory::AbstractString, config::MPSMACConfig) =
    operator_cache_path(cache_directory, operator_fingerprint(config))

function write_operator_cache(
    path,
    bundle,
    config::MPSMACConfig;
    fingerprint=operator_fingerprint(config),
)
    atomic_replace(path) do temporary
        h5open(temporary, "w") do file
            attrs(file)["format"] = "mixing-layer-mps-operator-cache"
            attrs(file)["version"] = OPERATOR_CACHE_VERSION
            attrs(file)["fingerprint"] = fingerprint
            attrs(file)["max_boson"] = MAX_BOSON
            attrs(file)["predictor_count"] = length(bundle.predictor)
            write(file, "site_template", bundle.site_template)
            for (index, operator) in enumerate(bundle.predictor)
                write(file, "predictor_$index", operator)
            end
            write(file, "pressure", bundle.pressure)
            write(file, "correction", bundle.correction)
        end
    end
    return path
end

function read_operator_cache(
    path,
    config::MPSMACConfig;
    expected=operator_fingerprint(config),
)
    return h5open(path, "r") do file
        attrs(file)["format"] == "mixing-layer-mps-operator-cache" ||
            error("unrecognized operator-cache format")
        attrs(file)["version"] == OPERATOR_CACHE_VERSION ||
            error("operator-cache schema mismatch")
        attrs(file)["fingerprint"] == expected ||
            error("operator-cache fingerprint mismatch")
        template = read(file, "site_template", MPS)
        predictor_count = attrs(file)["predictor_count"]
        predictor = [read(file, "predictor_$index", MPO) for index in 1:predictor_count]
        pressure = read(file, "pressure", MPO)
        correction = read(file, "correction", MPO)
        return (;
            site_template=template,
            sites=siteinds(template),
            predictor,
            pressure,
            correction,
            fingerprint=operator_fingerprint(config),
            cache_hit=true,
        )
    end
end

function build_operator_bundle(config::MPSMACConfig; sites=nothing)
    if isnothing(sites)
        sites = siteinds("MACBoson", 4number_of_cells(config); conserve_qns=false)
    end
    length(sites) == 4number_of_cells(config) || error("operator site count mismatch")
    template = MPS(sites, fill(1, length(sites)))
    predictor = build_predictor_mpo(sites, config)
    pressure = build_pressure_relaxation_mpo(sites, config)
    correction = build_pressure_correction_mpo(sites, config)
    return (;
        site_template=template,
        sites,
        predictor,
        pressure,
        correction,
        fingerprint=operator_fingerprint(config),
        cache_hit=false,
    )
end

function load_or_build_operators(
    config::MPSMACConfig;
    cache_directory=nothing,
    sites=nothing,
    rebuild::Bool=false,
    require_cache::Bool=false,
)
    path = isnothing(cache_directory) ? nothing : operator_cache_path(cache_directory, config)
    require_cache && isnothing(path) && error("a cache directory is required with require_cache")

    function load_or_build_unlocked()
        if !rebuild && !isnothing(path)
            current_fingerprint = operator_fingerprint(config)
            candidates = [(path, current_fingerprint)]
            for source in LEGACY_OPERATOR_SOURCE_DIGESTS
                legacy_fingerprint = legacy_operator_fingerprint(config, source)
                legacy_path = operator_cache_path(cache_directory, legacy_fingerprint)
                legacy_path == path || push!(candidates, (legacy_path, legacy_fingerprint))
            end
            for (candidate_path, expected) in candidates
                isfile(candidate_path) || continue
                try
                    bundle = read_operator_cache(candidate_path, config; expected)
                    if expected != current_fingerprint
                        @info "Using legacy-compatible operator cache" path=candidate_path legacy_fingerprint=expected current_fingerprint
                    end
                    return bundle, candidate_path
                catch exception
                    @warn "Ignoring invalid operator-cache candidate" path=candidate_path exception
                end
            end
        end

        bundle = build_operator_bundle(config; sites)
        if !isnothing(path)
            try
                write_operator_cache(path, bundle, config)
            catch exception
                require_cache && rethrow()
                @warn "Could not persist operator cache" path exception
            end
        end
        return bundle, path
    end

    isnothing(path) && return load_or_build_unlocked()
    return with_file_lock(path * ".lock") do
        load_or_build_unlocked()
    end
end

function serialize_metadata(metadata)
    buffer = IOBuffer()
    serialize(buffer, metadata)
    return take!(buffer)
end

deserialize_metadata(bytes) = deserialize(IOBuffer(bytes))

checkpoint_checksum(path::AbstractString) = open(path, "r") do io
    bytes2hex(sha256(io))
end

checkpoint_sidecar(path::AbstractString) = path * ".sha256"

function checkpoint_generation_paths(path::AbstractString)
    directory = dirname(path)
    isdir(directory) || return String[]
    prefix = basename(path) * ".generation."
    return [
        joinpath(directory, name) for name in readdir(directory)
        if startswith(name, prefix) && endswith(name, ".h5")
    ]
end

function read_checkpoint_manifest(path::AbstractString)
    isfile(path) || return nothing
    lines = try
        readlines(path)
    catch
        return nothing
    end
    isempty(lines) && return nothing
    lines[1] == "format=$CHECKPOINT_MANIFEST_FORMAT" || return nothing
    entries = Dict{String,String}()
    for line in lines[2:end]
        split_line = split(line, '='; limit=2)
        length(split_line) == 2 || error("malformed checkpoint manifest: $path")
        entries[split_line[1]] = split_line[2]
    end
    for key in ("generation", "checksum", "completed_step", "fingerprint")
        haskey(entries, key) || error("checkpoint manifest is missing $key")
    end
    return (;
        generation=joinpath(dirname(path), entries["generation"]),
        checksum=entries["checksum"],
        completed_step=parse(Int, entries["completed_step"]),
        fingerprint=entries["fingerprint"],
    )
end

function verified_checkpoint_checksum(path::AbstractString; expected=nothing)
    sidecar = checkpoint_sidecar(path)
    isfile(sidecar) || error("checkpoint checksum sidecar is missing: $sidecar")
    recorded = strip(read(sidecar, String))
    !isnothing(expected) && recorded != expected &&
        error("checkpoint manifest/sidecar checksum mismatch: $path")
    actual = checkpoint_checksum(path)
    actual == recorded || error("checkpoint content checksum mismatch: $path")
    return actual
end

function checkpoint_header(
    path::AbstractString,
    config::MPSMACConfig;
    verify_content::Bool=true,
    expected_checksum=nothing,
)
    expected = evolution_fingerprint(config)
    if verify_content
        verified_checkpoint_checksum(path; expected=expected_checksum)
    else
        isfile(checkpoint_sidecar(path)) ||
            error("checkpoint checksum sidecar is missing: $(checkpoint_sidecar(path))")
    end
    return h5open(path, "r") do file
        attrs(file)["format"] == "mixing-layer-mps-checkpoint" ||
            error("unrecognized checkpoint format")
        attrs(file)["version"] == CHECKPOINT_VERSION ||
            error("checkpoint schema mismatch")
        attrs(file)["fingerprint"] == expected ||
            error("checkpoint configuration/code fingerprint mismatch")
        metadata = deserialize_metadata(read(file, "metadata"))
        metadata.completed_step == attrs(file)["completed_step"] ||
            error("checkpoint metadata step mismatch")
        return (; completed_step=metadata.completed_step, metadata)
    end
end

function prune_checkpoint_generations(path, config; keep::Int=CHECKPOINT_GENERATIONS_TO_KEEP)
    keep >= 2 || error("at least two checkpoint generations must be retained")
    valid = Tuple{Int,Float64,String}[]
    for generation in checkpoint_generation_paths(path)
        try
            header = checkpoint_header(generation, config; verify_content=false)
            push!(valid, (header.completed_step, stat(generation).mtime, generation))
        catch exception
            @warn "Retaining unreadable checkpoint generation for manual recovery" generation exception
        end
    end
    sort!(valid; by=item -> (item[1], item[2]), rev=true)
    for (_, _, generation) in valid[keep+1:end]
        rm(generation; force=true)
        rm(checkpoint_sidecar(generation); force=true)
    end
    length(valid) > keep && fsync_directory(dirname(path))
    return nothing
end

function write_checkpoint_unlocked(path, state::MPS, metadata, config::MPSMACConfig)
    fingerprint = evolution_fingerprint(config)
    manifest = try
        read_checkpoint_manifest(path)
    catch exception
        @warn "Replacing malformed checkpoint manifest" path exception
        nothing
    end
    if !isnothing(manifest)
        manifest_is_valid = try
            header = checkpoint_header(
                manifest.generation,
                config;
                expected_checksum=manifest.checksum,
            )
            header.completed_step == manifest.completed_step ||
                error("manifest step does not match its checkpoint generation")
            manifest.fingerprint == fingerprint ||
                error("manifest fingerprint does not match the current trajectory")
            true
        catch exception
            @warn "Replacing manifest that points to an invalid generation" path exception
            false
        end
        manifest_is_valid && manifest.completed_step > metadata.completed_step && error(
            "refusing to regress checkpoint from step $(manifest.completed_step) " *
            "to $(metadata.completed_step)",
        )
    end
    generation = string(
        path,
        ".generation.",
        lpad(metadata.completed_step, 8, '0'),
        ".",
        time_ns(),
        ".h5",
    )
    atomic_replace(generation) do temporary
        h5open(temporary, "w") do file
            attrs(file)["format"] = "mixing-layer-mps-checkpoint"
            attrs(file)["version"] = CHECKPOINT_VERSION
            attrs(file)["fingerprint"] = fingerprint
            attrs(file)["completed_step"] = metadata.completed_step
            write(file, "state", state)
            write(file, "metadata", serialize_metadata(metadata))
        end
    end
    checksum = checkpoint_checksum(generation)
    atomic_replace(checkpoint_sidecar(generation)) do temporary
        open(temporary, "w") do io
            println(io, checksum)
        end
    end
    atomic_replace(path) do temporary
        open(temporary, "w") do io
            println(io, "format=$CHECKPOINT_MANIFEST_FORMAT")
            println(io, "generation=$(basename(generation))")
            println(io, "checksum=$checksum")
            println(io, "completed_step=$(metadata.completed_step)")
            println(io, "fingerprint=$fingerprint")
        end
    end
    prune_checkpoint_generations(path, config)
    return path
end

function write_checkpoint(path, state::MPS, metadata, config::MPSMACConfig)
    return with_file_lock(path * ".writer.lock") do
        write_checkpoint_unlocked(path, state, metadata, config)
    end
end

function read_checkpoint_file(
    path,
    config::MPSMACConfig;
    verify_content::Bool=true,
)
    expected = evolution_fingerprint(config)
    verify_content && verified_checkpoint_checksum(path)
    return h5open(path, "r") do file
        attrs(file)["format"] == "mixing-layer-mps-checkpoint" ||
            error("unrecognized checkpoint format")
        attrs(file)["version"] == CHECKPOINT_VERSION ||
            error("checkpoint schema mismatch")
        attrs(file)["fingerprint"] == expected ||
            error("checkpoint configuration/code fingerprint mismatch")
        state = read(file, "state", MPS)
        metadata = deserialize_metadata(read(file, "metadata"))
        metadata.completed_step == attrs(file)["completed_step"] ||
            error("checkpoint metadata step mismatch")
        return (; state, metadata, path)
    end
end

function load_checkpoint(path, config::MPSMACConfig)
    manifest = try
        read_checkpoint_manifest(path)
    catch exception
        @warn "Ignoring malformed checkpoint manifest and scanning generations" path exception
        nothing
    end
    candidates = String[]
    !isnothing(manifest) && push!(candidates, manifest.generation)
    append!(candidates, checkpoint_generation_paths(path))
    # Schema-v1 compatibility candidates are attempted only when they are HDF5,
    # not when `path` is the schema-v2 text manifest.
    isnothing(manifest) && isfile(path) && push!(candidates, path)
    isfile(path * ".previous") && push!(candidates, path * ".previous")
    unique!(candidates)
    failures = String[]
    ranked = Tuple{Int,Float64,String}[]
    for candidate in candidates
        isfile(candidate) || continue
        try
            header = checkpoint_header(
                candidate,
                config;
                verify_content=false,
            )
            push!(ranked, (header.completed_step, stat(candidate).mtime, candidate))
        catch exception
            push!(failures, string(candidate, ": ", sprint(showerror, exception)))
        end
    end
    sort!(ranked; by=item -> (item[1], item[2]), rev=true)
    for (_, _, candidate) in ranked
        try
            manifest_checksum = !isnothing(manifest) && candidate == manifest.generation ?
                manifest.checksum : nothing
            # Hash only the highest-ranked candidate (and a fallback on failure),
            # then load only that one full MPS into memory.
            verified_checkpoint_checksum(candidate; expected=manifest_checksum)
            return read_checkpoint_file(candidate, config; verify_content=false)
        catch exception
            push!(failures, string(candidate, ": ", sprint(showerror, exception)))
        end
    end
    isempty(failures) && error("checkpoint does not exist: $path")
    error("no valid checkpoint generation found:\n" * join(failures, "\n"))
end


# ---------------------------------------------------------------------------
# Chorin step and diagnostics
# ---------------------------------------------------------------------------

function evolve_tdvp(operator, total_time, state, config; nsteps=1, nsite=2)
    evolved = tdvp(
        operator,
        total_time,
        state;
        nsteps=nsteps,
        cutoff=config.cutoff,
        maxdim=config.maxdim,
        nsite,
        outputlevel=0,
        normalize=false,
        updater_kwargs=(;
            tol=config.krylov_tolerance,
            krylovdim=config.krylov_dimension,
            maxiter=config.krylov_maxiter,
        ),
    )
    normalize!(evolved)
    return evolved
end

function pressure_poisson_residual(fields, config::MPSMACConfig)
    h = grid_spacing(config)
    rhs = mac_divergence(fields.u, fields.v, h)
    laplacian_phi = periodic_laplacian(fields.phi, h)
    residual = laplacian_phi - rhs
    denominator = max(norm(rhs), norm(laplacian_phi), eps(Float64))
    return norm(residual) / denominator
end

function pressure_gauge_ratio(fields, config::MPSMACConfig)
    pressure_rms = norm(fields.phi) / sqrt(length(fields.phi))
    scale = max(
        pressure_rms,
        PRESSURE_GAUGE_SCALE_FLOOR * config.pressure_impulse_scale,
    )
    return abs(mean(fields.phi)) / scale
end

"""
Remove the arbitrary constant mode of periodic pressure from the MPS itself.

A uniform local displacement shifts every pressure coherent amplitude without
changing velocity observables, pressure gradients, or MPS bond dimensions.
Two iterations give finite-boson truncation a chance to remove any small
residual left by the first displacement.
"""
function enforce_pressure_gauge(
    state,
    fields,
    config::MPSMACConfig;
    include_ceiling::Bool=false,
    use_gpu::Bool=false,
)
    centered = state
    centered_fields = fields
    for _ in 1:2
        pressure_gauge_ratio(centered_fields, config) <= PRESSURE_GAUGE_TARGET && break
        displacement = -mean(centered_fields.phi) / config.pressure_impulse_scale
        sites = siteinds(centered)
        gates = [
            exp(
                displacement * op("adag", sites[psite(k)]) -
                conj(displacement) * op("a", sites[psite(k)])
            ) for k in 1:number_of_cells(config)
        ]
        use_gpu && (gates = [CUDA.cu(gate) for gate in gates])
        centered = apply(gates, centered)
        normalize!(centered)
        centered_fields = field_expectations(centered, config; include_ceiling)
    end
    return centered, centered_fields
end

function relax_pressure(state, tentative_fields, pressure_mpo, config::MPSMACConfig)
    fields = tentative_fields
    residual = pressure_poisson_residual(fields, config)
    blocks_used = 0
    residual <= config.poisson_tolerance && return state, fields, residual, blocks_used

    block_time = config.poisson_pseudo_dt * config.poisson_steps_per_block
    for block in 1:config.poisson_max_blocks
        state = evolve_tdvp(
            pressure_mpo,
            block_time,
            state,
            config;
            nsteps=config.poisson_steps_per_block,
            nsite=config.pressure_nsite,
        )
        fields = field_expectations(state, config)
        residual = pressure_poisson_residual(fields, config)
        blocks_used = block
        residual <= config.poisson_tolerance && break
    end
    return state, fields, residual, blocks_used
end

function relative_velocity_difference(first, second)
    numerator = sqrt(norm(first.u - second.u)^2 + norm(first.v - second.v)^2)
    denominator = max(sqrt(norm(first.u)^2 + norm(first.v)^2), eps(Float64))
    return numerator / denominator
end

function relative_scalar_difference(first, second)
    return norm(first.scalar - second.scalar) /
        max(norm(first.scalar), eps(Float64))
end

function chorin_step(
    state,
    predictor_mpo,
    pressure_mpo,
    correction_mpo,
    config;
    include_ceiling::Bool=false,
    use_gpu::Bool=false,
)
    predictor_seconds = @elapsed begin
        tentative = evolve_tdvp(
            predictor_mpo,
            config.dt,
            state,
            config;
            nsite=config.predictor_nsite,
        )
        tentative_fields = field_expectations(tentative, config)
    end
    pressure_seconds = @elapsed begin
        relaxed, relaxed_fields, pressure_residual, blocks =
            relax_pressure(tentative, tentative_fields, pressure_mpo, config)
        relaxed, relaxed_fields = enforce_pressure_gauge(
            relaxed,
            relaxed_fields,
            config;
            use_gpu,
        )
        pressure_residual = pressure_poisson_residual(relaxed_fields, config)
    end
    velocity_leakage = relative_velocity_difference(tentative_fields, relaxed_fields)
    pressure_scalar_leakage = relative_scalar_difference(tentative_fields, relaxed_fields)
    correction_seconds = @elapsed begin
        corrected = evolve_tdvp(
            correction_mpo,
            1.0,
            relaxed,
            config;
            nsteps=config.correction_steps,
            nsite=config.correction_nsite,
        )
        corrected_fields = field_expectations(
            corrected,
            config;
            include_ceiling,
        )
        corrected, corrected_fields = enforce_pressure_gauge(
            corrected,
            corrected_fields,
            config;
            include_ceiling,
            use_gpu,
        )
    end
    h = grid_spacing(config)
    grad_x, grad_y = mac_pressure_gradient(relaxed_fields.phi, h)
    correction_norm = sqrt(norm(grad_x)^2 + norm(grad_y)^2)
    correction_defect = sqrt(
        norm(corrected_fields.u - relaxed_fields.u + grad_x)^2 +
        norm(corrected_fields.v - relaxed_fields.v + grad_y)^2
    ) / max(correction_norm, eps(Float64))
    pressure_leakage = norm(corrected_fields.phi - relaxed_fields.phi) /
        max(norm(relaxed_fields.phi), eps(Float64))
    correction_scalar_leakage = relative_scalar_difference(
        relaxed_fields,
        corrected_fields,
    )
    defects = (;
        pressure_residual,
        velocity_leakage,
        pressure_scalar_leakage,
        correction_defect,
        pressure_leakage,
        correction_scalar_leakage,
    )
    timing = (; predictor_seconds, pressure_seconds, correction_seconds)
    return corrected, corrected_fields, defects, blocks, timing
end

function step_quality(state, fields, config::MPSMACConfig)
    h = grid_spacing(config)
    div = mac_divergence(fields.u, fields.v, h)
    return (
        max_divergence=maximum(abs.(div)),
        relative_divergence=norm(div) /
            max((norm(fields.u) + norm(fields.v)) / h, eps(Float64)),
        pressure_gauge=pressure_gauge_ratio(fields, config),
        scalar_mass=mean(fields.scalar),
        scalar_mass_error=abs(mean(fields.scalar) - scalar_reference_mass(config)) /
            max(abs(scalar_reference_mass(config)), eps(Float64)),
        scalar_minimum=minimum(fields.scalar),
        scalar_maximum=maximum(fields.scalar),
        max_imaginary=fields.max_imaginary,
        max_bond=maxlinkdim(state),
        ceiling_probability=fields.ceiling_probability,
    )
end

function check_step_quality(
    quality,
    defects,
    blocks,
    step,
    config::MPSMACConfig;
    strict::Bool=false,
)
    values = (
        quality.max_divergence,
        quality.relative_divergence,
        quality.pressure_gauge,
        quality.max_imaginary,
        defects.pressure_residual,
        defects.velocity_leakage,
        defects.pressure_scalar_leakage,
        defects.correction_defect,
        defects.pressure_leakage,
        defects.correction_scalar_leakage,
        quality.scalar_mass,
        quality.scalar_mass_error,
        quality.scalar_minimum,
        quality.scalar_maximum,
    )
    all(isfinite, values) || error("non-finite MPS diagnostic at physical step $step")
    isfinite(quality.ceiling_probability) && quality.ceiling_probability < 0 &&
        error("negative boson ceiling probability at physical step $step")

    defects.pressure_residual > config.poisson_tolerance &&
        @warn "pressure relaxation did not reach tolerance" step pressure_residual=defects.pressure_residual blocks
    quality.relative_divergence > 1e-4 &&
        @warn "post-projection divergence gate failed" step value=quality.relative_divergence
    defects.velocity_leakage > 1e-5 &&
        @warn "velocity changed during pressure relaxation" step value=defects.velocity_leakage
    defects.pressure_scalar_leakage > 1e-5 &&
        @warn "scalar changed during pressure relaxation" step value=defects.pressure_scalar_leakage
    defects.correction_defect > 1e-3 &&
        @warn "pressure-correction map defect is too large" step value=defects.correction_defect
    defects.pressure_leakage > 1e-5 &&
        @warn "pressure changed during velocity correction" step value=defects.pressure_leakage
    defects.correction_scalar_leakage > 1e-5 &&
        @warn "scalar changed during velocity correction" step value=defects.correction_scalar_leakage
    quality.scalar_mass_error > SCALAR_MASS_TOLERANCE &&
        @warn "conserved-scalar mean drifted" step value=quality.scalar_mass_error
    quality.pressure_gauge > 1e-4 &&
        @warn "pressure-impulse mean gauge drifted" step value=quality.pressure_gauge
    quality.max_imaginary > 1e-8 &&
        @warn "coherent amplitudes developed an imaginary component" step value=quality.max_imaginary
    isfinite(quality.ceiling_probability) && quality.ceiling_probability > 1e-4 &&
        @warn "boson ceiling occupation is too large" step value=quality.ceiling_probability
    quality.max_bond >= config.maxdim &&
        @warn "MPS reached the configured maximum bond dimension" step maxdim=config.maxdim

    failures = String[]
    defects.pressure_residual > config.poisson_tolerance &&
        push!(failures, "pressure residual")
    quality.relative_divergence > 1e-4 && push!(failures, "relative divergence")
    defects.velocity_leakage > 1e-5 && push!(failures, "pressure velocity leakage")
    defects.pressure_scalar_leakage > 1e-5 &&
        push!(failures, "pressure scalar leakage")
    defects.correction_defect > 1e-3 && push!(failures, "correction defect")
    defects.pressure_leakage > 1e-5 && push!(failures, "correction pressure leakage")
    defects.correction_scalar_leakage > 1e-5 &&
        push!(failures, "correction scalar leakage")
    quality.scalar_mass_error > SCALAR_MASS_TOLERANCE &&
        push!(failures, "scalar mass conservation")
    quality.pressure_gauge > 1e-4 && push!(failures, "pressure gauge")
    quality.max_imaginary > 1e-8 && push!(failures, "imaginary amplitude")
    isfinite(quality.ceiling_probability) && quality.ceiling_probability > 1e-4 &&
        push!(failures, "boson ceiling occupation")
    quality.max_bond >= config.maxdim && push!(failures, "maximum bond dimension")
    strict && !isempty(failures) && error(
        "strict MPS quality gate failed at physical step $step: " * join(failures, ", ")
    )
    return nothing
end

function snapshot_diagnostics(
    state,
    fields,
    defects,
    config::MPSMACConfig;
    quality=step_quality(state, fields, config),
)
    omega = mac_vorticity(fields.u, fields.v, grid_spacing(config))
    return (
        kinetic_energy=0.5 * mean(fields.u .^ 2 .+ fields.v .^ 2),
        cross_stream_energy=0.5 * mean(fields.v .^ 2),
        enstrophy=0.5 * mean(omega .^ 2),
        max_divergence=quality.max_divergence,
        relative_divergence=quality.relative_divergence,
        mean_u=mean(fields.u),
        mean_v=mean(fields.v),
        mean_pressure=mean(fields.pressure),
        scalar_mass=quality.scalar_mass,
        scalar_mass_error=quality.scalar_mass_error,
        scalar_minimum=quality.scalar_minimum,
        scalar_maximum=quality.scalar_maximum,
        scalar_variance=mean((fields.scalar .- quality.scalar_mass) .^ 2),
        pressure_gauge=quality.pressure_gauge,
        pressure_residual=defects.pressure_residual,
        pressure_velocity_leakage=defects.velocity_leakage,
        pressure_scalar_leakage=defects.pressure_scalar_leakage,
        correction_defect=defects.correction_defect,
        correction_pressure_leakage=defects.pressure_leakage,
        correction_scalar_leakage=defects.correction_scalar_leakage,
        max_imaginary=quality.max_imaginary,
        mps_norm=real(inner(state, state)),
        max_bond=quality.max_bond,
        ceiling_probability=quality.ceiling_probability,
    )
end


# ---------------------------------------------------------------------------
# Output and driver
# ---------------------------------------------------------------------------

function plot_vorticity_snapshots(times, omega_history, config::MPSMACConfig, path)
    Base.find_package("CairoMakie") === nothing && begin
        @warn "CairoMakie is unavailable; skipping figure"
        return
    end
    @eval using CairoMakie

    count = length(times)
    rows = count <= 4 ? 1 : 2
    columns = ceil(Int, count / rows)
    limit = max(maximum(maximum(abs.(omega)) for omega in omega_history), eps(Float64))
    h = grid_spacing(config)
    coordinates = collect((0:config.n-1) .* h)
    figure = CairoMakie.Figure(size=(360columns + 100, 340rows + 100))
    plots = Any[]

    for index in 1:count
        row = (index - 1) ÷ columns + 1
        column = (index - 1) % columns + 1
        axis = CairoMakie.Axis(
            figure[row, column],
            title=@sprintf("t = %.2f", times[index]),
            xlabel=row == rows ? "x" : "",
            ylabel=column == 1 ? "y" : "",
            aspect=CairoMakie.DataAspect(),
        )
        image = CairoMakie.heatmap!(
            axis,
            coordinates,
            coordinates,
            transpose(omega_history[index]);
            colormap=:RdBu,
            colorrange=(-limit, limit),
        )
        push!(plots, image)
    end
    CairoMakie.Colorbar(figure[1:rows, columns + 1], plots[1], label="vorticity")
    CairoMakie.Label(
        figure[0, 1:columns],
        "Bosonic MPS Chorin/MAC mixing layer: n=$(config.n), Re=$(config.reynolds)",
        fontsize=22,
    )
    mkpath(dirname(path))
    CairoMakie.save(path, figure; px_per_unit=2)
end

function save_results(
    times,
    field_history,
    omega_history,
    diagnostics,
    step_diagnostics,
    config,
    path,
    ;
    snapshot_step_history=Int[],
    completed_step=0,
    requested_steps=round(Int, config.final_time / config.dt),
    terminal_fields=nothing,
    run_metadata=Dict{String,Any}(),
)
    mkpath(dirname(path))
    u = cat((field.u for field in field_history)...; dims=3)
    v = cat((field.v for field in field_history)...; dims=3)
    pressure_impulse = cat((field.phi for field in field_history)...; dims=3)
    pressure = cat((field.pressure for field in field_history)...; dims=3)
    scalar = cat((field.scalar for field in field_history)...; dims=3)
    vorticity = cat(omega_history...; dims=3)
    diagnostic_data = Dict(
        String(name) => [getproperty(diagnostic, name) for diagnostic in diagnostics]
        for name in propertynames(first(diagnostics))
    )
    step_diagnostic_data = isempty(step_diagnostics) ? Dict{String,Any}() : Dict(
        String(name) => [getproperty(diagnostic, name) for diagnostic in step_diagnostics]
        for name in propertynames(first(step_diagnostics))
    )
    parameters = configuration_metadata(config)
    isnothing(terminal_fields) && (terminal_fields = last(field_history))
    atomic_replace(path) do temporary
        jldsave(
            temporary;
            times,
            snapshot_steps=collect(snapshot_step_history),
            completed_step,
            requested_steps,
            u,
            v,
            pressure,
            pressure_impulse,
            scalar,
            vorticity,
            terminal_u=terminal_fields.u,
            terminal_v=terminal_fields.v,
            terminal_pressure=terminal_fields.pressure,
            terminal_pressure_impulse=terminal_fields.phi,
            terminal_scalar=terminal_fields.scalar,
            diagnostics=diagnostic_data,
            step_diagnostics=step_diagnostic_data,
            parameters,
            run_metadata,
        )
    end
    return path
end

function maybe_enable_gpu(requested::Bool)
    requested || return false
    Base.find_package("CUDA") === nothing &&
        error("--gpu requested, but CUDA is not installed in this Julia environment")
    @eval import CUDA
    CUDA.functional() || error("--gpu requested, but CUDA.functional() is false")
    CUDA.allowscalar(false)
    @info "Using GPU" device=CUDA.device()
    return true
end

to_device(object, use_gpu::Bool) = use_gpu ? CUDA.cu(object) : object
to_device(objects::Vector{<:MPO}, use_gpu::Bool) =
    use_gpu ? [CUDA.cu(object) for object in objects] : objects

operator_maxlinkdim(operator::MPO) = maxlinkdim(operator)
operator_maxlinkdim(operators::Vector{<:MPO}) = maximum(maxlinkdim, operators)

function snapshot_steps(total_steps::Integer, requested_steps::Integer)
    0 <= total_steps <= requested_steps || error("invalid physical-step count")
    global_steps = unique(round.(Int, range(0, requested_steps; length=8)))
    return [step for step in global_steps if step <= total_steps]
end

function configure_cpu_threads(; blas_threads::Int=1, strided_threads::Int=Threads.nthreads())
    blas_threads > 0 || error("BLAS thread count must be positive")
    strided_threads > 0 || error("Strided thread count must be positive")
    BLAS.set_num_threads(blas_threads)
    ITensors.Strided.set_num_threads(strided_threads)
    @info "CPU threading" julia_threads=Threads.nthreads() blas_threads=BLAS.get_num_threads() strided_threads=ITensors.Strided.get_num_threads()
    return nothing
end

function execution_provenance(strict_quality::Bool)
    return (;
        julia_threads=Threads.nthreads(),
        blas_threads=BLAS.get_num_threads(),
        strided_threads=ITensors.Strided.get_num_threads(),
        strict_quality,
        max_boson=MAX_BOSON,
    )
end

function write_run_status(
    path::AbstractString;
    completed_step::Int,
    requested_steps::Int,
    snapshot_count::Int,
    expected_snapshot_count::Int,
    stopped_early::Bool,
    data_path::AbstractString,
)
    state = completed_step == requested_steps && snapshot_count == expected_snapshot_count ?
        "complete" : "partial"
    atomic_replace(path) do temporary
        open(temporary, "w") do io
            println(io, "format=mixing-layer-mps-run-status-v1")
            println(io, "state=$state")
            println(io, "completed_step=$completed_step")
            println(io, "requested_steps=$requested_steps")
            println(io, "snapshot_count=$snapshot_count")
            println(io, "expected_snapshot_count=$expected_snapshot_count")
            println(io, "stopped_early=$stopped_early")
            println(io, "data_file=$(basename(data_path))")
        end
    end
    return state
end

function run_simulation(
    config::MPSMACConfig;
    use_gpu::Bool=false,
    max_steps=nothing,
    make_plot::Bool=true,
    operator_cache_directory=nothing,
    rebuild_operators::Bool=false,
    require_operator_cache::Bool=false,
    checkpoint_path=nothing,
    resume_path=nothing,
    checkpoint_interval::Int=25,
    ceiling_interval::Int=10,
    progress_interval::Int=1,
    stop_after_seconds::Float64=Inf,
    shutdown_reserve_seconds::Float64=0.0,
    stop_file=nothing,
    strict_quality::Bool=false,
    run_status_path=joinpath(config.output_dir, "run_status.txt"),
)
    run_wall_start = time()
    validate_config(config)
    checkpoint_interval >= 0 || error("checkpoint_interval must be nonnegative")
    ceiling_interval >= 0 || error("ceiling_interval must be nonnegative")
    progress_interval > 0 || error("progress_interval must be positive")
    stop_after_seconds > 0 || error("stop_after_seconds must be positive")
    shutdown_reserve_seconds >= 0 || error("shutdown_reserve_seconds must be nonnegative")
    isfinite(stop_after_seconds) && shutdown_reserve_seconds >= stop_after_seconds &&
        error("shutdown reserve must be smaller than the wall-time limit")
    isfinite(stop_after_seconds) && isnothing(checkpoint_path) &&
        error("a checkpoint path is required with stop_after_seconds")
    !isnothing(stop_file) && isnothing(checkpoint_path) &&
        error("a checkpoint path is required with stop_file")
    use_gpu && (!isnothing(checkpoint_path) || !isnothing(resume_path)) &&
        error("HDF5 checkpoint/restart currently supports CPU MPS data only")

    gpu = maybe_enable_gpu(use_gpu)
    requested_steps = round(Int, config.final_time / config.dt)
    target_step = isnothing(max_steps) ? requested_steps : min(max_steps, requested_steps)
    target_step >= 0 || error("max_steps must be nonnegative")
    global_output_steps = snapshot_steps(requested_steps, requested_steps)
    output_set = Set(global_output_steps)
    current_provenance = execution_provenance(strict_quality)
    stop_threshold = stop_after_seconds - shutdown_reserve_seconds
    stop_requested() = (
        (!isnothing(stop_file) && isfile(stop_file)) ||
        (isfinite(stop_threshold) && time() - run_wall_start >= stop_threshold)
    )

    checkpoint = isnothing(resume_path) ? nothing : load_checkpoint(resume_path, config)
    checkpoint_state = isnothing(checkpoint) ? nothing : checkpoint.state
    sites_override = isnothing(checkpoint_state) ? nothing : siteinds(checkpoint_state)

    @info "Loading/building periodic Chorin/MAC/scalar operators" sites=4number_of_cells(config)
    operator_measurement = @timed load_or_build_operators(
        config;
        cache_directory=operator_cache_directory,
        sites=sites_override,
        rebuild=rebuild_operators,
        require_cache=require_operator_cache,
    )
    bundle, cache_path = operator_measurement.value
    sites = bundle.sites
    predictor_mpo = bundle.predictor
    pressure_mpo = bundle.pressure
    correction_mpo = bundle.correction
    cache_hit = bundle.cache_hit
    @info "MPO bond dimensions" cache_hit cache_path operator_seconds=operator_measurement.time operator_allocated_gib=operator_measurement.bytes / 2.0^30 max_rss_gib=Sys.maxrss() / 2.0^30 predictor_components=join(maxlinkdim.(predictor_mpo), ",") predictor_max=operator_maxlinkdim(predictor_mpo) pressure=maxlinkdim(pressure_mpo) correction=maxlinkdim(correction_mpo)

    initial_state_seconds = 0.0
    initial_state_allocated_bytes = 0
    if isnothing(checkpoint)
        initial = validate_initial_fields(config)
        @info "Building direct coherent product MPS" sites=length(sites) max_boson=MAX_BOSON
        state_measurement = @timed build_initial_mps(initial, sites, config)
        state = state_measurement.value
        initial_state_seconds = state_measurement.time
        initial_state_allocated_bytes = state_measurement.bytes
        completed_step = 0
        snapshot_step_history = Int[]
        times = Float64[]
        fields_history = Any[]
        omega_history = Matrix{Float64}[]
        diagnostics_history = Any[]
        step_diagnostics_history = Any[]
        defects = (
            pressure_residual=NaN,
            velocity_leakage=0.0,
            pressure_scalar_leakage=0.0,
            correction_defect=0.0,
            pressure_leakage=0.0,
            correction_scalar_leakage=0.0,
        )
        run_id = string("run-", getpid(), "-", time_ns())
    else
        state = checkpoint_state
        length(state) == length(sites) || error("checkpoint/operator site count mismatch")
        if siteinds(state) != sites
            for (old_site, new_site) in zip(siteinds(state), sites)
                dim(old_site) == dim(new_site) || error("checkpoint/operator site dimension mismatch")
            end
            replace_siteinds!(state, sites)
        end
        metadata = checkpoint.metadata
        metadata.operator_fingerprint == operator_fingerprint(config) ||
            error("checkpoint operator fingerprint mismatch")
        metadata.requested_steps == requested_steps ||
            error("checkpoint requested-step count mismatch")
        metadata.output_steps == global_output_steps ||
            error("checkpoint snapshot schedule mismatch")
        metadata.execution_provenance == current_provenance || error(
            "checkpoint CPU-thread/strict-quality provenance mismatch: " *
            "stored=$(metadata.execution_provenance), current=$current_provenance",
        )
        completed_step = metadata.completed_step
        snapshot_step_history = collect(metadata.snapshot_step_history)
        times = collect(metadata.times)
        fields_history = Any[metadata.fields_history...]
        omega_history = Matrix{Float64}[metadata.omega_history...]
        diagnostics_history = Any[metadata.diagnostics_history...]
        step_diagnostics_history = Any[metadata.step_diagnostics_history...]
        defects = metadata.defects
        run_id = metadata.run_id
        @info "Resumed MPS checkpoint" path=checkpoint.path completed_step run_id
    end
    completed_step <= target_step ||
        error("checkpoint step $completed_step exceeds requested target step $target_step")

    state = to_device(state, gpu)
    predictor_mpo = to_device(predictor_mpo, gpu)
    pressure_mpo = to_device(pressure_mpo, gpu)
    correction_mpo = to_device(correction_mpo, gpu)

    last_fields = nothing
    last_quality = nothing
    function record!(step; fields=nothing, quality=nothing)
        step in snapshot_step_history && return fields, quality
        isnothing(fields) &&
            (fields = field_expectations(state, config; include_ceiling=true))
        if !isfinite(fields.ceiling_probability)
            fields = field_expectations(state, config; include_ceiling=true)
        end
        isnothing(quality) && (quality = step_quality(state, fields, config))
        diagnostic = snapshot_diagnostics(state, fields, defects, config; quality)
        push!(snapshot_step_history, step)
        push!(times, step * config.dt)
        push!(fields_history, fields)
        push!(omega_history, mac_vorticity(fields.u, fields.v, grid_spacing(config)))
        push!(diagnostics_history, diagnostic)
        @printf(
            "snapshot %d/8 step=%d t=%6.3f E=%.6e mean(c)=%.6e dc=%.3e max|div|=%.3e Poisson=%.3e chi=%d ceiling=%.3e\n",
            length(times), step, times[end], diagnostic.kinetic_energy,
            diagnostic.scalar_mass, diagnostic.scalar_mass_error,
            diagnostic.max_divergence, diagnostic.pressure_residual,
            diagnostic.max_bond, diagnostic.ceiling_probability,
        )
        flush(stdout)
        return fields, quality
    end

    if isnothing(checkpoint)
        last_fields, last_quality = record!(0)
    end

    evolution_wall_start = time()
    last_checkpoint_step = -1
    stopped_for_walltime = false
    step_wall_seconds = Float64[]

    function checkpoint_metadata(step)
        return (;
            completed_step=step,
            requested_steps,
            output_steps=global_output_steps,
            snapshot_step_history=copy(snapshot_step_history),
            times=copy(times),
            fields_history=copy(fields_history),
            omega_history=copy(omega_history),
            diagnostics_history=copy(diagnostics_history),
            step_diagnostics_history=copy(step_diagnostics_history),
            defects,
            operator_fingerprint=operator_fingerprint(config),
            evolution_fingerprint=evolution_fingerprint(config),
            operator_cache_path=cache_path,
            execution_provenance=current_provenance,
            run_id,
        )
    end

    function checkpoint!(step)
        isnothing(checkpoint_path) && return 0.0
        seconds = @elapsed write_checkpoint(
            checkpoint_path,
            state,
            checkpoint_metadata(step),
            config,
        )
        @info "Committed restart checkpoint" checkpoint_path step seconds
        return seconds
    end

    for step in completed_step+1:target_step
        if stop_requested()
            stopped_for_walltime = true
            break
        end

        checkpoint_due = !isnothing(checkpoint_path) && checkpoint_interval > 0 &&
            step % checkpoint_interval == 0
        ceiling_due = ceiling_interval > 0 && step % ceiling_interval == 0
        sample_ceiling = ceiling_due || checkpoint_due || step in output_set || step == target_step
        step_start = time()
        state, fields, defects, blocks, stage_timing = chorin_step(
            state,
            predictor_mpo,
            pressure_mpo,
            correction_mpo,
            config;
            include_ceiling=sample_ceiling,
            use_gpu=gpu,
        )
        quality = step_quality(state, fields, config)
        check_step_quality(quality, defects, blocks, step, config; strict=strict_quality)
        step_seconds = time() - step_start
        push!(step_wall_seconds, step_seconds)
        push!(step_diagnostics_history, (
            step,
            time=step * config.dt,
            poisson_blocks=blocks,
            pressure_sweeps=blocks * config.poisson_steps_per_block,
            pressure_residual=defects.pressure_residual,
            max_divergence=quality.max_divergence,
            relative_divergence=quality.relative_divergence,
            pressure_velocity_leakage=defects.velocity_leakage,
            pressure_scalar_leakage=defects.pressure_scalar_leakage,
            correction_defect=defects.correction_defect,
            correction_pressure_leakage=defects.pressure_leakage,
            correction_scalar_leakage=defects.correction_scalar_leakage,
            scalar_mass=quality.scalar_mass,
            scalar_mass_error=quality.scalar_mass_error,
            scalar_minimum=quality.scalar_minimum,
            scalar_maximum=quality.scalar_maximum,
            pressure_gauge=quality.pressure_gauge,
            max_imaginary=quality.max_imaginary,
            max_bond=quality.max_bond,
            ceiling_probability=quality.ceiling_probability,
            ceiling_sampled=isfinite(quality.ceiling_probability),
            predictor_seconds=stage_timing.predictor_seconds,
            pressure_seconds=stage_timing.pressure_seconds,
            correction_seconds=stage_timing.correction_seconds,
            step_seconds,
        ))
        completed_step = step
        last_fields = fields
        last_quality = quality
        step in output_set && ((last_fields, last_quality) = record!(step; fields, quality))

        if step % progress_interval == 0 || step == target_step
            mean_step = mean(step_wall_seconds)
            eta = mean_step * (target_step - step)
            @printf(
                "step %d/%d wall=%.2fs pred=%.2fs pressure=%.2fs corr=%.2fs blocks=%d chi=%d ETA=%.1fh\n",
                step, target_step, step_seconds, stage_timing.predictor_seconds,
                stage_timing.pressure_seconds, stage_timing.correction_seconds,
                blocks, quality.max_bond, eta / 3600,
            )
            flush(stdout)
        end

        if checkpoint_due
            checkpoint!(step)
            last_checkpoint_step = step
        end

        if stop_requested()
            stopped_for_walltime = true
            break
        end
    end

    if isnothing(last_fields)
        last_fields = field_expectations(state, config; include_ceiling=true)
        last_quality = step_quality(state, last_fields, config)
    elseif !isfinite(last_fields.ceiling_probability)
        last_fields = field_expectations(state, config; include_ceiling=true)
        last_quality = step_quality(state, last_fields, config)
        if !isempty(step_diagnostics_history) && last(step_diagnostics_history).step == completed_step
            step_diagnostics_history[end] = merge(
                last(step_diagnostics_history),
                (;
                    ceiling_probability=last_quality.ceiling_probability,
                    ceiling_sampled=true,
                ),
            )
        end
    end

    if strict_quality && completed_step > 0
        final_blocks = isempty(step_diagnostics_history) ? 0 :
            last(step_diagnostics_history).poisson_blocks
        check_step_quality(
            last_quality,
            defects,
            final_blocks,
            completed_step,
            config;
            strict=true,
        )
    end

    if !isnothing(checkpoint_path) && last_checkpoint_step != completed_step
        checkpoint!(completed_step)
        last_checkpoint_step = completed_step
    end

    output_stem = completed_step == requested_steps ?
        "mixing_layer_mps_mac" : @sprintf("mixing_layer_mps_mac_steps_%06d", completed_step)
    data_path = joinpath(config.output_dir, output_stem * ".jld2")
    figure_path = joinpath(config.output_dir, output_stem * "_vorticity.png")
    run_metadata = Dict{String,Any}(
        "run_id" => run_id,
        "operator_cache_path" => cache_path,
        "operator_cache_hit" => cache_hit,
        "operator_seconds" => operator_measurement.time,
        "operator_allocated_bytes" => operator_measurement.bytes,
        "initial_state_seconds" => initial_state_seconds,
        "initial_state_allocated_bytes" => initial_state_allocated_bytes,
        "evolution_wall_seconds" => time() - evolution_wall_start,
        "total_wall_seconds" => time() - run_wall_start,
        "stopped_for_walltime" => stopped_for_walltime,
        "stop_file" => stop_file,
        "shutdown_reserve_seconds" => shutdown_reserve_seconds,
        "strict_quality" => strict_quality,
        "julia_threads" => Threads.nthreads(),
        "blas_threads" => BLAS.get_num_threads(),
        "strided_threads" => ITensors.Strided.get_num_threads(),
        "max_rss_bytes" => Sys.maxrss(),
    )
    save_results(
        times,
        fields_history,
        omega_history,
        diagnostics_history,
        step_diagnostics_history,
        config,
        data_path;
        snapshot_step_history,
        completed_step,
        requested_steps,
        terminal_fields=last_fields,
        run_metadata,
    )
    if make_plot
        plot_vorticity_snapshots(times, omega_history, config, figure_path)
        @info "Saved results" data_path figure_path completed_step
    else
        @info "Saved results" data_path completed_step
    end
    completion_state = write_run_status(
        run_status_path;
        completed_step,
        requested_steps,
        snapshot_count=length(snapshot_step_history),
        expected_snapshot_count=length(global_output_steps),
        stopped_early=stopped_for_walltime,
        data_path,
    )
    @info "Run resource summary" completed_step total_wall_seconds=run_metadata["total_wall_seconds"] max_rss_gib=Sys.maxrss() / 2.0^30
    return (;
        times,
        snapshot_step_history,
        fields_history,
        omega_history,
        diagnostics_history,
        step_diagnostics_history,
        state,
        completed_step,
        requested_steps,
        data_path,
        checkpoint_path,
        operator_cache_path=cache_path,
        run_metadata,
        completion_state,
        run_status_path,
    )
end

function smoke_config(; output_dir="outputs/mps_mac_smoke", steps::Int=1)
    production = MPSMACConfig()
    return MPSMACConfig(
        n=4,
        reynolds=production.reynolds,
        peclet=production.peclet,
        dt=production.dt,
        final_time=steps * production.dt,
        middle_fraction=production.middle_fraction,
        transition_thickness=production.transition_thickness,
        kh_width=production.kh_width,
        kh_mode=production.kh_mode,
        kh_amplitude=production.kh_amplitude,
        secondary_mode=production.secondary_mode,
        secondary_amplitude=production.secondary_amplitude,
        kh_phase=production.kh_phase,
        velocity_scale=production.velocity_scale,
        pressure_impulse_scale=production.pressure_impulse_scale,
        scalar_scale=production.scalar_scale,
        maxdim=8,
        cutoff=1e-8,
        poisson_pseudo_dt=0.01,
        poisson_steps_per_block=5,
        poisson_max_blocks=4,
        poisson_tolerance=0.02,
        correction_steps=2,
        predictor_nsite=production.predictor_nsite,
        predictor_chunks=1,
        krylov_tolerance=production.krylov_tolerance,
        krylov_dimension=production.krylov_dimension,
        krylov_maxiter=production.krylov_maxiter,
        output_dir=output_dir,
    )
end

function smoke_test(
    ;
    output_dir="outputs/mps_mac_smoke",
    operator_cache_directory=nothing,
    checkpoint_path=nothing,
)
    config = smoke_config(; output_dir)
    result = run_simulation(
        config;
        max_steps=1,
        make_plot=false,
        operator_cache_directory,
        checkpoint_path,
        checkpoint_interval=1,
        ceiling_interval=1,
    )
    final = last(result.diagnostics_history)
    isfinite(final.kinetic_energy) || error("smoke test produced non-finite energy")
    final.pressure_residual < 0.02 || error("smoke-test pressure relaxation failed")
    final.relative_divergence < 1e-4 || error("smoke-test projection failed")
    final.scalar_mass_error < SCALAR_MASS_TOLERANCE ||
        error("smoke-test scalar conservation failed")
    @printf("MPS Chorin/MAC smoke test passed; final max|div|=%.3e\n", final.max_divergence)
    return result
end

function argument_value(args, flag, default)
    index = findfirst(==(flag), args)
    index === nothing && return default
    index < length(args) || error("$flag requires a value")
    return parse(typeof(default), args[index + 1])
end

function optional_argument(args, flag)
    index = findfirst(==(flag), args)
    index === nothing && return nothing
    index < length(args) || error("$flag requires a value")
    return args[index + 1]
end

const CLI_VALUE_FLAGS = Set([
    "--steps", "--n", "--re", "--pe", "--dt", "--final-time", "--middle-fraction",
    "--transition", "--kh-width", "--kh-mode", "--kh-amplitude",
    "--secondary-mode", "--secondary-amplitude", "--maxdim", "--cutoff",
    "--poisson-tol", "--poisson-blocks", "--poisson-steps",
    "--predictor-nsite", "--pressure-nsite", "--correction-nsite",
    "--predictor-chunks", "--krylov-tol", "--krylov-dim", "--krylov-maxiter",
    "--output-dir", "--cache-dir", "--checkpoint", "--resume",
    "--checkpoint-interval", "--ceiling-interval", "--progress-interval",
    "--stop-after-seconds", "--shutdown-reserve-seconds", "--stop-file",
    "--run-status", "--blas-threads", "--strided-threads", "--scalar-scale",
])

const CLI_SWITCH_FLAGS = Set([
    "--validate", "--smoke-test", "--no-cache", "--rebuild-operators",
    "--require-cache", "--no-checkpoint", "--strict-quality", "--gpu",
    "--no-plot", "--help",
])

function validate_cli_arguments(args)
    seen = Set{String}()
    index = 1
    while index <= length(args)
        flag = args[index]
        flag in seen && error("duplicate command-line option: $flag")
        push!(seen, flag)
        if flag in CLI_SWITCH_FLAGS
            index += 1
        elseif flag in CLI_VALUE_FLAGS
            index < length(args) || error("$flag requires a value")
            startswith(args[index + 1], "--") &&
                error("$flag requires a value; found option $(args[index + 1])")
            index += 2
        else
            error("unknown command-line option: $flag (use --help)")
        end
    end
    return nothing
end

function print_help()
    println("""
    Usage:
      julia --project=. mixing_layer_mps_mac.jl [options]

    Options:
      --validate          validate the 32x32 divergence-free initial condition only
      --smoke-test        build a 4x4 MPS/MPO and execute one complete Chorin step
      --steps N           limit the default 32x32 run to N physical steps
      --n N               override the square grid size (default 32)
      --re VALUE          override Re (default 100)
      --pe VALUE          override scalar Pe (default: same value as Re)
      --dt VALUE          override physical dt (default 0.0025)
      --final-time VALUE  override final time (default 3.5)
      --middle-fraction F override middle-layer width fraction (default 0.30)
      --transition D      override tanh transition parameter (default 0.06)
      --kh-width VALUE    override KH envelope width (default 0.10)
      --kh-mode N         override primary KH mode (default 1)
      --kh-amplitude A    override primary KH amplitude (default 0.10)
      --secondary-mode N  override secondary KH mode (default 2)
      --secondary-amplitude A
                          override secondary KH amplitude (default 0.025)
      --scalar-scale S    coherent-amplitude scale for scalar (default 4)
      --maxdim N          override maximum MPS bond dimension (default 64)
      --cutoff VALUE      override TDVP truncation cutoff (default 1e-10)
      --poisson-tol VALUE override pressure-relaxation residual target (default 1e-3)
      --poisson-blocks N  override maximum pressure-relaxation blocks (default 12)
      --poisson-steps N   override pressure TDVP sweeps per residual check (default 20)
      --predictor-nsite N predictor TDVP update size, 1 or 2 (default 1)
      --pressure-nsite N  pressure TDVP update size, 1 or 2 (default 1)
      --correction-nsite N
                          correction TDVP update size, 1 or 2 (default 1)
      --predictor-chunks N
                          row chunks per predictor family (default 8)
      --krylov-tol VALUE  local exponential tolerance (default 1e-10)
      --krylov-dim N      local Krylov dimension (default 30)
      --krylov-maxiter N  local Krylov restart limit (default 100)
      --output-dir PATH   result directory (default outputs/mps_mac)
      --cache-dir PATH    persistent HDF5 operator-cache directory
      --no-cache          disable the static operator cache
      --rebuild-operators ignore and replace a compatible operator cache
      --require-cache     fail if the persistent cache cannot be written
      --checkpoint PATH   restart checkpoint path
      --no-checkpoint     disable periodic checkpoints
      --resume PATH       resume a compatible checkpoint
      --checkpoint-interval N
                          checkpoint every N physical steps (default 25)
      --ceiling-interval N
                          sample boson-ceiling occupation every N steps (default 10)
      --progress-interval N
                          print timing/ETA every N steps (default 1)
      --stop-after-seconds VALUE
                          allocation wall-time budget in seconds
      --shutdown-reserve-seconds VALUE
                          reserve for final diagnostics/checkpoint/output (default 0)
      --stop-file PATH    stop at the next complete Chorin step when PATH appears
      --run-status PATH   atomic machine-readable completion-status path
      --strict-quality    exit nonzero when any production quality gate fails
      --blas-threads N    BLAS threads (default 1; avoids nested threading)
      --strided-threads N Strided threads (default Julia thread count)
      --gpu               use CUDA if installed and functional
      --no-plot           skip CairoMakie output
      --help              show this message
    """)
end

function main(args=ARGS)
    validate_cli_arguments(args)
    "--help" in args && return print_help()

    output_dir = something(optional_argument(args, "--output-dir"), "outputs/mps_mac")
    reynolds = argument_value(args, "--re", 100.0)
    config = MPSMACConfig(
        n=argument_value(args, "--n", 32),
        reynolds=reynolds,
        peclet=argument_value(args, "--pe", reynolds),
        dt=argument_value(args, "--dt", 0.0025),
        final_time=argument_value(args, "--final-time", 3.5),
        middle_fraction=argument_value(args, "--middle-fraction", 0.30),
        transition_thickness=argument_value(args, "--transition", 0.06),
        kh_width=argument_value(args, "--kh-width", 0.10),
        kh_mode=argument_value(args, "--kh-mode", 1),
        kh_amplitude=argument_value(args, "--kh-amplitude", 0.10),
        secondary_mode=argument_value(args, "--secondary-mode", 2),
        secondary_amplitude=argument_value(args, "--secondary-amplitude", 0.025),
        scalar_scale=argument_value(args, "--scalar-scale", 4.0),
        maxdim=argument_value(args, "--maxdim", 64),
        cutoff=argument_value(args, "--cutoff", 1.0e-10),
        poisson_steps_per_block=argument_value(args, "--poisson-steps", 20),
        poisson_tolerance=argument_value(args, "--poisson-tol", 1.0e-3),
        poisson_max_blocks=argument_value(args, "--poisson-blocks", 12),
        predictor_nsite=argument_value(args, "--predictor-nsite", 1),
        pressure_nsite=argument_value(args, "--pressure-nsite", 1),
        correction_nsite=argument_value(args, "--correction-nsite", 1),
        predictor_chunks=argument_value(args, "--predictor-chunks", 8),
        krylov_tolerance=argument_value(args, "--krylov-tol", 1.0e-10),
        krylov_dimension=argument_value(args, "--krylov-dim", 30),
        krylov_maxiter=argument_value(args, "--krylov-maxiter", 100),
        output_dir=output_dir,
    )
    if "--validate" in args
        validate_initial_fields(config)
        return
    end
    configure_cpu_threads(
        ;
        blas_threads=argument_value(args, "--blas-threads", 1),
        strided_threads=argument_value(args, "--strided-threads", Threads.nthreads()),
    )

    cache_directory = "--no-cache" in args ? nothing : something(
        optional_argument(args, "--cache-dir"),
        joinpath(output_dir, "operator_cache"),
    )
    resume_path = optional_argument(args, "--resume")
    checkpoint_path = "--no-checkpoint" in args ? nothing : something(
        optional_argument(args, "--checkpoint"),
        resume_path,
        joinpath(output_dir, "mixing_layer_mps_checkpoint.h5"),
    )
    if "--smoke-test" in args
        return smoke_test(
            ;
            output_dir=output_dir,
            operator_cache_directory=cache_directory,
            checkpoint_path,
        )
    end

    max_steps = argument_value(args, "--steps", -1)
    run_simulation(
        config;
        use_gpu="--gpu" in args,
        max_steps=max_steps < 0 ? nothing : max_steps,
        make_plot=!("--no-plot" in args),
        operator_cache_directory=cache_directory,
        rebuild_operators="--rebuild-operators" in args,
        require_operator_cache="--require-cache" in args,
        checkpoint_path,
        resume_path,
        checkpoint_interval=argument_value(args, "--checkpoint-interval", 25),
        ceiling_interval=argument_value(args, "--ceiling-interval", 10),
        progress_interval=argument_value(args, "--progress-interval", 1),
        stop_after_seconds=argument_value(args, "--stop-after-seconds", Inf),
        shutdown_reserve_seconds=argument_value(args, "--shutdown-reserve-seconds", 0.0),
        stop_file=optional_argument(args, "--stop-file"),
        strict_quality="--strict-quality" in args,
        run_status_path=something(
            optional_argument(args, "--run-status"),
            joinpath(output_dir, "run_status.txt"),
        ),
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
