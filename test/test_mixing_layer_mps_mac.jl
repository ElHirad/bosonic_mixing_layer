using Test
using Random

include(joinpath(@__DIR__, "..", "mixing_layer_mps_mac.jl"))

@testset "serpentine MAC indexing" begin
    n = MPSMACConfig().n
    for k in 1:n^2
        j, i = grid_from_cell(k, n)
        @test cell_from_grid(j, i, n) == k
    end
end

@testset "periodic MAC operator compatibility" begin
    n = 7
    h = 1 / n
    p = [sin(2pi * (i - 1) / n) * cos(2pi * (j - 1) / n) for j in 1:n, i in 1:n]
    u = [cos(4pi * (i - 1) / n) + 0.1j for j in 1:n, i in 1:n]
    v = [sin(2pi * (j - 1) / n) - 0.2i for j in 1:n, i in 1:n]
    gx, gy = mac_pressure_gradient(p, h)
    lhs = sum(p .* mac_divergence(u, v, h))
    rhs = -sum(gx .* u .+ gy .* v)
    @test lhs ≈ rhs atol=1e-11 rtol=1e-11
    @test mac_divergence(gx, gy, h) ≈ periodic_laplacian(p, h) atol=1e-11 rtol=1e-11

    # The centered conservative flux form must not create kinetic energy for
    # any periodic, discretely divergence-free velocity field.
    Random.seed!(20260817)
    psi = randn(n, n)
    u_divfree = zeros(n, n)
    v_divfree = zeros(n, n)
    for j in 1:n, i in 1:n
        ip = next_index(i, n)
        jp = next_index(j, n)
        u_divfree[j, i] = 0.3 + (psi[jp, i] - psi[j, i]) / h
        v_divfree[j, i] = -(psi[j, ip] - psi[j, i]) / h
    end
    advection_u, advection_v = mac_conservative_advection(u_divfree, v_divfree, h)
    @test maximum(abs.(mac_divergence(u_divfree, v_divfree, h))) < 1e-11
    @test abs(sum(u_divfree .* advection_u .+ v_divfree .* advection_v)) < 1e-9

    scalar = randn(n, n)
    scalar_advection = mac_conservative_scalar_advection(
        u_divfree,
        v_divfree,
        scalar,
        h,
    )
    @test abs(sum(scalar_advection)) < 1e-10
    @test abs(sum(periodic_laplacian(scalar, h))) < 1e-10
end

@testset "32x32 visible-mixing initialization" begin
    config = MPSMACConfig()
    fields = validate_initial_fields(config; verbose=false)
    h = grid_spacing(config)
    @test maximum(abs.(mac_divergence(fields.u, fields.v, h))) < 1e-12
    @test maximum(abs.(vec(mean(fields.u; dims=2)) .- fields.target_u)) < 1e-12
    @test mean(fields.u) ≈ mean(fields.target_u) atol=1e-14
    @test minimum(fields.target_u) ≈ -1.0 atol=1e-14
    @test maximum(fields.target_u) ≈ 1.0 atol=1e-14
    @test all(iszero, fields.phi)
    @test minimum(fields.scalar) ≈ 0.0 atol=1e-14
    @test maximum(fields.scalar) ≈ 1.0 atol=1e-14
    @test vec(mean(fields.scalar; dims=2)) ≈ fields.target_scalar atol=1e-14
    @test mean(fields.scalar) ≈ scalar_reference_mass(config) atol=1e-14

    advection_u, advection_v = mac_conservative_advection(fields.u, fields.v, h)
    inviscid_energy_rate = mean(fields.u .* advection_u .+ fields.v .* advection_v)
    @test abs(inviscid_energy_rate) < 1e-12
end

@testset "eight snapshot schedule" begin
    @test snapshot_steps(1400, 1400) == collect(0:200:1400)
    @test length(snapshot_steps(1400, 1400)) == 8
    @test snapshot_steps(3, 1400) == [0]
    @test snapshot_steps(201, 1400) == [0, 200]
end

@testset "direct coherent product MPS" begin
    config = smoke_config()
    fields = initial_mac_fields(config)
    sites = siteinds("MACBoson", 4config.n^2; conserve_qns=false)
    direct = build_initial_mps(fields, sites, config)

    reference = MPS(sites, fill(1, length(sites)))
    for k in 1:config.n^2
        j, i = grid_from_cell(k, config.n)
        amplitudes = (
            fields.u[j, i] / config.velocity_scale,
            fields.v[j, i] / config.velocity_scale,
            fields.phi[j, i] / config.pressure_impulse_scale,
            fields.scalar[j, i] / config.scalar_scale,
        )
        for (alpha, site_number) in zip(
            amplitudes,
            (usite(k), vsite(k), psite(k), csite(k)),
        )
            gate = exp(
                alpha * op("adag", sites[site_number]) -
                conj(alpha) * op("a", sites[site_number])
            )
            reference = apply(gate, reference; cutoff=config.cutoff, maxdim=config.maxdim)
        end
    end
    normalize!(reference)

    @test maxlinkdim(direct) == 1
    @test norm(direct) ≈ 1 atol=1e-13
    @test abs(inner(direct, reference)) ≈ 1 atol=1e-12
    @test expect(direct, "a") ≈ expect(reference, "a") atol=1e-12 rtol=1e-12
end

@testset "periodic pressure gauge projection" begin
    config = smoke_config()
    initial = initial_mac_fields(config)
    h = grid_spacing(config)
    shifted_phi = [
        0.01 + 0.02sin(2pi * (i - 1) / config.n) * cos(2pi * (j - 1) / config.n)
        for j in 1:config.n, i in 1:config.n
    ]
    shifted_initial = merge(initial, (; phi=shifted_phi))
    sites = siteinds("MACBoson", 4config.n^2; conserve_qns=false)
    shifted = build_initial_mps(shifted_initial, sites, config)
    before = field_expectations(shifted, config)
    before_gx, before_gy = mac_pressure_gradient(before.phi, h)

    centered, after = enforce_pressure_gauge(shifted, before, config)
    after_gx, after_gy = mac_pressure_gradient(after.phi, h)

    @test pressure_gauge_ratio(before, config) > 1e-2
    @test pressure_gauge_ratio(after, config) <= PRESSURE_GAUGE_TARGET
    @test abs(mean(after.phi)) < 1e-13
    @test relative_velocity_difference(before, after) < 1e-13
    @test relative_scalar_difference(before, after) < 1e-13
    @test after_gx ≈ before_gx atol=1e-13 rtol=1e-13
    @test after_gy ≈ before_gy atol=1e-13 rtol=1e-13
    @test maxlinkdim(centered) == maxlinkdim(shifted)
    @test norm(centered) ≈ 1 atol=1e-13
end

@testset "one-site predictor regression" begin
    config = smoke_config()
    sites = siteinds("MACBoson", 4config.n^2; conserve_qns=false)
    initial = initial_mac_fields(config)
    state = build_initial_mps(initial, sites, config)
    predictor = build_predictor_mpo(sites, config)

    one_site = evolve_tdvp(predictor, config.dt, state, config; nsite=1)
    two_site = evolve_tdvp(predictor, config.dt, state, config; nsite=2)
    one_site_fields = field_expectations(one_site, config)
    two_site_fields = field_expectations(two_site, config)
    one_site_quality = step_quality(one_site, one_site_fields, config)
    two_site_quality = step_quality(two_site, two_site_fields, config)

    @test relative_velocity_difference(one_site_fields, two_site_fields) < 1e-7
    @test relative_scalar_difference(one_site_fields, two_site_fields) < 1e-7
    @test one_site_fields.phi ≈ two_site_fields.phi atol=1e-12 rtol=1e-12
    @test one_site_quality.relative_divergence ≈
        two_site_quality.relative_divergence atol=1e-8 rtol=1e-5
    @test one_site_quality.max_bond <= two_site_quality.max_bond
end

@testset "stable HPC fingerprints" begin
    first_config = MPSMACConfig(output_dir="first")
    second_config = MPSMACConfig(output_dir="second")
    more_pressure_blocks = MPSMACConfig(poisson_max_blocks=13)
    @test first_config.poisson_max_blocks == 12
    @test evolution_fingerprint(first_config) == evolution_fingerprint(second_config)
    @test operator_fingerprint(first_config) == operator_fingerprint(second_config)
    @test operator_fingerprint(first_config) == operator_fingerprint(more_pressure_blocks)
    @test evolution_fingerprint(first_config) != evolution_fingerprint(more_pressure_blocks)
    @test evolution_fingerprint(MPSMACConfig(dt=nextfloat(first_config.dt))) !=
        evolution_fingerprint(first_config)
    @test operator_fingerprint(MPSMACConfig(reynolds=101.0)) !=
        operator_fingerprint(first_config)
    @test operator_fingerprint(MPSMACConfig(peclet=101.0)) !=
        operator_fingerprint(first_config)
    @test operator_fingerprint(MPSMACConfig(predictor_chunks=4)) !=
        operator_fingerprint(first_config)
end

@testset "CLI Pe defaults and override" begin
    @test isnothing(main(["--validate", "--n", "16", "--re", "50"]))
    @test isnothing(main([
        "--validate",
        "--n", "16",
        "--re", "50",
        "--pe", "25",
        "--transition", "0.12",
        "--kh-width", "0.20",
    ]))
end
