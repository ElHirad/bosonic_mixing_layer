using Test

include("test_mixing_layer_mps_mac.jl")

@testset "bosonic MPS Chorin/MAC integration" begin
    mktempdir() do output_dir
        cache_directory = joinpath(output_dir, "operator_cache")
        checkpoint_path = joinpath(output_dir, "checkpoint.h5")
        result = smoke_test(; output_dir, operator_cache_directory=cache_directory, checkpoint_path)
        final = last(result.diagnostics_history)
        @test final.pressure_residual < 0.02
        @test final.relative_divergence < 1e-4
        @test final.pressure_velocity_leakage < 1e-5
        @test final.pressure_scalar_leakage < 1e-5
        @test final.correction_defect < 1e-3
        @test final.correction_pressure_leakage < 1e-5
        @test final.correction_scalar_leakage < 1e-5
        @test final.scalar_mass_error < SCALAR_MASS_TOLERANCE
        @test final.pressure_gauge < 1e-4
        @test final.ceiling_probability < 1e-4
        @test length(result.step_diagnostics_history) == 1
        data_path = joinpath(output_dir, "mixing_layer_mps_mac.jld2")
        @test isfile(data_path)
        saved = load(data_path)
        @test saved["times"] == result.times
        @test size(saved["u"]) == (4, 4, 2)
        @test size(saved["scalar"]) == (4, 4, 2)
        @test haskey(saved, "terminal_scalar")
        @test haskey(saved, "step_diagnostics")
        @test saved["snapshot_steps"] == [0, 1]
        @test saved["completed_step"] == 1
        @test saved["parameters"]["max_boson"] == MAX_BOSON
        @test saved["parameters"]["reynolds"] == 100.0
        @test saved["parameters"]["peclet"] == 100.0
        @test saved["parameters"]["kh_amplitude"] == 0.10
        @test saved["parameters"]["secondary_amplitude"] == 0.025

        loaded = load_checkpoint(checkpoint_path, smoke_config(; output_dir))
        @test loaded.metadata.completed_step == 1
        @test field_expectations(loaded.state, smoke_config(; output_dir)).u ≈
            field_expectations(result.state, smoke_config(; output_dir)).u atol=1e-12 rtol=1e-12

        cache_config = smoke_config(; output_dir)
        cached, cache_path = load_or_build_operators(
            cache_config;
            cache_directory,
        )
        @test cached.cache_hit
        @test isfile(cache_path)
        @test length(cached.predictor) == 8
        @test siteinds(cached.site_template) == siteinds(loaded.state)
    end
end

@testset "checkpoint resume matches continuous evolution" begin
    mktempdir() do root
        cache_directory = joinpath(root, "operator_cache")
        continuous_config = smoke_config(; output_dir=joinpath(root, "continuous"), steps=2)
        continuous = run_simulation(
            continuous_config;
            max_steps=2,
            make_plot=false,
            operator_cache_directory=cache_directory,
            ceiling_interval=1,
            progress_interval=10,
        )

        split_config = smoke_config(; output_dir=joinpath(root, "split"), steps=2)
        checkpoint_path = joinpath(root, "split", "checkpoint.h5")
        first_half = run_simulation(
            split_config;
            max_steps=1,
            make_plot=false,
            operator_cache_directory=cache_directory,
            checkpoint_path,
            checkpoint_interval=1,
            ceiling_interval=1,
            progress_interval=10,
        )
        @test first_half.completed_step == 1

        resumed = run_simulation(
            split_config;
            max_steps=2,
            make_plot=false,
            operator_cache_directory=cache_directory,
            checkpoint_path,
            resume_path=checkpoint_path,
            checkpoint_interval=1,
            ceiling_interval=1,
            progress_interval=10,
        )
        @test resumed.completed_step == 2
        @test resumed.snapshot_step_history == [0, 1, 2]
        @test length(resumed.step_diagnostics_history) == 2

        continuous_fields = field_expectations(continuous.state, continuous_config)
        resumed_fields = field_expectations(resumed.state, split_config)
        @test continuous_fields.u ≈ resumed_fields.u atol=1e-10 rtol=1e-10
        @test continuous_fields.v ≈ resumed_fields.v atol=1e-10 rtol=1e-10
        @test continuous_fields.phi ≈ resumed_fields.phi atol=1e-10 rtol=1e-10
        @test continuous_fields.scalar ≈ resumed_fields.scalar atol=1e-10 rtol=1e-10
        fidelity = abs(inner(continuous.state, resumed.state)) /
            (norm(continuous.state) * norm(resumed.state))
        @test fidelity ≈ 1 atol=1e-10
        @test getproperty.(continuous.step_diagnostics_history, :poisson_blocks) ==
            getproperty.(resumed.step_diagnostics_history, :poisson_blocks)

        latest = load_checkpoint(checkpoint_path, split_config)
        @test latest.metadata.completed_step == 2
        open(latest.path, "w") do io
            write(io, "deliberately truncated checkpoint generation")
        end
        fallback = load_checkpoint(checkpoint_path, split_config)
        @test fallback.metadata.completed_step == 1
        @test fallback.path != latest.path

        # A commit immediately after fallback must not destroy the known-good
        # generation even though the manifest still points at the corrupt one.
        write_checkpoint(checkpoint_path, fallback.state, fallback.metadata, split_config)
        after_fallback_commit = load_checkpoint(checkpoint_path, split_config)
        @test after_fallback_commit.metadata.completed_step == 1

        # A later recovered step supersedes both step-1 generations, and the
        # loader ranks valid generations by completed step rather than pointer age.
        write_checkpoint(checkpoint_path, resumed.state, latest.metadata, split_config)
        recovered = load_checkpoint(checkpoint_path, split_config)
        @test recovered.metadata.completed_step == 2
    end
end
