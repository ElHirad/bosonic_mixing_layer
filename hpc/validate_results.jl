#!/usr/bin/env julia

using JLD2

length(ARGS) == 1 || error("usage: validate_results.jl RESULT.jld2")
path = only(ARGS)
isfile(path) || error("final result does not exist: $path")

data = load(path)
completed = data["completed_step"]
requested = data["requested_steps"]
steps = data["snapshot_steps"]
times = data["times"]
parameters = data["parameters"]
diagnostics = data["diagnostics"]
step_diagnostics = data["step_diagnostics"]
n = parameters["n"]

completed == requested || error("trajectory is incomplete: $completed/$requested steps")
length(steps) == 8 || error("expected exactly 8 snapshots, found $(length(steps))")
steps[1] == 0 || error("the first snapshot is not the initial condition")
steps[end] == requested || error("the final snapshot is not the terminal step")
issorted(steps) && allunique(steps) || error("snapshot steps are not strictly ordered")
expected_steps = unique(round.(Int, range(0, requested; length=8)))
steps == expected_steps || error("snapshot schedule differs from $expected_steps")
length(times) == 8 || error("snapshot time count mismatch")
times ≈ steps .* parameters["dt"] || error("snapshot times do not match step*dt")

for key in ("u", "v", "pressure", "pressure_impulse", "vorticity")
    field = data[key]
    size(field) == (n, n, 8) || error("$key has size $(size(field)), expected ($n,$n,8)")
    all(isfinite, field) || error("$key contains a non-finite value")
end

gate(name, limit; skip_initial=false) = begin
    values = diagnostics[name]
    checked = skip_initial ? values[2:end] : values
    all(isfinite, checked) || error("$name contains a non-finite value")
    maximum(checked; init=-Inf) <= limit ||
        error("$name exceeds $limit (maximum=$(maximum(checked)))")
end

gate("relative_divergence", 1.0e-4)
gate("pressure_residual", parameters["poisson_tolerance"]; skip_initial=true)
gate("pressure_velocity_leakage", 1.0e-5)
gate("correction_defect", 1.0e-3)
gate("correction_pressure_leakage", 1.0e-5)
gate("pressure_gauge", 1.0e-4)
gate("max_imaginary", 1.0e-8)
gate("ceiling_probability", 1.0e-4)
maximum(diagnostics["max_bond"]) < parameters["maxdim"] ||
    error("a snapshot saturated the configured MPS maxdim")

length(step_diagnostics["step"]) == requested ||
    error("per-step diagnostic history is incomplete")
step_diagnostics["step"] == collect(1:requested) ||
    error("per-step diagnostic indices are not exactly 1:$requested")
step_diagnostics["time"] ≈ collect(1:requested) .* parameters["dt"] ||
    error("per-step diagnostic times are inconsistent")

function step_gate(name, limit)
    values = step_diagnostics[name]
    all(isfinite, values) || error("per-step $name contains a non-finite value")
    maximum(values) <= limit ||
        error("per-step $name exceeds $limit (maximum=$(maximum(values)))")
end

step_gate("pressure_residual", parameters["poisson_tolerance"])
step_gate("relative_divergence", 1.0e-4)
step_gate("pressure_velocity_leakage", 1.0e-5)
step_gate("correction_defect", 1.0e-3)
step_gate("correction_pressure_leakage", 1.0e-5)
step_gate("pressure_gauge", 1.0e-4)
step_gate("max_imaginary", 1.0e-8)
all(step_diagnostics["max_bond"] .< parameters["maxdim"]) ||
    error("a physical step saturated the configured MPS maxdim")

sampled = step_diagnostics["ceiling_sampled"]
any(sampled) || error("boson-ceiling occupation was never sampled")
sampled[end] || error("terminal boson-ceiling occupation was not sampled")
ceiling = step_diagnostics["ceiling_probability"][sampled]
all(isfinite, ceiling) || error("a sampled boson-ceiling value is non-finite")
maximum(ceiling) <= 1.0e-4 || error("boson-ceiling occupation exceeds 1e-4")

println(
    "validated final MPS result: steps=$completed snapshots=$(length(steps)) " *
    "n=$(parameters["n"]) Re=$(parameters["reynolds"])",
)
