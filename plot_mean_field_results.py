#!/usr/bin/env python3
"""Plot measured single-site bosonic fields; never run a DNS/reference solver."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mixing_layer_mean_field import MeanFieldConfig, atomic_json, validate_results
from compare_mean_field_dns import matched_dns_integrator


def load_fields(path):
    with np.load(path, allow_pickle=False) as data:
        fields = {name: data[name].copy() for name in ("times", "u", "v", "concentration", "vorticity")}
        for name in ("c1", "c2", "c3"):
            if name in data:
                fields[name] = data[name].copy()
        if "config_json" in data:
            fields["config"] = json.loads(str(data["config_json"]))
        for key in ("run_metadata", "validation", "history", "initial_diagnostics"):
            if key+"_json" in data:
                fields[key] = json.loads(str(data[key+"_json"]))
    n = fields["u"].shape[2]
    if fields["v"].shape[1] == n+1 and fields["vorticity"].shape[1] == n:
        # Historical MPS vorticity omits the zero top-wall vertex row.
        fields["vorticity"] = np.concatenate((fields["vorticity"], np.zeros((8, 1, n))), axis=1)
    return fields


def pairing_modes(omega):
    density = np.mean(np.minimum(omega, 0)**2, axis=1)
    density -= density.mean(axis=1, keepdims=True)
    return np.abs(np.fft.rfft(density, axis=1))/omega.shape[2]


def verify_dns_setup(config, reference):
    """Reject a reference from another physical case instead of mislabelling it."""
    if "config" not in reference:
        raise ValueError("DNS reference must include its physical configuration")
    expected = {
        "nx": config.n, "ny": config.n, "dx": 1/config.n, "dy": 1/config.n,
        "reynolds": config.reynolds, "peclet": config.peclet, "reference_velocity": 1.0,
        "boundary_y": config.boundary_y, "shear_center_fraction": config.shear_center,
        "middle_layer_fraction": config.middle_fraction, "transition_thickness": config.transition,
        "perturbation_width": config.kh_width, "perturbation_mode": config.kh_mode,
        "perturbation_amplitude": config.kh_amplitude, "subharmonic_mode": config.secondary_mode,
        "subharmonic_amplitude": config.secondary_amplitude, "perturbation_phase": config.phase,
        "max_dt": config.dt, "t_end": config.final_time,
    }
    if config.scalar_species == 3:
        expected.update(scalar_species=3, damkohler=config.damkohler)
    if config.scalar_species == 3 or config.time_integrator == "forward-euler" or "time_integrator" in reference["config"]:
        expected["time_integrator"] = matched_dns_integrator(config)
    for key, value in expected.items():
        if reference["config"].get(key) != value:
            raise ValueError(f"DNS reference parameter does not match: {key}")


def identical_initial_fields(primary, reference):
    keys = ("u", "v", "concentration") + (("c1", "c2", "c3") if "c1" in primary or "c1" in reference else ())
    return all(key in primary and key in reference and np.array_equal(primary[key][0], reference[key][0])
               for key in keys)


def comparison(primary, reference):
    if primary["times"].shape != reference["times"].shape:
        raise ValueError("reference output times do not match")
    if not np.allclose(primary["times"], reference["times"], rtol=0, atol=1e-12):
        raise ValueError("reference output times do not match")
    species = ("c1", "c2", "c3") if "c1" in primary or "c1" in reference else ()
    for field in ("u", "v", "concentration", "vorticity")+species:
        if field not in primary or field not in reference:
            raise ValueError(f"reference species missing: {field}")
        if primary[field].shape != reference[field].shape:
            raise ValueError(f"reference {field} grid/layout does not match")
        if not np.all(np.isfinite(primary[field])) or not np.all(np.isfinite(reference[field])):
            raise ValueError(f"non-finite comparison {field} data")
    rows = []
    for k, time in enumerate(primary["times"]):
        row = {"time": float(time)}
        for field in ("concentration", "vorticity")+species:
            delta = primary[field][k]-reference[field][k]
            row[field+"_relative_l2"] = float(np.linalg.norm(delta)/max(np.linalg.norm(reference[field][k]), 1e-30))
            row[field+"_max_abs"] = float(np.max(np.abs(delta)))
        du, dv = primary["u"][k]-reference["u"][k], primary["v"][k]-reference["v"][k]
        row["velocity_relative_l2"] = float(np.sqrt((np.sum(du*du)+np.sum(dv*dv))/
                                                    max(np.sum(reference["u"][k]**2)+np.sum(reference["v"][k]**2), 1e-30)))
        rows.append(row)
    return rows


def field_coordinates(array, config, field):
    """Preserve MAC coordinates and periodic vertex endpoints; no smoothing."""
    n = config.n
    if field == "vorticity":
        array = np.concatenate((array, array[:, :1]), axis=1)
        if config.boundary_y == "periodic":
            array = np.concatenate((array, array[:1]), axis=0)
        return np.arange(n+1)/n, np.arange(array.shape[0])/n, array
    coordinates = (np.arange(n)+.5)/n
    return coordinates, coordinates, array


def plot_dns_comparison(primary, reference, config, field, destination):
    """All eight times: matched colours for MF/DNS, separate signed-error scale."""
    rows = comparison(primary, reference)
    first, second = primary[field], reference[field]
    delta = first-second
    vort = field == "vorticity"
    bounds = (-20, 20) if vort else (min(0., float(first.min()), float(second.min())),
                                    max(1., float(first.max()), float(second.max())))
    error_limit = max(float(np.max(np.abs(delta))), 1e-15)
    fig, axes = plt.subplots(6, 4, figsize=(12.5, 17.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=.07, right=.98, bottom=.095, top=.945, hspace=.23, wspace=.14)
    for k, time_value in enumerate(primary["times"]):
        base, col = 3*(k//4), k % 4
        for offset, (values, label) in enumerate(((first, "Mean field"), (second, "DNS"), (delta, "MF − DNS"))):
            ax = axes[base+offset, col]
            xx, yy, array = field_coordinates(values[k], config, field)
            vmin, vmax = (-error_limit, error_limit) if offset == 2 else bounds
            artist = ax.pcolormesh(xx, yy, array, shading="nearest", rasterized=True,
                                  cmap="RdBu_r" if vort or offset == 2 else "viridis", vmin=vmin, vmax=vmax)
            ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xticks=[0, .5, 1], yticks=[0, .5, 1])
            if col == 0:
                ax.set_ylabel(f"{label}\ny")
            if offset == 0:
                ax.set_title(f"t = {time_value:.4f}")
            if offset == 2:
                error_artist = artist
                ax.set_title(f"Relative L2: {100*rows[k][field+'_relative_l2']:.3g}%", fontsize=10)
            else:
                value_artist = artist
    for ax in axes[-1]:
        ax.set_xlabel("x")
    value_bar_axes = fig.add_axes([.10, .047, .34, .012])
    error_bar_axes = fig.add_axes([.59, .047, .34, .012])
    bar = fig.colorbar(value_artist, cax=value_bar_axes, orientation="horizontal", extend="both" if vort else "neither")
    bar.set_label("Vorticity (colours saturated beyond ±20)" if vort else "Concentration")
    fig.colorbar(error_artist, cax=error_bar_axes, orientation="horizontal").set_label(f"Signed {field} difference")
    match_label = "identical initial fields" if identical_initial_fields(primary, reference) else "reference initial fields differ"
    fig.suptitle(f"Single-site mean field vs independent DNS: {field}\n"
                 f"{config.n}×{config.n}, Re={config.reynolds:g}, Pe={config.peclet:g}"
                 +(f", Da={config.damkohler:g}" if config.scalar_species == 3 else "")
                 +f"; {match_label}, matched output times")
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def plot_dns_diagnostics(primary, reference, config, destination):
    rows = comparison(primary, reference)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for field in ("velocity", "vorticity", "concentration"):
        axes[0, 0].plot(primary["times"], [100*r[field+"_relative_l2"] for r in rows], "o-", label=field)
    axes[0, 0].set_ylabel("MF − DNS relative L2 difference (%)")
    for data, label, style in ((primary, "Mean field", "-"), (reference, "DNS", "--")):
        modes = pairing_modes(data["vorticity"])
        ratio = modes[:, config.secondary_mode]/np.maximum(modes[:, config.kh_mode], 1e-30)
        axes[0, 1].semilogy(data["times"], ratio, "o"+style, label=label)
        if "history" in data:
            times = [row["time"] for row in data["history"]]
            axes[1, 0].semilogy(times, np.maximum([row["scalar_mass_error"] for row in data["history"]], 1e-17), style, label=label)
            axes[1, 1].plot(times, [row["scalar_minimum"] for row in data["history"]], style, label=label+" min(c)")
            axes[1, 1].plot(times, [row["scalar_maximum"] for row in data["history"]], style, label=label+" max(c)")
    axes[0, 1].axhline(1, color="black", linestyle=":")
    axes[0, 1].set_ylabel(f"Mode {config.secondary_mode} / mode {config.kh_mode}")
    axes[1, 0].set_ylabel("Raw relative scalar-mass drift")
    axes[1, 1].set_ylabel("Concentration range (no clipping)")
    for value in (0, 1):
        axes[1, 1].axhline(value, color="black", linestyle=":")
    for ax in axes.flat:
        ax.set_xlabel("Physical time")
        ax.grid(alpha=.25)
        ax.legend(fontsize=9)
    match_label = "Matched initial-condition DNS comparison" if identical_initial_fields(primary, reference) else "Mean-field / DNS reference comparison"
    dns_method = reference.get("run_metadata", {}).get("time_integrator", matched_dns_integrator(config))
    fig.suptitle(match_label+f"\nMF: Chorin split with local bosonic operators; DNS: projected {dns_method}")
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def plot_grid(data, config, field, destination, method="Explicit single-site bosonic mean field", bounds=None):
    n = config.n
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 7.1), sharex=True, sharey=True, constrained_layout=True)
    vort = field == "vorticity"
    values = data[field]
    vmin, vmax = (-20, 20) if vort else (min(0., float(values.min())), max(1., float(values.max())))
    if bounds is not None:
        vmin, vmax = bounds
    for k, ax in enumerate(axes.flat):
        array = values[k]
        xx, yy, array = field_coordinates(array, config, field)
        artist = ax.pcolormesh(xx, yy, array, shading="nearest", cmap="RdBu_r" if vort else "viridis",
                               vmin=vmin, vmax=vmax, rasterized=True)
        if vort:
            u = .5*(data["u"][k]+np.roll(data["u"][k], -1, axis=1))
            v = data["v"][k]
            v = .5*(v[1:]+v[:-1]) if v.shape[0] == n+1 else .5*(v+np.roll(v, -1, axis=0))
            coordinates = (np.arange(n)+.5)/n
            stride = max(1, n//8)
            ax.quiver(coordinates[::stride], coordinates[::stride], u[::stride, ::stride], v[::stride, ::stride],
                      color="black", alpha=.65, scale=17, pivot="mid", width=.004)
        ax.set(title=f"t = {data['times'][k]:.4f}", xlim=(0, 1), ylim=(0, 1), aspect="equal")
        ax.set_xticks([0, .5, 1])
        ax.set_yticks([0, .5, 1])
    for ax in axes[-1]:
        ax.set_xlabel("x")
    for ax in axes[:, 0]:
        ax.set_ylabel("y")
    bar = fig.colorbar(artist, ax=axes, shrink=.88, pad=.02, extend="both" if vort else "neither")
    bar.set_label("Vorticity (colour saturated beyond ±20)" if vort else
                  (f"Concentration {field}" if config.scalar_species == 3 else "Conserved concentration"))
    fig.suptitle(f"{method}: {field}\n"
                 f"{n}×{n}, Re={config.reynolds:g}, Pe={config.peclet:g}, "
                 f"{config.boundary_y} y"
                 +(f", Da={config.damkohler:g}" if config.scalar_species == 3 else "")
                 +(f", cutoff {config.boson_cutoff}" if "mean field" in method else ""))
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def make_plots(result_path, output_dir, references=None, convergence=None):
    validation = validate_results(result_path)
    config = MeanFieldConfig(**validation["config"])
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_fields(result_path)
    if config.scalar_species == 3:
        if convergence:
            raise ValueError("reactive convergence studies require separately matched time/configuration checks")
        return make_reaction_plots(data, config, validation, output_dir, references)
    with np.load(result_path, allow_pickle=False) as saved:
        history = json.loads(str(saved["history_json"]))
    for field in ("vorticity", "concentration"):
        plot_grid(data, config, field, output_dir/f"mean_field_{field}.png")
    modes = pairing_modes(data["vorticity"])
    ratio = modes[:, config.secondary_mode]/np.maximum(modes[:, config.kh_mode], 1e-30)
    summary = {"validation": validation, "pairing": {
        "definition": "Fourier amplitudes of y-mean negative-vorticity squared; includes both wall rows",
        "times": data["times"].tolist(), "primary_mode": modes[:, config.kh_mode].tolist(),
        "secondary_mode": modes[:, config.secondary_mode].tolist(), "ratio": ratio.tolist(),
        "caution": "A ratio crossing one is not by itself evidence of vortex merging; inspect spatial fields.",
    }, "concentration_range_all_steps": {
        "minimum": min(row["scalar_minimum"] for row in history),
        "maximum": max(row["scalar_maximum"] for row in history),
        "caution": "Centered conservative transport is not bound-preserving; no clipping is applied.",
    }, "references": {}, "convergence": {}}
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    axes[0, 0].semilogy(data["times"], modes[:, config.kh_mode], "o-", label=f"MF mode {config.kh_mode}")
    axes[0, 0].semilogy(data["times"], modes[:, config.secondary_mode], "s-", label=f"MF mode {config.secondary_mode}")
    axes[0, 0].set_ylabel("Negative-vorticity enstrophy mode amplitude")
    axes[0, 1].semilogy(data["times"], ratio, "o-", label="Single-site MF")
    axes[0, 1].axhline(1, color="black", linestyle=":")
    axes[0, 1].set_ylabel(f"Mode {config.secondary_mode} / mode {config.kh_mode}")
    for label, path in (references or {}).items():
        reference = load_fields(path)
        if label == "DNS":
            verify_dns_setup(config, reference)
        rows = comparison(data, reference)
        summary["references"][label] = {"file": str(path), "metrics": rows,
                                        "identical_initial_fields": identical_initial_fields(data, reference),
                                        "metadata": reference.get("run_metadata"), "validation": reference.get("validation")}
        if label == "DNS":
            for field in ("vorticity", "concentration"):
                plot_dns_comparison(data, reference, config, field, output_dir/f"mean_field_vs_dns_{field}.png")
            plot_dns_diagnostics(data, reference, config, output_dir/"mean_field_vs_dns_diagnostics.png")
        ref_modes = pairing_modes(reference["vorticity"])
        axes[0, 1].semilogy(reference["times"], ref_modes[:, config.secondary_mode]/
                            np.maximum(ref_modes[:, config.kh_mode], 1e-30), "--", label=label+" reference")
    times = [row["time"] for row in history]
    for key, label in (("scalar_mass_error", "Raw scalar-mass drift"),
                       ("relative_divergence", "Relative divergence"), ("pressure_residual", "Pressure residual")):
        axes[1, 0].semilogy(times, np.maximum([row[key] for row in history], 1e-17), label=label)
    axes[1, 0].set_ylabel("Error (no field correction)")
    for key, label in (("coherent_eigenstate_defect", "Coherent eigenstate defect"),
                       ("pressure_velocity_leakage", "Pressure → velocity leakage"),
                       ("correction_pressure_leakage", "Correction → pressure leakage")):
        axes[1, 1].semilogy(times, np.maximum([row[key] for row in history], 1e-17), label=label)
    axes[1, 1].set_ylabel("Local-state / stage consistency")
    axes[0, 2].plot(times, [row["scalar_minimum"] for row in history], label="Minimum c")
    axes[0, 2].plot(times, [row["scalar_maximum"] for row in history], label="Maximum c")
    axes[0, 2].axhline(0, color="black", linestyle=":")
    axes[0, 2].axhline(1, color="black", linestyle=":")
    axes[0, 2].set_ylabel("Concentration range (not clipped)")
    axes[1, 2].plot(times, [row["kinetic_energy"] for row in history], label="Kinetic energy")
    axes[1, 2].set_ylabel("Domain-integrated kinetic energy")
    for ax in axes.flat:
        ax.set_xlabel("Physical time")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle("Single-site mean-field mixing layer: pairing and uncorrected invariants")
    fig.savefig(output_dir/"mean_field_diagnostics.png", dpi=180)
    plt.close(fig)
    numerical_keys = {"boson_cutoff", "predictor_substeps", "correction_substeps", "pressure_cfl", "pressure_tolerance"}
    for label, path in (convergence or {}).items():
        other_validation = validate_results(path)
        for key, value in validation["config"].items():
            if key not in numerical_keys and value != other_validation["config"][key]:
                raise ValueError(f"convergence case changes physical parameter {key}")
        rows = comparison(data, load_fields(path))
        summary["convergence"][label] = {
            "validation": other_validation, "metrics": rows,
            "worst_snapshot_relative_l2": {field: max(row[field+"_relative_l2"] for row in rows)
                                            for field in ("velocity", "vorticity", "concentration")},
        }
    atomic_json(output_dir/"summary.json", summary)
    for category in ("references", "convergence"):
        for label, result in summary[category].items():
            with (output_dir/f"comparison_{label}.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=result["metrics"][0].keys(), lineterminator="\n")
                writer.writeheader()
                writer.writerows(result["metrics"])
    return summary


def reaction_sign_comparison(datasets, config, output_dir):
    """Export every physical step, including t=0; never infer missing history."""
    metrics = ("minimum", "maximum", "mass", "positive_cells", "negative_cells", "zero_cells",
               "positive_fraction", "negative_fraction", "zero_fraction", "material_negative_cells",
               "material_negative_fraction", "positive_integral", "negative_integral")
    records, summary = [], {}
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)
    for values, label, style in datasets:
        initial = values.get("initial_diagnostics")
        history = values.get("history", [])
        if initial is None or not history or "c1_negative_integral" not in history[0]:
            plt.close(fig)
            raise ValueError("complete signed-concentration history including t=0 is required")
        rows = [dict(step=0, time=0., **initial), *history]
        if [row["step"] for row in rows] != list(range(config.steps+1)):
            plt.close(fig)
            raise ValueError("incomplete signed-concentration step history")
        times = [row["time"] for row in rows]
        summary[label] = {}
        for column, name in enumerate(("c1", "c2", "c3")):
            for row in rows:
                records.append(dict(method=label, species=name, step=row["step"], time=row["time"],
                                    **{key: row[f"{name}_{key}"] for key in metrics}))
            summary[label][name] = {
                "maximum_negative_fraction": max(row[f"{name}_negative_fraction"] for row in rows),
                "maximum_material_negative_fraction": max(row[f"{name}_material_negative_fraction"] for row in rows),
                "minimum_negative_integral": min(row[f"{name}_negative_integral"] for row in rows),
                "maximum_positive_integral": max(row[f"{name}_positive_integral"] for row in rows),
                "first_negative_step": next((row["step"] for row in rows if row[f"{name}_negative_cells"]), None),
                "first_material_negative_step": next((row["step"] for row in rows if row[f"{name}_material_negative_cells"]), None),
                "final_positive_integral": rows[-1][f"{name}_positive_integral"],
                "final_negative_integral": rows[-1][f"{name}_negative_integral"],
            }
            for sign, color in (("positive", "C2"), ("negative", "C3")):
                axes[0, column].plot(times, [r[f"{name}_{sign}_fraction"] for r in rows],
                                     style, color=color, label=f"{label} {sign}")
                axes[1, column].plot(times, [r[f"{name}_{sign}_integral"] for r in rows],
                                     style, color=color, label=f"{label} {sign}")
            axes[0, column].set_title(name)
    for index, ax in enumerate(axes.flat):
        ax.set(xlabel="Physical time", ylabel="Raw cell fraction (>0 / <0)" if index < 3
               else "Signed concentration integral")
        ax.axhline(0., color="black", linewidth=.7)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle(f"Unclipped species signs: {config.n}×{config.n}, Re=Pe={config.reynolds:g}, Da={config.damkohler:g}")
    fig.savefig(output_dir/"concentration_signs.png", dpi=170)
    plt.close(fig)
    with (output_dir/"concentration_sign_history.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    return {"material_negative_threshold": -1e-12, "raw_sign_threshold": 0.,
            "integral_definition": "unit-area mean(max(c,0)) and signed mean(min(c,0)); observational only",
            "includes_initial_state": True, "methods": summary}


def make_reaction_plots(data, config, validation, output_dir, references):
    """Expose both reacting species, the product, and uncorrected invariants."""
    history = data["history"]
    ranges = validation["species_ranges_all_steps"]
    modes = pairing_modes(data["vorticity"])
    ratio = modes[:, config.secondary_mode]/np.maximum(modes[:, config.kh_mode], 1e-30)
    summary = {"validation": validation, "reaction": {"law": "c1+c2 -> c3; rate=Da*c1*c2",
        "damkohler": config.damkohler, "conserved_integrals": ["c1+c3", "c2+c3"],
        "caution": "Centered transport is not positivity-preserving; negative concentrations/rates are retained, not physical reverse chemistry."},
        "species_ranges_all_steps": ranges,
        "concentration_range_all_steps": {"minimum": min(v["minimum"] for v in ranges.values()),
                                           "maximum": max(v["maximum"] for v in ranges.values())},
        "pairing": {"ratio": ratio.tolist(), "times": data["times"].tolist(),
                    "caution": "Mode ratios do not by themselves establish vortex merger."},
        "references": {}, "convergence": {}}
    reference = None
    for label, path in (references or {}).items():
        if label != "DNS":
            raise ValueError("reactive plots currently accept only a matched DNS reference")
        reference = load_fields(path)
        verify_dns_setup(config, reference)
        if not identical_initial_fields(data, reference):
            raise ValueError("reactive DNS must match all three initial species and velocities exactly")
        if reference.get("run_metadata", {}).get("mean_field_fingerprint") != validation["fingerprint"]:
            raise ValueError("reactive DNS fingerprint does not match the mean-field source")
        rows = comparison(data, reference)
        summary["references"][label] = {"file": str(path), "metrics": rows, "identical_initial_fields": True,
            "metadata": reference.get("run_metadata"), "validation": reference.get("validation")}
        with (output_dir/"comparison_DNS.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys(), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    for field in ("c1", "c2", "c3", "vorticity"):
        bounds = None
        if reference is not None and field != "vorticity":
            bounds = (min(0., float(data[field].min()), float(reference[field].min())),
                      max(1., float(data[field].max()), float(reference[field].max())))
        plot_grid(data, config, field, output_dir/f"mean_field_{field}.png", bounds=bounds)
        if reference is not None:
            plot_grid(reference, config, field, output_dir/f"dns_{field}.png", method="Independent DNS", bounds=bounds)
            plot_dns_comparison(data, reference, config, field, output_dir/f"mean_field_vs_dns_{field}.png")
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    datasets = [(data, "MF", "-")] + ([(reference, "DNS", "--")] if reference is not None else [])
    for values, label, style in datasets:
        rows = values["history"]
        times = [r["time"] for r in rows]
        for name, color in (("c1", "C0"), ("c2", "C1"), ("c3", "C2")):
            axes[0, 0].plot(times, [r[f"{name}_mass"] for r in rows], style, color=color, label=f"{label} {name}")
            axes[1, 0].plot(times, [r[f"{name}_minimum"] for r in rows], style, color=color, label=f"{label} min {name}")
            axes[1, 1].plot(times, [r[f"{name}_maximum"] for r in rows], style, color=color, label=f"{label} max {name}")
        for invariant in ("13", "23"):
            axes[0, 1].semilogy(times, np.maximum([r[f"invariant_{invariant}_error"] for r in rows], 1e-17),
                                 style, label=f"{label} M{invariant}")
        axes[1, 2].plot(times, [r["reaction_rate_minimum"] for r in rows], style, label=f"{label} min rate")
        axes[1, 2].plot(times, [r["reaction_rate_mean"] for r in rows], style, label=f"{label} mean rate")
    if reference is not None:
        errors = summary["references"]["DNS"]["metrics"]
        for name in ("c1", "c2", "c3", "velocity", "vorticity"):
            axes[0, 2].plot(data["times"], [100*r[name+"_relative_l2"] for r in errors], "o-", label=name)
    for ax, label in zip(axes.flat, ("Domain-mean species amount", "Raw relative invariant drift",
        "MF − DNS relative L2 difference (%)", "Species minima (not clipped)", "Species maxima (not clipped)",
        "Da c1 c2 (negative rates are unphysical)")):
        ax.set(xlabel="Physical time", ylabel=label)
        ax.grid(alpha=.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    axes[1, 0].axhline(0, color="black", linestyle=":")
    axes[1, 1].axhline(1, color="black", linestyle=":")
    fig.suptitle(f"Reacting mixing layer: {config.n}×{config.n}, Re={config.reynolds:g}, "
                 f"Pe={config.peclet:g}, Da={config.damkohler:g}")
    fig.savefig(output_dir/"reaction_diagnostics.png", dpi=170)
    plt.close(fig)
    # Older archives without sign histories remain plottable, but no invented
    # all-step statistics are reported for them.
    if all("initial_diagnostics" in values and values["history"]
           and "c1_negative_integral" in values["history"][0] for values, _, _ in datasets):
        summary["sign_statistics"] = reaction_sign_comparison(datasets, config, output_dir)
    atomic_json(output_dir/"summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--saved-mps", type=Path)
    parser.add_argument("--saved-dns", type=Path)
    parser.add_argument("--larger-basis", type=Path)
    parser.add_argument("--finer-integration", type=Path)
    args = parser.parse_args()
    references = {name: path for name, path in (("MPS", args.saved_mps), ("DNS", args.saved_dns)) if path is not None}
    convergence = {name: path for name, path in (("larger_basis", args.larger_basis), ("finer_integration", args.finer_integration)) if path is not None}
    result = make_plots(args.result, args.output_dir, references, convergence)
    print(json.dumps({"final_pairing_ratio": result["pairing"]["ratio"][-1],
                      "reference_final_errors": {key: value["metrics"][-1] for key, value in result["references"].items()},
                      "convergence": {key: value["worst_snapshot_relative_l2"] for key, value in result["convergence"].items()}}, indent=2))


if __name__ == "__main__":
    main()
