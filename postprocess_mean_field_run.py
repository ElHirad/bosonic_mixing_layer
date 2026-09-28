#!/usr/bin/env python3
"""Validate a finished MF trajectory and publish its separate DNS comparison.

Designed for batch execution after the single-site solver has finished. This
program performs no mean-field evolution and is not part of its algorithm.
"""
import argparse
import json
from pathlib import Path
import shutil

from compare_mean_field_dns import run_reference
from mixing_layer_mean_field import atomic_json, validate_results
from plot_mean_field_results import make_plots


def reaction_report(validation, summary):
    cfg, meta = validation["config"], validation["run_metadata"]
    reference = summary["references"]["DNS"]
    errors = reference["metrics"][-1]
    text = (
        f"# Reacting {cfg['n']}×{cfg['n']} explicit single-site mean-field mixing layer\n\n"
        f"Completed and validated {validation['completed_steps']} steps to t={cfg['final_time']:g}. "
        f"Slurm job {meta['slurm_job_id']}; solver time {meta['elapsed_seconds']/3600:.3f} hours "
        f"after resuming from step {meta['resumed_from_step']} (earlier work additional).\n\n"
        f"Re={cfg['reynolds']:g}, Pe={cfg['peclet']:g}, Da={cfg['damkohler']:g}; thickness {cfg['transition']:g}, "
        f"dt={cfg['dt']:g}, cutoff {cfg['boson_cutoff']}. Periodic x, free-slip/no-flux y. "
        "Initial c2=1-c1, c3=0; all species have the same diffusivity. "
        "The nondimensional mass-action rate is Da c1 c2, with sources (-R,-R,+R).\n\n"
        "Reaction and transport evolve explicit single-site Fock vectors together in every "
        f"predictor {cfg['time_integrator']} substep. Pressure relaxation and correction use "
        f"the same integrator and local bosonic operators. The projected "
        f"{reference['metadata']['time_integrator']} DNS is a separate benchmark, never a mean-field "
        "substep. All three initial DNS species and both velocities match the measured "
        "mean-field fields exactly. No concentration clipping, mass repair, or coherent resets.\n\n"
        "## Species comparison plots\n\n"
    )
    for species in ("c1", "c2", "c3"):
        text += (f"- {species}: [Mean field](./mean_field_{species}.png), [DNS](./dns_{species}.png), "
                 f"[side-by-side and signed differences](./mean_field_vs_dns_{species}.png)\n")
    text += ("- [Reaction, invariant, and comparison diagnostics](./reaction_diagnostics.png)\n"
             "- [MF vs DNS vorticity](./mean_field_vs_dns_vorticity.png)\n\n"
             "## Validation and limitations\n\n"
             f"Maximum relative drift of the integrals of c1+c3 and c2+c3: "
             f"{validation['worst_step']['scalar_mass_error']:.3e}. "
             f"Maximum relative divergence: {validation['worst_step']['relative_divergence']:.3e}. "
             "Individual species are not conserved. The unweighted sum c1+c2+c3 is not a "
             "reaction invariant; c1+c2+2c3 is.\n\n"
             "| Species | MF all-step range | DNS all-step range | Final MF/DNS relative L2 |\n"
             "|---|---|---|---|\n")
    for species in ("c1", "c2", "c3"):
        first = summary["species_ranges_all_steps"][species]
        second = reference["validation"]["species_ranges_all_steps"][species]
        text += (f"| {species} | [{first['minimum']:.6g}, {first['maximum']:.6g}] | "
                 f"[{second['minimum']:.6g}, {second['maximum']:.6g}] | "
                 f"{100*errors[species+'_relative_l2']:.4g}% |\n")
    if "sign_statistics" in summary:
        text += ("\n## Positive and negative concentrations\n\n"
                 "[Every-step CSV, including t=0](./concentration_sign_history.csv) · "
                 "[Signed-amount and cell-fraction plot](./concentration_signs.png). "
                 "Counts/fractions distinguish c>0, c<0, c=0, and c<-1e-12. "
                 "Positive/negative integrals are the separate signed contributions over "
                 "the unit-area domain; they sum to each species amount. Negative integrals "
                 "are nonpositive, not absolute magnitudes. No clipping or repair is applied.\n\n"
                 "| Method | Species | Peak c<0 fraction | Peak c<-1e-12 fraction | Most negative integral |\n"
                 "|---|---|---|---|---|\n")
        for method, species in summary["sign_statistics"]["methods"].items():
            for name, metrics in species.items():
                text += (f"| {method} | {name} | {metrics['maximum_negative_fraction']:.6g} | "
                         f"{metrics['maximum_material_negative_fraction']:.6g} | "
                         f"{metrics['minimum_negative_integral']:.6g} |\n")
    text += (
        "\nCentered transport is not positivity-preserving. Negative concentrations and "
        "negative Da c1 c2 rates, if present, are numerical artifacts, not physical reverse "
        "chemistry. The displayed bounds include overshoots; data are not clipped. "
        f"Positivity preserved within 1e-12: MF={validation['positivity_preserved']}, "
        f"DNS={reference['validation']['positivity_preserved']}. Passing "
        "the conservation and local-state gates does not certify physical positivity or "
        "grid/timestep convergence. The two solvers retain different temporal splitting. "
        "RK4 in the MF substeps does not establish fourth-order accuracy of the complete "
        "split MF algorithm. For RK4 DNS, stored pressure is the stage-weighted step average.\n\n"
        f"Final velocity/vorticity MF/DNS relative L2 differences: "
        f"{100*errors['velocity_relative_l2']:.4g}% / {100*errors['vorticity_relative_l2']:.4g}%. "
        "Inspect spatial fields for roll-up/pairing; Fourier-mode ratios alone do not prove merger.\n\n"
        "[Full summary](./summary.json) · [MF validation](./validation.json) · "
        "[DNS validation](./dns_matched_snapshots.validation.json) · [Comparison table](./comparison_DNS.csv)\n"
    )
    return text


def postprocess(result_path, output_dir):
    result_path, output_dir = Path(result_path), Path(output_dir)
    validation = validate_results(result_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir/"postprocess_status.json"
    atomic_json(status_path, {"state": "running", "mean_field_source": str(result_path)})
    try:
        published = output_dir/"mean_field_snapshots.npz"
        if result_path.resolve() != published.resolve():
            shutil.copy2(result_path, published)
        atomic_json(output_dir/"validation.json", validation)
        reference = output_dir/"dns_matched_snapshots.npz"
        run_reference(published, reference)
        summary = make_plots(published, output_dir, references={"DNS": reference})
        cfg, meta = validation["config"], validation["run_metadata"]
        errors = summary["references"]["DNS"]["metrics"][-1]
        scalar_range = summary["concentration_range_all_steps"]
        ratios = summary["pairing"]["ratio"]
        text = (
            f"# {cfg['n']}×{cfg['n']} explicit single-site mean-field mixing layer\n\n"
            f"Completed and validated {validation['completed_steps']} steps to t={cfg['final_time']:g}. "
            f"Production Slurm job: {meta['slurm_job_id']}. "
            f"This job resumed from step {meta['resumed_from_step']}; its measured solver time "
            f"is {meta['elapsed_seconds']/60:.2f} minutes, excluding earlier checkpointed work.\n\n"
            f"Re={cfg['reynolds']:g}, Pe={cfg['peclet']:g}; {cfg['boundary_y']} y boundaries, "
            f"periodic x. One shear at y={cfg['shear_center']:g}, tanh thickness "
            f"{cfg['transition']:g}, KH width {cfg['kh_width']:g}. "
            f"Mode {cfg['kh_mode']} amplitude {cfg['kh_amplitude']:g}; "
            f"mode {cfg['secondary_mode']} amplitude {cfg['secondary_amplitude']:g}. "
            f"Physical dt={cfg['dt']:g}; boson cutoff={cfg['boson_cutoff']}.\n\n"
            "Predictor, pressure relaxation, and correction all evolve explicit single-site "
            f"bosonic operators using {cfg['time_integrator']}. DNS below is a separate projected "
            f"{summary['references']['DNS']['metadata']['time_integrator']} benchmark, "
            "initialized from the exact measured mean-field initial fields; it is not a "
            "stage of the mean-field solver. No scalar-mass repair or clipping is used.\n\n"
            "## Plots\n\n"
            "- [Vorticity and velocity](./mean_field_vorticity.png)\n"
            "- [Concentration](./mean_field_concentration.png)\n"
            "- [Pairing and numerical diagnostics](./mean_field_diagnostics.png)\n"
            "- [MF vs DNS vorticity](./mean_field_vs_dns_vorticity.png)\n"
            "- [MF vs DNS concentration](./mean_field_vs_dns_concentration.png)\n"
            "- [MF vs DNS errors, pairing, and conservation](./mean_field_vs_dns_diagnostics.png)\n\n"
            "## Validation and limitations\n\n"
            f"Maximum raw relative scalar-mass drift: {validation['worst_step']['scalar_mass_error']:.3e}. "
            f"Maximum relative divergence: {validation['worst_step']['relative_divergence']:.3e}. "
            f"All-step concentration range: [{scalar_range['minimum']:.6g}, {scalar_range['maximum']:.6g}]. "
            "Centered transport is not bound-preserving; any overshoots are retained.\n\n"
            f"Final MF/DNS relative L2 differences: velocity {100*errors['velocity_relative_l2']:.4g}%, "
            f"vorticity {100*errors['vorticity_relative_l2']:.4g}%, "
            f"concentration {100*errors['concentration_relative_l2']:.4g}%. "
            "The two methods use different temporal splitting.\n\n"
            f"Final mode-{cfg['secondary_mode']}/mode-{cfg['kh_mode']} ratio: {ratios[-1]:.5g}. "
            "Inspect the spatial evolution to assess roll-up and pairing: subharmonic dominance "
            "alone can also result from faster decay of the primary mode. These large initial "
            "perturbations do not test linear KH instability. This run by itself does not "
            "establish grid or timestep convergence.\n\n"
            "[Full numerical summary](./summary.json) · [Mean-field validation](./validation.json) · "
            "[DNS validation](./dns_matched_snapshots.validation.json)\n"
        )
        if cfg["scalar_species"] == 3:
            text = reaction_report(validation, summary)
        (output_dir/"RUN_REPORT.md").write_text(text)
        atomic_json(status_path, {"state": "complete", "mean_field_source": str(result_path),
                                  "completed_steps": validation["completed_steps"],
                                  "report": "RUN_REPORT.md"})
        return summary
    except BaseException as error:
        atomic_json(status_path, {"state": "failed", "mean_field_source": str(result_path), "error": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    summary = postprocess(args.result, args.output_dir)
    print(json.dumps({"validated": summary["validation"]["validated"], "report": str(args.output_dir/"RUN_REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()
