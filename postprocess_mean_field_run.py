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
            "bosonic operators. DNS below is a separate projected-midpoint benchmark, "
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
