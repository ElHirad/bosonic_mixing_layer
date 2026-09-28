#!/usr/bin/env python3
"""Report completed one-step MF pressure probes alongside full-time DNS checks.

This does not evolve fields and never labels a startup probe a full trajectory.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_status(path, value):
    """Publish status atomically so a disconnected session is not needed."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            json.dump(value, file, indent=2, allow_nan=False)
            file.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def make_report(output_dir, substeps=8):
    output_dir = Path(output_dir)
    probes, dns = [], []
    for da in (1, 10, 100):
        probe = json.loads((output_dir/f"mf_euler_probe_da{da}_substeps{substeps}.json").read_text())
        reference = json.loads((output_dir/f"dns_preflight_da{da}.json").read_text())
        if probe["state"] not in ("one_step_passed", "rejected_candidate"):
            raise ValueError("pressure probe has not completed")
        if not probe["pressure_history"]:
            raise ValueError("probe has no measured pressure history")
        if probe["fingerprint"] != reference["mean_field_source_fingerprint"] or probe["config"] != reference["config"]:
            raise ValueError("DNS and MF probe configurations/sources do not match")
        probes.append(probe)
        dns.append(reference)
    cfg = probes[0]["config"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True, constrained_layout=True)
    rows = []
    for i, (ax, probe) in enumerate(zip(axes, probes)):
        history = probe["pressure_history"]
        da = probe["config"]["damkohler"]
        ax.semilogy([r["pressure_iterations"] for r in history],
                    [max(r["pressure_residual"], 1e-30) for r in history], color=f"C{i}", label="Measured residual")
        ax.axhline(cfg["pressure_tolerance"], color="black", linestyle="--", label="Unchanged tolerance")
        ax.set(title=f"Da={da:g}; final {history[-1]['pressure_residual']:.3e}",
               xlabel="Pressure Euler iterations")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
        rows.extend({"damkohler": da, **r} for r in history)
    axes[0].set_ylabel("Relative pressure residual")
    fig.suptitle(f"Single-step bosonic MF pressure probes: {cfg['n']}×{cfg['n']}, "
                 f"Re={cfg['reynolds']:g}, Pe={cfg['peclet']:g}; forward Euler")
    fig.savefig(output_dir/"pressure_convergence.png", dpi=170)
    plt.close(fig)
    with (output_dir/"pressure_convergence.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    text = (f"# {cfg['n']}×{cfg['n']} forward-Euler DNS and mean-field pressure checks\n\n"
        f"Re={cfg['reynolds']:g}, Pe={cfg['peclet']:g}; Da=1,10,100. Physical dt={cfg['dt']:g}, "
        f"end time {cfg['final_time']:g}, tanh thickness {cfg['transition']:g}. "
        "KH seeds and periodic-x/free-slip-y boundaries are unchanged. "
        "c2(0)=1-c1(0), c3(0)=0; all three species share Pe. No clipping or mass repair.\n\n"
        "The DNS checks span the full physical interval. The mean-field results below "
        "are **one-step startup diagnostics only**, not completed trajectories or a full "
        "MF/DNS field comparison.\n\n"
        "## DNS negativity\n\n"
        "| Da | Completed steps | Min c1 | Min c2 | Min c3 | Max invariant drift |\n"
        "|---|---|---|---|---|---|\n")
    for da, reference in zip((1, 10, 100), dns):
        ranges = reference["species_ranges_all_steps"]
        text += (f"| {da} | {reference['completed_steps']} | {ranges['c1']['minimum']:.8g} | "
                 f"{ranges['c2']['minimum']:.8g} | {ranges['c3']['minimum']:.8g} | "
                 f"{reference['worst_invariant_error']:.3e} |\n")
    text += ("\nExtrema include initialization and every accepted DNS step. These runs "
             "do not establish general positivity or grid/timestep convergence.\n\n"
             "## Mean-field pressure convergence\n\n"
             f"All MF stages use explicit local Fock vectors and forward Euler. "
             f"The pressure tolerance is {cfg['pressure_tolerance']:.1e}, pseudo-step "
             f"{cfg['pressure_cfl']:g}/{cfg['n']}², and iteration cap {cfg['pressure_max_steps']:,}. "
             f"Predictor and correction each use {cfg['predictor_substeps']} and "
             f"{cfg['correction_substeps']} Euler subdivisions, respectively. "
             "No direct pressure solve, coherent reset, or tolerance relaxation is used.\n\n"
             "| Da | Slurm job | Iterations | Final residual | Pressure converged | Full candidate accepted |\n"
             "|---|---|---|---|---|---|\n")
    for da, probe in zip((1, 10, 100), probes):
        last = probe["pressure_history"][-1]
        text += (f"| {da} | {probe['slurm_job_id']} | {last['pressure_iterations']:,} | "
                 f"{last['pressure_residual']:.9e} | {probe['pressure_converged']} | "
                 f"{probe['accepted_steps'] == 1} |\n")
    text += "\n![Measured pressure residuals](./pressure_convergence.png)\n\n"
    for da, probe in zip((1, 10, 100), probes):
        text += f"### Da={da}\n\n"
        if probe.get("error"):
            text += f"Candidate rejection: `{probe['error']}`.\n\n"
        if probe.get("candidate_diagnostics"):
            row = probe["candidate_diagnostics"]
            text += (f"Final relative divergence: {row['relative_divergence']:.3e}; "
                     f"invariant drift: {row['scalar_mass_error']:.3e}; "
                     f"coherent-eigenstate defect: {row['coherent_eigenstate_defect']:.3e}.\n\n")
        if probe.get("failed_gates"):
            for name, values in probe["failed_gates"].items():
                text += f"- {name}: {values['value']:.6e}, limit {values['limit']:.6e}.\n"
            text += "\n"
        text += (f"[Complete probe](./mf_euler_probe_da{da}_substeps{substeps}.json) · "
                 f"[DNS screening report](./dns_preflight_da{da}.json).\n\n")
    text += ("Pressure convergence is necessary but not sufficient for accepting a mean-field "
             "step. Slurm exit 0 for a diagnostic means the probe finished, not that every "
             "gate passed. No full MF run is implied by this report.\n\n"
             "[Pressure history CSV](./pressure_convergence.csv) · [Setup and status](./STATUS.md)\n")
    (output_dir/"PRESSURE_CONVERGENCE.md").write_text(text)
    return probes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--substeps", type=int, default=8)
    args = parser.parse_args()
    status = args.output_dir/"pressure_report_status.json"
    write_status(status, {"state": "running", "slurm_job_id": os.environ.get("SLURM_JOB_ID")})
    try:
        probes = make_report(args.output_dir, args.substeps)
        result = {"state": "complete", "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                  "pressure_converged": [p["pressure_converged"] for p in probes],
                  "accepted_steps": [p["accepted_steps"] for p in probes],
                  "report": "PRESSURE_CONVERGENCE.md"}
        write_status(status, result)
        print(json.dumps(result))
    except BaseException as error:
        write_status(status, {"state": "failed", "error": str(error),
                              "slurm_job_id": os.environ.get("SLURM_JOB_ID")})
        raise


if __name__ == "__main__":
    main()
