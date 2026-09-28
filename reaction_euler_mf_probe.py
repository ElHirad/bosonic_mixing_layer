#!/usr/bin/env python3
"""Audit one full forward-Euler MF step and record pressure convergence.

This is a startup diagnostic, not a completed or accepted trajectory. It uses
the production stepper and gates unchanged and records rejected candidates.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import numpy as np

from mean_field_bosons import LocalBosons
from mean_field_operators import MeanFieldOperators
import mixing_layer_mean_field as mf


def probe(damkohler, substeps=8, *, n=64, reynolds=100., peclet=100., dt=.000625,
          pressure_max_steps=48000, progress_path=None):
    cfg = mf.MeanFieldConfig(n=n, reynolds=float(reynolds), peclet=float(peclet), transition=.01875,
        dt=dt, pressure_max_steps=pressure_max_steps, scalar_species=3, damkohler=float(damkohler),
        time_integrator="forward-euler", predictor_substeps=substeps, correction_substeps=substeps)
    started = time.perf_counter()
    bosons, ops = LocalBosons(cfg.boson_cutoff), MeanFieldOperators(cfg)
    states = bosons.coherent_states(ops.layout.pack(*mf.initial_amplitudes(cfg)))
    reference = mf.scalar_references(mf.read_fields(states, bosons, ops, cfg), cfg)
    stages = {"initial": mf.diagnostics(states, cfg, bosons, ops, reference)}
    report = {"kind": "one-step explicit single-site forward-Euler diagnostic, not a full trajectory",
              "config": asdict(cfg), "fingerprint": mf.source_fingerprint(cfg),
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "stages": stages,
              "attempted_steps": 1, "accepted_steps": 0, "pressure_history": [],
              "state": "running"}
    def publish():
        report["elapsed_seconds"] = time.perf_counter()-started
        if progress_path is not None:
            mf.atomic_json(progress_path, report)
    def pressure_observe(stats):
        report["pressure_history"].append({**stats, "elapsed_seconds": time.perf_counter()-started})
        report["pressure_converged"] = stats["pressure_residual"] <= cfg.pressure_tolerance
        if (stats["pressure_iterations"] % 5000 == 0 or report["pressure_converged"]
                or stats["pressure_iterations"] == cfg.pressure_max_steps):
            print(f"Da={damkohler:g} pressure_iterations={stats['pressure_iterations']} "
                  f"residual={stats['pressure_residual']:.9e}", flush=True)
            publish()
    def observe(name, candidate):
        stages[name] = mf.diagnostics(candidate, cfg, bosons, ops, reference)
        print(f"Da={damkohler:g} {name}: defect={stages[name]['coherent_eigenstate_defect']:.6e} "
              f"invariant_error={stages[name]['scalar_mass_error']:.6e}", flush=True)
        publish()
    publish()
    try:
        candidate, stats = mf.advance_one_step(states, cfg, bosons, ops, stage_observer=observe,
                                              pressure_observer=pressure_observe)
        row = mf.diagnostics(candidate, cfg, bosons, ops, reference, stats)
        report["candidate_diagnostics"] = row
        report["failed_gates"] = {key: {"value": row[key], "limit": limit}
                                  for key, limit in {**mf.GATES, "pressure_residual": cfg.pressure_tolerance}.items()
                                  if row[key] > limit*(1+1e-12)}
        mf.check_quality(row, cfg)
        report["accepted_steps"] = 1
        report["state"] = "one_step_passed"
    except (FloatingPointError, ValueError) as error:
        report.update(state="rejected_candidate", error=str(error))
    publish()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--re", type=float, default=100)
    parser.add_argument("--pe", type=float, default=100)
    parser.add_argument("--dt", type=float, default=.000625)
    parser.add_argument("--pressure-max-steps", type=int, default=48000)
    parser.add_argument("--damkohler", type=float, choices=(1, 10, 100), nargs="+", default=[1, 10, 100])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for da in args.damkohler:
        path = args.output_dir/f"mf_euler_probe_da{da:g}_substeps{args.substeps}.json"
        progress = args.output_dir/f"mf_euler_probe_da{da:g}_substeps{args.substeps}.progress.json"
        if path.exists() or progress.exists():
            raise FileExistsError("diagnostic output exists; choose a new output directory")
        result = probe(da, args.substeps, n=args.n, reynolds=args.re, peclet=args.pe, dt=args.dt,
                       pressure_max_steps=args.pressure_max_steps, progress_path=progress)
        mf.atomic_json(path, result)
        print(json.dumps({"Da": da, "state": result["state"], "elapsed": result["elapsed_seconds"],
                          "failed_gates": result.get("failed_gates"), "file": str(path)}), flush=True)


if __name__ == "__main__":
    main()
