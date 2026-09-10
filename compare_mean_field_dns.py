#!/usr/bin/env python3
"""Independent DNS benchmark initialized from measured mean-field observables.

This is a separate comparison program, never imported by the all-stage
single-site mean-field solver. DNS uses its original projected-midpoint
method; the mean-field trajectory uses predictor/relaxation/correction.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy

import mixing_layer_dns as dns
from mixing_layer_mean_field import MeanFieldConfig, atomic_json, atomic_npz, validate_results


def dns_config(config):
    return dns.SimulationConfig(
        nx=config.n, ny=config.n, dx=1/config.n, dy=1/config.n,
        reynolds=config.reynolds, peclet=config.peclet, boundary_y=config.boundary_y,
        shear_center_fraction=config.shear_center, middle_layer_fraction=config.middle_fraction,
        transition_thickness=config.transition, perturbation_width=config.kh_width,
        perturbation_mode=config.kh_mode, perturbation_amplitude=config.kh_amplitude,
        subharmonic_mode=config.secondary_mode, subharmonic_amplitude=config.secondary_amplitude,
        perturbation_phase=config.phase, max_dt=config.dt, t_end=config.final_time,
    )


def diagnostics(u, v, c, p, config, mass_reference):
    if not all(np.all(np.isfinite(a)) for a in (u, v, c, p)):
        raise FloatingPointError("non-finite DNS reference fields")
    div_fn = dns.channel_divergence if config.boundary_y == "free-slip" else dns.divergence
    div = div_fn(u, v, config.dx, config.dy)
    row = {
        "relative_divergence": float(np.linalg.norm(div)/max(config.nx*(np.linalg.norm(u)+np.linalg.norm(v)), 1e-12)),
        "max_divergence": float(np.max(np.abs(div))),
        "scalar_mass_error": float(abs(np.mean(c)-mass_reference)/max(abs(mass_reference), 1e-12)),
        "scalar_mass": float(np.mean(c)), "scalar_minimum": float(c.min()), "scalar_maximum": float(c.max()),
        "wall_normal_velocity": float(np.max(np.abs(v[[0, -1]]))) if config.boundary_y == "free-slip" else 0.0,
        "pressure_mean": float(abs(p.mean())),
    }
    for key, limit in {"relative_divergence": 1e-12, "scalar_mass_error": 1e-12,
                       "wall_normal_velocity": 1e-14, "pressure_mean": 1e-10}.items():
        if row[key] > limit:
            raise FloatingPointError(f"DNS reference {key}={row[key]:.6e} exceeds {limit:.6e}")
    return row


def run_reference(mean_field_path, output_path):
    """Run only DNS, starting from the exact t=0 measured MF u, v, and c."""
    validation = validate_results(mean_field_path)
    config = MeanFieldConfig(**validation["config"])
    physical = dns_config(config)
    with np.load(mean_field_path, allow_pickle=False) as source:
        u, v, c = (source[name][0].copy() for name in ("u", "v", "concentration"))
        steps = source["snapshot_steps"].copy()
        source_fingerprint = str(source["fingerprint"])
    pressure = np.zeros_like(u)
    mass_reference = float(c.mean())
    history, snapshots = [], [(u.copy(), v.copy(), c.copy(), pressure.copy())]
    initial = diagnostics(u, v, c, pressure, physical, mass_reference)
    started = time.perf_counter()
    for step in range(1, config.steps+1):
        if config.dt > dns.stable_timestep(u, v, physical)*(1+1e-12):
            raise FloatingPointError("matched fixed DNS timestep exceeds the stability bound")
        u, v, pressure, c = dns.advance_one_step_with_scalar(u, v, c, config.dt, physical)
        row = diagnostics(u, v, c, pressure, physical, mass_reference)
        history.append(dict(step=step, time=step*config.dt, **row))
        if step in steps:
            snapshots.append((u.copy(), v.copy(), c.copy(), pressure.copy()))
    if len(snapshots) != len(steps):
        raise ValueError("incorrect number of matched DNS snapshots")
    validation = {
        "validated": True, "completed_steps": config.steps, "snapshots": len(snapshots),
        "initial": initial,
        "worst_step": {key: max(row[key] for row in history) for key in
                       ("relative_divergence", "max_divergence", "scalar_mass_error", "wall_normal_velocity", "pressure_mean")},
        "concentration_range_all_steps": {
            "minimum": min(row["scalar_minimum"] for row in history),
            "maximum": max(row["scalar_maximum"] for row in history),
        },
    }
    metadata = {
        "method": "independent DNS: projected midpoint with direct discrete pressure solve",
        "initialization": "exact measured u, v, concentration from the mean-field t=0 snapshot",
        "mean_field_source_file": str(mean_field_path), "mean_field_fingerprint": source_fingerprint,
        "source_sha256": {Path(file).name: hashlib.sha256(Path(file).read_bytes()).hexdigest()
                          for file in (__file__, dns.__file__)},
        "elapsed_seconds": time.perf_counter()-started, "hostname": platform.node(),
        "numpy": np.__version__, "scipy": scipy.__version__, "python": platform.python_version(),
        "mass_projection": False, "scalar_clipping": False,
    }
    atomic_npz(output_path, times=steps*config.dt, snapshot_steps=steps,
               completed_step=np.array(config.steps), requested_steps=np.array(config.steps),
               config_json=np.array(json.dumps(asdict(physical), sort_keys=True)),
               history_json=np.array(json.dumps(history)), validation_json=np.array(json.dumps(validation)),
               run_metadata_json=np.array(json.dumps(metadata)),
               **{name: np.stack([snapshot[k] for snapshot in snapshots])
                  for k, name in enumerate(("u", "v", "concentration", "pressure"))},
               vorticity=np.stack([dns.vorticity(s[0], s[1], physical.dx, physical.dy) for s in snapshots]))
    atomic_json(Path(output_path).with_suffix(".validation.json"), {**validation, "config": asdict(physical), "metadata": metadata})
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mean_field_result", type=Path)
    parser.add_argument("dns_output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_reference(args.mean_field_result, args.dns_output), indent=2))


if __name__ == "__main__":
    main()
