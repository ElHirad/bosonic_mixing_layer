#!/usr/bin/env python3
"""Single-site bosonic mean-field mixing layer with conserved concentration.

Every dynamical stage evolves explicit local Fock vectors under mean-field
operators f_j a_j^dagger + b_j a_j + c_j I. No TDVP, DNS call, amplitude-only
integrator, Poisson inversion, or post-step scalar-mass repair is used.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import time

import numpy as np
import scipy

import mean_field_bosons
import mean_field_operators
from mean_field_bosons import LocalBosons, relax_pressure
from mean_field_operators import MeanFieldOperators


GATES = {
    "relative_divergence": 1e-7, "scalar_mass_error": 1e-7, "pressure_mean": 1e-8,
    "wall_normal_velocity": 1e-14, "coherent_eigenstate_defect": 1e-5,
    "maximum_ceiling_probability": 1e-8, "maximum_norm_error": 1e-12,
    "maximum_imaginary_amplitude": 1e-12,
    "pressure_velocity_leakage": 1e-7, "pressure_scalar_leakage": 1e-7,
    "correction_pressure_leakage": 1e-7, "correction_scalar_leakage": 1e-7,
}


@dataclass(frozen=True)
class MeanFieldConfig:
    n: int = 16
    reynolds: float = 50.0
    peclet: float = 50.0
    dt: float = 0.0025
    final_time: float = 0.65
    boundary_y: str = "free-slip"
    shear_center: float = 0.5
    middle_fraction: float = 0.3
    transition: float = 0.04
    kh_width: float = 0.12
    kh_mode: int = 2
    kh_amplitude: float = 2.5
    secondary_mode: int = 1
    secondary_amplitude: float = 0.5
    phase: float = 0.0
    velocity_scale: float = 8.0
    pressure_scale: float = 4.0
    scalar_scale: float = 4.0
    boson_cutoff: int = 12
    predictor_substeps: int = 8
    correction_substeps: int = 8
    pressure_cfl: float = 0.125
    pressure_tolerance: float = 1e-8
    pressure_max_steps: int = 12000

    def __post_init__(self):
        for value in asdict(self).values():
            if isinstance(value, (float, int)) and not np.isfinite(value):
                raise ValueError("all numeric parameters must be finite")
        if min(self.dt, self.final_time, self.velocity_scale, self.pressure_scale,
               self.scalar_scale, self.reynolds, self.peclet, self.transition,
               self.kh_width, self.pressure_tolerance) <= 0:
            raise ValueError("times, scales, thicknesses, Re, Pe, and tolerance must be positive")
        if self.boundary_y not in ("free-slip", "periodic"):
            raise ValueError("unknown y boundary condition")
        for key in ("n", "boson_cutoff", "predictor_substeps", "correction_substeps",
                    "pressure_max_steps", "kh_mode", "secondary_mode"):
            value = getattr(self, key)
            if int(value) != value or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.n < 4 or self.boson_cutoff < 2:
            raise ValueError("n >= 4 and boson_cutoff >= 2 required")
        if max(self.kh_mode, self.secondary_mode) > self.n//2:
            raise ValueError("perturbation mode exceeds the Nyquist limit")
        if not 0 < self.shear_center < 1 or not 0 < self.middle_fraction < 1:
            raise ValueError("shear location/fraction must be in (0,1)")
        if not 0 < self.pressure_cfl <= 0.30:
            raise ValueError("pressure_cfl must be in (0,0.30] for explicit RK4 stability")
        ratio = self.final_time/self.dt
        if not np.isclose(ratio, round(ratio), rtol=1e-12, atol=1e-12) or round(ratio) < 7:
            raise ValueError("final_time/dt must be an integer >= 7 (eight snapshots)")

    @property
    def steps(self):
        return round(self.final_time/self.dt)

    @property
    def pseudo_dt(self):
        return self.pressure_cfl/self.n**2


def source_fingerprint(config):
    digest = hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode())
    for source in (Path(__file__), Path(mean_field_bosons.__file__), Path(mean_field_operators.__file__)):
        digest.update(source.read_bytes())
    digest.update(f"{np.__version__}/{scipy.__version__}".encode())
    return digest.hexdigest()


def initial_amplitudes(config):
    """Analytic initial encoding; discrete streamfunction curl needs no solve."""
    n, h = config.n, 1/config.n
    y = (np.arange(n)+0.5)*h
    channel = config.boundary_y == "free-slip"
    if channel:
        base = np.tanh((y-config.shear_center)/config.transition)
        profile = (1+base)/2
        psi_base = np.zeros(n+1)
        psi_base[1:] = h*np.cumsum(base)
        x, yy = np.meshgrid(np.arange(n)*h, np.arange(n+1)*h)
        envelope = np.exp(-((yy-config.shear_center)/config.kh_width)**2)
        envelope[[0, -1]] = 0
    else:
        lo, hi = (1-config.middle_fraction)/2, (1+config.middle_fraction)/2
        profile = (np.tanh((y-lo)/config.transition)-np.tanh((y-hi)/config.transition))/2
        base = 2*profile-1
        psi_base = np.zeros(n)
        psi_base[1:] = h*np.cumsum((base-base.mean())[:-1])
        x, yy = np.meshgrid(np.arange(n)*h, np.arange(n)*h)
        envelope = sum(np.exp(-(np.sin(np.pi*(yy-center))/(np.pi*config.kh_width))**2)
                       for center in (lo, hi))
    k, ks = 2*np.pi*config.kh_mode, 2*np.pi*config.secondary_mode
    psi = psi_base[:, None]+envelope*(config.kh_amplitude/k*np.cos(k*x)
                                     + config.secondary_amplitude/ks*np.cos(ks*x+config.phase))
    u = np.diff(psi, axis=0)/h if channel else (np.roll(psi, -1, axis=0)-psi)/h+base.mean()
    v = -(np.roll(psi, -1, axis=1)-psi)/h
    c = np.repeat(((profile-profile.min())/(profile.max()-profile.min()))[:, None], n, axis=1)
    return u/config.velocity_scale, v/config.velocity_scale, c/config.scalar_scale


def read_fields(states, bosons, operators, config):
    """Physical observables are s_j <psi_j|a_j|psi_j>/<psi_j|psi_j>."""
    fields = operators.layout.unpack(bosons.amplitudes(states))
    if any(np.max(np.abs(value.imag)) > 1e-12 for value in fields.values()):
        raise FloatingPointError("real physical fields developed imaginary amplitudes")
    return {f: value.real*({"u": config.velocity_scale, "v": config.velocity_scale,
                           "phi": config.pressure_scale, "c": config.scalar_scale}[f])
            for f, value in fields.items()}


def divergence(u, v, config):
    dy = v[1:]-v[:-1] if config.boundary_y == "free-slip" else np.roll(v, -1, axis=0)-v
    return config.n*(np.roll(u, -1, axis=1)-u+dy)


def vorticity(u, v, config):
    if config.boundary_y == "free-slip":
        du = np.zeros_like(v)
        du[1:-1] = u[1:]-u[:-1]
    else:
        du = u-np.roll(u, 1, axis=0)
    return config.n*(v-np.roll(v, 1, axis=1)-du)


def relative_difference(a, b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(a), 1e-12))


def advance_one_step(states, config, bosons, operators):
    """All three stages evolve explicit local kets under their MF generators."""
    tentative = bosons.advance(states, config.dt, operators.predictor, config.predictor_substeps)
    star = bosons.amplitudes(tentative)
    relaxed, stats = relax_pressure(tentative, bosons, operators, config.pseudo_dt,
                                    config.pressure_tolerance, config.pressure_max_steps)
    alpha_relaxed = bosons.amplitudes(relaxed)
    corrected = bosons.advance(relaxed, 1.0, operators.correction, config.correction_substeps)
    alpha_corrected = bosons.amplitudes(corrected)
    s = operators.layout.slices
    velocity = slice(s["u"].start, s["v"].stop)
    stats.update({
        "pressure_velocity_leakage": relative_difference(star[velocity], alpha_relaxed[velocity]),
        "pressure_scalar_leakage": relative_difference(star[s["c"]], alpha_relaxed[s["c"]]),
        "correction_pressure_leakage": relative_difference(alpha_relaxed[s["phi"]], alpha_corrected[s["phi"]]),
        "correction_scalar_leakage": relative_difference(alpha_relaxed[s["c"]], alpha_corrected[s["c"]]),
    })
    return corrected, stats


def diagnostics(states, config, bosons, operators, mass_reference, stage=None):
    fields = read_fields(states, bosons, operators, config)
    u, v, c, phi = (fields[f] for f in ("u", "v", "c", "phi"))
    if not all(np.all(np.isfinite(a)) for a in fields.values()):
        raise FloatingPointError("non-finite mean-field observables")
    div = divergence(u, v, config)
    result = {
        "relative_divergence": float(np.linalg.norm(div)/max(config.n*(np.linalg.norm(u)+np.linalg.norm(v)), 1e-12)),
        "max_divergence": float(np.max(np.abs(div))),
        "scalar_mass_error": float(abs(np.mean(c)-mass_reference)/max(abs(mass_reference), 1e-12)),
        "scalar_mass": float(np.mean(c)), "scalar_minimum": float(c.min()), "scalar_maximum": float(c.max()),
        "pressure_mean": float(abs(np.mean(phi))),
        "wall_normal_velocity": float(np.max(np.abs(v[[0, -1]]))) if config.boundary_y == "free-slip" else 0.0,
        "kinetic_energy": float(0.5*(np.sum(u*u)+np.sum(v*v))/config.n**2),
        **bosons.quality(states),
    }
    result.update({k: 0.0 for k in ("pressure_residual", "pressure_iterations", "pressure_pseudo_time",
                                   "pressure_velocity_leakage", "pressure_scalar_leakage",
                                   "correction_pressure_leakage", "correction_scalar_leakage")})
    if stage is not None:
        result.update(stage)
    return result


def check_quality(row, config):
    if not all(np.isfinite(v) for v in row.values()):
        raise FloatingPointError("non-finite diagnostic")
    for key, limit in {**GATES, "pressure_residual": config.pressure_tolerance}.items():
        if row[key] > limit*(1+1e-12):
            raise FloatingPointError(f"{key}={row[key]:.6e} exceeds {limit:.6e}")


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as file:
        temporary = Path(file.name)
        try:
            np.savez_compressed(file, **arrays)
            file.flush()
            os.fsync(file.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            json.dump(value, file, indent=2, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_simulation(config, output_dir, *, resume=False, checkpoint_interval=10,
                   max_steps=None, stop_file=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint_interval < 1 or (max_steps is not None and max_steps < 0):
        raise ValueError("invalid checkpoint interval or step limit")
    with (output_dir/".trajectory.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_locked(config, output_dir, resume, checkpoint_interval, max_steps, stop_file)


def _run_locked(config, output_dir, resume, checkpoint_interval, max_steps, stop_file):
    checkpoint = output_dir/"checkpoint.npz"
    fingerprint = source_fingerprint(config)
    schedule = np.rint(np.linspace(0, config.steps, 8)).astype(int)
    target = min(config.steps, config.steps if max_steps is None else max_steps)
    bosons, operators = LocalBosons(config.boson_cutoff), MeanFieldOperators(config)
    states = bosons.coherent_states(operators.layout.pack(*initial_amplitudes(config)))
    mass_reference = float(np.mean(read_fields(states, bosons, operators, config)["c"]))
    steps, snapshots, history, completed = [], [], [], 0
    if resume:
        with np.load(checkpoint, allow_pickle=False) as saved:
            if str(saved["fingerprint"]) != fingerprint:
                raise ValueError("checkpoint configuration/source/version mismatch")
            completed = int(saved["completed_step"])
            states = saved["terminal_local_states"].copy()
            mass_reference = float(saved["mass_reference"])
            steps = saved["snapshot_steps"].tolist()
            history = json.loads(str(saved["history_json"]))
            snapshots = [s.copy() for s in saved["local_states"]]
            if len(history) != completed or steps != [int(s) for s in schedule if s <= completed]:
                raise ValueError("inconsistent checkpoint history")
            if states.shape != (operators.layout.size, bosons.dimension):
                raise ValueError("incorrect checkpoint local-state shape")
            for row in history:
                check_quality(row, config)
    elif checkpoint.exists() or (output_dir/"mean_field_snapshots.npz").exists():
        raise FileExistsError("run data already exists; use --resume or a new output directory")
    if completed > target:
        raise ValueError("checkpoint exceeds requested step limit")
    check_quality(diagnostics(states, config, bosons, operators, mass_reference), config)
    if not resume:
        steps.append(0)
        snapshots.append(states.copy())
    resumed_from = completed
    started = time.perf_counter()

    def payload():
        fields = [read_fields(s, bosons, operators, config) for s in snapshots]
        return dict(
            format_version=np.array(2), config_json=np.array(json.dumps(asdict(config), sort_keys=True)),
            fingerprint=np.array(fingerprint), completed_step=np.array(completed), requested_steps=np.array(config.steps),
            mass_reference=np.array(mass_reference), snapshot_steps=np.array(steps), times=np.array(steps)*config.dt,
            history_json=np.array(json.dumps(history, allow_nan=False)), terminal_local_states=states,
            local_states=np.stack(snapshots),
            **{name: np.stack([s[key] for s in fields]) for name, key in
               (("u", "u"), ("v", "v"), ("concentration", "c"), ("pressure_impulse", "phi"))},
        )

    atomic_json(output_dir/"run_status.json", {"state": "running", "completed_step": completed})
    try:
        for step in range(completed+1, target+1):
            if stop_file is not None and Path(stop_file).exists():
                break
            fields = read_fields(states, bosons, operators, config)
            rate = config.n*(np.max(np.abs(fields["u"]))+np.max(np.abs(fields["v"])))
            safe_dt = min(0.35/max(rate, 1e-12), 0.2*min(config.reynolds, config.peclet)/config.n**2)
            if config.dt > safe_dt*(1+1e-12):
                raise FloatingPointError("physical dt exceeds advective/diffusive safety bound")
            candidate, stats = advance_one_step(states, config, bosons, operators)
            row = diagnostics(candidate, config, bosons, operators, mass_reference, stats)
            check_quality(row, config)
            states, completed = candidate, step
            history.append(dict(step=step, time=step*config.dt, **row))
            if step in schedule:
                steps.append(step)
                snapshots.append(states.copy())
            if step % checkpoint_interval == 0 or step in schedule:
                atomic_npz(checkpoint, **payload())
                print(f"step={step}/{config.steps} t={step*config.dt:.4f} "
                      f"mass_error={row['scalar_mass_error']:.3e} div={row['max_divergence']:.3e} "
                      f"pressure_steps={row['pressure_iterations']} wall={time.perf_counter()-started:.2f}s", flush=True)
        atomic_npz(checkpoint, **payload())
        result = payload()
        result["pressure"] = result["pressure_impulse"]/config.dt
        result["vorticity"] = np.stack([vorticity(u, v, config) for u, v in zip(result["u"], result["v"])])
        result["run_metadata_json"] = np.array(json.dumps({
            "elapsed_seconds": time.perf_counter()-started, "hostname": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "numpy": np.__version__, "scipy": scipy.__version__,
            "python": platform.python_version(), "resumed_from_step": resumed_from,
            "state_representation": "product of explicit single-site Fock vectors",
            "local_generator": "f_j a_j^dagger + b_j a_j + c_j I",
            "pressure_method": "single-site mean-field operator relaxation (RK4)",
            "mass_projection": False, "coherent_state_reset": False, "tdvp": False,
            "direct_poisson_solve": False, "operator_counts": operators.summary(),
        }))
        state = "complete" if completed == config.steps else "partial"
        result_path = output_dir/("mean_field_snapshots.npz" if state == "complete" else "partial_snapshots.npz")
        atomic_npz(result_path, **result)
        if state == "complete":
            atomic_json(output_dir/"validation.json", validate_results(result_path))
        atomic_json(output_dir/"run_status.json", {
            "state": state, "completed_step": completed, "requested_steps": config.steps, "data_file": result_path.name,
        })
        print(f"{state}: {completed}/{config.steps} steps, {time.perf_counter()-started:.3f}s", flush=True)
        return result_path
    except BaseException as error:
        atomic_npz(checkpoint, **payload())
        atomic_json(output_dir/"run_status.json", {"state": "failed", "completed_step": completed, "error": str(error)})
        raise


def validate_results(path):
    """Recompute observables from stored local Fock vectors, then check gates."""
    with np.load(path, allow_pickle=False) as data:
        config = MeanFieldConfig(**json.loads(str(data["config_json"])))
        bosons, operators = LocalBosons(config.boson_cutoff), MeanFieldOperators(config)
        schedule = np.rint(np.linspace(0, config.steps, 8)).astype(int)
        if int(data["completed_step"]) != config.steps or int(data["requested_steps"]) != config.steps:
            raise ValueError("trajectory is incomplete")
        if not np.array_equal(data["snapshot_steps"], schedule):
            raise ValueError("incorrect snapshot schedule")
        if not np.allclose(data["times"], schedule*config.dt, rtol=0, atol=1e-14):
            raise ValueError("incorrect snapshot times")
        if data["local_states"].shape != (8, operators.layout.size, bosons.dimension):
            raise ValueError("incorrect snapshot local-state shapes")
        if not np.array_equal(data["local_states"][-1], data["terminal_local_states"]):
            raise ValueError("terminal local states disagree with final snapshot")
        initial = bosons.coherent_states(operators.layout.pack(*initial_amplitudes(config)))
        if not np.allclose(initial, data["local_states"][0], rtol=0, atol=1e-14):
            raise ValueError("initial state does not match configuration")
        reference = float(np.mean(read_fields(initial, bosons, operators, config)["c"]))
        if abs(reference-float(data["mass_reference"])) > 1e-14:
            raise ValueError("incorrect initial scalar mass reference")
        history = json.loads(str(data["history_json"]))
        if [row["step"] for row in history] != list(range(1, config.steps+1)):
            raise ValueError("incomplete diagnostic history")
        for row in history:
            check_quality(row, config)
            if abs(row["time"]-row["step"]*config.dt) > 1e-14:
                raise ValueError("incorrect step time")
        for k, states in enumerate(data["local_states"]):
            check_quality(diagnostics(states, config, bosons, operators, reference), config)
            fields = read_fields(states, bosons, operators, config)
            fields.update(pressure=fields["phi"]/config.dt, concentration=fields["c"],
                          pressure_impulse=fields["phi"], vorticity=vorticity(fields["u"], fields["v"], config))
            for name in ("u", "v", "pressure", "pressure_impulse", "concentration", "vorticity"):
                if data[name][k].shape != fields[name].shape or not np.allclose(data[name][k], fields[name], rtol=0, atol=1e-12):
                    raise ValueError(f"{name} does not match stored bosonic states")
        return {
            "validated": True, "completed_steps": config.steps, "snapshots": 8,
            "config": asdict(config), "fingerprint": str(data["fingerprint"]),
            "worst_step": {key: max(row[key] for row in history) for key in (*GATES, "pressure_residual")},
            "total_pressure_iterations": sum(row["pressure_iterations"] for row in history),
            "final_step": history[-1], "run_metadata": json.loads(str(data["run_metadata_json"])),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, value in asdict(MeanFieldConfig()).items():
        flag = {"reynolds": "--re", "peclet": "--pe"}.get(name, "--"+name.replace("_", "-"))
        parser.add_argument(flag, dest=name, type=type(value), default=value)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mean_field_16x16_re50"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--validate-results", type=Path)
    args = vars(parser.parse_args())
    validation = args.pop("validate_results")
    if validation is not None:
        print(json.dumps(validate_results(validation), indent=2))
        return
    options = {name: args.pop(name) for name in ("output_dir", "resume", "checkpoint_interval", "max_steps", "stop_file")}
    config = MeanFieldConfig(**args)
    print(json.dumps(asdict(config), sort_keys=True), flush=True)
    print(f"saved: {run_simulation(config, **options)}", flush=True)


if __name__ == "__main__":
    main()
