#!/usr/bin/env python3
"""Run a matched DNS from an MPS initial state and plot their vorticity.

JLD2 stores Julia arrays in column-major order.  HDF5 readers expose the
snapshot axis first but reverse the two spatial axes, so the loaded MPS fields
are transposed into the DNS convention ``(time, y, x)`` before comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mixing_layer_dns import (
    FlowSnapshot,
    SimulationConfig,
    advance_one_step,
    cell_centered_velocity,
    flow_diagnostics,
    save_snapshots,
    stable_timestep,
    vorticity,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mps_data", type=Path, help="completed MPS JLD2 result")
    parser.add_argument("output_dir", type=Path, help="directory for plots and DNS data")
    parser.add_argument("--re", type=float, default=50.0)
    parser.add_argument("--transition-thickness", type=float, default=0.12)
    parser.add_argument("--perturbation-width", type=float, default=0.20)
    parser.add_argument("--middle-layer-fraction", type=float, default=0.30)
    parser.add_argument("--kh-mode", type=int, default=1)
    parser.add_argument("--kh-amplitude", type=float, default=0.10)
    parser.add_argument("--secondary-mode", type=int, default=2)
    parser.add_argument("--secondary-amplitude", type=float, default=0.025)
    parser.add_argument("--phase", type=float, default=float(np.pi / 5.0))
    parser.add_argument("--max-dt", type=float, default=0.0025)
    return parser.parse_args()


def load_mps(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as file:
        completed = int(file["completed_step"][()])
        requested = int(file["requested_steps"][()])
        if completed != requested:
            raise ValueError(f"incomplete MPS result: {completed}/{requested} steps")
        result = {
            "times": np.asarray(file["times"], dtype=float),
            "steps": np.asarray(file["snapshot_steps"], dtype=int),
            "u": np.asarray(file["u"], dtype=float).transpose(0, 2, 1),
            "v": np.asarray(file["v"], dtype=float).transpose(0, 2, 1),
            "vorticity": np.asarray(file["vorticity"], dtype=float).transpose(0, 2, 1),
        }
    if result["times"].shape != (8,):
        raise ValueError(f"expected 8 MPS snapshots, found {result['times'].shape}")
    if result["u"].shape != result["v"].shape or result["u"].shape != result["vorticity"].shape:
        raise ValueError("MPS u, v, and vorticity arrays have inconsistent shapes")
    if result["u"].shape[1] != result["u"].shape[2]:
        raise ValueError("this comparison expects a square MPS grid")
    return result


def run_matched_dns(
    mps: dict[str, np.ndarray], config: SimulationConfig
) -> tuple[list[FlowSnapshot], int]:
    """Advance DNS from the exact staggered MPS velocity at t=0."""

    u = mps["u"][0].copy()
    v = mps["v"][0].copy()
    pressure = np.zeros_like(u)
    snapshots = [
        FlowSnapshot(0.0, u.copy(), v.copy(), pressure.copy(), flow_diagnostics(u, v, config))
    ]
    time = 0.0
    steps = 0
    for target_time in mps["times"][1:]:
        while time < target_time - 16.0 * np.finfo(float).eps * max(1.0, target_time):
            dt = min(stable_timestep(u, v, config), target_time - time)
            u, v, pressure = advance_one_step(u, v, dt, config)
            time += dt
            steps += 1
            if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v))):
                raise FloatingPointError(f"non-finite DNS velocity at t={time:.8f}")
        time = float(target_time)
        snapshots.append(
            FlowSnapshot(
                time,
                u.copy(),
                v.copy(),
                pressure.copy(),
                flow_diagnostics(u, v, config),
            )
        )
    return snapshots, steps


def plot_field_grid(
    fields: np.ndarray,
    times: np.ndarray,
    output_path: Path,
    title: str,
    color_limit: float,
    velocities: tuple[np.ndarray, np.ndarray] | None = None,
    annotations: list[str] | None = None,
) -> None:
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(15.5, 7.6),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    images = []
    n = fields.shape[1]
    coordinates = (np.arange(n) + 0.5) / n
    stride = max(1, n // 8)
    for index, (axis, field, time) in enumerate(zip(axes.flat, fields, times)):
        image = axis.imshow(
            field,
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
            cmap="RdBu_r",
            vmin=-color_limit,
            vmax=color_limit,
            interpolation="bilinear",
            aspect="equal",
            rasterized=True,
        )
        images.append(image)
        if velocities is not None:
            u_center, v_center = cell_centered_velocity(
                velocities[0][index], velocities[1][index]
            )
            axis.quiver(
                coordinates[::stride],
                coordinates[::stride],
                u_center[::stride, ::stride],
                v_center[::stride, ::stride],
                color="black",
                alpha=0.42,
                pivot="mid",
                scale=18.0,
                width=0.003,
                headwidth=3.2,
            )
        panel_title = f"t = {time:.2f}"
        if annotations is not None:
            panel_title += f"\n{annotations[index]}"
        axis.set_title(panel_title)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_xticks([0.0, 0.5, 1.0])
        axis.set_yticks([0.0, 0.5, 1.0])
    for axis in axes[-1, :]:
        axis.set_xlabel("x")
    for axis in axes[:, 0]:
        axis.set_ylabel("y")
    colorbar = fig.colorbar(images[0], ax=axes, shrink=0.90, pad=0.015)
    colorbar.set_label(r"vorticity $\omega = \partial_x v - \partial_y u$")
    fig.suptitle(title, fontsize=14)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def comparison_metrics(
    mps: dict[str, np.ndarray], dns_u: np.ndarray, dns_v: np.ndarray, dns_omega: np.ndarray
) -> list[dict[str, float]]:
    rows = []
    for index, time in enumerate(mps["times"]):
        omega_delta = mps["vorticity"][index] - dns_omega[index]
        u_delta = mps["u"][index] - dns_u[index]
        v_delta = mps["v"][index] - dns_v[index]
        dns_omega_norm = max(float(np.linalg.norm(dns_omega[index])), np.finfo(float).tiny)
        dns_velocity_norm = max(
            float(np.hypot(np.linalg.norm(dns_u[index]), np.linalg.norm(dns_v[index]))),
            np.finfo(float).tiny,
        )
        rows.append(
            {
                "time": float(time),
                "vorticity_rmse": float(np.sqrt(np.mean(omega_delta**2))),
                "vorticity_relative_l2": float(np.linalg.norm(omega_delta) / dns_omega_norm),
                "vorticity_max_abs": float(np.max(np.abs(omega_delta))),
                "velocity_relative_l2": float(
                    np.hypot(np.linalg.norm(u_delta), np.linalg.norm(v_delta))
                    / dns_velocity_norm
                ),
            }
        )
    return rows


def main() -> None:
    args = parse_arguments()
    mps = load_mps(args.mps_data)
    n = mps["u"].shape[1]
    config = SimulationConfig(
        nx=n,
        ny=n,
        dx=1.0 / n,
        dy=1.0 / n,
        reynolds=args.re,
        middle_layer_fraction=args.middle_layer_fraction,
        transition_thickness=args.transition_thickness,
        perturbation_width=args.perturbation_width,
        perturbation_mode=args.kh_mode,
        perturbation_amplitude=args.kh_amplitude,
        subharmonic_mode=args.secondary_mode,
        subharmonic_amplitude=args.secondary_amplitude,
        perturbation_phase=args.phase,
        max_dt=args.max_dt,
        t_end=float(mps["times"][-1]),
        n_snapshots=len(mps["times"]),
    )

    # The saved MPS vorticity must agree with the common MAC difference stencil.
    reconstructed_initial = vorticity(mps["u"][0], mps["v"][0], config.dx, config.dy)
    if not np.allclose(reconstructed_initial, mps["vorticity"][0], rtol=0.0, atol=1e-12):
        raise ValueError("MPS array orientation or vorticity convention does not match DNS")

    dns_snapshots, dns_steps = run_matched_dns(mps, config)
    dns_u = np.stack([snapshot.u for snapshot in dns_snapshots])
    dns_v = np.stack([snapshot.v for snapshot in dns_snapshots])
    dns_omega = np.stack(
        [vorticity(snapshot.u, snapshot.v, config.dx, config.dy) for snapshot in dns_snapshots]
    )
    difference = mps["vorticity"] - dns_omega
    metrics = comparison_metrics(mps, dns_u, dns_v, dns_omega)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_snapshots(dns_snapshots, config, args.output_dir / "dns_snapshots.npz")
    common_limit = max(
        float(np.max(np.abs(mps["vorticity"]))),
        float(np.max(np.abs(dns_omega))),
        np.finfo(float).eps,
    )
    difference_limit = max(float(np.max(np.abs(difference))), np.finfo(float).eps)
    case_label = (
        f"16 x 16, Re = {config.reynolds:g}, transition = "
        f"{config.transition_thickness:g}, KH width = {config.perturbation_width:g}"
    )
    plot_field_grid(
        mps["vorticity"],
        mps["times"],
        args.output_dir / "mps_vorticity.png",
        f"Bosonic MPS Chorin/MAC mixing layer\n{case_label}",
        common_limit,
        velocities=(mps["u"], mps["v"]),
    )
    plot_field_grid(
        dns_omega,
        mps["times"],
        args.output_dir / "dns_vorticity.png",
        f"Classical DNS Chorin/MAC mixing layer\n{case_label}",
        common_limit,
        velocities=(dns_u, dns_v),
    )
    annotations = [f"rel. L2 = {row['vorticity_relative_l2']:.2e}" for row in metrics]
    plot_field_grid(
        difference,
        mps["times"],
        args.output_dir / "mps_minus_dns_vorticity.png",
        f"Vorticity difference: MPS - DNS\n{case_label}",
        difference_limit,
        annotations=annotations,
    )

    with (args.output_dir / "comparison_metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(metrics)
    summary = {
        "mps_data": args.mps_data.name,
        "dns_steps": dns_steps,
        "config": {
            "n": n,
            "reynolds": config.reynolds,
            "transition_thickness": config.transition_thickness,
            "perturbation_width": config.perturbation_width,
            "middle_layer_fraction": config.middle_layer_fraction,
            "kh_mode": config.perturbation_mode,
            "kh_amplitude": config.perturbation_amplitude,
            "secondary_mode": config.subharmonic_mode,
            "secondary_amplitude": config.subharmonic_amplitude,
            "phase": config.perturbation_phase,
            "max_dt": config.max_dt,
        },
        "metrics": metrics,
    }
    (args.output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"DNS steps: {dns_steps}")
    for row in metrics:
        print(
            f"t={row['time']:.2f} omega_rel_l2={row['vorticity_relative_l2']:.6e} "
            f"velocity_rel_l2={row['velocity_relative_l2']:.6e}"
        )
    for name in ("mps_vorticity.png", "dns_vorticity.png", "mps_minus_dns_vorticity.png"):
        print(f"saved: {args.output_dir / name}")


if __name__ == "__main__":
    main()
