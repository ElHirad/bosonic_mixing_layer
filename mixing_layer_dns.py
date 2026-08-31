#!/usr/bin/env python3
"""Two-dimensional incompressible mixing-layer DNS on a periodic MAC grid.

The velocity equations are advanced with a second-order midpoint method.  Each
stage uses Chorin's fractional-step projection: an explicit tentative velocity
is formed, a cell-centered pressure Poisson equation is solved, and the normal
face velocities are corrected to make the discrete divergence zero.

All arrays have shape ``(ny, nx)`` and store one copy of each periodic degree
of freedom:

* ``p[j, i]`` is at the cell center ``((i+1/2) dx, (j+1/2) dy)``;
* ``u[j, i]`` is at the west/east face ``(i dx, (j+1/2) dy)``;
* ``v[j, i]`` is at the south/north face ``((i+1/2) dx, j dy)``.

The periodic pressure equation is inverted with an FFT using the eigenvalues
of the *discrete* five-point Laplacian.  Spatial derivatives in the momentum
equations remain second-order finite differences on the staggered grid.

The matched MPS comparison also advances a cell-centered passive concentration
with centered conservative MAC fluxes and diffusivity ``1/Pe``. Its explicit
midpoint stages use the corresponding projected velocity stages.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
try:
    from numpy.typing import NDArray
except ImportError:  # NumPy 1.20 on the cluster predates numpy.typing.NDArray.
    NDArray = None


Array = np.ndarray if NDArray is None else NDArray[np.float64]


@dataclass(frozen=True)
class SimulationConfig:
    """Physical and numerical parameters for the mixing-layer calculation."""

    nx: int = 128
    ny: int = 128
    dx: float = 1.0 / 128.0
    dy: float = 1.0 / 128.0
    reynolds: float = 1000.0
    peclet: float | None = None
    reference_velocity: float = 1.0
    middle_layer_fraction: float = 0.30
    transition_thickness: float = 0.03
    perturbation_width: float = 0.06
    perturbation_mode: int = 2
    perturbation_amplitude: float = 0.020
    subharmonic_mode: int = 1
    subharmonic_amplitude: float = 0.005
    perturbation_phase: float = np.pi / 5.0
    cfl: float = 0.35
    diffusion_safety: float = 0.80
    max_dt: float = 0.0025
    t_end: float = 3.5
    n_snapshots: int = 8

    def __post_init__(self) -> None:
        if self.nx < 4 or self.ny < 4:
            raise ValueError("nx and ny must both be at least 4")
        if self.dx <= 0.0 or self.dy <= 0.0:
            raise ValueError("dx and dy must be positive")
        if self.reynolds <= 0.0:
            raise ValueError("reynolds must be positive")
        if self.peclet is not None and self.peclet <= 0.0:
            raise ValueError("peclet must be positive")
        if self.transition_thickness <= 0.0 or self.perturbation_width <= 0.0:
            raise ValueError("layer thicknesses must be positive")
        if not (0.0 < self.middle_layer_fraction < 1.0):
            raise ValueError("middle_layer_fraction must lie in (0, 1)")
        if self.perturbation_mode <= 0 or self.subharmonic_mode <= 0:
            raise ValueError("perturbation modes must be positive integers")
        if not (0.0 < self.cfl <= 1.0):
            raise ValueError("cfl must lie in (0, 1]")
        if self.max_dt <= 0.0 or self.t_end <= 0.0:
            raise ValueError("max_dt and t_end must be positive")
        if self.n_snapshots != 8:
            raise ValueError("this calculation is configured for exactly 8 snapshots")

    @property
    def lx(self) -> float:
        return self.nx * self.dx

    @property
    def ly(self) -> float:
        return self.ny * self.dy

    @property
    def viscosity(self) -> float:
        # Re = U L / nu, with L equal to the periodic streamwise length.
        return self.reference_velocity * self.lx / self.reynolds

    @property
    def scalar_diffusivity(self) -> float:
        """Passive-scalar diffusivity from ``Pe = U L / kappa``."""

        return self.reference_velocity * self.lx / self.effective_peclet

    @property
    def effective_peclet(self) -> float:
        """Return explicit Pe, or Re when Pe was not independently specified."""

        return self.reynolds if self.peclet is None else self.peclet


@dataclass
class FlowSnapshot:
    """One independently stored flow state and its scalar diagnostics."""

    time: float
    u: Array
    v: Array
    pressure: Array
    diagnostics: dict[str, float]


def divergence(u: Array, v: Array, dx: float, dy: float) -> Array:
    """Cell-centered MAC divergence of periodic face velocities."""

    return (
        (np.roll(u, -1, axis=1) - u) / dx
        + (np.roll(v, -1, axis=0) - v) / dy
    )


def pressure_gradient(pressure: Array, dx: float, dy: float) -> tuple[Array, Array]:
    """Pressure gradients at u and v faces, respectively."""

    grad_x = (pressure - np.roll(pressure, 1, axis=1)) / dx
    grad_y = (pressure - np.roll(pressure, 1, axis=0)) / dy
    return grad_x, grad_y


def laplacian(field: Array, dx: float, dy: float) -> Array:
    """Periodic second-order five-point Laplacian."""

    return (
        (np.roll(field, -1, axis=1) - 2.0 * field + np.roll(field, 1, axis=1))
        / dx**2
        + (np.roll(field, -1, axis=0) - 2.0 * field + np.roll(field, 1, axis=0))
        / dy**2
    )


def solve_periodic_poisson(rhs: Array, dx: float, dy: float) -> Array:
    """Solve the mean-zero periodic discrete Poisson equation ``L p = rhs``."""

    ny, nx = rhs.shape
    compatible_rhs = rhs - np.mean(rhs)
    rhs_hat = np.fft.fft2(compatible_rhs)

    mode_x = np.fft.fftfreq(nx)
    mode_y = np.fft.fftfreq(ny)
    lambda_x = -4.0 * np.sin(np.pi * mode_x) ** 2 / dx**2
    lambda_y = -4.0 * np.sin(np.pi * mode_y) ** 2 / dy**2
    eigenvalues = lambda_y[:, None] + lambda_x[None, :]

    pressure_hat = np.zeros_like(rhs_hat)
    nonzero = eigenvalues != 0.0
    pressure_hat[nonzero] = rhs_hat[nonzero] / eigenvalues[nonzero]
    pressure_hat[0, 0] = 0.0
    pressure = np.fft.ifft2(pressure_hat).real
    return pressure - np.mean(pressure)


def project_velocity(
    u_star: Array,
    v_star: Array,
    dt: float,
    dx: float,
    dy: float,
) -> tuple[Array, Array, Array]:
    """Apply one Chorin pressure projection to tentative MAC velocities."""

    if dt <= 0.0:
        raise ValueError("projection dt must be positive")
    pressure = solve_periodic_poisson(divergence(u_star, v_star, dx, dy) / dt, dx, dy)
    grad_x, grad_y = pressure_gradient(pressure, dx, dy)
    u = u_star - dt * grad_x
    v = v_star - dt * grad_y
    return u, v, pressure


def conservative_advection(u: Array, v: Array, dx: float, dy: float) -> tuple[Array, Array]:
    """Second-order centered momentum-flux divergence on the MAC grid.

    The returned arrays approximate ``div(u u)`` and ``div(u v)`` at the u and
    v faces.  All face/corner interpolations are centered and periodic.
    """

    # Fluxes through the east/west and north/south sides of a u control volume.
    u_e = 0.5 * (u + np.roll(u, -1, axis=1))
    u_w = 0.5 * (np.roll(u, 1, axis=1) + u)
    u_n = 0.5 * (u + np.roll(u, -1, axis=0))
    u_s = 0.5 * (np.roll(u, 1, axis=0) + u)

    v_n_row = np.roll(v, -1, axis=0)
    v_at_north_u_corner = 0.5 * (v_n_row + np.roll(v_n_row, 1, axis=1))
    v_at_south_u_corner = 0.5 * (v + np.roll(v, 1, axis=1))

    advection_u = (
        (u_e**2 - u_w**2) / dx
        + (u_n * v_at_north_u_corner - u_s * v_at_south_u_corner) / dy
    )

    # Fluxes through the east/west and north/south sides of a v control volume.
    v_e = 0.5 * (v + np.roll(v, -1, axis=1))
    v_w = 0.5 * (np.roll(v, 1, axis=1) + v)
    v_n = 0.5 * (v + np.roll(v, -1, axis=0))
    v_s = 0.5 * (np.roll(v, 1, axis=0) + v)

    u_e_column = np.roll(u, -1, axis=1)
    u_at_east_v_corner = 0.5 * (u_e_column + np.roll(u_e_column, 1, axis=0))
    u_at_west_v_corner = 0.5 * (u + np.roll(u, 1, axis=0))

    advection_v = (
        (u_at_east_v_corner * v_e - u_at_west_v_corner * v_w) / dx
        + (v_n**2 - v_s**2) / dy
    )
    return advection_u, advection_v


def conservative_scalar_advection(
    u: Array,
    v: Array,
    concentration: Array,
    dx: float,
    dy: float,
) -> Array:
    """Centered conservative ``div(u*c)`` at scalar cell centers."""

    if u.shape != v.shape or u.shape != concentration.shape:
        raise ValueError("u, v, and concentration must have identical shapes")
    concentration_e = 0.5 * (concentration + np.roll(concentration, -1, axis=1))
    concentration_w = 0.5 * (np.roll(concentration, 1, axis=1) + concentration)
    concentration_n = 0.5 * (concentration + np.roll(concentration, -1, axis=0))
    concentration_s = 0.5 * (np.roll(concentration, 1, axis=0) + concentration)
    return (
        (
            np.roll(u, -1, axis=1) * concentration_e
            - u * concentration_w
        )
        / dx
        + (
            np.roll(v, -1, axis=0) * concentration_n
            - v * concentration_s
        )
        / dy
    )


def scalar_rhs(
    u: Array,
    v: Array,
    concentration: Array,
    diffusivity: float,
    dx: float,
    dy: float,
) -> Array:
    """Conservative passive-scalar advection-diffusion right-hand side."""

    return -conservative_scalar_advection(u, v, concentration, dx, dy) + (
        diffusivity * laplacian(concentration, dx, dy)
    )


def momentum_rhs(
    u: Array,
    v: Array,
    viscosity: float,
    dx: float,
    dy: float,
) -> tuple[Array, Array]:
    """Pressure-free right-hand side for the incompressible momentum equations."""

    advection_u, advection_v = conservative_advection(u, v, dx, dy)
    rhs_u = -advection_u + viscosity * laplacian(u, dx, dy)
    rhs_v = -advection_v + viscosity * laplacian(v, dx, dy)
    return rhs_u, rhs_v


def periodic_double_tanh_profile(y: Array, config: SimulationConfig) -> Array:
    """Periodic -1/+1/-1 double layer with two resolved tanh transitions."""

    lower_interface = 0.5 * (1.0 - config.middle_layer_fraction) * config.ly
    upper_interface = 0.5 * (1.0 + config.middle_layer_fraction) * config.ly
    return config.reference_velocity * (
        np.tanh((y - lower_interface) / config.transition_thickness)
        - np.tanh((y - upper_interface) / config.transition_thickness)
        - 1.0
    )


def periodic_double_tanh_concentration(y: Array, config: SimulationConfig) -> Array:
    """Unnormalized 0/1/0 double-tanh concentration profile."""

    lower_interface = 0.5 * (1.0 - config.middle_layer_fraction) * config.ly
    upper_interface = 0.5 * (1.0 + config.middle_layer_fraction) * config.ly
    return 0.5 * (
        np.tanh((y - lower_interface) / config.transition_thickness)
        - np.tanh((y - upper_interface) / config.transition_thickness)
    )


def initialize_concentration(config: SimulationConfig) -> Array:
    """Return the sampled double-tanh scalar with exact 0 and 1 plateaus."""

    y_centers = (np.arange(config.ny, dtype=float) + 0.5) * config.dy
    profile = periodic_double_tanh_concentration(y_centers, config)
    minimum = float(np.min(profile))
    maximum = float(np.max(profile))
    if maximum <= minimum:
        raise ValueError("degenerate sampled concentration profile")
    normalized = (profile - minimum) / (maximum - minimum)
    return np.repeat(normalized[:, None], config.nx, axis=1)


def initialize_velocity(config: SimulationConfig) -> tuple[Array, Array, Array]:
    """Construct a discretely divergence-free double shear layer plus KH seed.

    A streamfunction is stored at cell vertices.  Taking its discrete curl puts
    u and v directly on their MAC faces and makes ``D u = 0`` by cancellation.
    """

    x_vertices = np.arange(config.nx, dtype=float) * config.dx
    y_vertices = np.arange(config.ny, dtype=float) * config.dy
    y_u_faces = (np.arange(config.ny, dtype=float) + 0.5) * config.dy

    target_base_u = periodic_double_tanh_profile(y_u_faces, config)
    mean_base_u = float(np.mean(target_base_u))
    # A periodic streamfunction represents the mean-zero part.  The uniform
    # mean is added back after taking the curl, preserving the requested
    # -1/+1/-1 plateaus even when their occupied areas are unequal.
    base_u_fluctuation = target_base_u - mean_base_u
    base_streamfunction = np.zeros(config.ny, dtype=float)
    base_streamfunction[1:] = config.dy * np.cumsum(base_u_fluctuation[:-1])

    x_mesh, y_mesh = np.meshgrid(x_vertices, y_vertices)
    lower_interface = 0.5 * (1.0 - config.middle_layer_fraction) * config.ly
    upper_interface = 0.5 * (1.0 + config.middle_layer_fraction) * config.ly

    def periodic_envelope(center: float) -> Array:
        scaled_distance = np.sin(np.pi * (y_mesh - center) / config.ly)
        scaled_width = np.pi * config.perturbation_width / config.ly
        return np.exp(-(scaled_distance / scaled_width) ** 2)

    envelope = periodic_envelope(lower_interface) + periodic_envelope(upper_interface)
    dominant_k = 2.0 * np.pi * config.perturbation_mode / config.lx
    subharmonic_k = 2.0 * np.pi * config.subharmonic_mode / config.lx
    perturbation_streamfunction = envelope * (
        config.perturbation_amplitude / dominant_k * np.cos(dominant_k * x_mesh)
        + config.subharmonic_amplitude
        / subharmonic_k
        * np.cos(subharmonic_k * x_mesh + config.perturbation_phase)
    )

    streamfunction = base_streamfunction[:, None] + perturbation_streamfunction
    u = (
        (np.roll(streamfunction, -1, axis=0) - streamfunction) / config.dy
        + mean_base_u
    )
    v = -(np.roll(streamfunction, -1, axis=1) - streamfunction) / config.dx
    pressure = np.zeros_like(u)
    return u, v, pressure


def advance_one_step(
    u: Array,
    v: Array,
    dt: float,
    config: SimulationConfig,
) -> tuple[Array, Array, Array]:
    """Second-order midpoint update with a Chorin projection at each stage."""

    rhs_u, rhs_v = momentum_rhs(u, v, config.viscosity, config.dx, config.dy)
    u_half_star = u + 0.5 * dt * rhs_u
    v_half_star = v + 0.5 * dt * rhs_v
    u_half, v_half, _ = project_velocity(
        u_half_star, v_half_star, 0.5 * dt, config.dx, config.dy
    )

    rhs_u_half, rhs_v_half = momentum_rhs(
        u_half, v_half, config.viscosity, config.dx, config.dy
    )
    u_star = u + dt * rhs_u_half
    v_star = v + dt * rhs_v_half
    return project_velocity(u_star, v_star, dt, config.dx, config.dy)


def advance_one_step_with_scalar(
    u: Array,
    v: Array,
    concentration: Array,
    dt: float,
    config: SimulationConfig,
) -> tuple[Array, Array, Array, Array]:
    """Second-order midpoint update of velocity and its passive scalar."""

    rhs_u, rhs_v = momentum_rhs(u, v, config.viscosity, config.dx, config.dy)
    u_half_star = u + 0.5 * dt * rhs_u
    v_half_star = v + 0.5 * dt * rhs_v
    u_half, v_half, _ = project_velocity(
        u_half_star, v_half_star, 0.5 * dt, config.dx, config.dy
    )
    concentration_half = concentration + 0.5 * dt * scalar_rhs(
        u,
        v,
        concentration,
        config.scalar_diffusivity,
        config.dx,
        config.dy,
    )

    rhs_u_half, rhs_v_half = momentum_rhs(
        u_half, v_half, config.viscosity, config.dx, config.dy
    )
    u_star = u + dt * rhs_u_half
    v_star = v + dt * rhs_v_half
    concentration_next = concentration + dt * scalar_rhs(
        u_half,
        v_half,
        concentration_half,
        config.scalar_diffusivity,
        config.dx,
        config.dy,
    )
    u_next, v_next, pressure = project_velocity(
        u_star, v_star, dt, config.dx, config.dy
    )
    return u_next, v_next, pressure, concentration_next


def stable_timestep(u: Array, v: Array, config: SimulationConfig) -> float:
    """Return a conservative explicit advection/diffusion time step."""

    advective_rate = np.max(np.abs(u)) / config.dx + np.max(np.abs(v)) / config.dy
    dt_advection = config.cfl / max(advective_rate, np.finfo(float).tiny)
    strongest_diffusivity = max(config.viscosity, config.scalar_diffusivity)
    dt_diffusion = config.diffusion_safety * 0.5 / (
        strongest_diffusivity * (1.0 / config.dx**2 + 1.0 / config.dy**2)
    )
    return min(config.max_dt, dt_advection, dt_diffusion)


def vorticity(u: Array, v: Array, dx: float, dy: float) -> Array:
    """Vertex-centered scalar vorticity ``dv/dx - du/dy``."""

    return (
        (v - np.roll(v, 1, axis=1)) / dx
        - (u - np.roll(u, 1, axis=0)) / dy
    )


def cell_centered_velocity(u: Array, v: Array) -> tuple[Array, Array]:
    """Interpolate staggered velocities to pressure-cell centers."""

    u_center = 0.5 * (u + np.roll(u, -1, axis=1))
    v_center = 0.5 * (v + np.roll(v, -1, axis=0))
    return u_center, v_center


def flow_diagnostics(u: Array, v: Array, config: SimulationConfig) -> dict[str, float]:
    """Compute compact resolution and incompressibility diagnostics."""

    omega = vorticity(u, v, config.dx, config.dy)
    div = divergence(u, v, config.dx, config.dy)
    return {
        "kinetic_energy": 0.5 * float(np.mean(u**2 + v**2)),
        "cross_stream_energy": 0.5 * float(np.mean(v**2)),
        "enstrophy": 0.5 * float(np.mean(omega**2)),
        "max_divergence": float(np.max(np.abs(div))),
        "mean_u": float(np.mean(u)),
        "mean_v": float(np.mean(v)),
        "max_speed_component": float(max(np.max(np.abs(u)), np.max(np.abs(v)))),
    }


def _make_snapshot(
    time: float,
    u: Array,
    v: Array,
    pressure: Array,
    config: SimulationConfig,
) -> FlowSnapshot:
    return FlowSnapshot(
        time=float(time),
        u=u.copy(),
        v=v.copy(),
        pressure=pressure.copy(),
        diagnostics=flow_diagnostics(u, v, config),
    )


def run_simulation(
    config: SimulationConfig,
    progress: Callable[[int, int, FlowSnapshot], None] | None = None,
) -> list[FlowSnapshot]:
    """Advance the flow and return states at eight exact, evenly spaced times."""

    u, v, pressure = initialize_velocity(config)
    snapshot_times = np.linspace(0.0, config.t_end, config.n_snapshots)
    snapshots = [_make_snapshot(0.0, u, v, pressure, config)]
    steps = 0
    if progress is not None:
        progress(0, steps, snapshots[0])

    time = 0.0
    for snapshot_index, target_time in enumerate(snapshot_times[1:], start=1):
        while time < target_time - 16.0 * np.finfo(float).eps * max(1.0, target_time):
            dt = min(stable_timestep(u, v, config), target_time - time)
            u, v, pressure = advance_one_step(u, v, dt, config)
            time += dt
            steps += 1
            if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v))):
                raise FloatingPointError(f"non-finite velocity encountered at t={time:.8f}")
        time = float(target_time)
        snapshot = _make_snapshot(time, u, v, pressure, config)
        snapshots.append(snapshot)
        if progress is not None:
            progress(snapshot_index, steps, snapshot)

    return snapshots


def plot_snapshots(
    snapshots: Sequence[FlowSnapshot],
    config: SimulationConfig,
    output_path: Path,
) -> None:
    """Save a two-by-four vorticity figure with sparse velocity arrows."""

    if len(snapshots) != 8:
        raise ValueError("plot_snapshots requires exactly 8 snapshots")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    omega_fields = [vorticity(s.u, s.v, config.dx, config.dy) for s in snapshots]
    color_limit = max(float(np.max(np.abs(field))) for field in omega_fields)

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(15.5, 7.6),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    images = []
    x_centers = (np.arange(config.nx) + 0.5) * config.dx
    y_centers = (np.arange(config.ny) + 0.5) * config.dy
    arrow_stride_x = max(1, config.nx // 8)
    arrow_stride_y = max(1, config.ny // 8)

    for axis, snapshot, omega in zip(axes.flat, snapshots, omega_fields, strict=True):
        image = axis.imshow(
            omega,
            origin="lower",
            extent=(0.0, config.lx, 0.0, config.ly),
            cmap="RdBu_r",
            vmin=-color_limit,
            vmax=color_limit,
            interpolation="bilinear",
            aspect="equal",
            rasterized=True,
        )
        images.append(image)
        u_center, v_center = cell_centered_velocity(snapshot.u, snapshot.v)
        axis.quiver(
            x_centers[::arrow_stride_x],
            y_centers[::arrow_stride_y],
            u_center[::arrow_stride_y, ::arrow_stride_x],
            v_center[::arrow_stride_y, ::arrow_stride_x],
            color="black",
            alpha=0.42,
            pivot="mid",
            scale=18.0,
            width=0.003,
            headwidth=3.2,
        )
        axis.set_title(f"t = {snapshot.time:.2f}")
        axis.set_xlim(0.0, config.lx)
        axis.set_ylim(0.0, config.ly)
        axis.set_xticks([0.0, 0.5 * config.lx, config.lx])
        axis.set_yticks([0.0, 0.5 * config.ly, config.ly])

    for axis in axes[-1, :]:
        axis.set_xlabel("x")
    for axis in axes[:, 0]:
        axis.set_ylabel("y")

    colorbar = fig.colorbar(images[0], ax=axes, shrink=0.90, pad=0.015)
    colorbar.set_label(r"vorticity $\omega = \partial_x v - \partial_y u$")
    fig.suptitle(
        "2-D periodic mixing layer: MAC grid + Chorin projection\n"
        f"Re = {config.reynolds:g},  {config.nx} x {config.ny},  "
        f"dx = dy = {config.dx:.6f},  middle layer = "
        f"{config.middle_layer_fraction:.2f} Ly",
        fontsize=14,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_snapshots(
    snapshots: Sequence[FlowSnapshot],
    config: SimulationConfig,
    output_path: Path,
    *,
    concentration_snapshots: Sequence[Array] | None = None,
) -> None:
    """Save face velocities, pressure, vorticity, and diagnostics to a compressed NPZ."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_names = tuple(snapshots[0].diagnostics)
    arrays = dict(
        times=np.array([snapshot.time for snapshot in snapshots]),
        u=np.stack([snapshot.u for snapshot in snapshots]),
        v=np.stack([snapshot.v for snapshot in snapshots]),
        pressure=np.stack([snapshot.pressure for snapshot in snapshots]),
        vorticity=np.stack(
            [vorticity(snapshot.u, snapshot.v, config.dx, config.dy) for snapshot in snapshots]
        ),
        diagnostic_names=np.array(diagnostic_names),
        diagnostics=np.array(
            [
                [snapshot.diagnostics[name] for name in diagnostic_names]
                for snapshot in snapshots
            ]
        ),
        config_json=np.array(json.dumps(asdict(config), sort_keys=True)),
    )
    if concentration_snapshots is not None:
        if len(concentration_snapshots) != len(snapshots):
            raise ValueError("concentration snapshot count does not match flow snapshots")
        arrays["concentration"] = np.stack(concentration_snapshots)
    np.savez_compressed(output_path, **arrays)


def _progress(snapshot_index: int, steps: int, snapshot: FlowSnapshot) -> None:
    diagnostics = snapshot.diagnostics
    print(
        f"snapshot {snapshot_index + 1}/8  t={snapshot.time:7.4f}  steps={steps:5d}  "
        f"E={diagnostics['kinetic_energy']:.8e}  "
        f"Z={diagnostics['enstrophy']:.8e}  "
        f"||div u||inf={diagnostics['max_divergence']:.3e}",
        flush=True,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="directory for the PNG and NPZ files (default: outputs)",
    )
    parser.add_argument(
        "--nx",
        type=int,
        default=None,
        help="number of periodic cells in x; dx is adjusted to keep Lx=1",
    )
    parser.add_argument(
        "--ny",
        type=int,
        default=None,
        help="number of periodic cells in y; dy is adjusted to keep Ly=1",
    )
    parser.add_argument(
        "--re",
        type=float,
        default=None,
        help="override the Reynolds number",
    )
    parser.add_argument(
        "--pe",
        type=float,
        default=None,
        help="override Pe (default: same value as Re)",
    )
    parser.add_argument(
        "--t-end",
        type=float,
        default=None,
        help="override the default final nondimensional time 3.5",
    )
    parser.add_argument(
        "--middle-layer-fraction",
        type=float,
        default=None,
        help="middle-layer thickness divided by Ly (default: 0.30)",
    )
    parser.add_argument(
        "--transition-thickness",
        type=float,
        default=None,
        help="override the tanh transition parameter",
    )
    parser.add_argument(
        "--perturbation-width",
        type=float,
        default=None,
        help="override the periodic KH envelope width",
    )
    parser.add_argument(
        "--kh-mode",
        type=int,
        default=None,
        help="override the dominant KH streamwise mode",
    )
    parser.add_argument(
        "--kh-amplitude",
        type=float,
        default=None,
        help="override the dominant KH cross-stream velocity amplitude",
    )
    parser.add_argument(
        "--secondary-mode",
        type=int,
        default=None,
        help="override the secondary KH streamwise mode",
    )
    parser.add_argument(
        "--secondary-amplitude",
        type=float,
        default=None,
        help="override the secondary KH cross-stream velocity amplitude",
    )
    parser.add_argument(
        "--no-data",
        action="store_true",
        help="do not save the compressed snapshot arrays",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config = SimulationConfig()
    nx = config.nx if args.nx is None else args.nx
    ny = config.ny if args.ny is None else args.ny
    config = replace(
        config,
        nx=nx,
        ny=ny,
        dx=config.lx / nx,
        dy=config.ly / ny,
    )
    if args.re is not None:
        config = replace(config, reynolds=args.re, peclet=args.re)
    if args.pe is not None:
        config = replace(config, peclet=args.pe)
    if args.t_end is not None:
        config = replace(config, t_end=args.t_end)
    if args.middle_layer_fraction is not None:
        config = replace(config, middle_layer_fraction=args.middle_layer_fraction)
    if args.transition_thickness is not None:
        config = replace(config, transition_thickness=args.transition_thickness)
    if args.perturbation_width is not None:
        config = replace(config, perturbation_width=args.perturbation_width)
    if args.kh_mode is not None:
        config = replace(config, perturbation_mode=args.kh_mode)
    if args.kh_amplitude is not None:
        config = replace(config, perturbation_amplitude=args.kh_amplitude)
    if args.secondary_mode is not None:
        config = replace(config, subharmonic_mode=args.secondary_mode)
    if args.secondary_amplitude is not None:
        config = replace(config, subharmonic_amplitude=args.secondary_amplitude)

    print(
        f"Running {config.nx}x{config.ny} periodic MAC DNS: "
        f"Re={config.reynolds:g}, Pe={config.effective_peclet:g}, "
        f"nu={config.viscosity:.6g}, kappa={config.scalar_diffusivity:.6g}, "
        f"t_end={config.t_end:g}"
    )
    snapshots = run_simulation(config, progress=_progress)

    figure_path = args.output_dir / "mixing_layer_mac_vorticity.png"
    plot_snapshots(snapshots, config, figure_path)
    print(f"saved figure: {figure_path}")

    if not args.no_data:
        data_path = args.output_dir / "mixing_layer_mac_snapshots.npz"
        save_snapshots(snapshots, config, data_path)
        print(f"saved data:   {data_path}")


if __name__ == "__main__":
    main()
