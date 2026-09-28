#!/usr/bin/env python3
"""Snapshot-only Reynolds-stress, thickness and covariance comparisons.

This program measures existing MF/DNS output; it never advances either solver.
MF fields are validated against their stored single-site bosonic states first.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_mean_field_dns import diagnostics as dns_diagnostics, dns_config
from mixing_layer_mean_field import MeanFieldConfig, validate_results
from plot_mean_field_results import load_fields, verify_dns_setup, identical_initial_fields


DAMKOHLERS = (1, 10, 100)
COLORS = {1: "#0072B2", 10: "#D55E00", 100: "#009E73"}
MARKERS = {1: "o", 10: "s", 100: "^"}
STYLES = {"DNS": "-", "MF": "--"}
METRICS = ("R11", "R22", "R12", "vorticity_thickness",
           "covariance_global", "covariance_streamwise")
THICKNESS_SOURCE = (
    "https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/"
    "variabledensity-effects-in-incompressible-nonbuoyant-sheardriven-turbulent-"
    "mixing-layers/250CE5A774864C97D4C73FB0742C2797"
)


def snapshot_statistics(u, v, c1, c2, dy):
    """Unit-density channel statistics, with periodic x and free-slip y.

    Reynolds fluctuations are relative to the x mean at each y, *not* the
    domain mean shear. Both velocities are interpolated to scalar cell centers
    before all three velocity moments are taken. There is no time averaging.
    """
    u, v, c1, c2 = (np.asarray(a, dtype=float) for a in (u, v, c1, c2))
    if (u.ndim != 2 or min(u.shape) < 2 or v.shape != (u.shape[0]+1, u.shape[1])
            or c1.shape != u.shape or c2.shape != u.shape):
        raise ValueError("expected channel MAC u=(ny,nx), v=(ny+1,nx), c1/c2=(ny,nx)")
    if not np.isfinite(dy) or dy <= 0 or not all(np.isfinite(a).all() for a in (u, v, c1, c2)):
        raise ValueError("finite fields and positive dy required")
    uc = .5*(u+np.roll(u, -1, axis=1))
    vc = .5*(v[:-1]+v[1:])
    mean_u, mean_v = uc.mean(axis=1), vc.mean(axis=1)
    up, vp = uc-mean_u[:, None], vc-mean_v[:, None]
    mean_c1, mean_c2 = c1.mean(axis=1), c2.mean(axis=1)
    profiles = {
        "y": (np.arange(u.shape[0])+.5)*dy,
        "mean_u": mean_u, "mean_v": mean_v,
        "R11": np.mean(up*up, axis=1), "R22": np.mean(vp*vp, axis=1),
        "R12": np.mean(up*vp, axis=1),
        "mean_c1": mean_c1, "mean_c2": mean_c2,
        "covariance_streamwise": np.mean((c1-mean_c1[:, None])*(c2-mean_c2[:, None]), axis=1),
    }
    # Two neighboring cell-center means give the second-order derivative at
    # their intervening y face. Zero wall gradients implement free-slip y.
    gradient = np.r_[0., np.diff(mean_u)/dy, 0.]
    peak_gradient = float(np.max(np.abs(gradient)))
    velocity_jump = float(abs(mean_u[-1]-mean_u[0]))
    if peak_gradient == 0 or velocity_jump == 0:
        raise ValueError("vorticity thickness undefined without a mean shear/velocity jump")
    global_covariance = float(np.mean((c1-c1.mean())*(c2-c2.mean())))
    between_y_covariance = float(np.mean((mean_c1-mean_c1.mean())*(mean_c2-mean_c2.mean())))
    scalar = {name: float(profiles[name].mean()) for name in ("R11", "R22", "R12", "covariance_streamwise")}
    scalar.update(
        vorticity_thickness=velocity_jump/peak_gradient,
        velocity_jump=velocity_jump, maximum_mean_shear=peak_gradient,
        covariance_global=global_covariance, covariance_between_y=between_y_covariance,
        mean_c1=float(c1.mean()), mean_c2=float(c2.mean()),
    )
    if not np.isclose(global_covariance, scalar["covariance_streamwise"]+between_y_covariance,
                      rtol=1e-12, atol=1e-14):
        raise ValueError("covariance decomposition failed")
    # Observational only: neither negative concentrations nor covariances clip.
    return scalar, profiles


def load_case(folder, damkohler):
    """Fail closed for incomplete, mismatched or corrupted snapshot series."""
    mf_path, dns_path = folder/"mean_field_snapshots.npz", folder/"dns_matched_snapshots.npz"
    validation = validate_results(mf_path)
    config = MeanFieldConfig(**validation["config"])
    if (config.n != 64 or config.reynolds != 100 or config.peclet != 100
            or config.scalar_species != 3 or config.damkohler != damkohler
            or config.time_integrator != "rk4" or config.boundary_y != "free-slip"):
        raise ValueError("expected the 64x64 Re=Pe=100 reacting RK4 case")
    primary, reference = load_fields(mf_path), load_fields(dns_path)
    verify_dns_setup(config, reference)
    if not identical_initial_fields(primary, reference):
        raise ValueError("MF and DNS initial conditions do not match")
    if not np.array_equal(primary["times"], reference["times"]):
        raise ValueError("MF and DNS snapshot times do not match")
    expected_shape = (8, config.n, config.n)
    for values in (primary, reference):
        for name in ("u", "c1", "c2", "c3"):
            if values[name].shape != expected_shape or not np.isfinite(values[name]).all():
                raise ValueError(f"invalid {name} snapshot fields")
        if values["v"].shape != (8, config.n+1, config.n) or not np.isfinite(values["v"]).all():
            raise ValueError("invalid v snapshot fields")
        if values["run_metadata"].get("scalar_clipping") is not False:
            raise ValueError("unclipped source metadata required")
    meta = reference["run_metadata"]
    if meta.get("time_integrator") != "rk4" or meta.get("mean_field_fingerprint") != validation["fingerprint"]:
        raise ValueError("DNS method/source mismatch")
    recorded = reference["validation"]
    if recorded.get("validated") is not True or recorded.get("completed_steps") != config.steps:
        raise ValueError("incomplete DNS validation")
    history = reference["history"]
    if [row["step"] for row in history] != list(range(1, config.steps+1)):
        raise ValueError("incomplete DNS diagnostic history")
    physical = dns_config(config)
    with np.load(dns_path, allow_pickle=False) as saved:
        for k, time in enumerate(reference["times"]):
            scalars = np.stack([reference[name][k] for name in ("c1", "c2", "c3")])
            checks = dns_diagnostics(reference["u"][k], reference["v"][k], scalars,
                                     saved["pressure"][k], physical, np.array([.5, .5]), damkohler)
            record = reference["initial_diagnostics"] if k == 0 else history[round(time/config.dt)-1]
            for key, value in checks.items():
                if not np.isclose(value, record[key], rtol=1e-12, atol=1e-13):
                    raise ValueError(f"DNS snapshot/diagnostic mismatch: {key}")
    provenance = {
        "damkohler": damkohler, "config": asdict(config), "mf_fingerprint": validation["fingerprint"],
        "files": {label: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                  for label, path in (("MF", mf_path), ("DNS", dns_path))},
        "mf_validation_recomputed": True, "dns_snapshot_diagnostics_recomputed": True,
    }
    return config, {"DNS": reference, "MF": primary}, provenance


def draw_metric(ax, rows, key):
    """Six correctly styled lines per panel; never offset coincident curves."""
    for index, da in enumerate(DAMKOHLERS):
        for method in ("DNS", "MF"):
            series = [row for row in rows if row["Da"] == da and row["method"] == method]
            ax.plot([row["time"] for row in series], [row[key] for row in series],
                    color=COLORS[da], linestyle=STYLES[method], linewidth=1.8,
                    marker=MARKERS[da], markersize=6, markevery=list(range(index, len(series), 3)),
                    markerfacecolor=COLORS[da] if method == "DNS" else "white",
                    markeredgewidth=1.2, label=f"Da = {da}, {method}")
    ax.set_xlabel("Time t")
    ax.grid(alpha=.23)
    ax.legend(ncol=2, fontsize=8, framealpha=.9)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)


def make_plots(rows, output_dir):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white", "pdf.fonttype": 42})
    plots = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    for ax, key, title, label in zip(axes, ("R11", "R22", "R12"),
            ("Streamwise normal stress", "Transverse normal stress", "Shear stress (signed)"),
            (r"$\langle u'^2\rangle_{xy}$", r"$\langle v'^2\rangle_{xy}$", r"$\langle u'v'\rangle_{xy}$")):
        draw_metric(ax, rows, key)
        ax.set(title=title, ylabel=label)
        ax.legend(ncol=2, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.23), frameon=False)
    fig.suptitle("Reynolds stresses: 64×64, Re = Pe = 100, RK4", y=.985)
    fig.text(.5, .015, "Fluctuations about the streamwise mean at each y; averaged over y.  Da curves overlap within each method.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .06, 1, .95))
    plots.append((fig, "reynolds_stresses"))

    fig, ax = plt.subplots(figsize=(8, 5.4))
    draw_metric(ax, rows, "vorticity_thickness")
    ax.set(title="Vorticity thickness: 64×64, Re = Pe = 100, RK4",
           ylabel=r"$\delta_\omega = \Delta U(t)\,/\,\max_y |\partial_y\overline{u}|$")
    fig.text(.5, .025, "Eight saved times; no temporal smoothing. Da curves overlap within each method.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, 1))
    plots.append((fig, "vorticity_thickness"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    draw_metric(axes[0], rows, "covariance_global")
    axes[0].set(title="Full-domain means and covariance",
                ylabel=r"$\langle(c_1-\langle c_1\rangle_{xy})(c_2-\langle c_2\rangle_{xy})\rangle_{xy}$")
    draw_metric(axes[1], rows, "covariance_streamwise")
    axes[1].set(title="Streamwise fluctuations, averaged over y",
                ylabel=r"$\langle\langle(c_1-\overline{c}_1)(c_2-\overline{c}_2)\rangle_x\rangle_y$")
    fig.suptitle("Unmixedness: signed c₁–c₂ covariance, 64×64, Re = Pe = 100", y=.985)
    fig.text(.5, .015, "Solid: DNS. Dashed: MF. Negative values retained; no absolute value or normalization. Lines connect stored snapshots.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .95))
    plots.append((fig, "unmixedness"))
    for fig, name in plots:
        for suffix in ("png", "pdf"):
            fig.savefig(output_dir/f"{name}.{suffix}", dpi=200)
        plt.close(fig)


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def generate(input_root, output_dir):
    rows, profiles, sources = [], [], []
    baseline_config, baseline_times = None, None
    for da in DAMKOHLERS:
        folder = input_root/f"mean_field_sites_64x64_re100_pe100_da{da}_reaction_rk4_ptol1e7"
        config, data, source = load_case(folder, da)
        same_config = {key: value for key, value in asdict(config).items() if key != "damkohler"}
        if baseline_config is None:
            baseline_config, baseline_times = same_config, data["MF"]["times"]
        if same_config != baseline_config or not np.array_equal(baseline_times, data["MF"]["times"]):
            raise ValueError("the Damkohler series must differ only in Da")
        sources.append(source)
        for method, fields in data.items():
            for k, time in enumerate(fields["times"]):
                scalar, profile = snapshot_statistics(*(fields[name][k] for name in ("u", "v", "c1", "c2")), 1/config.n)
                prefix = dict(Da=da, method=method, snapshot=k, time=float(time))
                rows.append(dict(**prefix, **scalar))
                profiles.extend(dict(**prefix, **{key: float(value[j]) for key, value in profile.items()})
                                for j in range(config.n))
        print(f"Validated and measured Da={da}: MF/DNS, eight snapshots each.", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir/"statistics.csv", rows)
    write_csv(output_dir/"profiles.csv", profiles)
    make_plots(rows, output_dir)
    errors = {}
    for da in DAMKOHLERS:
        dns = [row for row in rows if row["Da"] == da and row["method"] == "DNS"]
        mf = [row for row in rows if row["Da"] == da and row["method"] == "MF"]
        errors[str(da)] = {key: {
            "final_DNS": dns[-1][key], "final_MF": mf[-1][key],
            "final_relative_difference_percent": 100*abs(mf[-1][key]-dns[-1][key])/max(abs(dns[-1][key]), 1e-30),
            "maximum_snapshot_absolute_difference": max(abs(m[key]-d[key]) for m, d in zip(mf, dns)),
        } for key in METRICS}
    overlap = {}
    for method in STYLES:
        overlap[method] = {key: max(np.ptp([row[key] for row in rows if row["method"] == method and row["snapshot"] == k])
                                   for k in range(8)) for key in ("R11", "R22", "R12", "vorticity_thickness")}
    summary = {"sources": sources, "times": baseline_times.tolist(), "snapshot_count_per_method_case": 8,
               "line_styles": STYLES, "final_comparisons": errors, "maximum_spread_across_Da": overlap,
               "thickness_definition_source": THICKNESS_SOURCE,
               "postprocessor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "scalar_clipping": False, "simulation_rerun": False}
    (output_dir/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    final_table = "\n".join(
        f"| {row['Da']} | {row['method']} | "+" | ".join(f"{row[key]:.8g}" for key in METRICS)+" |"
        for row in rows if row["snapshot"] == 7)
    error_table = "\n".join(
        f"| {da} | "+" | ".join(f"{errors[str(da)][key]['final_relative_difference_percent']:.3f}%"
                                for key in METRICS)+" |" for da in DAMKOHLERS)
    report = r"""# Reacting mixing-layer statistics: MF versus DNS

64×64, Re=Pe=100, RK4, Da=1,10,100, t=0…0.65. Postprocessing only;
no trajectory was changed or rerun. All MF results were revalidated against
the stored explicit local Fock kets; DNS snapshot diagnostics, configurations,
initial fields and output times were checked. Input SHA-256 hashes are in
[summary.json](./summary.json).

## Three figures

1. [Reynolds stresses](./reynolds_stresses.png) ([PDF](./reynolds_stresses.pdf)): R11, R22 and signed R12 versus time.
2. [Vorticity thickness](./vorticity_thickness.png) ([PDF](./vorticity_thickness.pdf)): delta_omega versus time.
3. [Unmixedness](./unmixedness.png) ([PDF](./unmixedness.pdf)): signed c1/c2 covariance using both explicit averaging conventions.

Every panel has Da=1,10,100; **solid DNS, dashed MF**. Colors/marker shapes
identify Da. Marker locations are staggered *among existing saved times* to
show coincident curves; no data, coordinates or curves are offset. Only eight
field snapshots were saved per trajectory. Straight connections are visual
guides, not additional measurements or temporal interpolation claims.

## Definitions and discrete evaluation

Let overbar mean the periodic-x spatial mean at fixed y and time. Angle
brackets with xy denote the full-domain mean. These are spatial statistics
of a deterministic 2D realization, not time/ensemble statistics or quantum
connected correlations. The MF observables are measured physical fields.

MAC velocities are collocated at scalar cell centers before all stress products:

$$u^c_{j,i}=(u_{j,i}+u_{j,i+1})/2,\qquad
v^c_{j,i}=(v_{j,i}+v_{j+1,i})/2.$$

$$u'=u^c-\overline{u^c}(y,t),\quad v'=v^c-\overline{v^c}(y,t),\qquad
(R_{11},R_{22},R_{12})(t)=\langle(u'^2,v'^2,u'v')\rangle_{xy}.$$

These are kinematic Reynolds stresses (no density factor, no minus sign on
R12). Subtracting the local x mean prevents the mean shear itself from being
counted as a fluctuation. No extra velocity-jump normalization is applied;
the fields already use the simulation's nondimensional reference velocity.

$$\delta_\omega(t)=\frac{\Delta U(t)}{\max_y|\partial_y\bar u|},\qquad
\Delta U(t)=|\bar u(y_{N-1},t)-\bar u(y_0,t)|.$$

The outermost cell-center means estimate the two external velocities. The
derivative at each interior y face is (bar_u[j]-bar_u[j-1])/dy, with zero wall
derivatives for free slip; no smoothing, fitting or spectral differentiation.
This is thickness of the *mean shear*, not inverse peak instantaneous
vorticity. Finite-grid initial thickness need not equal the continuum 2*delta
for a tanh profile. The conventional gradient-based definition is described
in [Baltzer & Livescu, JFM, equation (4.13)](THICKNESS_SOURCE).

The user's signed unmixedness is shown with both possible meanings of the brackets:

$$C_{12}^{xy}=\langle(c_1-\langle c_1\rangle_{xy})(c_2-\langle c_2\rangle_{xy})\rangle_{xy},$$

$$C_{12}^{x|y}=\langle\overline{(c_1-\bar c_1(y,t))(c_2-\bar c_2(y,t))}\rangle_y.$$

The first includes stratification between different y positions; the second
isolates within-x-plane fluctuations before averaging over y. Both use raw,
independently evolved c1 and c2; no clipping, absolute value, sign reversal or
division by mean concentrations is applied. Negative covariance represents
anticorrelation, not by itself a negative-concentration artifact. At t=0,
c2=1-c1: the global covariance is minus Var_xy(c1), whereas the streamwise
covariance is zero. At every snapshot the code checks the total-covariance identity

$$C_{12}^{xy}=C_{12}^{x|y}+\langle(\bar c_1-\langle c_1\rangle_{xy})
(\bar c_2-\langle c_2\rangle_{xy})\rangle_y.$$

Reaction is passive to momentum, so all three Da velocity-statistic curves
coincide within each method up to numerical roundoff. This is expected, not
a missing case. Scalar covariances do depend on Da. Agreement with DNS is
not a grid/timestep-convergence or positivity certificate.

## Final-time values

| Da | Method | R11 | R22 | R12 | Vorticity thickness | Global covariance | Streamwise covariance |
|---|---|---|---|---|---|---|---|
FINAL_TABLE

Relative differences below are abs(MF-DNS)/abs(DNS) at the final time, not
errors against an exact solution. The shear stress changes sign over the
trajectory, so a relative error near a zero crossing would be misleading;
the exported summary also gives maximum absolute differences at saved times.

| Da | R11 difference | R22 difference | R12 difference | Thickness difference | Global covariance difference | Streamwise covariance difference |
|---|---|---|---|---|---|---|
ERROR_TABLE

## Reusable data and reproduction

- [statistics.csv](./statistics.csv): 48 rows (three Da × two methods × eight times), all unrounded values.
- [profiles.csv](./profiles.csv): 3,072 rows, including mean fields, R11(y), R22(y), R12(y), and streamwise scalar covariance(y) at every snapshot.
- [summary.json](./summary.json): provenance, MF–DNS differences, and cross-Da flow-statistic spread.

From the repository root, run `python3 plot_reaction_statistics.py` using the
same NumPy/SciPy/Matplotlib environment as the existing postprocessing.
"""
    (output_dir/"README.md").write_text(report.replace("THICKNESS_SOURCE", THICKNESS_SOURCE)
                                      .replace("FINAL_TABLE", final_table).replace("ERROR_TABLE", error_table))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/reaction_64x64_re100_pe100_rk4_series/statistics"))
    args = parser.parse_args()
    summary = generate(args.input_root, args.output_dir)
    print(json.dumps(summary["final_comparisons"], indent=2))


if __name__ == "__main__":
    main()
