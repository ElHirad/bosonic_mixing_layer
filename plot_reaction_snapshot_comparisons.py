#!/usr/bin/env python3
"""Five combined, all-snapshot y-profile and DNS/MF/difference figures.

Only existing snapshots are measured. Neither simulation is advanced.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mixing_layer_mean_field import vorticity
from plot_mean_field_results import comparison, field_coordinates
from plot_reaction_statistics import (
    COLORS, DAMKOHLERS, MARKERS, STYLES, load_case, snapshot_statistics, write_csv,
)


FIELDS = ("c1", "c2", "vorticity")
PROFILE_LABELS = {
    "shear_stress": r"$\langle(u-\langle u\rangle_x)(v-\langle v\rangle_x)\rangle_x$",
    "unmixedness": r"$\langle(c_1-\langle c_1\rangle_x)(c_2-\langle c_2\rangle_x)\rangle_x$",
}


def save_figure(fig, base):
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=180)
    plt.close(fig)


def profile_figure(rows, quantity, times, limits):
    """All saved times in one figure; all three Da comparisons in each panel."""
    if quantity not in PROFILE_LABELS:
        raise ValueError("only shear_stress and unmixedness profiles are requested")
    columns = min(4, len(times))
    panel_rows = (len(times)+columns-1)//columns
    fig, axes = plt.subplots(panel_rows, columns, figsize=(4.3*columns, 3.8*panel_rows+1.2),
                             sharex=True, sharey=True, squeeze=False)
    for k, time in enumerate(times):
        ax = axes.flat[k]
        for index, da in enumerate(DAMKOHLERS):
            for method in ("DNS", "MF"):
                selected = sorted((r for r in rows if r["Da"] == da and r["method"] == method
                                   and r["time"] == time), key=lambda r: r["y"])
                if not selected:
                    plt.close(fig)
                    raise ValueError("all six profiles at every snapshot time are required")
                ax.plot([r["y"] for r in selected], [r[quantity] for r in selected],
                        color=COLORS[da], linestyle=STYLES[method], linewidth=1.8,
                        marker=MARKERS[da], markersize=3.8,
                        markevery=list(range(4*index, len(selected), 12)),
                        markerfacecolor=COLORS[da] if method == "DNS" else "white",
                        label=f"Da = {da}, {method}")
        ax.set(xlabel="y", xlim=(0, 1), ylim=limits, title=f"t = {time:.6f}")
        if k % columns == 0:
            ax.set_ylabel(PROFILE_LABELS[quantity], fontsize=10)
        ax.grid(alpha=.23)
    for ax in list(axes.flat)[len(times):]:
        ax.set_visible(False)
    title = "Reynolds shear stress" if quantity == "shear_stress" else "Unmixedness (signed covariance)"
    fig.suptitle(f"{title} vs y — all saved snapshots\n64×64, Re = Pe = 100, RK4", fontsize=15)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, fontsize=10, loc="lower center", bbox_to_anchor=(.5, .038), frameon=False)
    fig.text(.5, .015, "Solid DNS; dashed MF. Streamwise means at fixed y; no y or time averaging."
             +( " Da shear-stress curves overlap." if quantity == "shear_stress" else " Covariance is not sign-reversed."),
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .13, 1, .91))
    return fig


def field_figure(cases, field):
    """All times, three Da, DNS/MF/difference: one 9-row comparison sheet."""
    if field not in FIELDS:
        raise ValueError("expected c1, c2, or vorticity")
    times = cases[DAMKOHLERS[0]][1]["DNS"]["times"]
    fig, axes = plt.subplots(9, len(times), figsize=(2.65*len(times)+1.4, 22.5),
                             sharex=True, sharey=True, squeeze=False)
    fig.subplots_adjust(left=.065, right=.985, top=.948,
                        bottom=.14 if field == "vorticity" else .085, wspace=.13, hspace=.22)
    metrics, physical_artists = [], {}
    for case_index, da in enumerate(DAMKOHLERS):
        config, data = cases[da]
        if not np.array_equal(times, data["DNS"]["times"]) or not np.array_equal(times, data["MF"]["times"]):
            plt.close(fig)
            raise ValueError("all cases and methods must share snapshot times")
        references = comparison(data["MF"], data["DNS"])
        for k, time in enumerate(times):
            bounds, error_limit = color_limits(cases, field, k)
            dns, mf = data["DNS"][field][k], data["MF"][field][k]
            difference = mf-dns
            relative_l2 = float(np.linalg.norm(difference)/max(np.linalg.norm(dns), 1e-30))
            if not np.isclose(relative_l2, references[k][field+"_relative_l2"], rtol=1e-13, atol=1e-14):
                plt.close(fig)
                raise ValueError("field comparison metric mismatch")
            metrics.append(dict(Da=da, snapshot=k, step=round(time/config.dt), time=float(time), field=field,
                                relative_l2=relative_l2, max_abs_difference=float(np.max(np.abs(difference))),
                                physical_vmin=bounds[0], physical_vmax=bounds[1], difference_limit=error_limit))
            for offset, (values, method) in enumerate(zip((dns, mf, difference), ("DNS", "MF", "MF − DNS"))):
                row = 3*case_index+offset
                ax = axes[row, k]
                x, y, array = field_coordinates(values, config, field)
                artist = ax.pcolormesh(x, y, array, shading="nearest", rasterized=True,
                                      cmap="RdBu_r" if field == "vorticity" or offset == 2 else "viridis",
                                      vmin=bounds[0] if offset < 2 else -error_limit,
                                      vmax=bounds[1] if offset < 2 else error_limit)
                ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xticks=[0, .5, 1], yticks=[0, .5, 1])
                ax.tick_params(labelsize=7)
                if k == 0:
                    ax.set_ylabel(f"Da = {da}\n{method}\ny", fontsize=10)
                if row == 0:
                    ax.set_title(f"t = {time:.6f}", fontsize=10)
                if row == 8:
                    ax.set_xlabel("x", fontsize=9)
                if offset == 2:
                    error_artist = artist
                    ax.text(.03, .04, f"L2: {100*relative_l2:.3g}%", transform=ax.transAxes,
                            fontsize=8, bbox=dict(facecolor="white", alpha=.8, edgecolor="none", pad=1))
                else:
                    physical_artists[k] = artist
    label = "Vorticity" if field == "vorticity" else field.upper()
    if field == "vorticity":
        for k, time in enumerate(times):
            pos = axes[-1, k].get_position()
            cax = fig.add_axes([pos.x0, .077, pos.width, .009])
            limit = physical_artists[k].get_clim()[1]
            bar = fig.colorbar(physical_artists[k], cax=cax, orientation="horizontal", ticks=[-limit, 0, limit])
            bar.ax.set_xticklabels([f"{-limit:.1f}", "0", f"{limit:.1f}"])
            bar.ax.tick_params(labelsize=7)
            bar.set_label(f"Vorticity at t={time:.6f}", fontsize=8)
        error_axes = fig.add_axes([.32, .026, .36, .009])
    else:
        value_axes = fig.add_axes([.12, .033, .32, .009])
        fig.colorbar(physical_artists[0], cax=value_axes, orientation="horizontal").set_label(label)
        error_axes = fig.add_axes([.60, .033, .32, .009])
    fig.colorbar(error_artist, cax=error_axes, orientation="horizontal").set_label(f"Signed {label} difference (MF − DNS)")
    fig.suptitle(f"{label}: all eight saved snapshots, all three Damkohler cases\n"
                 "64×64, Re = Pe = 100, RK4; DNS / MF / signed MF − DNS in each three-row block", fontsize=16)
    return fig, metrics


def color_limits(cases, field, index):
    """No color clipping: shared scales across Da and methods.

    Concentration scales and each field's signed-difference scale are fixed
    across all times. Vorticity physical scales adapt per time to show the
    decaying vortices without saturating the initial extrema.
    """
    arrays = [data[method][field] for _, data in cases.values() for method in ("DNS", "MF")]
    if field == "vorticity":
        maximum = max(1e-15, max(float(np.max(np.abs(a[index]))) for a in arrays))
        bounds = (-maximum, maximum)
    else:
        bounds = (min(0., min(float(a.min()) for a in arrays)),
                  max(1., max(float(a.max()) for a in arrays)))
    error = max(1e-15, max(float(np.max(np.abs(data["MF"][field]-data["DNS"][field])))
                          for _, data in cases.values()))
    return bounds, error


def generate(input_root, output_dir):
    cases, sources, profiles = {}, [], []
    common, times = None, None
    for da in DAMKOHLERS:
        folder = input_root/f"mean_field_sites_64x64_re100_pe100_da{da}_reaction_rk4_ptol1e7"
        config, data, source = load_case(folder, da)
        physical = {key: value for key, value in asdict(config).items() if key != "damkohler"}
        if common is None:
            common, times = physical, data["DNS"]["times"]
        if physical != common or not np.array_equal(data["DNS"]["times"], times):
            raise ValueError("the three cases must share parameters and snapshot times except Da")
        for method, values in data.items():
            for k, time in enumerate(times):
                if not np.allclose(values["vorticity"][k], vorticity(values["u"][k], values["v"][k], config),
                                   rtol=0, atol=1e-12):
                    raise ValueError("saved vorticity does not match the measured velocities")
                _, profile = snapshot_statistics(*(values[key][k] for key in ("u", "v", "c1", "c2")), 1/config.n)
                for j, y in enumerate(profile["y"]):
                    profiles.append(dict(Da=da, method=method, snapshot=k, step=round(time/config.dt),
                                         time=float(time), y=float(y), shear_stress=float(profile["R12"][j]),
                                         unmixedness=float(profile["covariance_streamwise"][j])))
        cases[da], sources = (config, data), [*sources, source]
        print(f"Validated Da={da}: local states, DNS diagnostics, field matching, and vorticity.", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir/"profiles_all_snapshots.csv", profiles)
    limits = {}
    for quantity in PROFILE_LABELS:
        low = min(0., min(row[quantity] for row in profiles))
        high = max(0., max(row[quantity] for row in profiles))
        margin = .08*max(high-low, 1e-12)
        limits[quantity] = (low-margin, high+margin)
    files, metrics = [], []
    for quantity in PROFILE_LABELS:
        save_figure(profile_figure(profiles, quantity, times, limits[quantity]), output_dir/quantity)
        files.append(dict(kind=quantity, base=quantity, snapshots=len(times),
                          Damkohler=list(DAMKOHLERS), panels=len(times)))
    print("Saved combined shear-stress and unmixedness figures: eight times each.", flush=True)
    for field in FIELDS:
        fig, field_metrics = field_figure(cases, field)
        save_figure(fig, output_dir/field)
        metrics.extend(field_metrics)
        files.append(dict(kind=field, base=field, snapshots=len(times),
                          Damkohler=list(DAMKOHLERS), panels=9*len(times)))
        print(f"Saved combined {field}: all three Da, all eight times, DNS/MF/difference.", flush=True)
    metrics.sort(key=lambda row: (row["Da"], row["snapshot"], FIELDS.index(row["field"])))
    write_csv(output_dir/"field_comparison_metrics.csv", metrics)
    manifest = dict(sources=sources, times=times.tolist(), files=files, image_format="png",
                    profile_limits=limits, line_styles=STYLES, difference="MF - DNS",
                    averaging="x mean at fixed y and time; no y/time averaging",
                    scalar_clipping=False, simulation_rerun=False,
                    field_color_scales="Shared across Da and methods; vorticity per time, all other scales across time",
                    postprocessor_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
                                          (Path(__file__).name, "plot_reaction_statistics.py", "plot_mean_field_results.py")})
    (output_dir/"manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    report = r"""# Combined all-snapshot comparisons

64×64, Re=Pe=100, RK4, Da=1,10,100. All **eight saved field snapshots** are
included, from t=0 to t=0.65. There are not full-field data for every one of
the 1,040 integration steps; no missing fields have been interpolated or rerun.
There are now **five combined PNG files**, one per quantity. The
snapshot-per-file figures and their per-file profile CSVs
have been removed. Original simulation snapshots and consolidated CSVs are
unchanged; the earlier time-history/overview plots remain available.

| Quantity | All-snapshot figure |
|---|---|
| Reynolds shear stress vs y | [PNG](./shear_stress.png) |
| Unmixedness vs y | [PNG](./unmixedness.png) |
| C1: DNS, MF, MF−DNS | [PNG](./c1.png) |
| C2: DNS, MF, MF−DNS | [PNG](./c2.png) |
| Vorticity: DNS, MF, MF−DNS | [PNG](./vorticity.png) |

## Shear stress and unmixedness versus y

Each profile figure contains eight time panels, with **all three Da in every
panel, solid DNS / dashed MF**. Only the shear Reynolds stress is plotted here,
not the normal components. The horizontal coordinate is y. Angle brackets
mean the spatial average over periodic x at fixed y and time:

$$R_{12}(y,t)=\langle(u-\langle u\rangle_x)(v-\langle v\rangle_x)\rangle_x,$$
$$C_{12}(y,t)=\langle(c_1-\langle c_1\rangle_x)(c_2-\langle c_2\rangle_x)\rangle_x.$$

There is **no y or temporal averaging**, absolute value, sign reversal, or
normalization. Staggered MAC velocities are collocated to scalar cell centers
before stress products, as in the [statistics definitions](../statistics/README.md).
Concentrations are the raw independently evolved species, without clipping.
The three Da shear-stress curves coincide within each method because reaction
does not feed back into momentum. Markers distinguish overlapping cases at
interleaved existing grid locations; no curves or data are offset.

Each quantity uses the same vertical-axis limits in all eight panels, so the
initial zero covariance and roundoff-level shear stress are not magnified.
All six 64-point profiles at every time are in the consolidated profile CSV.

## C1, C2 and vorticity: DNS, MF, and signed differences

Each field has a single combined figure with **eight time columns** and
**nine rows**: DNS, MF, and MF−DNS for Da=1, then Da=10, then Da=100. Thus
each field figure has 72 panels and includes every case/time/method/difference.
Difference panels report relative L2 against DNS. The difference sign is
identical everywhere. Coordinates retain the native MAC
layout, including vertex-centered vorticity and the periodic endpoint.
No smoothing is applied. Physical color limits match DNS/MF and all Da.
Concentration limits are fixed across all snapshots and include all recorded
extrema. Vorticity limits adapt per snapshot to display decaying vortices
without saturating the initial extrema. Each field's symmetric difference
scale is fixed across all times and Da. The vorticity figure has one physical
colorbar under each time column, plus a shared difference colorbar. C1 and C2
use a single physical colorbar and a separate difference colorbar. Color limits
are exported in the CSV. The PNG figures are large so individual panels
remain readable when zoomed.

## Verification and reusable data

MF fields were revalidated against stored local bosonic states; matched DNS
snapshot diagnostics and vorticity from both methods' velocities were checked.
Input hashes, plotting-source hashes, and the complete figure index are in
[manifest.json](./manifest.json). No simulation inputs or results were modified.

- [All y profiles](./profiles_all_snapshots.csv): 3,072 rows (8 times × 3 Da × 2 methods × 64 y points).
- [Field comparison metrics and color limits](./field_comparison_metrics.csv): 72 rows.
- Reproduce with `python3 plot_reaction_snapshot_comparisons.py` from the repo root, using the established plotting environment.

Passing numerical checks is not a grid/timestep-convergence certificate.
"""
    (output_dir/"README.md").write_text(report)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/reaction_64x64_re100_pe100_rk4_series/snapshot_comparisons"))
    args = parser.parse_args()
    result = generate(args.input_root, args.output_dir)
    print(f"Complete: {len(result['files'])} combined PNG figures; {args.output_dir/'README.md'}", flush=True)


if __name__ == "__main__":
    main()
