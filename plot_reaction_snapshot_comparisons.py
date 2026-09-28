#!/usr/bin/env python3
"""Per-snapshot y profiles and DNS/MF/signed-difference field figures.

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


def snapshot_name(index, step):
    return f"snapshot_{index:02d}_step_{step:04d}"


def save_figure(fig, base):
    base.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(base.with_suffix("."+extension), dpi=180)
    plt.close(fig)


def profile_figure(rows, quantity, time, limits):
    """One x-averaged profile at fixed time, with all three Da comparisons."""
    if quantity not in PROFILE_LABELS:
        raise ValueError("only shear_stress and unmixedness profiles are requested")
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    for index, da in enumerate(DAMKOHLERS):
        for method in ("DNS", "MF"):
            selected = sorted((r for r in rows if r["Da"] == da and r["method"] == method), key=lambda r: r["y"])
            if not selected or any(r["time"] != time for r in selected):
                plt.close(fig)
                raise ValueError("all six profiles at exactly one snapshot time are required")
            ax.plot([r["y"] for r in selected], [r[quantity] for r in selected],
                    color=COLORS[da], linestyle=STYLES[method], linewidth=1.9,
                    marker=MARKERS[da], markersize=4.8,
                    markevery=list(range(4*index, len(selected), 12)),
                    markerfacecolor=COLORS[da] if method == "DNS" else "white",
                    label=f"Da = {da}, {method}")
    title = "Reynolds shear stress" if quantity == "shear_stress" else "Unmixedness (signed covariance)"
    ax.set(xlabel="y", ylabel=PROFILE_LABELS[quantity], xlim=(0, 1), ylim=limits,
           title=f"{title} vs y\n64×64, Re = Pe = 100, RK4; t = {time:.6f}")
    ax.grid(alpha=.23)
    ax.legend(ncol=3, fontsize=9, loc="upper center", bbox_to_anchor=(.5, -.18), frameon=False)
    fig.text(.5, .015, "Solid DNS; dashed MF. Streamwise means at fixed y; no y or time averaging."
             +( " Da shear-stress curves overlap." if quantity == "shear_stress" else " Covariance is not sign-reversed."),
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .055, 1, 1))
    return fig


def field_figure(data, config, field, index, bounds, error_limit):
    """DNS, MF, and MF-DNS at one time; matched physical color scales."""
    if field not in FIELDS:
        raise ValueError("expected c1, c2, or vorticity")
    dns, mf = data["DNS"][field][index], data["MF"][field][index]
    if dns.shape != mf.shape or not np.isfinite(dns).all() or not np.isfinite(mf).all():
        raise ValueError("finite, matching field arrays required")
    difference = mf-dns
    relative_l2 = float(np.linalg.norm(difference)/max(np.linalg.norm(dns), 1e-30))
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 5.0), layout="constrained")
    artists = []
    for i, (ax, values, title) in enumerate(zip(axes, (dns, mf, difference), ("DNS", "MF", "MF − DNS"))):
        x, y, array = field_coordinates(values, config, field)
        artist = ax.pcolormesh(x, y, array, shading="nearest", rasterized=True,
                              cmap="RdBu_r" if field == "vorticity" or i == 2 else "viridis",
                              vmin=bounds[0] if i < 2 else -error_limit,
                              vmax=bounds[1] if i < 2 else error_limit)
        ax.set(xlabel="x", ylabel="y", xlim=(0, 1), ylim=(0, 1), aspect="equal",
               title=title if i < 2 else f"{title}\nRelative L2 = {100*relative_l2:.3g}%")
        artists.append(artist)
    label = "Vorticity" if field == "vorticity" else field.upper()
    fig.colorbar(artists[0], ax=axes[:2], orientation="horizontal", shrink=.8, pad=.08).set_label(label)
    fig.colorbar(artists[2], ax=axes[2], orientation="horizontal", shrink=.9, pad=.08).set_label(f"Signed {label} difference")
    time = data["DNS"]["times"][index]
    fig.suptitle(f"{label}: DNS / MF / signed difference — Da = {config.damkohler:g}\n"
                 f"{config.n}×{config.n}, Re = Pe = 100, RK4; t = {time:.6f}")
    return fig, relative_l2


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
    profile_links = []
    for k, time in enumerate(times):
        step = round(time/config.dt)
        name = snapshot_name(k, step)
        selected = [row for row in profiles if row["snapshot"] == k]
        for quantity in PROFILE_LABELS:
            base = output_dir/"profiles"/quantity/name
            save_figure(profile_figure(selected, quantity, time, limits[quantity]), base)
            write_csv(base.with_suffix(".csv"), [
                {key: value for key, value in row.items() if key not in PROFILE_LABELS or key == quantity}
                for row in selected])
            files.append(dict(kind=quantity, snapshot=k, step=step, time=float(time),
                              base=str(base.relative_to(output_dir))))
        profile_links.append(f"| {step} | {time:.6f} | [PNG](./profiles/shear_stress/{name}.png) / "
                             f"[PDF](./profiles/shear_stress/{name}.pdf) | [PNG](./profiles/unmixedness/{name}.png) / "
                             f"[PDF](./profiles/unmixedness/{name}.pdf) |")
    print("Saved eight shear-stress and eight unmixedness profile figures (PNG/PDF/CSV).", flush=True)
    field_sections = []
    for da, (config, data) in cases.items():
        references = comparison(data["MF"], data["DNS"])
        table = []
        for k, time in enumerate(times):
            step = round(time/config.dt)
            name = snapshot_name(k, step)
            links = []
            for field in FIELDS:
                bounds, error = color_limits(cases, field, k)
                fig, relative_l2 = field_figure(data, config, field, k, bounds, error)
                if not np.isclose(relative_l2, references[k][field+"_relative_l2"], rtol=1e-13, atol=1e-14):
                    raise ValueError("field comparison metric mismatch")
                base = output_dir/"fields"/f"da{da}"/field/name
                save_figure(fig, base)
                relative = str(base.relative_to(output_dir))
                files.append(dict(kind=field, Da=da, snapshot=k, step=step, time=float(time), base=relative))
                metrics.append(dict(Da=da, snapshot=k, step=step, time=float(time), field=field,
                                    relative_l2=relative_l2, max_abs_difference=references[k][field+"_max_abs"],
                                    physical_vmin=bounds[0], physical_vmax=bounds[1], difference_limit=error))
                links.append(f"[PNG](./{relative}.png) / [PDF](./{relative}.pdf)")
            table.append(f"| {step} | {time:.6f} | "+" | ".join(links)+" |")
        field_sections.append(f"### Da = {da}\n\n| Step | Time | C1 | C2 | Vorticity |\n"
                              "|---|---|---|---|---|\n"+"\n".join(table))
        print(f"Saved Da={da}: 24 individual DNS/MF/difference field figures (PNG/PDF).", flush=True)
    write_csv(output_dir/"field_comparison_metrics.csv", metrics)
    manifest = dict(sources=sources, times=times.tolist(), files=files,
                    profile_limits=limits, line_styles=STYLES, difference="MF - DNS",
                    averaging="x mean at fixed y and time; no y/time averaging",
                    scalar_clipping=False, simulation_rerun=False,
                    field_color_scales="Shared across Da and methods; vorticity per time, all other scales across time",
                    postprocessor_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
                                          (Path(__file__).name, "plot_reaction_statistics.py", "plot_mean_field_results.py")})
    (output_dir/"manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    report = r"""# Per-snapshot shear stress, unmixedness, and field comparisons

64×64, Re=Pe=100, RK4, Da=1,10,100. All **eight saved field snapshots** are
included, from t=0 to t=0.65. There are not full-field data for every one of
the 1,040 integration steps; no missing fields have been interpolated or rerun.
The earlier time-history/overview plots are preserved.

## Shear stress and unmixedness versus y

One separate file per quantity per snapshot; **all three Da in every plot,
solid DNS / dashed MF**. Only the shear Reynolds stress is plotted here,
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

Each quantity uses the same vertical-axis limits in all eight files, so the
initial zero covariance and roundoff-level shear stress are not magnified.
Every PNG has a matching PDF and a CSV containing all six 64-point profiles.

| Step | Time | Shear stress vs y | Unmixedness vs y |
|---|---|---|---|
PROFILE_TABLE

## C1, C2 and vorticity: DNS, MF, and signed differences

Each linked file has three panels: **DNS | MF | MF − DNS**. These are 72
individual comparisons: three fields × three Da × eight snapshots. The
difference sign is identical everywhere. Coordinates retain the native MAC
layout, including vertex-centered vorticity and the periodic endpoint.
No smoothing is applied. Physical color limits match DNS/MF and all Da.
Concentration limits are fixed across all snapshots and include all recorded
extrema. Vorticity limits adapt per snapshot to display decaying vortices
without saturating the initial extrema. Each field's symmetric difference
scale is fixed across all times and Da. Color limits are exported in the CSV.

FIELD_TABLES

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
    (output_dir/"README.md").write_text(report.replace("PROFILE_TABLE", "\n".join(profile_links))
                                      .replace("FIELD_TABLES", "\n\n".join(field_sections)))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/reaction_64x64_re100_pe100_rk4_series/snapshot_comparisons"))
    args = parser.parse_args()
    result = generate(args.input_root, args.output_dir)
    print(f"Complete: {len(result['files'])} figures, each in PNG and PDF; {args.output_dir/'README.md'}", flush=True)


if __name__ == "__main__":
    main()
