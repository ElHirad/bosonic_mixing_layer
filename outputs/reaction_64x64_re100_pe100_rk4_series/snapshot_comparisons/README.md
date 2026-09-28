# Per-snapshot shear stress, unmixedness, and field comparisons

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
| 0 | 0.000000 | [PNG](./profiles/shear_stress/snapshot_00_step_0000.png) / [PDF](./profiles/shear_stress/snapshot_00_step_0000.pdf) | [PNG](./profiles/unmixedness/snapshot_00_step_0000.png) / [PDF](./profiles/unmixedness/snapshot_00_step_0000.pdf) |
| 149 | 0.093125 | [PNG](./profiles/shear_stress/snapshot_01_step_0149.png) / [PDF](./profiles/shear_stress/snapshot_01_step_0149.pdf) | [PNG](./profiles/unmixedness/snapshot_01_step_0149.png) / [PDF](./profiles/unmixedness/snapshot_01_step_0149.pdf) |
| 297 | 0.185625 | [PNG](./profiles/shear_stress/snapshot_02_step_0297.png) / [PDF](./profiles/shear_stress/snapshot_02_step_0297.pdf) | [PNG](./profiles/unmixedness/snapshot_02_step_0297.png) / [PDF](./profiles/unmixedness/snapshot_02_step_0297.pdf) |
| 446 | 0.278750 | [PNG](./profiles/shear_stress/snapshot_03_step_0446.png) / [PDF](./profiles/shear_stress/snapshot_03_step_0446.pdf) | [PNG](./profiles/unmixedness/snapshot_03_step_0446.png) / [PDF](./profiles/unmixedness/snapshot_03_step_0446.pdf) |
| 594 | 0.371250 | [PNG](./profiles/shear_stress/snapshot_04_step_0594.png) / [PDF](./profiles/shear_stress/snapshot_04_step_0594.pdf) | [PNG](./profiles/unmixedness/snapshot_04_step_0594.png) / [PDF](./profiles/unmixedness/snapshot_04_step_0594.pdf) |
| 743 | 0.464375 | [PNG](./profiles/shear_stress/snapshot_05_step_0743.png) / [PDF](./profiles/shear_stress/snapshot_05_step_0743.pdf) | [PNG](./profiles/unmixedness/snapshot_05_step_0743.png) / [PDF](./profiles/unmixedness/snapshot_05_step_0743.pdf) |
| 891 | 0.556875 | [PNG](./profiles/shear_stress/snapshot_06_step_0891.png) / [PDF](./profiles/shear_stress/snapshot_06_step_0891.pdf) | [PNG](./profiles/unmixedness/snapshot_06_step_0891.png) / [PDF](./profiles/unmixedness/snapshot_06_step_0891.pdf) |
| 1040 | 0.650000 | [PNG](./profiles/shear_stress/snapshot_07_step_1040.png) / [PDF](./profiles/shear_stress/snapshot_07_step_1040.pdf) | [PNG](./profiles/unmixedness/snapshot_07_step_1040.png) / [PDF](./profiles/unmixedness/snapshot_07_step_1040.pdf) |

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

### Da = 1

| Step | Time | C1 | C2 | Vorticity |
|---|---|---|---|---|
| 0 | 0.000000 | [PNG](./fields/da1/c1/snapshot_00_step_0000.png) / [PDF](./fields/da1/c1/snapshot_00_step_0000.pdf) | [PNG](./fields/da1/c2/snapshot_00_step_0000.png) / [PDF](./fields/da1/c2/snapshot_00_step_0000.pdf) | [PNG](./fields/da1/vorticity/snapshot_00_step_0000.png) / [PDF](./fields/da1/vorticity/snapshot_00_step_0000.pdf) |
| 149 | 0.093125 | [PNG](./fields/da1/c1/snapshot_01_step_0149.png) / [PDF](./fields/da1/c1/snapshot_01_step_0149.pdf) | [PNG](./fields/da1/c2/snapshot_01_step_0149.png) / [PDF](./fields/da1/c2/snapshot_01_step_0149.pdf) | [PNG](./fields/da1/vorticity/snapshot_01_step_0149.png) / [PDF](./fields/da1/vorticity/snapshot_01_step_0149.pdf) |
| 297 | 0.185625 | [PNG](./fields/da1/c1/snapshot_02_step_0297.png) / [PDF](./fields/da1/c1/snapshot_02_step_0297.pdf) | [PNG](./fields/da1/c2/snapshot_02_step_0297.png) / [PDF](./fields/da1/c2/snapshot_02_step_0297.pdf) | [PNG](./fields/da1/vorticity/snapshot_02_step_0297.png) / [PDF](./fields/da1/vorticity/snapshot_02_step_0297.pdf) |
| 446 | 0.278750 | [PNG](./fields/da1/c1/snapshot_03_step_0446.png) / [PDF](./fields/da1/c1/snapshot_03_step_0446.pdf) | [PNG](./fields/da1/c2/snapshot_03_step_0446.png) / [PDF](./fields/da1/c2/snapshot_03_step_0446.pdf) | [PNG](./fields/da1/vorticity/snapshot_03_step_0446.png) / [PDF](./fields/da1/vorticity/snapshot_03_step_0446.pdf) |
| 594 | 0.371250 | [PNG](./fields/da1/c1/snapshot_04_step_0594.png) / [PDF](./fields/da1/c1/snapshot_04_step_0594.pdf) | [PNG](./fields/da1/c2/snapshot_04_step_0594.png) / [PDF](./fields/da1/c2/snapshot_04_step_0594.pdf) | [PNG](./fields/da1/vorticity/snapshot_04_step_0594.png) / [PDF](./fields/da1/vorticity/snapshot_04_step_0594.pdf) |
| 743 | 0.464375 | [PNG](./fields/da1/c1/snapshot_05_step_0743.png) / [PDF](./fields/da1/c1/snapshot_05_step_0743.pdf) | [PNG](./fields/da1/c2/snapshot_05_step_0743.png) / [PDF](./fields/da1/c2/snapshot_05_step_0743.pdf) | [PNG](./fields/da1/vorticity/snapshot_05_step_0743.png) / [PDF](./fields/da1/vorticity/snapshot_05_step_0743.pdf) |
| 891 | 0.556875 | [PNG](./fields/da1/c1/snapshot_06_step_0891.png) / [PDF](./fields/da1/c1/snapshot_06_step_0891.pdf) | [PNG](./fields/da1/c2/snapshot_06_step_0891.png) / [PDF](./fields/da1/c2/snapshot_06_step_0891.pdf) | [PNG](./fields/da1/vorticity/snapshot_06_step_0891.png) / [PDF](./fields/da1/vorticity/snapshot_06_step_0891.pdf) |
| 1040 | 0.650000 | [PNG](./fields/da1/c1/snapshot_07_step_1040.png) / [PDF](./fields/da1/c1/snapshot_07_step_1040.pdf) | [PNG](./fields/da1/c2/snapshot_07_step_1040.png) / [PDF](./fields/da1/c2/snapshot_07_step_1040.pdf) | [PNG](./fields/da1/vorticity/snapshot_07_step_1040.png) / [PDF](./fields/da1/vorticity/snapshot_07_step_1040.pdf) |

### Da = 10

| Step | Time | C1 | C2 | Vorticity |
|---|---|---|---|---|
| 0 | 0.000000 | [PNG](./fields/da10/c1/snapshot_00_step_0000.png) / [PDF](./fields/da10/c1/snapshot_00_step_0000.pdf) | [PNG](./fields/da10/c2/snapshot_00_step_0000.png) / [PDF](./fields/da10/c2/snapshot_00_step_0000.pdf) | [PNG](./fields/da10/vorticity/snapshot_00_step_0000.png) / [PDF](./fields/da10/vorticity/snapshot_00_step_0000.pdf) |
| 149 | 0.093125 | [PNG](./fields/da10/c1/snapshot_01_step_0149.png) / [PDF](./fields/da10/c1/snapshot_01_step_0149.pdf) | [PNG](./fields/da10/c2/snapshot_01_step_0149.png) / [PDF](./fields/da10/c2/snapshot_01_step_0149.pdf) | [PNG](./fields/da10/vorticity/snapshot_01_step_0149.png) / [PDF](./fields/da10/vorticity/snapshot_01_step_0149.pdf) |
| 297 | 0.185625 | [PNG](./fields/da10/c1/snapshot_02_step_0297.png) / [PDF](./fields/da10/c1/snapshot_02_step_0297.pdf) | [PNG](./fields/da10/c2/snapshot_02_step_0297.png) / [PDF](./fields/da10/c2/snapshot_02_step_0297.pdf) | [PNG](./fields/da10/vorticity/snapshot_02_step_0297.png) / [PDF](./fields/da10/vorticity/snapshot_02_step_0297.pdf) |
| 446 | 0.278750 | [PNG](./fields/da10/c1/snapshot_03_step_0446.png) / [PDF](./fields/da10/c1/snapshot_03_step_0446.pdf) | [PNG](./fields/da10/c2/snapshot_03_step_0446.png) / [PDF](./fields/da10/c2/snapshot_03_step_0446.pdf) | [PNG](./fields/da10/vorticity/snapshot_03_step_0446.png) / [PDF](./fields/da10/vorticity/snapshot_03_step_0446.pdf) |
| 594 | 0.371250 | [PNG](./fields/da10/c1/snapshot_04_step_0594.png) / [PDF](./fields/da10/c1/snapshot_04_step_0594.pdf) | [PNG](./fields/da10/c2/snapshot_04_step_0594.png) / [PDF](./fields/da10/c2/snapshot_04_step_0594.pdf) | [PNG](./fields/da10/vorticity/snapshot_04_step_0594.png) / [PDF](./fields/da10/vorticity/snapshot_04_step_0594.pdf) |
| 743 | 0.464375 | [PNG](./fields/da10/c1/snapshot_05_step_0743.png) / [PDF](./fields/da10/c1/snapshot_05_step_0743.pdf) | [PNG](./fields/da10/c2/snapshot_05_step_0743.png) / [PDF](./fields/da10/c2/snapshot_05_step_0743.pdf) | [PNG](./fields/da10/vorticity/snapshot_05_step_0743.png) / [PDF](./fields/da10/vorticity/snapshot_05_step_0743.pdf) |
| 891 | 0.556875 | [PNG](./fields/da10/c1/snapshot_06_step_0891.png) / [PDF](./fields/da10/c1/snapshot_06_step_0891.pdf) | [PNG](./fields/da10/c2/snapshot_06_step_0891.png) / [PDF](./fields/da10/c2/snapshot_06_step_0891.pdf) | [PNG](./fields/da10/vorticity/snapshot_06_step_0891.png) / [PDF](./fields/da10/vorticity/snapshot_06_step_0891.pdf) |
| 1040 | 0.650000 | [PNG](./fields/da10/c1/snapshot_07_step_1040.png) / [PDF](./fields/da10/c1/snapshot_07_step_1040.pdf) | [PNG](./fields/da10/c2/snapshot_07_step_1040.png) / [PDF](./fields/da10/c2/snapshot_07_step_1040.pdf) | [PNG](./fields/da10/vorticity/snapshot_07_step_1040.png) / [PDF](./fields/da10/vorticity/snapshot_07_step_1040.pdf) |

### Da = 100

| Step | Time | C1 | C2 | Vorticity |
|---|---|---|---|---|
| 0 | 0.000000 | [PNG](./fields/da100/c1/snapshot_00_step_0000.png) / [PDF](./fields/da100/c1/snapshot_00_step_0000.pdf) | [PNG](./fields/da100/c2/snapshot_00_step_0000.png) / [PDF](./fields/da100/c2/snapshot_00_step_0000.pdf) | [PNG](./fields/da100/vorticity/snapshot_00_step_0000.png) / [PDF](./fields/da100/vorticity/snapshot_00_step_0000.pdf) |
| 149 | 0.093125 | [PNG](./fields/da100/c1/snapshot_01_step_0149.png) / [PDF](./fields/da100/c1/snapshot_01_step_0149.pdf) | [PNG](./fields/da100/c2/snapshot_01_step_0149.png) / [PDF](./fields/da100/c2/snapshot_01_step_0149.pdf) | [PNG](./fields/da100/vorticity/snapshot_01_step_0149.png) / [PDF](./fields/da100/vorticity/snapshot_01_step_0149.pdf) |
| 297 | 0.185625 | [PNG](./fields/da100/c1/snapshot_02_step_0297.png) / [PDF](./fields/da100/c1/snapshot_02_step_0297.pdf) | [PNG](./fields/da100/c2/snapshot_02_step_0297.png) / [PDF](./fields/da100/c2/snapshot_02_step_0297.pdf) | [PNG](./fields/da100/vorticity/snapshot_02_step_0297.png) / [PDF](./fields/da100/vorticity/snapshot_02_step_0297.pdf) |
| 446 | 0.278750 | [PNG](./fields/da100/c1/snapshot_03_step_0446.png) / [PDF](./fields/da100/c1/snapshot_03_step_0446.pdf) | [PNG](./fields/da100/c2/snapshot_03_step_0446.png) / [PDF](./fields/da100/c2/snapshot_03_step_0446.pdf) | [PNG](./fields/da100/vorticity/snapshot_03_step_0446.png) / [PDF](./fields/da100/vorticity/snapshot_03_step_0446.pdf) |
| 594 | 0.371250 | [PNG](./fields/da100/c1/snapshot_04_step_0594.png) / [PDF](./fields/da100/c1/snapshot_04_step_0594.pdf) | [PNG](./fields/da100/c2/snapshot_04_step_0594.png) / [PDF](./fields/da100/c2/snapshot_04_step_0594.pdf) | [PNG](./fields/da100/vorticity/snapshot_04_step_0594.png) / [PDF](./fields/da100/vorticity/snapshot_04_step_0594.pdf) |
| 743 | 0.464375 | [PNG](./fields/da100/c1/snapshot_05_step_0743.png) / [PDF](./fields/da100/c1/snapshot_05_step_0743.pdf) | [PNG](./fields/da100/c2/snapshot_05_step_0743.png) / [PDF](./fields/da100/c2/snapshot_05_step_0743.pdf) | [PNG](./fields/da100/vorticity/snapshot_05_step_0743.png) / [PDF](./fields/da100/vorticity/snapshot_05_step_0743.pdf) |
| 891 | 0.556875 | [PNG](./fields/da100/c1/snapshot_06_step_0891.png) / [PDF](./fields/da100/c1/snapshot_06_step_0891.pdf) | [PNG](./fields/da100/c2/snapshot_06_step_0891.png) / [PDF](./fields/da100/c2/snapshot_06_step_0891.pdf) | [PNG](./fields/da100/vorticity/snapshot_06_step_0891.png) / [PDF](./fields/da100/vorticity/snapshot_06_step_0891.pdf) |
| 1040 | 0.650000 | [PNG](./fields/da100/c1/snapshot_07_step_1040.png) / [PDF](./fields/da100/c1/snapshot_07_step_1040.pdf) | [PNG](./fields/da100/c2/snapshot_07_step_1040.png) / [PDF](./fields/da100/c2/snapshot_07_step_1040.pdf) | [PNG](./fields/da100/vorticity/snapshot_07_step_1040.png) / [PDF](./fields/da100/vorticity/snapshot_07_step_1040.pdf) |

## Verification and reusable data

MF fields were revalidated against stored local bosonic states; matched DNS
snapshot diagnostics and vorticity from both methods' velocities were checked.
Input hashes, plotting-source hashes, and the complete figure index are in
[manifest.json](./manifest.json). No simulation inputs or results were modified.

- [All y profiles](./profiles_all_snapshots.csv): 3,072 rows (8 times × 3 Da × 2 methods × 64 y points).
- [Field comparison metrics and color limits](./field_comparison_metrics.csv): 72 rows.
- Reproduce with `python3 plot_reaction_snapshot_comparisons.py` from the repo root, using the established plotting environment.

Passing numerical checks is not a grid/timestep-convergence certificate.
