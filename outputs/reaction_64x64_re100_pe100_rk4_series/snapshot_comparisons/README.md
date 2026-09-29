# Combined all-snapshot comparisons

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
