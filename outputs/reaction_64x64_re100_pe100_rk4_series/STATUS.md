# 64×64 reacting mean field and DNS: RK4, Re=Pe=100

## Completed results and new statistics (2026-09-28)

All three full MF jobs and their independent DNS comparisons completed,
covering 1,040 steps to t=0.65. Fresh revalidation of the stored local Fock
states passed for Da=1,10,100. The launch-time status below is historical.

New [Reynolds-stress, vorticity-thickness, and unmixedness comparisons](./statistics/README.md)
include all three Da values in each panel: **solid DNS, dashed MF**. They use
the eight saved field snapshots, without changing/rerunning the trajectories.
Both averaging conventions for the signed c1/c2 covariance are documented;
CSV time series, y profiles, PNG/PDF figures, and input provenance are included.

The [combined comparison index](./snapshot_comparisons/README.md) now
provides **five all-snapshot figures**: shear stress vs y, unmixedness vs y,
C1, C2, and vorticity. Each includes all eight saved times and all three Da.
The profiles use solid DNS/dashed MF; the fields include DNS, MF, and signed
MF−DNS panels. Only x averaging is used for the y profiles, without y/time
averaging or concentration clipping. Snapshot-per-file plots and per-snapshot
CSVs were removed; raw simulation data and consolidated CSVs are preserved.

The user clarified **MF**, not the original MPS/TDVP solver, and requested
64×64, Re=Pe=100, RK4, and positive/negative concentration tracking. Da=1,10,100
and c1+c2 -> c3 are retained. All older datasets and failed runs are preserved.

## Setup

- n=64; Re=Pe=100; dt=0.000625; t=0.65 (1,040 physical steps).
- One shear, thickness 0.01875; KH mode 2/1 amplitudes 2.5/0.5, width 0.12,
  phase zero; periodic x, free-slip/no-flux y.
- Initial c2=1-c1, c3=0; all three species use Pe=100.
- Explicit local Fock kets, cutoff 12, velocity/pressure/scalar scales 8/4/4.
- Predictor/correction: eight RK4 subdivisions each. Pressure: RK4,
  pseudo-step 0.125/64², cap 48,000, approved tolerance **1e-7**.
- All other gates unchanged, including coherence 1e-5 and invariant drift 1e-7.
- No coherent resets, amplitude-only evolution, classical MF pressure solve,
  species clipping, or mass repair.

Reacting DNS uses four-stage RK4 on the coupled projected-momentum and species
equations, not the old midpoint reference. Projected RHS stages keep velocity
divergence-free. Stored DNS pressure is the RK-stage-weighted step average.
MF retains Chorin splitting: RK4 in its stages does not establish globally
fourth-order MF accuracy or equivalence to DNS.

## Signed diagnostics

Both solvers record c1/c2/c3 minima, maxima, positive/negative/zero cell counts
and fractions, and positive/negative signed integrals at t=0 and every accepted
physical step. Negative integrals are nonpositive; adding both signed parts
recovers each species amount. Raw c<0 and material c<-1e-12 fractions are both
stored. No data are clipped; internal RK-stage extrema are not claimed.

DNS screening writes `dns_preflight_daDA.json` and `dns_sign_history_daDA.csv`.
Completed MF/DNS postprocessing writes `concentration_sign_history.csv`,
`concentration_signs.png`, c1/c2/c3 snapshot/difference plots, flow diagnostics,
and `RUN_REPORT.md` separately for each Da.

## Conditional workflow

Each preflight runs full-time DNS screening, then requires **ten consecutive
accepted MF steps**. It requests one Turin CPU, 4 GiB, and six hours. All three
must pass before any full MF job starts. A failed, short, or interrupted test
cannot release production; waiting full jobs are cancelled on invalid dependencies.

Full jobs resume their ten-step checkpoints, target all 1,040 steps,
validate the complete result, then run independent matched RK4 DNS and plots.
Each requests one Turin CPU, 4 GiB, and three days, with per-step checkpointing.
There is no automatic wall-limit resubmission. Numerical/plotting sources
and scripts are frozen before submission. Old checkpoints are not reused.

All **91 Python tests passed** before submission, including RK4 convergence,
raw-sign preservation, exact restart, rejection of midpoint mislabelling,
end-to-end comparison plots, and fail-closed preflight guards.

## Submitted jobs (2026-09-25)

| Da | DNS screen + ten MF steps | Conditional full MF + DNS comparison |
|---|---|---|
| 1 | 11453847 | 11453849 |
| 10 | 11453846 | 11453851 |
| 100 | 11453848 | 11453850 |

All three preflights started on htc-n77. All full jobs were confirmed PENDING
with `afterok:11453847:11453846:11453848` and
`KillOInInvalidDependent=Yes`. Compute-node plotting imports passed, and the
printed MF configurations matched their DNS screen configuration exactly.
MF validation is still pending at this launch handoff.

The frozen launch bundle is
`cluster_runs/mf64-reaction-rk4-launch-20260925-9HRDq6` (nine Python modules
and five batch scripts). Both tests and full jobs use it as `PROJECT_ROOT`,
with `/usr/bin/python3` and the original workspace's `cluster_runs/mf-plot-support`.
For each DA=1,10,100, the explicitly selected paths are:

- Scratch: `/ix/jmendoza-arenas/hia21/mixing-layer/mean-field-sites64-re100-pe100-daDA-three-species-rk4-delta001875-nb12-ptol1e7`.
- Publication: `outputs/mean_field_sites_64x64_re100_pe100_daDA_reaction_rk4_ptol1e7` under the original workspace.
- Logs: `mf64_rk4_daDA_test.JOBID.log` / `mf64_rk4_daDA_full.JOBID.log` in the original workspace.

The operator evolution modules remain byte-identical to the previous series:
`mean_field_bosons.py` and `mean_field_operators.py`. The main solver source
changed only for signed diagnostic collection/persistence/validation, not
its time-stepping algorithm; its new SHA-256 is
`affe3c89db7690d61e03e60945c763f8352e42be4289a08927a095fb1901ebe2`.
Source fingerprints therefore correctly prevent reuse of older checkpoints.

## Completed full-time DNS screening

All three independent **RK4 DNS** screens completed 1,040 steps to t=0.65,
passing conservation, divergence, boundary, gauge, and timestep checks.
All-step extrema include initialization. Small negative values remain:

| Da | Min c1/c2 | Max c1/c2 | Min c3 | Max c3 |
|---|---|---|---|---|
| 1 | -0.00824371 | 1.00855549 | -0.000221973 | 0.0997204 |
| 10 | -0.00707095 | 1.00987835 | -0.00193001 | 0.3439352 |
| 100 | -0.00412520 | 1.01590849 | -0.00764368 | 0.4592345 |

c1/c2 extrema agree to the displayed precision. Maximum relative drift of
the c1+c3 / c2+c3 integrals was 2.22e-16 in all three cases. Peak negative-cell
fractions for each reactant were 0.2686%, 0.2686%, and 0.4639%, respectively;
for c3 the peak fraction was 0.5859% in every case. Raw and c<-1e-12 peak
fractions agree here. No positivity guarantee is implied.

Every DNS CSV contains **1,041 rows including t=0**. A separate read-back audit
verified all step/time entries, positive+negative+zero counts=4,096 per species,
and signed-integral sums equal the species amounts. DNS fingerprints match
the current MF sources/configurations. These are DNS results only, not a
completed MF comparison.

- Da=1: [summary](./dns_preflight_da1.json), [signed history](./dns_sign_history_da1.csv).
- Da=10: [summary](./dns_preflight_da10.json), [signed history](./dns_sign_history_da10.csv).
- Da=100: [summary](./dns_preflight_da100.json), [signed history](./dns_sign_history_da100.csv).
