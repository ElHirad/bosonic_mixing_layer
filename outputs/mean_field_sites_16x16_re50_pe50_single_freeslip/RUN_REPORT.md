# Explicit single-site mean-field mixing layer: cluster verification

The production trajectory completed all 260 physical steps to $t=0.65$ on
HTC, job **11215322**, and passed the independent local-state validator.
It used one CPU and about 57 MiB maximum resident memory. The solver process
took 2 minutes 13 seconds (scheduler allocation: 2 minutes 14 seconds).

This run evolves explicit single-site Fock vectors under
$f_j a_j^\dagger+b_j a_j+c_jI$ in **every** dynamical stage. It uses no TDVP,
amplitude-only time integration, direct Poisson solve, coherent-state reset,
or scalar-mass repair. Source/configuration fingerprints and compute-node
provenance are embedded in the NPZ files. The three frozen production
modules remain in the cluster run directories under `/ix/jmendoza-arenas/hia21/mixing-layer/`.

## Physical setup and local-state accuracy

The physical parameters are unchanged: 16×16 unit-square MAC cells,
Re=Pe=50, periodic $x$, free-slip/no-flux $y$, one shear at $y=0.5$,
thickness 0.04, KH width 0.12, mode-2 amplitude 2.5, mode-1 amplitude 0.5,
phase zero, and physical timestep 0.0025. The initial seed is deliberately
finite-amplitude, with a peak velocity component about 2.66.

There are 1,040 local states (including the explicit wall-vacuum sites).
Production uses occupation cutoff 12, eight RK4 subdivisions of each
predictor/correction interval, pressure pseudo-step $0.125/16^2$, and
pressure residual tolerance $10^{-8}$.

| Maximum over all 260 steps | Production value |
|---|---:|
| Raw relative scalar-mass drift | $5.05\times10^{-13}$ |
| Relative divergence | $2.92\times10^{-10}$ |
| Absolute divergence | $4.32\times10^{-8}$ |
| Pressure-relaxation relative residual | $9.99\times10^{-9}$ |
| Absolute mean pressure impulse | $2.61\times10^{-10}$ |
| Coherent eigenstate defect | $2.01\times10^{-10}$ |
| Ceiling occupation probability | $4.61\times10^{-21}$ |
| Pressure-stage velocity leakage | $1.46\times10^{-13}$ |
| Correction-stage pressure leakage | $4.22\times10^{-9}$ |
| Wall-normal velocity | Exactly zero |

The pressure stage required 166,380 local mean-field pseudo-time steps in
total. The final raw concentration mean is 0.49999999999974737.
All 38 Python tests pass, including explicit local-matrix evolution,
mean-field factorization, independent MAC coefficients, passive-scalar
compatibility, pressure/correction evolution, cutoff effects, and exact
checkpoint restart.
The four DNS-comparison tests cover parameter matching, exact initial
observables, independent reference evolution, field norms, coordinate
alignment, incompatible inputs, and plot generation.

## Completed convergence checks

All three HTC jobs finished with exit code `0:0`, completed 260 steps, and
passed validation by reconstructing fields from their stored local kets.

| Job | Local-state settings | Scheduler elapsed |
|---|---|---|
| 11215322 | Cutoff 12; 8 predictor/correction substeps; pressure CFL 0.125 | 00:02:14 |
| 11215323 | Cutoff 16; otherwise identical to production | 00:10:36 |
| 11215324 | Cutoff 12; 16 predictor/correction substeps; pressure CFL 0.0625 | 00:06:34 |

The maximum relative L2 differences across the eight stored snapshots are:

| Check against production | Velocity | Vorticity | Concentration |
|---|---:|---:|---:|
| Larger basis, cutoff 16 | $1.15\times10^{-15}$ | $3.12\times10^{-15}$ | $6.20\times10^{-16}$ |
| Finer local integration | $1.36\times10^{-10}$ | $4.08\times10^{-10}$ | $1.15\times10^{-10}$ |

The basis comparison is at roundoff level. The integration comparison
confirms that resolving the explicit local-state evolution more finely does
not materially change these fields. Neither comparison removes the scalar
overshoot or establishes grid convergence.

All three complete local-state trajectories are included as NPZ files in
this directory. Their per-run validations and per-snapshot comparison
metrics are also embedded in [summary.json](./summary.json).

## Dynamics, comparison, and limitations

- [Vorticity and velocity snapshots](./mean_field_vorticity.png)
- [Concentration snapshots](./mean_field_concentration.png)
- [Pairing and all-step diagnostics](./mean_field_diagnostics.png)
- [Mean-field/DNS vorticity with signed difference maps](./mean_field_vs_dns_vorticity.png)
- [Mean-field/DNS concentration with signed difference maps](./mean_field_vs_dns_concentration.png)
- [Mean-field/DNS errors, pairing, conservation, and concentration ranges](./mean_field_vs_dns_diagnostics.png)

The spatial snapshots show two seeded rollers developing, tilting, and
broadening into a larger-scale structure. Mode 1 becomes dominant between
$t=0.3725$ and $t=0.465$; the final mode-1/mode-2 ratio is 5.5203. This is
consistent with the previous run's pairing behavior, but at 16×16 with
substantial viscous decay it is not a grid-converged demonstration of KH
instability or a diagnostic based solely on counting the Fourier modes.
The vorticity plots show the actual sampled grid without smoothing; the
fixed colour range saturates beyond ±20 without changing the saved data.

| Final relative L2 difference | Saved MPS reference | Matched DNS benchmark |
|---|---:|---:|
| Velocity | 0.01661% | 0.88924% |
| Vorticity | 0.03524% | 2.06454% |
| Concentration | 0.02247% | 0.62345% |

The MPS reference is the **previously saved** cutoff-4 trajectory, with TDVP,
looser pressure tolerance, and scalar-mean repair. Its initial observables
differ slightly from the new mean-field encoding.

The DNS benchmark is a **new independent calculation**, initialized from
the exact measured mean-field $t=0$ velocity and concentration arrays. Its
initial field differences are identically zero. It completes 260 steps on
the same 16×16 grid at the same Re, Pe, physical timestep, boundary
conditions, and eight output times. It uses the existing projected-midpoint
DNS integrator, while the mean-field trajectory uses a pressure-free
predictor followed by relaxation and correction. Thus their temporal
splitting differs; the field differences are not isolated estimates of
the mean-field approximation error. Neither trajectory establishes spatial
convergence, and this same-grid DNS is a numerical reference rather than a
grid-converged physical solution.

DNS is run only by the separate comparison program, **never as a stage of
the mean-field solver**. The existing mean-field trajectory is not changed.
Its source fingerprints still match all three validated cluster runs.
The benchmark records its own configuration, source hashes, per-step
history, and provenance in [dns_matched_snapshots.npz](./dns_matched_snapshots.npz).
Its maximum relative scalar-mass drift is $2.22\times10^{-16}$, maximum
absolute divergence is $7.11\times10^{-15}$, and both wall velocities
remain zero. See [DNS validation](./dns_matched_snapshots.validation.json)
and [the per-snapshot comparison table](./comparison_DNS.csv).

**Concentration is conserved but not bound-preserving.** Across all physical
steps, its range briefly reaches $[-0.0726354,1.0817469]$. This is exposed in
the diagnostics; neither clipping nor scalar-mass repair is applied. A
physical requirement of $0\le c\le1$ needs a new bound-preserving transport
formulation. At the final time the range is $[0.0077733,0.9932955]$.
The independent DNS also overshoots, with all-step range
$[-0.0733810,1.0733810]$; its scalar flux uses the same non-bound-preserving
centered discretization. Both unmodified ranges are plotted.

Local integration convergence does not establish physical-timestep or grid
convergence. The finer check holds the physical Chorin timestep fixed and
only refines the integration of its three operator stages.

## Reproduce the production plots

From the repository root:

```bash
python3 compare_mean_field_dns.py \
  outputs/mean_field_sites_16x16_re50_pe50_single_freeslip/mean_field_snapshots.npz \
  outputs/mean_field_sites_16x16_re50_pe50_single_freeslip/dns_matched_snapshots.npz

python3 plot_mean_field_results.py \
  outputs/mean_field_sites_16x16_re50_pe50_single_freeslip/mean_field_snapshots.npz \
  outputs/mean_field_sites_16x16_re50_pe50_single_freeslip \
  --saved-mps outputs/mps_dns_16x16_re50_pe50_single_freeslip_delta004_largekh/mps_snapshots.npz \
  --saved-dns outputs/mean_field_sites_16x16_re50_pe50_single_freeslip/dns_matched_snapshots.npz \
  --larger-basis outputs/mean_field_sites_16x16_re50_pe50_single_freeslip/larger_basis_snapshots.npz \
  --finer-integration outputs/mean_field_sites_16x16_re50_pe50_single_freeslip/finer_integration_snapshots.npz
```

See [the complete operator algorithm](../../SINGLE_SITE_MEAN_FIELD_README.md),
[machine-readable validation](./validation.json), and
[full comparison/convergence metrics](./summary.json).
