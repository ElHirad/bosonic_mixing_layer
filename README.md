# 2-D incompressible mixing-layer solvers

## Completed 64×64 higher-Re/Pe run

The higher-Re/Pe case uses **Re=Pe=200**, reducing viscosity and scalar diffusivity
from 0.02 to 0.005. Grid size (64×64), shear thickness (0.01875), timestep
(0.000625), KH seeds, and all single-site bosonic integration settings are
unchanged. It completed 1,040 steps to time 0.65 in a separate directory, preserving
the completed Re=Pe=50 results.

```bash
sbatch --clusters=htc --export=ALL,MF_REYNOLDS=200,MF_PECLET=200 hpc/mean_field64_cpu.sbatch
```

Job **11244737** completed the trajectory, validation, separate matched DNS
comparison, and plotting in **24 hours 36 minutes** (earlier preflight work is
additional). The mean-field fields show clear spiral roll-up and stronger
vortex-pair rotation than at Re=Pe=50. Two distinct cores remain at t=0.65,
so a completed merger is not demonstrated.

Final MF/DNS relative L2 differences are **4.58% for velocity, 11.71% for
vorticity, and 2.63% for concentration**. These larger discrepancies warrant
a timestep-convergence study before claiming quantitative accuracy. All-step
concentration spans [-0.059699, 1.0625]: centered transport is not bound-preserving,
and no clipping or mass repair is used. Passing the numerical gates does not
establish grid or timestep convergence. See the
[completed report and plots](./outputs/mean_field_sites_64x64_re200_pe200_single_freeslip_delta001875/RUN_REPORT.md),
[concentration roll-up](./outputs/mean_field_sites_64x64_re200_pe200_single_freeslip_delta001875/mean_field_concentration.png),
and [launch history and archived DNS preview](./outputs/mean_field_sites_64x64_re200_pe200_single_freeslip_delta001875/LAUNCH_NOTES.md).

## Completed 64×64 Re=Pe=50 run

The new preset uses a 64×64 grid, tanh thickness **0.01875** (25% thinner
than the 32×32 case), and timestep **0.000625**: 1,040 steps to time 0.65.
Re=Pe=50 and the KH seeds remain unchanged by default. All dynamical stages
continue to evolve explicit single-site bosonic states. The original 16×16
and 32×32 results are preserved.

```bash
sbatch --clusters=htc hpc/mean_field64_cpu.sbatch
```

Job **11217985** completed all 1,040 steps, validation, DNS comparison, and
plotting in **26 hours 2 minutes** (earlier preflight work is additional).
Final MF/DNS relative L2 differences are 0.30% for velocity, 0.68% for vorticity,
and 0.17% for concentration. The fields show roll-up followed by broadening,
not strong clean pairing. See the
[completed 64×64 report and plots](./outputs/mean_field_sites_64x64_re50_pe50_single_freeslip_delta001875/RUN_REPORT.md),
[launch history](./outputs/mean_field_sites_64x64_re50_pe50_single_freeslip_delta001875/LAUNCH_NOTES.md),
and [cluster instructions](./hpc/README.md).

## Completed 32×32 thinner-shear mean-field case

The new cluster preset keeps Re=Pe=50 and the explicit single-site algorithm,
with a 32×32 grid, tanh thickness **0.025**, and physical timestep **0.00125**.
It retains KH mode amplitudes 2.5/0.5 and runs 520 steps to time 0.65.
All 520 steps completed and passed validation. The final Slurm allocation,
job 11216428, took 59 minutes 12 seconds including DNS comparison and plotting;
earlier checkpointed work is additional. The original 16×16 results remain unchanged.

```bash
sbatch --clusters=htc hpc/mean_field32_cpu.sbatch
```

The batch workflow validates the full single-site result, then produces a
separate matched DNS comparison and plots automatically. See the completed
[32×32 report and plots](./outputs/mean_field_sites_32x32_re50_pe50_single_freeslip_delta0025/RUN_REPORT.md),
[32×32 launch notes](./outputs/mean_field_sites_32x32_re50_pe50_single_freeslip_delta0025/LAUNCH_NOTES.md)
and [cluster instructions](./hpc/README.md). Final MF/DNS relative L2 differences
are 0.55% for velocity, 1.26% for vorticity, and 0.33% for concentration.
The fields show early roll-up followed by broadening and weakening, not a
clear, strong vortex-pairing event. Transient concentration overshoots remain
(-0.020 to 1.025); the conservative centered scheme is not bound-preserving.

## Explicit single-site bosonic mean-field solver (current)

[`mixing_layer_mean_field.py`](./mixing_layer_mean_field.py) replaces TDVP
with self-consistent **single-site Fock-vector evolution** under explicit
creation, annihilation, and identity matrices. Momentum and concentration,
pressure relaxation, and velocity correction all use mean-field operators.
There is no amplitude-only evolution, direct Poisson solver, DNS dependency,
coherent-state resetting, or scalar-mass repair in this solver.

The defaults retain the last 16×16 single-layer case: Re=Pe=50, thickness
0.04, mode-2/mode-1 seed amplitudes 2.5/0.5, and 260 steps to time 0.65.
The local cutoff is increased to 12 and checked against 16.
See the [algorithm and operator derivation](./SINGLE_SITE_MEAN_FIELD_README.md)
and [cluster workflow](./hpc/README.md).

The completed single-site production run passes all 260 steps with maximum
raw relative scalar-mass drift `5.1e-13`, without a mass correction. Its final
vorticity differs from the saved MPS trajectory by `0.035%`.
The centered scalar transport remains non-bound-preserving; its brief
concentration overshoots are retained and displayed in the diagnostics.

- [Single-site vorticity snapshots](./outputs/mean_field_sites_16x16_re50_pe50_single_freeslip/mean_field_vorticity.png)
- [Single-site concentration snapshots](./outputs/mean_field_sites_16x16_re50_pe50_single_freeslip/mean_field_concentration.png)
- [Pairing, conservation, and local-state diagnostics](./outputs/mean_field_sites_16x16_re50_pe50_single_freeslip/mean_field_diagnostics.png)
- [Mean-field vs DNS vorticity and signed differences](./outputs/mean_field_sites_16x16_re50_pe50_single_freeslip/mean_field_vs_dns_vorticity.png)
- [Mean-field vs DNS concentration and signed differences](./outputs/mean_field_sites_16x16_re50_pe50_single_freeslip/mean_field_vs_dns_concentration.png)
- [DNS comparison: errors, pairing, and conservation](./outputs/mean_field_sites_16x16_re50_pe50_single_freeslip/mean_field_vs_dns_diagnostics.png)
- [Run verification and convergence report](./outputs/mean_field_sites_16x16_re50_pe50_single_freeslip/RUN_REPORT.md)

The separate matched DNS benchmark starts from the **exact measured initial
mean-field fields**, uses the same physical setup and output times, and
advances with the existing projected-midpoint DNS method. Final relative
differences are `0.889%` in velocity, `2.065%` in vorticity, and `0.623%` in
concentration. This benchmark is not a stage of the mean-field algorithm;
different time splitting is retained and explicitly documented.

```bash
OPENBLAS_NUM_THREADS=1 python3 mixing_layer_mean_field.py --output-dir /path/to/new/run
sbatch --clusters=htc hpc/mean_field_cpu.sbatch
```

The original DNS and TDVP/MPS programs below are retained as reference
implementations and for reproducing earlier results.

## Finite-amplitude 16×16 single-layer DNS

[`mixing_layer_dns.py`](./mixing_layer_dns.py) runs one centered shear layer on
a MAC grid, with periodicity only in `x` and free-slip boundaries in `y`:

```text
du/dy = 0,  v = 0,  dp/dy = 0  at y = 0, Ly.
```

At `16 x 16` and `Re=50`, small KH disturbances are viscously damped. A
strongly nonlinear mode-2 seed (`A2=2.5`) plus its mode-1 pairing subharmonic
(`A1=0.5`) nevertheless produces two resolved rolls followed by a two-to-one
merger. This remains a nominal `Re=50` calculation (`nu=0.02`, based on the
unit reference stream speed), but it is intentionally not a linear-instability
test: the finite-amplitude initial field reaches a peak component speed of
`2.66`.

- [Finite-amplitude vorticity snapshots](./outputs/dns_single_layer_16x16_re50_finite_amplitude/single_shear_layer_vorticity.png)
- [Mode-2 to mode-1 pairing diagnostic](./outputs/dns_single_layer_16x16_re50_finite_amplitude/single_shear_layer_pairing.png)
- [Compressed 16×16 DNS fields](./outputs/dns_single_layer_16x16_re50_finite_amplitude/single_shear_layer_snapshots.npz)

Regenerate this case with:

```bash
python mixing_layer_dns.py --nx 16 --ny 16 --re 50 \
  --transition-thickness 0.04 --perturbation-width 0.12 \
  --kh-mode 2 --kh-amplitude 2.5 \
  --secondary-mode 1 --secondary-amplitude 0.5 --phase 0 \
  --t-end 0.65 --vorticity-limit 20 \
  --output-dir outputs/dns_single_layer_16x16_re50_finite_amplitude
```

## Bosonic MPS Chorin/MAC solver

[`mixing_layer_mps_mac.jl`](./mixing_layer_mps_mac.jl) follows the bosonic-MPS
procedure in `ldc.jl`, but implements a staggered Chorin projection with
interleaved `u`, `v`, pressure-impulse, and conserved-concentration sites. It
supports both the legacy fully periodic double layer and the single-layer
periodic-x/free-slip-y channel. The bounded case uses conservative zero-normal
scalar flux and Neumann scalar diffusion at the y walls. Setup, numerical
details, convergence gates, and run commands are in
[`MPS_MAC_README.md`](./MPS_MAC_README.md).

Quick validation:

```bash
julia --project=. test/runtests.jl
julia --project=. mixing_layer_mps_mac.jl --validate
julia --project=. mixing_layer_mps_mac.jl --smoke-test
```

The `32 x 32` MPS contains 4096 truncated-boson sites and is a computationally
expensive research calculation. The code now has row-chunked MPO construction,
a locked persistent HDF5 operator cache, one-site CPU fast path, timing
controls, and checksummed generation-based checkpoint/restart.
The complete four-field 16×16 production case described below has been run and
validated. See [`hpc/README.md`](./hpc/README.md) before submitting a larger
case. A direct CPU run is:

```bash
julia --project=. mixing_layer_mps_mac.jl --strict-quality --no-plot
```

## Completed 16×16 free-slip roll-up and pairing comparison

The single-layer MPS calculation completed all 260 steps at `Re=Pe=50` with
transition thickness `0.04`, mode-2 amplitude `2.5`, and mode-1 pairing
amplitude `0.5`. It passed the independent result validator and every strict
per-step gate. The largest raw scalar-mean correction was `7.45e-6`, the
largest corrected mass error was `1.00e-11`, the largest relative divergence
was `1.24e-5`, and the largest boson-ceiling occupation was `1.74e-6`.

Both MPS and matched DNS form two rollers and then merge them into one. The
mode-1/mode-2 ratio crosses one between `t=0.3725` and `t=0.465`; at the final
time it is `5.52` for MPS and `5.44` for DNS. At `t=0.65`, the MPS/DNS relative
differences are `2.08%` in vorticity, `0.90%` in velocity, and `0.63%` in
concentration.

- [MPS vorticity and velocity snapshots](./outputs/mps_dns_16x16_re50_pe50_single_freeslip_delta004_largekh/mps_vorticity.png)
- [DNS vorticity and velocity snapshots](./outputs/mps_dns_16x16_re50_pe50_single_freeslip_delta004_largekh/dns_vorticity.png)
- [MPS concentration snapshots](./outputs/mps_dns_16x16_re50_pe50_single_freeslip_delta004_largekh/mps_concentration.png)
- [Roll-up and pairing compatibility](./outputs/mps_dns_16x16_re50_pe50_single_freeslip_delta004_largekh/pairing_compatibility.png)
- [Per-snapshot comparison metrics](./outputs/mps_dns_16x16_re50_pe50_single_freeslip_delta004_largekh/comparison_metrics.csv)

Regenerate the matched DNS and plots from the completed MPS result with:

```bash
python plot_mps_dns_comparison.py /path/to/mixing_layer_mps_mac.jld2 \
  outputs/mps_dns_16x16_re50_pe50_single_freeslip_delta004_largekh \
  --re 50 --pe 50 --boundary-y free-slip --shear-center 0.5 \
  --transition-thickness 0.04 --perturbation-width 0.12 \
  --kh-mode 2 --kh-amplitude 2.5 \
  --secondary-mode 1 --secondary-amplitude 0.5 --phase 0
```

## Earlier 16×16 periodic MPS–DNS comparison

A production MPS trajectory was completed and independently validated for a
`16 x 16` grid with `Re=Pe=50`, transition thickness `0.12`, KH-envelope width
`0.20`, and final time `3.5`. The MPS trajectory completed 1,400 steps in
21 h 59 min and passed every strict production gate. Its final relative scalar
mass drift was `8.91e-5`; the matched DNS conserves scalar mass to roundoff.

For a fair method comparison, the classical DNS starts from the exact
staggered velocity and concentration fields stored in the MPS `t=0` snapshot.
It evolves the scalar with the same conservative MAC fluxes and diffusivity
`1/Pe`. At `t=3.5`, the concentration relative L2 difference is `0.0281%`, the
vorticity relative L2 difference is `0.4105%`, and the velocity relative L2
difference is `0.0655%`.

- [MPS concentration snapshots](./outputs/mps_dns_16x16_re50_pe50_scalar_delta012/mps_concentration.png)
- [DNS concentration snapshots](./outputs/mps_dns_16x16_re50_pe50_scalar_delta012/dns_concentration.png)
- [MPS − DNS concentration difference](./outputs/mps_dns_16x16_re50_pe50_scalar_delta012/mps_minus_dns_concentration.png)
- [MPS vorticity snapshots](./outputs/mps_dns_16x16_re50_pe50_scalar_delta012/mps_vorticity.png)
- [DNS vorticity snapshots](./outputs/mps_dns_16x16_re50_pe50_scalar_delta012/dns_vorticity.png)
- [MPS − DNS vorticity difference](./outputs/mps_dns_16x16_re50_pe50_scalar_delta012/mps_minus_dns_vorticity.png)
- [Per-snapshot comparison metrics](./outputs/mps_dns_16x16_re50_pe50_scalar_delta012/comparison_metrics.csv)

Regenerate the DNS comparison and plots from a completed MPS JLD2 result with:

```bash
python plot_mps_dns_comparison.py /path/to/mixing_layer_mps_mac.jld2 \
  outputs/mps_dns_16x16_re50_pe50_scalar_delta012 --re 50 --pe 50
```

## MAC layout and Chorin projection

Pressure and `u` have shape `(Ny, Nx)`. The normal velocity contains both
physical boundary faces and has shape `(Ny + 1, Nx)`:

```text
             v[j+1,i]
                 ^
                 |
     p[j,i-1] -- u[j,i] -- p[j,i]
                 |
                 v[j,i]

p[j,i] : ((i+1/2) dx, (j+1/2) dy)
u[j,i] : ( i      dx, (j+1/2) dy)
v[j,i] : ((i+1/2) dx,  j      dy),  j = 0, ..., Ny
```

At each midpoint stage the code:

1. forms explicit tentative face velocities from centered conservative
   momentum fluxes and second-order viscous differences;
2. solves `laplacian(p) = divergence(u*) / dt` at cell centers;
3. applies `u = u* - dt grad(p)` on the faces.

The pressure solve diagonalizes the discrete five-point Laplacian with an FFT
in periodic `x` and a DCT-II in bounded `y`. These transforms are linear
solvers; momentum advection and diffusion remain second-order finite-volume
differences. The projection removes MAC divergence to roundoff while retaining
zero normal pressure gradient and zero boundary-normal velocity.

The initial velocity is obtained as the discrete curl of a vertex-centered
streamfunction.  It is therefore discretely divergence-free even before the
first pressure projection.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python mixing_layer_dns.py
```

The default run advances to `t = 3.5`, lands exactly on eight evenly spaced
output times, and writes to the selected output directory:

```text
single_shear_layer_vorticity.png
single_shear_layer_snapshots.npz
```

The compressed data file contains all eight face-velocity fields, pressure,
vorticity, total/cross-stream energy, enstrophy, divergence diagnostics, times,
and the JSON-encoded configuration.  To change
the final time or omit the data file:

```bash
python mixing_layer_dns.py --t-end 2.0 --no-data
```

The old periodic double-layer DNS remains available for reproducing the MPS
comparison or earlier reference outputs:

```bash
python mixing_layer_dns.py --periodic-y --nx 128 --ny 128 --re 1000 \
  --transition-thickness 0.03 --kh-mode 2 --kh-amplitude 0.02 \
  --secondary-mode 1 --secondary-amplitude 0.005
```

## Tests

The tests cover both boundary treatments: discrete Poisson equations,
gradient/divergence composition, projection accuracy, free-slip wall values,
inviscid energy conservation, divergence-free initialization, periodic scalar
and wall-bounded concentration conservation, and exact eight-snapshot
scheduling.

```bash
python -m unittest discover -s test -p 'test_*.py' -v
```
