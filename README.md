# 2-D incompressible mixing-layer solvers

## Single-layer DNS with free-slip y boundaries (current deliverable)

[`mixing_layer_dns.py`](./mixing_layer_dns.py) now runs one centered shear layer
instead of the periodic double layer. The production case uses a `256 x 256`
MAC grid, periodicity only in `x`, and free-slip boundaries in `y`:

```text
du/dy = 0,  v = 0,  dp/dy = 0  at y = 0, Ly.
```

The initial profile is `u0(y) = tanh[(y - Ly/2)/delta]`, with the reduced
thickness `delta = 0.02`. At `Re = 5000`, a mode-4 Kelvin--Helmholtz seed rolls
into four vortices and a weak mode-2 subharmonic produces a visible four-to-two
pairing. The saved eight-panel result shows the roll-up at `t = 0.5`, pairing
at `t = 1.0`, and two paired vortices at `t = 1.5`:

- [Vorticity and velocity snapshots](./outputs/dns_single_layer_freeslip/single_shear_layer_vorticity.png)
- [Compressed DNS fields and diagnostics](./outputs/dns_single_layer_freeslip/single_shear_layer_snapshots.npz)

Regenerate the production result with:

```bash
python mixing_layer_dns.py --output-dir outputs/dns_single_layer_freeslip
```

## Bosonic MPS Chorin/MAC solver (periodic comparison)

[`mixing_layer_mps_mac.jl`](./mixing_layer_mps_mac.jl) follows the bosonic-MPS
procedure in `ldc.jl`, but implements a periodic staggered Chorin projection
with interleaved `u`, `v`, pressure-impulse, and scalar sites on a `32 x 32`
grid. The solver carries a cell-centered conserved scalar initialized to 1 in
the middle stream and 0 in both outer streams through a double-tanh shear. The
coarse visible-mixing default is `Re=Pe=100`, and the full run schedules eight
vorticity and concentration snapshots. Setup, numerical details, convergence
gates, and run commands are in [`MPS_MAC_README.md`](./MPS_MAC_README.md).

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

## Completed 16×16 MPS–DNS velocity and concentration comparison

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
conservation, and exact eight-snapshot scheduling.

```bash
python -m unittest discover -s test -p 'test_*.py' -v
```
