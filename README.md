# 2-D incompressible mixing-layer solvers

## Bosonic MPS Chorin/MAC solver (current deliverable)

[`mixing_layer_mps_mac.jl`](./mixing_layer_mps_mac.jl) follows the bosonic-MPS
procedure in `ldc.jl`, but implements a periodic staggered Chorin projection
with interleaved `u`, `v`, and pressure-impulse sites on a `32 x 32` grid. The
coarse visible-mixing default is `Re=100`, and the full run schedules eight
vorticity snapshots. Setup, numerical details, convergence gates, and run
commands are in [`MPS_MAC_README.md`](./MPS_MAC_README.md).

Quick validation:

```bash
julia --project=. test/runtests.jl
julia --project=. mixing_layer_mps_mac.jl --validate
julia --project=. mixing_layer_mps_mac.jl --smoke-test
```

The `32 x 32` MPS contains 3072 truncated-boson sites and is a computationally
expensive research calculation. The code now has row-chunked MPO construction,
a locked persistent HDF5 operator cache, one-site CPU fast path, timing
controls, and checksummed generation-based checkpoint/restart.
Only reduced tensor tests were run locally; the full 1400-step evolution was
not. See [`hpc/README.md`](./hpc/README.md) before submitting it. A direct CPU
run is:

```bash
julia --project=. mixing_layer_mps_mac.jl --strict-quality --no-plot
```

## Completed 16×16 MPS–DNS comparison

A production MPS trajectory was completed and independently validated for a
`16 x 16` grid with `Re=50`, transition thickness `0.12`, KH-envelope width
`0.20`, and final time `3.5`. These parameters preserve the original 32×32
case's grid-scale Reynolds number and resolve each 10–90% shear transition with
approximately 4.2 cells.

For a fair method comparison, the classical DNS starts from the exact
staggered velocity fields stored in the MPS `t=0` snapshot. At `t=3.5`, the
vorticity relative L2 difference is `0.4105%`, while the velocity relative L2
difference is `0.0655%`.

- [MPS vorticity snapshots](./outputs/mps_dns_16x16_re50_delta012/mps_vorticity.png)
- [DNS vorticity snapshots](./outputs/mps_dns_16x16_re50_delta012/dns_vorticity.png)
- [MPS − DNS vorticity difference](./outputs/mps_dns_16x16_re50_delta012/mps_minus_dns_vorticity.png)
- [Per-snapshot comparison metrics](./outputs/mps_dns_16x16_re50_delta012/comparison_metrics.csv)

Regenerate the DNS comparison and plots from a completed MPS JLD2 result with:

```bash
python plot_mps_dns_comparison.py /path/to/mixing_layer_mps_mac.jld2 \
  outputs/mps_dns_16x16_re50_delta012
```

## Classical 128×128 DNS reference

This project solves the hydrodynamic incompressible Navier--Stokes equations for
a periodic double mixing layer on a `128 x 128` marker-and-cell (MAC) grid.  It
uses Chorin's fractional-step pressure projection and produces eight vorticity
snapshots with velocity arrows.

## Requested case

- `Nx = Ny = 128`, `dx = dy = 1/128`, so `Lx = Ly = 1`.
- Periodic boundary conditions in both directions.
- `Re = UL/nu = 1000` with `U = L = 1`, hence `nu = 0.001`.
- Bottom and top streams have `u = -1`; the middle stream has `u = +1`.
- The thinner middle layer occupies `0.30 Ly`, placing its interfaces at
  `y = 0.35` and `y = 0.65`; each transition has thickness `delta = 0.03`.
- A localized, divergence-free KH seed contains a dominant streamwise mode 2
  and a weaker mode-1 subharmonic to encourage later vortex pairing.

The double-tanh profile is

```text
u0(y) = tanh[(y-y1)/delta] - tanh[(y-y2)/delta] - 1,
y1 = (1-f) Ly/2,  y2 = (1+f) Ly/2,  f = 0.30.
```

The tails are saturated at the periodic seam, and the values match there by
symmetry. Because unequal layer widths give a nonzero mean velocity, the code
builds the mean-zero part from the streamfunction and then restores the uniform
mean; this retains the requested `-1/+1/-1` plateaus without affecting discrete
incompressibility.

## MAC layout and Chorin projection

Every field stores one periodic copy in an array of shape `(Ny, Nx)`:

```text
             v[j+1,i]
                 ^
                 |
     p[j,i-1] -- u[j,i] -- p[j,i]
                 |
                 v[j,i]

p[j,i] : ((i+1/2) dx, (j+1/2) dy)
u[j,i] : ( i      dx, (j+1/2) dy)
v[j,i] : ((i+1/2) dx,  j      dy)
```

At each midpoint stage the code:

1. forms explicit tentative face velocities from centered conservative
   momentum fluxes and second-order viscous differences;
2. solves `laplacian(p) = divergence(u*) / dt` at cell centers;
3. applies `u = u* - dt grad(p)` on the faces.

The Poisson solve uses an FFT only as a periodic linear solver.  Its symbols are
the eigenvalues of the *discrete* five-point Laplacian, so the MAC divergence is
removed to roundoff.  The momentum operators themselves are finite differences,
not pseudo-spectral derivatives.

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
output times, and writes:

```text
outputs/mixing_layer_mac_vorticity.png
outputs/mixing_layer_mac_snapshots.npz
```

The compressed data file contains all eight face-velocity fields, pressure,
vorticity, total/cross-stream energy, enstrophy, divergence diagnostics, times,
and the JSON-encoded configuration.  To change
the final time or omit the data file:

```bash
python mixing_layer_dns.py --t-end 2.0 --no-data
```

For a coarse reference using the MPS grid, Reynolds number, and transition
thickness:

```bash
python mixing_layer_dns.py --nx 32 --ny 32 --re 100 \
  --transition-thickness 0.06 --output-dir outputs/dns_32x32_re100
```

To use the same KH seed parameters as the MPS case, add:

```bash
--perturbation-width 0.10 --kh-mode 1 --kh-amplitude 0.10 \
  --secondary-mode 2 --secondary-amplitude 0.025
```

The middle width is exposed as a command-line parameter, for example:

```bash
python mixing_layer_dns.py --middle-layer-fraction 0.25
```

## Tests

The tests cover the staggered operator adjoint identity, the discrete Poisson
equation, projection accuracy and energy orthogonality, pure-gradient removal,
the divergence-free KH initialization, the requested mean velocity profile,
and exact eight-snapshot scheduling.

```bash
python -m unittest discover -s test -p 'test_*.py' -v
```
