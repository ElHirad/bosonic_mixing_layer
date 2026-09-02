# Bosonic-MPS Chorin/MAC mixing layer

[`mixing_layer_mps_mac.jl`](./mixing_layer_mps_mac.jl) implements periodic and
free-slip mixing layers with the coherent bosonic-MPS procedure used by
`ldc.jl`, but with a staggered marker-and-cell grid and Chorin pressure
projection.

## Formulation

Each grid cell contributes four interleaved sites along a serpentine chain:

```text
u[j,i]   : ((i-1) dx,   (j-1/2) dy)  vertical face
v[j,i]   : ((i-1/2) dx, (j-1) dy)    horizontal face
phi[j,i] : ((i-1/2) dx, (j-1/2) dy)  cell center, phi=dt*p
c[j,i]   : ((i-1/2) dx, (j-1/2) dy)  conserved passive scalar
```

The default `32x32` grid therefore has 4096 bosonic sites. The stored coherent
amplitudes normally use `u=4<a_u>`, `v=4<a_v>`, `phi=4<a_phi>`, and
`c=4<a_c>`. `--velocity-scale` changes only this encoding, not the physical
field; the finite-amplitude 16x16 channel uses 8 to stay below the local boson
ceiling.

One Chorin step consists of:

1. a one-site TDVP predictor containing conservative MAC momentum advection,
   viscosity, and conservative scalar advection-diffusion;
2. adaptive one-site TDVP relaxation of
   `laplacian(phi)=divergence(u_star)` (up to 12 blocks by default), followed
   by a uniform local displacement that enforces the zero-mean periodic or
   Neumann pressure gauge in the MPS itself;
3. a one-site TDVP correction `u=u_star-gradient(phi)`.

The coherent-product manifold is invariant under the ideal bosonic generator,
so the CPU default uses one-site TDVP for all three substeps and avoids
unnecessary two-site SVDs. Finite boson truncation can break that ideal
property. For quantitative work, compare against `--predictor-nsite 2`; this
allows predictor bonds to grow and is the more conservative convergence check.

The default double layer wraps in both directions. With
`--boundary-y free-slip`, only x wraps: `du/dy=0`, `v=0`, and `dphi/dy=0` are
imposed at the y walls. To retain exactly four sites per cell, the channel MPS
stores the bottom and `n-1` interior v faces; the top zero-valued face is implicit. The KH
perturbation is a discrete curl of a corner streamfunction and is divergence
free before projection.

The passive scalar obeys

```text
dc/dt + div(u c) = (1/Pe) laplacian(c).
```

The periodic case starts from a normalized double-tanh profile. The channel
starts from a normalized single tanh, from 0 below the shear to 1 above it.
Centered scalar values are averaged to the MAC faces before forming
conservative `u*c` and `v*c` fluxes. The channel has zero advective wall flux
and Neumann diffusion, so total concentration is conserved by the discrete
equations. The default `Pe=Re`; use `--pe` to choose it independently.

Finite local-boson truncation and one-site TDVP can introduce a small drift in
the constant scalar mode even though the discrete flux sum is zero. After each
predictor, a uniform scalar displacement restores the reference mean with the
minimum L2 correction. The uncorrected per-step drift is retained as
`scalar_mass_projection` and independently hard-gated at `1e-4`; the
projection therefore cannot hide an under-resolved scalar evolution.

The roll-up/pairing channel configuration is:

```bash
julia --project=. mixing_layer_mps_mac.jl --n 16 --re 50 --pe 50 \
  --boundary-y free-slip --transition 0.04 --kh-width 0.12 \
  --kh-mode 2 --kh-amplitude 2.5 \
  --secondary-mode 1 --secondary-amplitude 0.5 --phase 0 \
  --velocity-scale 8 --final-time 0.65 --strict-quality --no-plot
```

## Default visible-mixing case

```text
n = 32, dx = dy = 1/32
Re = 100, nu = 0.01
Pe = 100, scalar diffusivity = 0.01
dt = 0.0025, final time = 3.5, physical steps = 1400
middle width = 0.30 Ly, tanh transition parameter = 0.06
KH modes = 1 and 2, amplitudes = 0.10 and 0.025
nmax = 4, max bond dimension = 64, cutoff = 1e-10
Krylov tolerance = 1e-10, dimension = 30
pressure residual target = 1e-3
```

The sampled base profile is normalized to attain exactly `-1` and `+1`; its
10--90% transition spans approximately 4.2 cells. Here `Re=UL/nu`, with
`U=L=1`. This is a coarse research calculation, not a resolved DNS.

## CPU optimizations

The production path includes:

- direct dimension-one coherent-product construction instead of 4096 generic
  MPS `apply` calls;
- in-place `OpSum` insertion for all 56,320 default operator terms;
- an exact predictor sum split into eight row chunks per algebraic family,
  reducing generic OpSum-to-MPO workspace;
- a persistent, fingerprinted HDF5 cache for the site template and all MPOs;
- a pressure warm-start residual check that can skip all pressure TDVP work;
- reuse of the final pressure-block field extraction;
- one-site predictor/pressure/correction sweeps by default, with two-site
  predictor convergence available from the CLI;
- joint amplitude/ceiling expectation sweeps and configurable ceiling sampling;
- separate predictor, pressure, correction, step, build, allocation, and ETA
  measurements in logs and result metadata;
- configurable Julia/Strided, BLAS, Krylov, checkpoint, and progress controls.

On the local single-thread `8x8` construction benchmark, splitting each
predictor family from one chunk to eight reduced cumulative allocation from
43.7 GiB to 5.47 GiB and wall time from 12.6 s to 1.28 s. The resulting
predictor fields agreed to `2.6e-15`. In a warm `8x8` one-step comparison,
one-site predictor TDVP was 4.27 times faster than two-site TDVP; fields differed
by `4.3e-9` and both states remained at bond dimension one.

With the finished code, a cold `16x16` build using eight Julia threads and
eight predictor chunks took 60.6 s, reported 337 GiB of cumulative allocation,
and peaked at 2.94 GiB resident memory. Cumulative allocation is total allocator
traffic, not simultaneous RAM. These are local relative benchmarks, not
estimates of the unmeasured `32x32` evolution.

## Setup and verification

Use Julia 1.12.6 and the committed manifest:

```bash
export PATH="$HOME/.juliaup/bin:$PATH"
export JULIA_DEPOT_PATH="$PWD/cluster_runs/julia-depot-1.12"
julia +1.12.6 --project=. --startup-file=no -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
JULIA_PKG_OFFLINE=true julia +1.12.6 --project=. --startup-file=no test/runtests.jl
JULIA_PKG_OFFLINE=true julia +1.12.6 --project=. --startup-file=no mixing_layer_mps_mac.jl --validate
JULIA_PKG_OFFLINE=true julia +1.12.6 --project=. --startup-file=no mixing_layer_mps_mac.jl --smoke-test
```

The tests include periodic and channel MAC identities, conservative
concentration fluxes, direct/coherent-state equivalence,
stable fingerprints, one-site/two-site predictor agreement, cold/warm operator
caching, checksummed HDF5 checkpoint round trips, corrupt-generation recovery,
and continuous-versus-resumed TDVP equivalence.

## Checkpointed CPU execution

A normal CLI run enables a persistent operator cache and writes a restart
checkpoint every 25 complete physical steps:

```bash
julia --project=. --startup-file=no mixing_layer_mps_mac.jl --no-plot
```

Resume without rebuilding the trajectory:

```bash
julia --project=. --startup-file=no mixing_layer_mps_mac.jl \
  --resume outputs/mps_mac/mixing_layer_mps_checkpoint.h5 --no-plot
```

The checkpoint stores the full four-field MPS, snapshot history, scalar
diagnostics, pressure warm start, and run fingerprints. Each commit is an
immutable HDF5 generation with a SHA-256 sidecar; file and directory data are
fsynced before an atomic manifest update. The loader validates and ranks
generations by completed step, loads only the newest valid MPS, and retains
three valid generations. A restart rejects changes to numerical configuration,
threading/strict mode, source, manifest, package versions, layout, or
`MPS_MAX_BOSON`.

Operator caches use a separate explicit operator-definition version plus the
operator-relevant configuration and package manifest. Changes confined to the
driver, diagnostics, or pressure-gauge enforcement therefore do not force a
multi-gigabyte MPO rebuild. The loader also accepts the known source-hash cache
written before this separation when its full legacy fingerprint matches.

For a scheduler allocation, use `--stop-after-seconds` for the allocation
budget and `--shutdown-reserve-seconds` for final diagnostics, checkpoint hash,
and output. `--stop-file` also requests a stop at the next complete Chorin
boundary. The global snapshot schedule remains steps `0:200:1400` across
chunks. The supplied batch job uses an exclusive run lock and an advance USR1
signal. A partial chunk deliberately exits `75`; only a complete, independently
validated trajectory exits `0`.

Production should add `--strict-quality`, which returns a nonzero status when
projection, velocity/scalar leakage, scalar-mass conservation, correction,
pressure-gauge, imaginary-amplitude, sampled boson-ceiling, or bond-dimension
gates fail.

CPU SLURM probe and restart-aware job templates are in
[`hpc/README.md`](./hpc/README.md).

## Staged sizing

Do not submit all 1400 steps before measuring the target machine:

```bash
# Build/cache only
julia --project=. mixing_layer_mps_mac.jl --n 8  --steps 0 --no-plot
julia --project=. mixing_layer_mps_mac.jl --n 16 --steps 0 --no-plot
julia --project=. mixing_layer_mps_mac.jl --n 32 --steps 0 --no-plot

# Then time complete physical steps
julia --project=. mixing_layer_mps_mac.jl --n 32 --steps 1 --no-plot
julia --project=. mixing_layer_mps_mac.jl --n 32 --steps 5 --no-plot
```

The full 4096-site operator set and 1400-step trajectory have not been run
locally. Use `/usr/bin/time -v` and SLURM `sacct` to measure cache size, MaxRSS,
one-step time, pressure blocks, and checkpoint latency before production.

## Output and convergence

A completed run writes:

```text
outputs/mps_mac/mixing_layer_mps_mac.jld2
outputs/mps_mac/mixing_layer_mps_mac_vorticity.png
outputs/mps_mac/mixing_layer_mps_checkpoint.h5                 # manifest
outputs/mps_mac/mixing_layer_mps_checkpoint.h5.generation.*.h5 # up to 3 valid
outputs/mps_mac/run_status.txt
outputs/mps_mac/operator_cache/operators_<fingerprint>.h5
```

The JLD2 result contains exactly eight scheduled velocity, pressure,
vorticity, and scalar snapshots, terminal fields, per-step scalar mass and
projection diagnostics, sampled ceiling diagnostics, timing records,
threading provenance, configuration, and SHA-256 fingerprints.

Before quantitative use, repeat with a two-site predictor, `MPS_MAX_BOSON=5`,
bond dimension 96, cutoff `1e-12`, and `dt=0.00125`:

```bash
MPS_MAX_BOSON=5 julia --project=. mixing_layer_mps_mac.jl \
  --dt 0.00125 --maxdim 96 --cutoff 1e-12 --predictor-nsite 2 \
  --strict-quality --no-plot
```

Finite boson truncation means the Chorin projection is a convergent
approximation rather than an algebraic identity. A sampled ceiling occupation
or maximum bond dimension at its configured limit means the result is not
converged.
