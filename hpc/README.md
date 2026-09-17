# CPU/SLURM workflow

## Current explicit single-site mean-field calculation

### 64×64 thinner-shear preset

The completed Re=Pe=50 job 11217985 took 26 hours 2 minutes including its
automatic DNS comparison and plotting. The completed **Re=Pe=200** case keeps its
grid, thickness, timestep, KH seeds, cutoff, and all integration/gate settings:

```bash
sbatch --clusters=htc --export=ALL,MF_REYNOLDS=200,MF_PECLET=200 hpc/mean_field64_cpu.sbatch
```

Both viscosity and scalar diffusivity decrease from 0.02 to 0.005. Default
run/output paths include `re200-pe200` / `re200_pe200`, so completed Re=Pe=50
data are preserved. If you have exported `MF_RUN_DIR` or `MF_OUTPUT_DIR`,
override those too: never reuse a directory for changed physical parameters.
The same automatic validation and separate matched DNS/plotting pipeline is
used. Job **11244737** completed all 1,040 steps and post-processing in
**24 hours 36 minutes**, excluding its earlier three-step preflight. See the
[completed higher-Re/Pe report](../outputs/mean_field_sites_64x64_re200_pe200_single_freeslip_delta001875/RUN_REPORT.md)
and [launch history](../outputs/mean_field_sites_64x64_re200_pe200_single_freeslip_delta001875/LAUNCH_NOTES.md).
Roll-up and mutual vortex rotation are clearer, but two cores remain at t=0.65.
Final vorticity differs from the matched DNS by 11.71% in relative L2; passing
the acceptance gates does not establish quantitative or convergence accuracy.
The archived DNS preview is not a mean-field result.

For the original Re=Pe=50 configuration:

```bash
# Three-step timing and numerical preflight; saves resumable local kets.
sbatch --clusters=htc --job-name=mf64_preflight --time=01:00:00 --signal=B:USR1@600 hpc/mean_field64_cpu.sbatch --max-steps 3 --checkpoint-interval 1
# Continue the same configuration after the preflight has finished and passed.
sbatch --clusters=htc hpc/mean_field64_cpu.sbatch
```

The 64×64 preset keeps Re=Pe=50, one free-slip/no-flux-y shear layer, and KH
amplitudes 2.5/0.5 with width 0.12. Tanh thickness is 0.01875, reduced from
0.025. The initial concentration 10–90% transition spans about 2.64 cells
(previously 1.76); it remains a thin, not demonstrably converged profile.
The physical timestep is halved to 0.000625 (1,040 steps to time 0.65), below
the initial advective bound of approximately 0.001032.

There are 16,448 local Fock vectors, each of dimension 13. The pressure
pseudo-step remains 0.125/64² and its residual tolerance remains 1e-8. Only
the maximum allowed pressure iterations increase from 12,000 to 48,000 to
allow for the finer grid's slower relaxation. Eight RK4 subdivisions are
retained in both predictor and correction. No solver equations, dynamical
source files, acceptance tolerances, or scalar corrections are changed.

The script requests one `turin` CPU, 4 GiB, and 48 hours, with a checkpoint
stop signal 15 minutes before the limit. The allocation is headroom, not a
measured runtime prediction. The raw scaling estimate is around 32 times
the 32×32 runtime (four times the cells, about four times the pressure
iterations per step, and twice the physical steps); use the cluster preflight
for an actual estimate. Checkpoints are saved every five accepted steps.

Preflight job 11217973 passed three steps in about 310 seconds of stepping
time on `htc-n86`; warm steps took 88–89 seconds and `/usr/bin/time` recorded
about 118 MiB maximum resident memory. This suggested about 25–30 hours for
the full trajectory. Production job 11217985 resumed at step 3 and completed
in 26 hours 2 minutes. All 42 Python tests pass. Preflight measurements are
startup checks; the [completed Re=Pe=50 report](../outputs/mean_field_sites_64x64_re50_pe50_single_freeslip_delta001875/RUN_REPORT.md)
links the full-trajectory validation and independent DNS comparison.

This wrapper reuses the tested 32×32 batch workflow for environment checks,
frozen sources, restart, validation, and automatic independent DNS comparison
and plotting; its final CLI arguments select the actual 64×64 configuration.
Results go to `outputs/mean_field_sites_64x64_re50_pe50_single_freeslip_delta001875`.
`MF_RUN_DIR` and `MF_OUTPUT_DIR` can override the default directories.
Optional `MF_REYNOLDS` and `MF_PECLET` select a different physical case and
separate default paths; Pe defaults to Re when only Re is supplied. Do not
reuse a run directory for changed parameters. Changing Re/Pe is a separate
physics choice, not a requirement for increasing the grid size.

Re=Pe=50 remains strongly damped in the preceding calculation. More cells and
thinner initial shear do not establish better roll-up or pairing by themselves;
inspect the completed spatial fields rather than just the Fourier-mode ratio.

### 32×32 thinner-shear preset

```bash
sbatch --clusters=htc hpc/mean_field32_cpu.sbatch
```

This preset retains Re=Pe=50, the single free-slip-y layer, and KH amplitudes
2.5/0.5. It increases the grid to 32×32, reduces tanh thickness from 0.04 to
0.025, and halves the physical timestep to 0.00125 (520 steps to time 0.65).
The 32×32 initial advective safety bound is approximately 0.00210, so the
old physical timestep 0.0025 is not used.

A ten-step HTC preflight took 67 seconds of solver time on one CPU and passed
every gate. Budget approximately one hour for the full trajectory, depending
on compute-node performance; the script requests the same `turin` CPU type
as that preflight, four hours and 2 GiB for headroom and checkpoint/restart.
An older `cascade_lake` node took 187 seconds for ten resumed steps, about
three times longer. Override the scheduler constraint if desired, allowing
for the different runtime. The script auto-resumes a matching checkpoint.
No dynamical mean-field source changes or relaxed tolerances are needed.

After the complete mean-field result validates, the batch script runs the
**separate** matched DNS benchmark and generates all plots plus `RUN_REPORT.md`
under `outputs/mean_field_sites_32x32_re50_pe50_single_freeslip_delta0025`.
Set `MF_OUTPUT_DIR` to publish elsewhere. `postprocess_status.json` distinguishes
successful plotting from incomplete or failed post-processing. A partial
solver trajectory is retained as a checkpoint and is not presented as a final
result. Both the solver and post-processing sources are frozen in `MF_RUN_DIR`.

The plotting environment needs NumPy, SciPy, Matplotlib and their dependencies
available on compute nodes. `MF_PLOT_SUPPORT` optionally adds a persistent
supplemental package directory to `PYTHONPATH` (default
`cluster_runs/mf-plot-support`). The full plotting import chain and an in-memory
PNG render are checked on the compute node before the long run. This cluster's
login and compute nodes have different system plotting dependencies, so the
supplemental packages are installed in that shared, Git-ignored directory:

```bash
python3 -m pip install --target cluster_runs/mf-plot-support --only-binary=:all: --no-deps -r hpc/requirements-plot-support.txt
```

These pins supplement the existing Python 3.9 / NumPy 1.23.5 / SciPy 1.11.2 /
Matplotlib 3.7.1 installation. They do not replace the solver's numerical
packages or alter its checkpoint fingerprint. Nothing is downloaded during
the batch job.

Thinner shear and more cells do not guarantee sharp, sustained pairing at
Re=50. Inspect the actual vorticity/concentration fields together with mode
diagnostics. The large seeds test finite-amplitude interaction, and the
initial transition remains only a few cells wide; this is not a spatially
converged calculation.

### Original 16×16 preset

Use `mean_field_cpu.sbatch` for the all-stage single-site operator solver:

```bash
sbatch --clusters=htc hpc/mean_field_cpu.sbatch
sbatch --clusters=htc --export=ALL,MF_CUTOFF=16 hpc/mean_field_cpu.sbatch
```

The default case is 16×16, Re=Pe=50, one free-slip-y shear layer, thickness
0.04, KH width 0.12, amplitudes 2.5 and 0.5, and 260 steps to time 0.65.
It requests one CPU, 2 GiB, and one hour. No Julia/MPS environment or operator
cache is needed; Python, NumPy, and SciPy must already be installed.

Set `MF_RUN_DIR` for persistent results, `MF_CUTOFF` for maximum local boson
occupation (default 12), `MF_PRESSURE_CFL` for the pressure pseudo-time factor
(default 0.125), and `MF_SUBSTEPS` for predictor/correction RK4 subdivisions
(default 8). Use a **different** directory for changed parameters. For a
local-integration convergence check, halve `MF_PRESSURE_CFL` and double
`MF_SUBSTEPS`; do not change the physical case.

The batch script freezes the production modules, auto-resumes matching local
state checkpoints, and validates complete results from saved Fock vectors.
The solver has no DNS or direct pressure-solve stage. See
[the full operator algorithm](../SINGLE_SITE_MEAN_FIELD_README.md).

## Historical MPS/TDVP workflow

The MPS solver is single-process and single-node. It does not use MPI or Julia
workers. Request one SLURM task and use physical CPU cores within that task.

## Prepare the environment once

On a login or build node with package access, use Julia 1.12.6 and a persistent
depot. The supplied batch scripts default to the depot shown here:

```bash
export PATH="$HOME/.juliaup/bin:$PATH"
export JULIA_DEPOT_PATH="$PWD/cluster_runs/julia-depot-1.12"
export MPS_MAX_BOSON=4
julia +1.12.6 --project=. --startup-file=no -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
JULIA_PKG_OFFLINE=true julia +1.12.6 --project=. --startup-file=no test/runtests.jl
```

Compute jobs set `JULIA_PKG_OFFLINE=true`; they must not resolve or install
packages. The scripts require the persistent `JULIA_DEPOT_PATH`, assert Julia
1.12.6, and pin/log `MPS_MAX_BOSON` (default 4). If `JULIA_DEPOT_PATH` is not
exported, they use `cluster_runs/julia-depot-1.12` below the project root. Keep
`Project.toml`, `Manifest.toml`, `mixing_layer_mps_mac.jl`, thread counts,
strict mode, and `MPS_MAX_BOSON` unchanged while a checkpointed trajectory is
active. Stable SHA-256 fingerprints reject incompatible restarts.

## Measure before production

Start with cold and warm operator-cache probes:

```bash
sbatch --export=ALL,GRID=8,STEPS=0 hpc/mps_cpu_probe.sbatch
sbatch --export=ALL,GRID=16,STEPS=0 hpc/mps_cpu_probe.sbatch
sbatch --export=ALL,GRID=16,STEPS=1 hpc/mps_cpu_probe.sbatch
```

Then request a high-memory node for `GRID=32,STEPS=0`, followed by one and five
steps. Record elapsed time and `MaxRSS` with:

```bash
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed,AllocCPUS,ReqMem,MaxRSS
```

The default predictor is an exact sum of eight row chunks per algebraic family.
This greatly reduces construction memory. Benchmark `--predictor-chunks 4`,
`8`, and `16` on the target machine; changing it creates a distinct operator
cache and checkpoint fingerprint.

Pressure relaxation stops as soon as its residual reaches tolerance, with a
default safety cap of 12 blocks. After relaxation and correction, uniform
one-site displacement gates remove the arbitrary periodic/Neumann pressure
constant from the MPS without changing velocity observables or pressure
gradients.

The scalar predictor similarly removes only its small uniform mass-error mode.
The pre-projection correction is recorded and strictly limited to `1e-4`, so
the exact conserved mean does not mask loss of scalar resolution.

The code exposes two non-nested threading modes. The supplied scripts start
with Julia/Strided threading and one BLAS thread. Also benchmark the converse
(`--threads=1 --blas-threads=N --strided-threads=1`) before choosing production
resources.

The CPU default is `--predictor-nsite 1`. It was 4.27 times faster than the
two-site predictor in a warm local `8x8` step, with a `4.3e-9` relative field
difference. Include `--predictor-nsite 2` in the convergence campaign because
that update can grow bonds if finite boson truncation creates entanglement.

## Chunked production

Set persistent run and cache directories, then submit:

```bash
export MPS_RUN_DIR=/cluster/scratch/$USER/mixing-layer/mps32-re100
export MPS_CACHE_ROOT=/cluster/scratch/$USER/mixing-layer/operator-cache
export MPS_REYNOLDS=100
export MPS_PECLET=100
sbatch hpc/mps_cpu_chunk.sbatch
```

The batch template also accepts `MPS_DT`, `MPS_FINAL_TIME`, `BOUNDARY_Y`,
`SHEAR_CENTER`, `TRANSITION_THICKNESS`, `KH_WIDTH`, `KH_MODE`, `KH_AMPLITUDE`,
`SECONDARY_MODE`, `SECONDARY_AMPLITUDE`, `KH_PHASE`, `VELOCITY_SCALE`, and
`CHECKPOINT_INTERVAL`. For the 16x16 free-slip pairing/concentration case,
use:

```bash
export GRID=16 MPS_REYNOLDS=50 MPS_PECLET=50 MPS_FINAL_TIME=0.65
export BOUNDARY_Y=free-slip TRANSITION_THICKNESS=0.04 KH_WIDTH=0.12
export KH_MODE=2 KH_AMPLITUDE=2.5 SECONDARY_MODE=1 SECONDARY_AMPLITUDE=0.5
export KH_PHASE=0 VELOCITY_SCALE=8
sbatch hpc/mps_cpu_chunk.sbatch
```

The batch script takes a nonblocking `flock` on the run directory, so duplicate
submissions cannot write the same trajectory. It stops starting new steps after
23 hours of a 24-hour allocation, and SLURM sends USR1 one hour before the hard
limit. The solver finishes the current Chorin step, writes an immutable,
checksummed HDF5 generation, saves partial arrays, and exits. This reserve must
exceed the measured worst step plus checkpoint/output time from the probes; no
application can checkpoint after an unconditional scheduler kill.

Resubmit the same command to continue. A partial chunk exits with code `75`,
while exit `0` is reserved for a completed result that passes
`hpc/validate_results.jl`. Three valid checkpoint generations are retained.
Do not place them only in `$SLURM_TMPDIR`; that directory disappears with the
job. Filesystems without reliable POSIX `flock`, atomic rename, and `fsync`
semantics need a site-specific checkpoint directory.

If `MPS_PECLET` is omitted, the script sets `Pe=Re`. The job uses
`--strict-quality`, so a failed projection, scalar-mass conservation, scalar
stage-isolation check, boson ceiling, pressure gauge, correction map,
imaginary-amplitude, or bond-dimension gate returns a nonzero exit instead of
silently producing an unconverged result. Plotting is deliberately disabled on
compute nodes and can be done afterward from the final JLD2 arrays.
