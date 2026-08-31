# CPU/SLURM workflow

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
one-site displacement gates remove the arbitrary periodic pressure constant
from the MPS without changing velocity observables or pressure gradients.

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
