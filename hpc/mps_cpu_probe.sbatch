#!/bin/bash
#SBATCH --job-name=mps-mac-probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=mps-mac-probe-%j.out
#SBATCH --error=mps-mac-probe-%j.err

set -euo pipefail

# Override these with, for example:
# sbatch --export=ALL,GRID=16,STEPS=1 hpc/mps_cpu_probe.sbatch
GRID="${GRID:-16}"
STEPS="${STEPS:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}"
RESULT_ROOT="${RESULT_ROOT:-${SLURM_SUBMIT_DIR}/cluster_runs/probe_${GRID}_${SLURM_JOB_ID}}"
CACHE_ROOT="${MPS_CACHE_ROOT:-${SLURM_SUBMIT_DIR}/cluster_runs/operator_cache}"
EXPECTED_JULIA_VERSION="${EXPECTED_JULIA_VERSION:-1.12.6}"
MPS_MAX_BOSON="${MPS_MAX_BOSON:-4}"
JULIA_MINOR_VERSION="${EXPECTED_JULIA_VERSION%.*}"

# Prefer the exact Juliaup installation so a later default-channel change does
# not alter benchmark results. JULIA_CMD remains available for site overrides.
export PATH="$HOME/.juliaup/bin:$PATH"
JULIA_CMD="${JULIA_CMD:-$HOME/.julia/juliaup/julia-${EXPECTED_JULIA_VERSION}+0.x64.linux.gnu/bin/julia}"
if [[ ! -x "${JULIA_CMD}" ]]; then
    JULIA_CMD="$(command -v julia)"
fi

export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-${PROJECT_ROOT}/cluster_runs/julia-depot-${JULIA_MINOR_VERSION}}"
[[ -d "${JULIA_DEPOT_PATH%%:*}" ]] || {
    echo "Julia depot does not exist: ${JULIA_DEPOT_PATH%%:*}" >&2
    exit 72
}

mkdir -p "${RESULT_ROOT}" "${CACHE_ROOT}"

export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JULIA_PKG_OFFLINE=true
export JULIA_PKG_PRECOMPILE_AUTO=0
export MPS_MAX_BOSON
export EXPECTED_JULIA_VERSION

"${JULIA_CMD}" --startup-file=no --project="${PROJECT_ROOT}" -e '
    using Pkg
    VERSION == VersionNumber(ENV["EXPECTED_JULIA_VERSION"]) ||
        error("expected Julia $(ENV["EXPECTED_JULIA_VERSION"]), got $VERSION")
    Pkg.instantiate(; update_registry=false, allow_autoprecomp=false)
'

echo "project=${PROJECT_ROOT}"
echo "results=${RESULT_ROOT}"
echo "cache=${CACHE_ROOT}"
echo "grid=${GRID} steps=${STEPS} cpus=${SLURM_CPUS_PER_TASK}"
echo "max_boson=${MPS_MAX_BOSON} depot=${JULIA_DEPOT_PATH}"
"${JULIA_CMD}" --version

srun --ntasks=1 --cpu-bind=cores /usr/bin/time -v \
    "${JULIA_CMD}" --threads="${SLURM_CPUS_PER_TASK}" \
    --startup-file=no \
    --project="${PROJECT_ROOT}" \
    "${PROJECT_ROOT}/mixing_layer_mps_mac.jl" \
    --n "${GRID}" \
    --steps "${STEPS}" \
    --output-dir "${RESULT_ROOT}" \
    --cache-dir "${CACHE_ROOT}" \
    --require-cache \
    --checkpoint "${RESULT_ROOT}/checkpoint.h5" \
    --checkpoint-interval 1 \
    --ceiling-interval 1 \
    --progress-interval 1 \
    --blas-threads 1 \
    --strided-threads "${SLURM_CPUS_PER_TASK}" \
    --strict-quality \
    --no-plot

echo "Probe completed. Inspect MaxRSS in the time output and with:"
echo "sacct -j ${SLURM_JOB_ID} --format=JobID,State,ExitCode,Elapsed,AllocCPUS,ReqMem,MaxRSS"
