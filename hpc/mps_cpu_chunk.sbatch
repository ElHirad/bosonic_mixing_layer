#!/bin/bash
#SBATCH --job-name=mixing_layer_32x32_Re_100
#SBATCH --output=%x.%j.log
#SBATCH --clusters=htc
#SBATCH --partition=htc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# The 32x32 cold operator build exceeded 64 GiB while serializing its cache.
# Match the high-memory probe allocation so the first cache write has headroom.
#SBATCH --mem=128G
#SBATCH --time=6-00:00:00
#SBATCH --signal=B:USR1@7200
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# MPS_RUN_DIR and MPS_CACHE_ROOT must be on persistent storage. Resubmit this
# same script with the same paths to continue from the latest checkpoint.
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}"
GRID="${GRID:-32}"
MPS_REYNOLDS="${MPS_REYNOLDS:-100}"
TRANSITION_THICKNESS="${TRANSITION_THICKNESS:-0.06}"
KH_WIDTH="${KH_WIDTH:-0.10}"
CASE_TAG="${CASE_TAG:-mps${GRID}_re${MPS_REYNOLDS}_delta${TRANSITION_THICKNESS}}"
RUN_ROOT="${MPS_RUN_DIR:-${SLURM_SUBMIT_DIR}/cluster_runs/${CASE_TAG}}"
CACHE_ROOT="${MPS_CACHE_ROOT:-${SLURM_SUBMIT_DIR}/cluster_runs/operator_cache}"
CHECKPOINT="${RUN_ROOT}/mixing_layer_mps_checkpoint.h5"
RUN_STATUS="${RUN_ROOT}/run_status.txt"
STOP_FILE="${RUN_ROOT}/STOP_REQUESTED"
STOP_AFTER_SECONDS="${STOP_AFTER_SECONDS:-518400}"
SHUTDOWN_RESERVE_SECONDS="${SHUTDOWN_RESERVE_SECONDS:-7200}"
EXPECTED_JULIA_VERSION="${EXPECTED_JULIA_VERSION:-1.12.6}"
MPS_MAX_BOSON="${MPS_MAX_BOSON:-4}"
JULIA_MINOR_VERSION="${EXPECTED_JULIA_VERSION%.*}"

# Prefer the exact Juliaup installation so a later default-channel change does
# not alter a checkpointed run. JULIA_CMD remains available for site overrides.
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

mkdir -p "${RUN_ROOT}" "${CACHE_ROOT}"
command -v flock >/dev/null
exec 9>"${RUN_ROOT}/.trajectory.lock"
flock -n 9 || {
    echo "Another job owns ${RUN_ROOT}; refusing concurrent checkpoint writers." >&2
    exit 73
}
rm -f "${STOP_FILE}"

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

resume_arguments=()
if [[ -f "${CHECKPOINT}" || -f "${CHECKPOINT}.previous" ]] || \
   compgen -G "${CHECKPOINT}.generation.*.h5" >/dev/null; then
    resume_arguments=(--resume "${CHECKPOINT}")
    echo "Resuming ${CHECKPOINT}"
else
    echo "Starting a new trajectory"
fi

echo "project=${PROJECT_ROOT}"
echo "grid=${GRID} reynolds=${MPS_REYNOLDS} transition=${TRANSITION_THICKNESS} kh_width=${KH_WIDTH}"
echo "run=${RUN_ROOT}"
echo "cache=${CACHE_ROOT}"
echo "cpus=${SLURM_CPUS_PER_TASK} application_wall_limit=${STOP_AFTER_SECONDS}s"
echo "shutdown_reserve=${SHUTDOWN_RESERVE_SECONDS}s max_boson=${MPS_MAX_BOSON}"
echo "depot=${JULIA_DEPOT_PATH}"
"${JULIA_CMD}" --version

request_stop() {
    echo "Scheduler stop signal received; requesting a checkpoint at the next Chorin boundary."
    : > "${STOP_FILE}"
}
trap request_stop USR1 TERM

solver_status=0
srun --ntasks=1 --cpu-bind=cores /usr/bin/time -v \
    "${JULIA_CMD}" --threads="${SLURM_CPUS_PER_TASK}" \
    --startup-file=no \
    --project="${PROJECT_ROOT}" \
    "${PROJECT_ROOT}/mixing_layer_mps_mac.jl" \
    --n "${GRID}" \
    --re "${MPS_REYNOLDS}" \
    --dt 0.0025 \
    --final-time 3.5 \
    --transition "${TRANSITION_THICKNESS}" \
    --kh-width "${KH_WIDTH}" \
    --predictor-chunks 8 \
    --poisson-blocks 12 \
    --output-dir "${RUN_ROOT}" \
    --cache-dir "${CACHE_ROOT}" \
    --require-cache \
    --checkpoint "${CHECKPOINT}" \
    --checkpoint-interval 25 \
    --ceiling-interval 10 \
    --progress-interval 1 \
    --stop-after-seconds "${STOP_AFTER_SECONDS}" \
    --shutdown-reserve-seconds "${SHUTDOWN_RESERVE_SECONDS}" \
    --stop-file "${STOP_FILE}" \
    --run-status "${RUN_STATUS}" \
    --blas-threads 1 \
    --strided-threads "${SLURM_CPUS_PER_TASK}" \
    --strict-quality \
    --no-plot \
    "${resume_arguments[@]}" &
solver_pid=$!

# `wait` can be interrupted by USR1 while the job step is still alive.
while true; do
    set +e
    wait "${solver_pid}"
    solver_status=$?
    set -e
    kill -0 "${solver_pid}" 2>/dev/null || break
done
(( solver_status == 0 )) || exit "${solver_status}"

status_state="$(awk -F= '$1 == "state" { print $2 }' "${RUN_STATUS}")"
if [[ "${status_state}" == "complete" ]]; then
    "${JULIA_CMD}" --startup-file=no --project="${PROJECT_ROOT}" \
        "${PROJECT_ROOT}/hpc/validate_results.jl" \
        "${RUN_ROOT}/mixing_layer_mps_mac.jld2"
    echo "Trajectory complete and validated."
    exit 0
fi

echo "Chunk checkpointed cleanly but the trajectory is incomplete; resubmit the same RUN_ROOT."
exit 75
