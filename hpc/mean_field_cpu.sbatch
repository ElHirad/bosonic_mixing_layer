#!/bin/bash
#SBATCH --job-name=mixing_layer_mean_field
#SBATCH --output=%x.%j.log
#SBATCH --clusters=htc
#SBATCH --partition=htc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --mail-type=END,FAIL

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}"
MF_CUTOFF="${MF_CUTOFF:-12}"
MF_PRESSURE_CFL="${MF_PRESSURE_CFL:-0.125}"
MF_SUBSTEPS="${MF_SUBSTEPS:-8}"
MF_RUN_DIR="${MF_RUN_DIR:-/ix/jmendoza-arenas/hia21/mixing-layer/mean-field-sites16-re50-pe50-single-freeslip-delta004-largekh-nb${MF_CUTOFF}}"
MF_PYTHON="${MF_PYTHON:-/usr/bin/python3}"
MF_STOP_FILE="${MF_RUN_DIR}/STOP_REQUESTED.${SLURM_JOB_ID}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

mkdir -p "${MF_RUN_DIR}/source"
# Freeze the exact three production modules for reproducible runs/restarts.
# Existing frozen source is never overwritten by a resubmission.
for source in mixing_layer_mean_field.py mean_field_bosons.py mean_field_operators.py; do
    if [[ -f "${MF_RUN_DIR}/source/${source}" ]]; then
        cmp "${PROJECT_ROOT}/${source}" "${MF_RUN_DIR}/source/${source}" || {
            echo "Source changed: use a new MF_RUN_DIR, or submit from the frozen source directory." >&2
            exit 72
        }
    else
        cp "${PROJECT_ROOT}/${source}" "${MF_RUN_DIR}/source/${source}"
    fi
done
resume_arguments=()
if [[ -f "${MF_RUN_DIR}/checkpoint.npz" ]]; then
    resume_arguments=(--resume)
fi
request_stop() {
    touch "${MF_STOP_FILE}"
    echo "Stop requested: checkpoint at the next physical-step boundary."
}
trap request_stop USR1 TERM

echo "job=${SLURM_JOB_ID} node=${SLURMD_NODENAME} output=${MF_RUN_DIR}"
"${MF_PYTHON}" --version
srun --ntasks=1 --cpu-bind=cores /usr/bin/time -v "${MF_PYTHON}" \
    "${MF_RUN_DIR}/source/mixing_layer_mean_field.py" \
    --n 16 --re 50 --pe 50 --dt 0.0025 --final-time 0.65 \
    --boundary-y free-slip --shear-center 0.5 --transition 0.04 --kh-width 0.12 \
    --kh-mode 2 --kh-amplitude 2.5 --secondary-mode 1 --secondary-amplitude 0.5 --phase 0 \
    --velocity-scale 8 --pressure-scale 4 --scalar-scale 4 \
    --boson-cutoff "${MF_CUTOFF}" --pressure-cfl "${MF_PRESSURE_CFL}" \
    --predictor-substeps "${MF_SUBSTEPS}" --correction-substeps "${MF_SUBSTEPS}" \
    --output-dir "${MF_RUN_DIR}" --checkpoint-interval 10 --stop-file "${MF_STOP_FILE}" \
    "${resume_arguments[@]}" "$@" &
solver_pid=$!
solver_status=0
while kill -0 "${solver_pid}" 2>/dev/null; do
    wait "${solver_pid}" && solver_status=0 || solver_status=$?
done
[[ "${solver_status}" -eq 0 ]] || exit "${solver_status}"
if [[ -f "${MF_RUN_DIR}/mean_field_snapshots.npz" ]]; then
    "${MF_PYTHON}" "${MF_RUN_DIR}/source/mixing_layer_mean_field.py" \
        --validate-results "${MF_RUN_DIR}/mean_field_snapshots.npz"
else
    echo "Partial trajectory only; inspect run_status.json before resuming."
fi
