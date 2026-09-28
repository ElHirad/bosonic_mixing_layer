#!/bin/bash
#SBATCH --job-name=mf64_euler_probe
#SBATCH --output=%x.%j.log
#SBATCH --clusters=htc
#SBATCH --partition=htc
#SBATCH --constraint=turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}"
PROBE_DIR="${PROBE_DIR:-${PROJECT_ROOT}/cluster_runs/reaction-euler-probe-${SLURM_JOB_ID}}"
PROBE_OUTPUT="${PROBE_OUTPUT:-${PROJECT_ROOT}/outputs/reaction_64x64_re100_pe100_euler_series}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "${PROBE_DIR}/source" "${PROBE_OUTPUT}"
for source in mixing_layer_mean_field.py mean_field_bosons.py mean_field_operators.py reaction_euler_mf_probe.py; do
    if [[ -e "${PROBE_DIR}/source/${source}" ]]; then
        cmp "${PROJECT_ROOT}/${source}" "${PROBE_DIR}/source/${source}"
    else
        cp "${PROJECT_ROOT}/${source}" "${PROBE_DIR}/source/${source}"
    fi
done
srun --ntasks=1 --cpu-bind=cores /usr/bin/python3 "${PROBE_DIR}/source/reaction_euler_mf_probe.py" "${PROBE_OUTPUT}" "$@"
