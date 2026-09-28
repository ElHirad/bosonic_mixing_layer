#!/bin/bash
#SBATCH --job-name=mf128_pressure_report
#SBATCH --output=%x.%j.log
#SBATCH --clusters=htc
#SBATCH --partition=htc
#SBATCH --constraint=turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}"
REPORT_DIR="${REPORT_DIR:-${PROJECT_ROOT}/outputs/reaction_128x128_re200_pe200_euler_series}"
REPORT_SOURCE_DIR="${REPORT_SOURCE_DIR:-${PROJECT_ROOT}/cluster_runs/pressure-report-${SLURM_JOB_ID}}"
MF_PLOT_SUPPORT="${MF_PLOT_SUPPORT:-${PROJECT_ROOT}/cluster_runs/mf-plot-support}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export MPLBACKEND=Agg MPLCONFIGDIR="${REPORT_SOURCE_DIR}/matplotlib-config"
if [[ -d "${MF_PLOT_SUPPORT}" ]]; then
    export PYTHONPATH="${MF_PLOT_SUPPORT}${PYTHONPATH:+:${PYTHONPATH}}"
fi
mkdir -p "${REPORT_SOURCE_DIR}" "${MPLCONFIGDIR}"
if [[ -f "${REPORT_SOURCE_DIR}/report_pressure_probes.py" ]]; then
    cmp "${PROJECT_ROOT}/report_pressure_probes.py" "${REPORT_SOURCE_DIR}/report_pressure_probes.py"
else
    cp "${PROJECT_ROOT}/report_pressure_probes.py" "${REPORT_SOURCE_DIR}/report_pressure_probes.py"
fi
# Submit with afterok dependencies on all diagnostic jobs. A rejected physical
# candidate still produces a diagnostic report; it is never labelled accepted.
srun --ntasks=1 --cpu-bind=cores /usr/bin/python3 \
    "${REPORT_SOURCE_DIR}/report_pressure_probes.py" "${REPORT_DIR}"
