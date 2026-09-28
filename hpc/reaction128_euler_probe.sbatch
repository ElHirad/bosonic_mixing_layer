#!/bin/bash
#SBATCH --job-name=mf128_euler_probe
#SBATCH --output=%x.%j.log
#SBATCH --clusters=htc
#SBATCH --partition=htc
#SBATCH --constraint=turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00

set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}"
export PROBE_OUTPUT="${PROBE_OUTPUT:-${PROJECT_ROOT}/outputs/reaction_128x128_re200_pe200_euler_series}"
# The finer grid reduces pseudo_dt by four. Four times the iteration cap
# preserves the previous maximum pseudo-time, without relaxing tolerance.
# Run separate --damkohler values in separate Slurm allocations if desired.
exec bash "${PROJECT_ROOT}/hpc/reaction_euler_probe.sbatch" \
    --n 128 --re 200 --pe 200 --dt .0003125 --pressure-max-steps 192000 "$@"
