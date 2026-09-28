#!/bin/bash
#SBATCH --job-name=mf128_reaction_ptol1e7
#SBATCH --output=%x.%j.log
#SBATCH --clusters=htc
#SBATCH --partition=htc
#SBATCH --qos=htc-htc-ll
#SBATCH --constraint=turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=21-00:00:00
#SBATCH --signal=B:USR1@3600
#SBATCH --mail-type=END,FAIL

set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}"
MF_DAMKOHLER="${MF_DAMKOHLER:?Set MF_DAMKOHLER to 1, 10, or 100}"
case "${MF_DAMKOHLER}" in
    1|10|100) ;;
    *) echo "This series requires MF_DAMKOHLER=1, 10, or 100" >&2; exit 64 ;;
esac
export MF_RUN_DIR="${MF_RUN_DIR:-/ix/jmendoza-arenas/hia21/mixing-layer/mean-field-sites128-re200-pe200-da${MF_DAMKOHLER}-three-species-euler-delta001875-nb12-ptol1e7}"
export MF_OUTPUT_DIR="${MF_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/mean_field_sites_128x128_re200_pe200_da${MF_DAMKOHLER}_reaction_euler_ptol1e7}"

# Only the pressure stopping tolerance changes. Retain every other parameter
# and numerical gate, including explicit local kets and forward Euler.
# Distinct directories prevent incompatible restarts from the 1e-8 series.
exec bash "${PROJECT_ROOT}/hpc/mean_field128_reaction_cpu.sbatch" \
    --pressure-tolerance 1e-7 "$@"
