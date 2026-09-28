#!/bin/bash
#SBATCH --job-name=mf64_reaction
#SBATCH --output=%x.%j.log
#SBATCH --clusters=htc
#SBATCH --partition=htc
#SBATCH --constraint=turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --mail-type=END,FAIL

set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}"
MF_DAMKOHLER="${MF_DAMKOHLER:?Set MF_DAMKOHLER to 1, 10, or 100}"
case "${MF_DAMKOHLER}" in
    1|10|100) ;;
    *) echo "This series requires MF_DAMKOHLER=1, 10, or 100" >&2; exit 64 ;;
esac
export MF_REYNOLDS=100 MF_PECLET=100
export MF_RUN_DIR="${MF_RUN_DIR:-/ix/jmendoza-arenas/hia21/mixing-layer/mean-field-sites64-re100-pe100-da${MF_DAMKOHLER}-three-species-euler-delta001875-nb12}"
export MF_OUTPUT_DIR="${MF_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/mean_field_sites_64x64_re100_pe100_da${MF_DAMKOHLER}_reaction_euler}"

# Separate, fingerprinted local-state trajectory for each reaction rate.
# The existing pipeline performs independent DNS and species plotting only
# after the mean-field trajectory is complete and validated.
exec bash "${PROJECT_ROOT}/hpc/mean_field64_cpu.sbatch" \
    --scalar-species 3 --damkohler "${MF_DAMKOHLER}" --time-integrator forward-euler "$@"
