#!/bin/bash
#SBATCH --job-name=mf128_reaction
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
export MF_REYNOLDS=200 MF_PECLET=200
# Match the successful startup probes, even if an older preset was exported.
export MF_CUTOFF=12 MF_PRESSURE_CFL=0.125 MF_SUBSTEPS=8
export MF_RUN_DIR="${MF_RUN_DIR:-/ix/jmendoza-arenas/hia21/mixing-layer/mean-field-sites128-re200-pe200-da${MF_DAMKOHLER}-three-species-euler-delta001875-nb12}"
export MF_OUTPUT_DIR="${MF_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/mean_field_sites_128x128_re200_pe200_da${MF_DAMKOHLER}_reaction_euler}"

# Preserve the tested solver unchanged. The shared pipeline freezes sources,
# resumes matching checkpoints, validates the full trajectory, and only then
# runs the independent DNS benchmark and plotting. These are not MF stages.
# Every accepted physical step is checkpointed. The early stop signal allows
# the current step to finish; resubmitting the same preset resumes its kets.
exec bash "${PROJECT_ROOT}/hpc/mean_field32_cpu.sbatch" \
    --n 128 --re 200 --pe 200 --transition 0.01875 \
    --dt 0.0003125 --final-time 0.65 \
    --scalar-species 3 --damkohler "${MF_DAMKOHLER}" \
    --time-integrator forward-euler --pressure-tolerance 1e-8 \
    --pressure-max-steps 192000 --checkpoint-interval 1 "$@"
