#!/bin/bash
#SBATCH --job-name=mixing_layer_mean_field64
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
MF_REYNOLDS="${MF_REYNOLDS:-50}"
MF_PECLET="${MF_PECLET:-${MF_REYNOLDS}}"
export MF_RUN_DIR="${MF_RUN_DIR:-/ix/jmendoza-arenas/hia21/mixing-layer/mean-field-sites64-re${MF_REYNOLDS}-pe${MF_PECLET}-single-freeslip-delta001875-largekh-nb12}"
export MF_OUTPUT_DIR="${MF_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/mean_field_sites_64x64_re${MF_REYNOLDS}_pe${MF_PECLET}_single_freeslip_delta001875}"

# Reuse the tested environment checks, frozen sources, checkpoint/restart,
# validation, and independent DNS/plotting pipeline. Only this script's
# SBATCH directives apply; the nested batch files run as ordinary shell code.
# exec preserves signal forwarding to the physical-step checkpoint boundary.
# The larger iteration cap does not change the 1e-8 pressure tolerance or
# the 0.125/n^2 pseudo-step, and introduces no classical pressure solve.
exec bash "${PROJECT_ROOT}/hpc/mean_field32_cpu.sbatch" \
    --n 64 --re "${MF_REYNOLDS}" --pe "${MF_PECLET}" \
    --transition 0.01875 --dt 0.000625 --final-time 0.65 \
    --pressure-max-steps 48000 --checkpoint-interval 5 "$@"
