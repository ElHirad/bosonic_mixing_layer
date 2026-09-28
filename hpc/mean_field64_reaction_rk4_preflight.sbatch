#!/bin/bash
#SBATCH --job-name=mf64_rk4_preflight
#SBATCH --output=%x.%j.log
#SBATCH --clusters=htc
#SBATCH --partition=htc
#SBATCH --constraint=turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --mail-type=END,FAIL

set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}"
MF_DAMKOHLER="${MF_DAMKOHLER:?Set MF_DAMKOHLER to 1, 10, or 100}"
case "${MF_DAMKOHLER}" in
    1|10|100) ;;
    *) echo "This series requires MF_DAMKOHLER=1, 10, or 100" >&2; exit 64 ;;
esac
export MF_RUN_DIR="${MF_RUN_DIR:-/ix/jmendoza-arenas/hia21/mixing-layer/mean-field-sites64-re100-pe100-da${MF_DAMKOHLER}-three-species-rk4-delta001875-nb12-ptol1e7}"
export MF_PYTHON="${MF_PYTHON:-/usr/bin/python3}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
DNS_PREFLIGHT_OUTPUT="${DNS_PREFLIGHT_OUTPUT:-${SLURM_SUBMIT_DIR}/outputs/reaction_64x64_re100_pe100_rk4_series}"
if [[ "$#" -ne 0 ]]; then
    echo "This preflight accepts no CLI overrides: ten RK4 steps at pressure tolerance 1e-7." >&2
    exit 64
fi

# Independent full-time DNS screening, with per-step signed-species CSV.
# It is not used in the MF algorithm or as a replacement for MF validation.
"${MF_PYTHON}" "${PROJECT_ROOT}/reaction_dns_preflight.py" "${DNS_PREFLIGHT_OUTPUT}" \
    --n 64 --re 100 --pe 100 --time-integrator rk4 --dt .000625 \
    --pressure-max-steps 48000 --pressure-tolerance 1e-7 --damkohler "${MF_DAMKOHLER}"

bash "${PROJECT_ROOT}/hpc/mean_field64_reaction_rk4_cpu.sbatch" --max-steps 10 &
wrapper_pid=$!
forward_stop() { kill -USR1 "${wrapper_pid}" 2>/dev/null || true; }
trap forward_stop USR1 TERM
solver_status=0
while true; do
    wait "${wrapper_pid}" && solver_status=0 || solver_status=$?
    kill -0 "${wrapper_pid}" 2>/dev/null || break
done
[[ "${solver_status}" -eq 0 ]] || exit "${solver_status}"

# Exit zero only after ten genuinely accepted steps, not a graceful early stop.
"${MF_PYTHON}" - "${MF_RUN_DIR}" "${MF_DAMKOHLER}" <<'PY'
import json
from pathlib import Path
import sys
import numpy as np

root = Path(sys.argv[1])
status = json.loads((root/"run_status.json").read_text())
if (status.get("state") != "partial" or status.get("completed_step") != 10
        or status.get("requested_steps") != 1040):
    raise RuntimeError(f"Ten-step preflight did not pass: {status}")
with np.load(root/"checkpoint.npz", allow_pickle=False) as checkpoint:
    if int(checkpoint["completed_step"]) != 10 or int(checkpoint["requested_steps"]) != 1040:
        raise RuntimeError("Preflight checkpoint does not contain ten accepted steps")
    history = json.loads(str(checkpoint["history_json"]))
    if [row["step"] for row in history] != list(range(1, 11)):
        raise RuntimeError("Preflight accepted-step history is incomplete")
    config = json.loads(str(checkpoint["config_json"]))
    expected = dict(n=64, reynolds=100., peclet=100., scalar_species=3,
                    damkohler=float(sys.argv[2]), dt=.000625, final_time=.65,
                    time_integrator="rk4", pressure_tolerance=1e-7)
    if any(config.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Preflight checkpoint has an unexpected configuration")
print("PREFLIGHT_PASSED: ten consecutive accepted RK4 steps; checkpoint ready for production.")
PY
