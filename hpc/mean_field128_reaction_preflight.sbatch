#!/bin/bash
#SBATCH --job-name=mf128_reaction_preflight
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
export MF_RUN_DIR="${MF_RUN_DIR:-/ix/jmendoza-arenas/hia21/mixing-layer/mean-field-sites128-re200-pe200-da${MF_DAMKOHLER}-three-species-euler-delta001875-nb12-ptol1e7}"
export MF_PYTHON="${MF_PYTHON:-/usr/bin/python3}"
if [[ "$#" -ne 0 ]]; then
    echo "This preflight accepts no CLI overrides: exactly five steps at pressure tolerance 1e-7." >&2
    exit 64
fi

bash "${PROJECT_ROOT}/hpc/mean_field128_reaction_ptol1e7_cpu.sbatch" --max-steps 5 &
wrapper_pid=$!
forward_stop() { kill -USR1 "${wrapper_pid}" 2>/dev/null || true; }
trap forward_stop USR1 TERM
solver_status=0
while true; do
    wait "${wrapper_pid}" && solver_status=0 || solver_status=$?
    # Always wait at least once, including when the child exits immediately.
    # A signal can interrupt wait while the solver is still checkpointing.
    kill -0 "${wrapper_pid}" 2>/dev/null || break
done
[[ "${solver_status}" -eq 0 ]] || exit "${solver_status}"

# A graceful early stop returns exit 0 from the shared solver. It must NOT
# release an afterok-dependent production job unless all five steps passed.
"${MF_PYTHON}" - "${MF_RUN_DIR}" "${MF_DAMKOHLER}" <<'PY'
import json
from pathlib import Path
import sys
import numpy as np

root = Path(sys.argv[1])
status = json.loads((root/"run_status.json").read_text())
if (status.get("state") != "partial" or status.get("completed_step") != 5
        or status.get("requested_steps") != 2080):
    raise RuntimeError(f"Five-step preflight did not pass: {status}")
with np.load(root/"checkpoint.npz", allow_pickle=False) as checkpoint:
    if int(checkpoint["completed_step"]) != 5 or int(checkpoint["requested_steps"]) != 2080:
        raise RuntimeError("Preflight checkpoint does not contain five accepted steps")
    history = json.loads(str(checkpoint["history_json"]))
    if [row["step"] for row in history] != list(range(1, 6)):
        raise RuntimeError("Preflight accepted-step history is incomplete")
    config = json.loads(str(checkpoint["config_json"]))
    expected = dict(n=128, reynolds=200., peclet=200., scalar_species=3,
                    damkohler=float(sys.argv[2]), dt=.0003125, final_time=.65,
                    time_integrator="forward-euler", pressure_tolerance=1e-7)
    if any(config.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Preflight checkpoint has an unexpected configuration")
print("PREFLIGHT_PASSED: five consecutive accepted steps; checkpoint ready for production.")
PY
