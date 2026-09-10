#!/bin/bash
#SBATCH --job-name=mixing_layer_mean_field32
#SBATCH --output=%x.%j.log
#SBATCH --clusters=htc
#SBATCH --partition=htc
#SBATCH --constraint=turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --mail-type=END,FAIL

set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}"
export MF_RUN_DIR="${MF_RUN_DIR:-/ix/jmendoza-arenas/hia21/mixing-layer/mean-field-sites32-re50-pe50-single-freeslip-delta0025-largekh-nb12}"
MF_OUTPUT_DIR="${MF_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/mean_field_sites_32x32_re50_pe50_single_freeslip_delta0025}"
export MF_PYTHON="${MF_PYTHON:-/usr/bin/python3}"
MF_PLOT_SUPPORT="${MF_PLOT_SUPPORT:-${PROJECT_ROOT}/cluster_runs/mf-plot-support}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR="${MF_RUN_DIR}/matplotlib-config"
if [[ -d "${MF_PLOT_SUPPORT}" ]]; then
    export PYTHONPATH="${MF_PLOT_SUPPORT}${PYTHONPATH:+:${PYTHONPATH}}"
fi
mkdir -p "${MF_RUN_DIR}/source" "${MPLCONFIGDIR}"
# Check the full plotting import chain and render on this compute node before
# spending time on the larger calculation. No mean-field evolution occurs here.
"${MF_PYTHON}" - "${PROJECT_ROOT}" <<'PY'
import io
import sys
sys.path.insert(0, sys.argv[1])
import numpy
import scipy
import matplotlib
import matplotlib.pyplot as plt
import postprocess_mean_field_run
fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1])
ax.set_title(r"MF / DNS environment check: $a^\dagger$")
fig.savefig(io.BytesIO(), format="png")
plt.close(fig)
print("Plotting check passed. NumPy", numpy.__version__, "SciPy", scipy.__version__,
      "Matplotlib", matplotlib.__version__)
PY
# Freeze the standalone benchmark and plotting sources too. The base batch
# script freezes/validates the three actual mean-field production modules.
for source in mixing_layer_dns.py compare_mean_field_dns.py plot_mean_field_results.py postprocess_mean_field_run.py; do
    if [[ -f "${MF_RUN_DIR}/source/${source}" ]]; then
        cmp "${PROJECT_ROOT}/${source}" "${MF_RUN_DIR}/source/${source}" || {
            echo "Post-processing source changed; use a new MF_RUN_DIR or its frozen sources." >&2
            exit 72
        }
    else
        cp "${PROJECT_ROOT}/${source}" "${MF_RUN_DIR}/source/${source}"
    fi
done

bash "${PROJECT_ROOT}/hpc/mean_field_cpu.sbatch" \
    --n 32 --transition 0.025 --dt 0.00125 --final-time 0.65 "$@" &
wrapper_pid=$!
forward_stop() { kill -USR1 "${wrapper_pid}" 2>/dev/null || true; }
trap forward_stop USR1 TERM
solver_status=0
while kill -0 "${wrapper_pid}" 2>/dev/null; do
    wait "${wrapper_pid}" && solver_status=0 || solver_status=$?
done
[[ "${solver_status}" -eq 0 ]] || exit "${solver_status}"
if ! "${MF_PYTHON}" -c 'import json,sys; sys.exit(json.load(open(sys.argv[1]))["state"] != "complete")' "${MF_RUN_DIR}/run_status.json"; then
    echo "Trajectory is partial; retained the checkpoint and skipped final plots."
    exit 0
fi

echo "Running the independent matched DNS benchmark and generating plots."
srun --ntasks=1 --cpu-bind=cores "${MF_PYTHON}" \
    "${MF_RUN_DIR}/source/postprocess_mean_field_run.py" \
    "${MF_RUN_DIR}/mean_field_snapshots.npz" "${MF_OUTPUT_DIR}"
echo "Finished: ${MF_OUTPUT_DIR}/RUN_REPORT.md"
