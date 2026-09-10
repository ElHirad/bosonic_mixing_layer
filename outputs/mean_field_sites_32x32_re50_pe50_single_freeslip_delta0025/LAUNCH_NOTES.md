# 32×32 thinner-shear mean-field run: launch record

**Completion update:** job **11216428** completed successfully (exit 0) in
59 minutes 12 seconds. All 520 steps to t=0.65 passed validation, and the
separate DNS comparison and plotting finished. See the
[completed report and plots](./RUN_REPORT.md). Spatial inspection shows early
roll-up and subsequent broadening, but does not establish strong, clean pairing.

The following is the archived launch/preflight record, including the runtime
estimates and partial checkpoints available at submission time.

The full trajectory was continued by HTC job **11216428**, using the checkpoint
from the clean stop at step **28** of job **11216412**. The initial ten accepted steps came
from preflight job **11215503**.
The batch job generated `RUN_REPORT.md` and the plots only after the complete
mean-field trajectory passed validation; numerical validation alone does not
establish that vortex pairing occurred.

## Setup

| Parameter | Value |
|---|---|
| Grid | 32×32 unit-square MAC cells |
| Reynolds / Péclet | Re=Pe=50 (unchanged) |
| Shear | One tanh layer at y=0.5, thickness 0.025 (previously 0.04) |
| Boundaries | Periodic x; free-slip/no-flux y |
| KH perturbations | Width 0.12; mode-2 amplitude 2.5; mode-1 amplitude 0.5, phase zero |
| Physical timestep | 0.00125 (previously 0.0025) |
| End time | 0.65; 520 steps |
| Saved steps | 0, 74, 149, 223, 297, 371, 446, 520 |
| Local bosons | Cutoff 12, dimension 13; 4,128 single-site Fock vectors |
| Pressure relaxation | Single-site mean-field operators, tolerance 1e-8; pseudo-step 0.125/32² |
| Predictor/correction integration | Eight local RK4 subdivisions per stage |

The larger grid's initial advective timestep bound is about 0.00210, so
the previous physical dt=0.0025 is not retained. No evolution equations,
operator matrices, scalar-mass repairs, or acceptance tolerances were changed.
All three dynamical stages still evolve explicit local bosonic states.

## Preflight evidence

The ten-step test completed successfully on `htc-n79` in about 67 seconds of
solver time, with maximum resident memory about 62 MiB. Its maximum errors
were:

- Raw relative scalar-mass drift: 1.64e-14.
- Relative divergence: 8.32e-11.
- Pressure residual: 9.89e-9.
- Coherent-eigenstate defect: 5.15e-11.

The checkpoint contains all 4,128 local kets and ten accepted steps; its
configuration/source fingerprint was checked before resubmission.
The full trajectory is expected to take roughly an hour on one `turin` CPU,
depending on node performance. The allocation allows four hours and 2 GiB
for headroom and checkpoint/restart. All 39 Python tests pass, including an
end-to-end test of automatic DNS comparison, validation, plotting, and report
generation.

The first full-job submission, **11215670**, failed in its plotting import
check before any solver evolution: the compute node lacked `packaging`.
Supplemental plotting dependencies were installed in the shared project
support directory without changing NumPy/SciPy or the solver sources.
Replacement job **11216412** passed the expanded compute-node import and
in-memory PNG-render check on `htc-n28` before resuming the saved local states.
Its step-20 checkpoint passed every gate: maximum raw relative scalar-mass
drift was 2.66e-14 and maximum relative divergence was 8.32e-11. However, this
older `cascade_lake` node took 187 seconds for ten steps, approximately three
times longer than the preflight node. It was asked to stop cleanly at a
physical-step boundary. Job **11216428** requests the faster `turin` CPU type
and depends on the previous job's successful checkpointed exit; no accepted
work is discarded and no dynamics are changed. The new batch preset retains
this CPU constraint. The old job exited successfully, and job **11216428**
started on `htc-n86`, passing the compute-node plotting check and resuming
from step 28. That checkpoint's maximum raw relative mass drift was 3.18e-14;
its source/configuration fingerprint and all numerical gates were verified.
The replacement job has since reached step **30/520** successfully: its first
two resumed steps took 16.61 seconds. This supports a roughly one-to-one-and-a-half
hour remaining-runtime estimate, not including queueing or any later restart.

## Automatic outputs and status

The persistent simulation directory is
`/ix/jmendoza-arenas/hia21/mixing-layer/mean-field-sites32-re50-pe50-single-freeslip-delta0025-largekh-nb12`.
It holds the frozen sources, `run_status.json`, and local-state checkpoints.
The production log is `mixing_layer_mean_field32.11216428.log` in the repository
root. To inspect the live job:

```bash
squeue --clusters=htc -j 11216428
tail -40 mixing_layer_mean_field32.11216428.log
```

After the mean-field trajectory completes and validates, the batch job runs
the **separate** matched DNS benchmark and writes into this directory:

- `RUN_REPORT.md`, `summary.json`, and `validation.json`.
- `mean_field_snapshots.npz` (including explicit local-state snapshots).
- `dns_matched_snapshots.npz` and its validation report.
- Vorticity, concentration, pairing, and numerical-diagnostic plots.
- Side-by-side MF/DNS plots and signed-difference maps.
- `postprocess_status.json`, which must say `complete` before treating the
  publication step as finished.

The DNS reference is not a substep of the mean-field solver. A stopped or
failed mean-field trajectory keeps its checkpoint and is not plotted as a
completed result. No assistant session needs to remain open for the Slurm
job and its post-processing to run.

Sharper-looking plots are not proof of physical convergence. At Re=50,
viscous damping remains strong; judge roll-up and pairing from the evolving
spatial fields as well as the mode-1/mode-2 ratio. The thin initial transition
is still only a few cells wide. The large perturbations test finite-amplitude
interaction, not linear KH instability, and concentration transport remains
conservative but not bound-preserving.
