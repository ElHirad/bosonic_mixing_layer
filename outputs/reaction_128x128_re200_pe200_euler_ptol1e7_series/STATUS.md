# 128×128 reacting Euler: pressure tolerance 1e-7

**Final outcome:** all three tests failed at step 2 on the unchanged coherence
gate: 1.053654e-5 for Da=1,10 and 1.391990e-5 for Da=100, versus 1e-5.
Pressure converged at the relaxed tolerance. All three conditional full jobs
were CANCELLED before starting. The user switched to
[64×64/Re=Pe=100 with RK4](../reaction_64x64_re100_pe100_rk4_series/STATUS.md).
The launch information below is historical.

The user approved relaxing the pressure stopping tolerance from 1e-8 to
**1e-7**, with every other numerical gate unchanged. The first full runs
failed on step 2 at residual 1.545023e-8; they are preserved in their original
directories. This change accepts a larger pressure residual; it does not
claim a smaller error or repair the separate coherence issue.

## Configuration and conditional workflow

Three independent Da=1,10,100 cases use n=128, Re=Pe=200, forward Euler,
dt=0.0003125, final time 0.65 (2,080 steps), shear thickness 0.01875, the same
KH seeds, periodic x and free-slip/no-flux y. c2(0)=1-c1(0), c3(0)=0.
All MF stages still evolve explicit single-site Fock kets with cutoff 12;
predictor/correction each use eight Euler subdivisions, pressure pseudo-step
0.125/128² and iteration cap 192,000. No direct pressure solve, coherent reset,
mass repair, clipping, or other relaxed acceptance threshold is introduced.

Each preflight attempts **five consecutive physical steps from initialization**
in a fresh directory. Only if all three preflights pass will the dependent
full MF jobs resume their respective five-step checkpoints. Each completed,
validated full trajectory triggers its separate matched forward-Euler DNS
benchmark and c1/c2/c3 comparison plots. If any preflight fails, the full-job
dependencies cannot be satisfied and Slurm is instructed to cancel them.
No failed preflight is treated as an accepted trajectory.

Preflights request one Turin CPU, 4 GiB, and six hours per case. Full jobs
request one Turin CPU, 4 GiB, and up to 21 days. Every accepted step is
checkpointed. The numerical solver is unchanged; wrappers enforce the test
length, parameter selection, checkpoint guard, and conditional continuation.

The previous second-predictor replay had Da=100 coherent-eigenstate defect
1.39199e-5 versus its unchanged 1e-5 gate. It remains to be tested along the
new trajectory. The pressure tolerance change does not guarantee that this
or other later checks will pass.

## Paths

For each DA=1,10,100:

- Scratch: `/ix/jmendoza-arenas/hia21/mixing-layer/mean-field-sites128-re200-pe200-daDA-three-species-euler-delta001875-nb12-ptol1e7`.
- Publication: `outputs/mean_field_sites_128x128_re200_pe200_daDA_reaction_euler_ptol1e7` under the original workspace.
- Per-case `run_status.json` and the checkpoint/log describe numerical progress; the running status's step count is only updated at invocation start.

## Submitted jobs (2026-09-25)

| Da | Five-step test | Conditional full MF + DNS/plots |
|---|---|---|
| 1 | 11453662 | 11453665 |
| 10 | 11453663 | 11453666 |
| 100 | 11453664 | 11453667 |

At launch verification all three tests were RUNNING on htc-n77 and all three
full jobs were PENDING (Dependency). Every full job has the same dependency
`afterok:11453662:11453663:11453664`, with `KillOInInvalidDependent=Yes`
confirmed by Slurm. Thus a failing test cancels the conditional full runs;
otherwise they resume from their respective five-step checkpoints without
another assistant session or user command. No validation success or full
result is claimed at this launch handoff.

All eight solver/benchmark/plotting modules and five batch scripts were
frozen before submission at
`cluster_runs/mf128-reaction-ptol1e7-launch-20260925-VWnaod`.
Both tests and full jobs use this frozen `PROJECT_ROOT` and explicitly select
the original-workspace publication paths above. `MF_PYTHON=/usr/bin/python3`
and `MF_PLOT_SUPPORT` is the original workspace's `cluster_runs/mf-plot-support`.
Logs in the workspace are `mf128_ptol1e7_daDA_test.JOBID.log` and
`mf128_ptol1e7_daDA_full.JOBID.log` using the table's DA/JOBID.

All **79 regression tests passed**. The 14 preset/preflight-guard tests also
passed after the final signal/wait handling adjustment. Compute-node plotting
checks passed for all three test jobs. The three numerical solver files have
the same SHA-256 hashes as the original 1e-8 series; only the runtime pressure
tolerance differs. Their hashes are:

- `mixing_layer_mean_field.py`: `898ef5e623915b6ca2e91977f2992cc76e2055c3a05f9799c2a5e1c83eacd826`
- `mean_field_bosons.py`: `65a066c5aef814d47201a0e15befb9c559768c7e6f462d6ad62cd12ba6fca0ba`
- `mean_field_operators.py`: `8154e0995e5bdfd470e328952742702d91c060f9783b4b377e8bad2cdedb3fce`
