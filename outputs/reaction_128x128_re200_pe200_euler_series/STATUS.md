# 128×128 reacting Euler screening and pressure probes

The user accepted the 64×64/Re=Pe=100 undershoots and requested a 128×128,
Re=Pe=200 DNS check first, followed by mean-field pressure diagnostics if
the negative concentrations remained small. This directory is separate from
all prior results.

Forward Euler is retained throughout. The physical shear thickness remains
0.01875; KH mode-2/mode-1 amplitudes are 2.5/0.5, width 0.12, phase zero.
The physical timestep is halved to **0.0003125** for the finer grid, giving
**2,080 steps to t=0.65**. All three species have Pe=200; c2(0)=1-c1(0), c3(0)=0.
x is periodic and y is free-slip/no-flux. No clipping, mass repair, or changed
spatial flux scheme is used.

## DNS completed first

All three full-time independent DNS checks completed and passed conservation,
divergence, wall, gauge, and fixed-timestep checks. Extrema span every step,
including initialization, rather than just selected snapshots.

| Da | Minimum c1/c2 | Maximum c1/c2 | Minimum c3 | Maximum c3 |
|---|---|---|---|---|
| 1 | 0 | 1 | 0 | 0.102096 |
| 10 | 0 | 1 | 0 | 0.350076 |
| 100 | -0.00183914 | 1 | 0 | 0.469851 |

c1/c2 extrema agree to the displayed precision. Da=1 and 10 stayed
nonnegative. Da=100 has a small reactant undershoot (0.184% of the unit
concentration scale), but no negative product. Its first negative value
beyond 1e-12 is at step 91 (t=0.0284375). Maximum relative drift of the
c1+c3 and c2+c3 integrals is 2.22e-16 in every case.

These negatives are smaller, species by species, than the corresponding
64×64/Re=Pe=100 values the user accepted. This satisfies the stated condition
for proceeding to mean-field pressure probes. It is not proof of general
positivity or a grid-convergence study: grid, Re/Pe, and timestep changed.

Full reports: [Da=1](./dns_preflight_da1.json),
[Da=10](./dns_preflight_da10.json), [Da=100](./dns_preflight_da100.json).

## Mean-field pressure probes completed

| Da | Slurm diagnostic job |
|---|---|
| 1 | 11431433 |
| 10 | 11431434 |
| 100 | 11431435 |

Each job attempts **one full physical mean-field step**, not a full trajectory,
with 98,432 explicit local Fock vectors of dimension 13 (cutoff 12). Predictor,
pressure relaxation, and velocity correction all use forward Euler and
explicit local bosonic operators. Predictor/correction retain eight subdivisions.
All numerical gates, including pressure tolerance **1e-8**, are unchanged.

Pressure pseudo-step is 0.125/128² = 7.62939453125e-6. The cap increases from
48,000 to **192,000 iterations**, maintaining the same maximum pressure
pseudo-time 1.46484375 as the 64×64 probe. The larger cap is not a tolerance
relaxation. Residual history is measured every 20 iterations and progress is
published at least every 5,000 iterations. The optional observer has a
regression test proving that it does not change the trajectory or statistics.

All three diagnostic jobs completed successfully. Each pressure solve took
51,480 iterations and reached a residual of approximately 9.9906e-9, below
the unchanged 1e-8 tolerance. Every complete one-step candidate passed all
acceptance gates. Measured runtimes were 17m35s (Da=1), 17m33s (Da=10), and
15m30s (Da=100). These are startup results, not full MF trajectories.
See [the completed convergence report](./PRESSURE_CONVERGENCE.md).

### Automatic report after log-off

Report job **11431547** completed after the three diagnostics and saved
`PRESSURE_CONVERGENCE.md`, `pressure_convergence.png`, and
`pressure_convergence.csv`. [pressure_report_status.json](./pressure_report_status.json)
confirms pressure convergence and one accepted step for all three cases.

## Full trajectory launch, 2026-09-25

The user subsequently requested the full mean-field trajectories. The new
`hpc/mean_field128_reaction_cpu.sbatch` preset retains the exact successful
probe configuration, including all tolerances and explicit single-site
operator evolution. All three current source/configuration fingerprints
were checked against the successful probes before submission.

Each independent Da case targets all 2,080 steps to t=0.65, saves a checkpoint
after every accepted step, and requests one Turin CPU, 4 GiB, and up to
21 days. A numerical quality failure stops the run and retains the last
accepted state. An early scheduler signal requests a checkpoint stop; a
partial result can be resumed with the same sources/configuration. No
automatic resubmission is enabled. A complete validated result triggers the
separate DNS comparison and species plots. Startup success does not guarantee
full-time acceptance.

| Da | Full-trajectory Slurm job | State at launch verification | Node |
|---|---|---|---|
| 1 | 11451107 | RUNNING | htc-n77 |
| 10 | 11451108 | RUNNING | htc-n77 |
| 100 | 11451109 | RUNNING | htc-n77 |

All three started on 2026-09-25 at 13:10:16 (Slurm timestamp), passed the
compute-node numerical/plotting import checks, and entered the solver. No
physical step had finished at this initial verification. The logs confirmed
that every configuration field matches its successful probe. The three
solver modules in each run directory are byte-identical to the validated
probe sources. All **72 Python tests passed** before submission, including
the new exact Euler restart and production-preset checks.

The launch bundle was frozen before submission at
`cluster_runs/mf128-reaction-euler-launch-20260925-5KU6cq`. It contains both
solver/postprocessing sources and the batch scripts, so later workspace edits
cannot change queued or active jobs. Submitted `PROJECT_ROOT` points to that
bundle, `MF_PYTHON=/usr/bin/python3`, and `MF_PLOT_SUPPORT` points to the
original workspace's `cluster_runs/mf-plot-support` directory.

For each Da, the explicitly selected paths are:

- Scratch/checkpoints: `/ix/jmendoza-arenas/hia21/mixing-layer/mean-field-sites128-re200-pe200-daDA-three-species-euler-delta001875-nb12`.
- Completed comparison/plots: `outputs/mean_field_sites_128x128_re200_pe200_daDA_reaction_euler` under the original workspace, not the frozen bundle.
- Log: `mf128_euler_daDA.JOBID.log` in the original workspace.

Replace `DA` and `JOBID` using the table. Use Slurm state, the latest log step,
and the checkpoint's `completed_step` for live progress; `run_status.json`
records running/partial/complete/failed, but its running step count is only
updated when the invocation starts. For a later restart, retain the frozen
`PROJECT_ROOT` and explicit scratch/publication/plot-support paths above;
never resume using modified solver sources.

### Final outcome of the original full runs

All three jobs failed during physical step 2, after accepting and saving step 1.
The failure in every case was pressure residual **1.545023e-8** after the
192,000-iteration cap, above the 1e-8 tolerance. Wall times were 1h14m05s,
1h21m30s, and 1h21m45s for Da=1,10,100. This was a numerical acceptance
failure, not an out-of-memory or scheduler timeout. No full MF trajectory,
matched-DNS comparison, or production plots were produced.

A read-only replay of the second predictor from each saved checkpoint found
coherent-eigenstate defects 9.70357e-6 (Da=1,10) and 1.39199e-5 (Da=100).
The latter exceeds the unchanged 1e-5 gate. These predictor-only diagnostics
are not accepted full steps; they identify an additional potential blocker.

The user approved a separate **1e-7 pressure-tolerance five-step test** before
any new full MF/DNS comparisons. All other checks remain unchanged. See
[the new series status](../reaction_128x128_re200_pe200_euler_ptol1e7_series/STATUS.md).
The old checkpoints and frozen launch bundle are retained unchanged.

## Reproduction

```bash
OPENBLAS_NUM_THREADS=1 python3 reaction_dns_preflight.py outputs/reaction_128x128_re200_pe200_euler_series --n 128 --re 200 --pe 200 --dt .0003125 --pressure-max-steps 192000
sbatch --clusters=htc --job-name=mf128_euler_da1_probe hpc/reaction128_euler_probe.sbatch --damkohler 1
sbatch --clusters=htc --job-name=mf128_euler_da10_probe hpc/reaction128_euler_probe.sbatch --damkohler 10
sbatch --clusters=htc --job-name=mf128_euler_da100_probe hpc/reaction128_euler_probe.sbatch --damkohler 100
```

The probe refuses to overwrite existing diagnostic files. For a new run set
`PROBE_OUTPUT` to a fresh directory. After all probes finish, the convergence
plot/report can be generated with `report_pressure_probes.py` using the same
plotting environment as the normal postprocessor.

67 Python tests passed before probe submission; all 69 tests passed after
adding the report generator, and its two dedicated tests passed again after
adding automatic report status. Completed DNS source fingerprints
and configurations will be checked against the final MF probes before
reporting them together. Earlier outputs remain intact.
