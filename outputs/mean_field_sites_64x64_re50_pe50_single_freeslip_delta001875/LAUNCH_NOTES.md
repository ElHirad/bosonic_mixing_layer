# 64×64 thinner-shear mean-field run: launch record

## Completion update

Production job **11217985** completed successfully (exit 0) on `htc-n86` in
**26 hours 1 minute 55 seconds**. It resumed the step-3 checkpoint from
successful preflight job **11217973** with the same source/configuration
fingerprint. All 1,040 steps to t=0.65, validation, matched DNS comparison,
and plotting completed; `postprocess_status.json` is `complete`.
See the [completed report and plots](./RUN_REPORT.md).

The fields show early roll-up followed by broadening, not a clear strong
pairing or merger. The configuration and startup timings below preserve the
launch history; preflight estimates are not the current run status.

## Parameters

| Parameter | Value |
|---|---|
| Grid | 64×64 unit-square MAC cells |
| Reynolds / Péclet | Re=Pe=50, unchanged |
| Shear | One tanh layer at y=0.5; thickness 0.01875, previously 0.025 |
| Boundaries | Periodic x; free-slip/no-flux y |
| KH perturbations | Width 0.12; mode-2 amplitude 2.5; mode-1 amplitude 0.5; phase zero |
| Physical timestep | 0.000625, previously 0.00125 |
| End time | 0.65; 1,040 physical steps |
| Bosonic state | 16,448 explicit single-site Fock vectors; cutoff 12, dimension 13 |
| Predictor/correction | Eight local-state RK4 subdivisions each |
| Pressure | Local-state mean-field RK4 relaxation, pseudo-step 0.125/64² |
| Pressure acceptance | Residual at most 1e-8, unchanged; cap raised to 48,000 iterations |
| Production allocation | One `turin` CPU, 4 GiB, 48 hours; checkpoint every five steps |

The initial advective timestep bound is about 0.001032, so the previous
dt=0.00125 is not retained. The concentration 10–90% transition spans about
2.64 grid cells instead of 1.76 on the 32×32 case. The layer is physically
thinner but somewhat better sampled; this is not a grid-convergence study.

All three mean-field stages retain explicit local creation, annihilation,
and identity operators. The three production Python modules are unchanged.
The larger pressure iteration cap allows the original residual tolerance to
be reached at the finer grid; it does not loosen it or replace the relaxation
with a classical Poisson solve. No scalar-mass repair or clipping is added.

## Runtime and output

The first two accepted preflight steps passed every numerical gate. Step 1
took 132.78 seconds and 12,900 pressure iterations; step 2 took 87.92 seconds
and 8,640 iterations. This confirms that the old 12,000-iteration cap would
have failed at startup, even though the unchanged tolerance can be satisfied.
These timings suggested roughly 25–30 hours for the full run on this CPU type,
subject to later pressure convergence and node performance. All 42 Python
tests pass, including new tests of the 64×64 batch argument and directory setup.

Step 3 took a further 89.37 seconds, for 310.07 seconds of stepping time.
The complete preflight allocation lasted 5 minutes 51 seconds on `htc-n86`,
including startup and checkpoint output. `/usr/bin/time` recorded 120,832 KiB
(118 MiB) maximum resident memory. The accepted checkpoint had:

- Maximum raw relative concentration-mass drift: 2.22e-16.
- Maximum relative divergence: 2.50e-11.
- Maximum pressure residual: 9.89e-9 (tolerance 1e-8).
- Maximum coherent-eigenstate defect: 6.23e-11.
- Exactly zero wall-normal velocity and no imaginary amplitudes.

The configuration/source fingerprint and every accepted-step quality check
were rechecked before submitting the full run. No pressure tolerance or
other acceptance gate was relaxed.

The grid increases the number of local states by approximately four, pressure
relaxation typically takes about four times as many iterations, and twice as
many physical steps are needed. The resulting rough cost estimate is about
32 times the 32×32 case. The preflight provided the measured estimate above;
the allocation was headroom rather than a runtime prediction.
Checkpoint/restart is enabled.

The persistent directory is
`/ix/jmendoza-arenas/hia21/mixing-layer/mean-field-sites64-re50-pe50-single-freeslip-delta001875-largekh-nb12`.
It contains frozen sources, `run_status.json`, and local-ket checkpoints.
The preflight log is `mf64_preflight.11217973.log` in the repository root.
The production log is `mixing_layer_mean_field64.11217985.log`. Check it with:

```bash
sacct --clusters=htc -j 11217985 --format=JobID,State,ExitCode,Elapsed
tail -40 mixing_layer_mean_field64.11217985.log
```

The batch pipeline checks imports and an in-memory plot on the compute node
before starting the solver. Only a completed and validated trajectory triggers
the separate matched DNS comparison and publication of plots and `RUN_REPORT.md`
in this directory. `postprocess_status.json` must say `complete` before that
publication is considered finished. Partial checkpoints are not final results.
The Slurm job and its post-processing do not require an open assistant session.

The 32×32 case showed early roll-up followed by broadening, not a clean strong
pairing event. Keeping Re=Pe=50 does not remove that damping. More resolution
and thinner initial shear alone cannot guarantee the requested pairing action.
Judge the spatial evolution alongside mode diagnostics; a large mode-1/mode-2
ratio alone is not proof of vortex merger.

A separate, exploratory DNS preview of the 64×64/Re=Pe=50 setup to t=0.65
also showed early interface roll-up followed by broadening, without a clear
strong merger. It used initial fields measured from the initial local Fock
vectors, passed the DNS conservation/divergence checks, and retained
concentration within [0,1]. This preview is not a mean-field result and does
not replace the automatically validated post-run matched DNS comparison.
