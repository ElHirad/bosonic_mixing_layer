# Forward-Euler reaction checks: 64×64, Re=Pe=100

The requested integrator change is implemented for every mean-field stage
(predictor, pressure relaxation, and velocity correction) and for the
independent reacting DNS. All are literal forward-Euler updates, not RK,
midpoint, exponentials, or coherent-state resets. Legacy RK4/midpoint options
remain only for reproducing earlier results. The current reaction batch
preset explicitly selects forward Euler and Re=Pe=100.

The physical timestep remains 0.000625, the final time 0.65 (1,040 steps),
the shear thickness 0.01875, and the existing KH seeds/boundary conditions are
unchanged. Initial c2=1-c1 and c3=0; all species have Pe=100. The mean-field
probe retains eight Euler subdivisions for predictor and correction, and
pressure pseudo-step 0.125/64². No clipping, mass repair, or relaxed gates.

## Full-time DNS results

All three independent forward-Euler DNS checks completed 1,040 steps.
The extrema below include every step, not just the eight output times.
c1 and c2 have the same extrema to the displayed precision.

| Da | Minimum c1/c2 | Maximum c1/c2 | Minimum c3 | Maximum c3 |
|---|---|---|---|---|
| 1 | -0.0132481 | 1.013699 | -0.000376856 | 0.0994747 |
| 10 | -0.0115525 | 1.015897 | -0.00324036 | 0.343655 |
| 100 | -0.00870408 | 1.026217 | -0.0128468 | 0.458935 |

**Reducing Re and Pe to 100 does not eliminate negative concentrations.**
Negative concentrations beyond 1e-12 first occur at step 17 for Da=1 and 10
(t=0.010625), and step 16 for Da=100 (t=0.010000). The largest relative
drift of the c1+c3 and c2+c3 integral invariants is 2.22e-16 for each case.
Divergence, wall, pressure-gauge, and timestep safety checks pass, but physical
positivity does not. Negative reaction rates are numerical artifacts.

Reports: [Da=1](./dns_preflight_da1.json), [Da=10](./dns_preflight_da10.json),
[Da=100](./dns_preflight_da100.json). These are independent DNS results,
not completed mean-field trajectories or MF/DNS comparisons.

## Controls

To distinguish the Re/Pe effect from the integrator change, all three cases
were also run with forward Euler at Re=Pe=200. Those configurations differ
only in Reynolds and Péclet numbers; source fingerprints were checked.

| Da | Min c1/c2 at Re=Pe=200, Euler | Min c3 at Re=Pe=200, Euler |
|---|---|---|
| 1 | -0.0838989 | -0.00366978 |
| 10 | -0.0673264 | -0.0289021 |
| 100 | -0.0361959 | -0.0845574 |

Thus the lower Re/Pe substantially reduces undershoots under the same
forward-Euler integrator. These controls are in
[`reaction_64x64_re200_pe200_euler_control`](../reaction_64x64_re200_pe200_euler_control).
They are distinct from the older Re=Pe=200 midpoint results.

A separate Da=100, Re=Pe=100 check halves dt to 0.0003125 (2,080 Euler steps):
minimum c1/c2=-0.00609137, minimum c3=-0.00996538. This improves the extrema
but still does not remove negativity. See
[the half-timestep check](./dns_da100_half_dt_check.json). It is a diagnostic
variation, not a change to the requested production timestep or a full
convergence study.

## Why Euler and Pe=100 do not guarantee positivity

For centered conservative scalar advection and diffusion, a neighboring cell
coefficient can be kappa/h² - u_face/(2h). A sufficient requirement for all
such off-diagonal coefficients to be nonnegative is |u_face|<=2 kappa/h,
and similarly for v. At Pe=100 and h=1/64 this speed threshold is 1.28.
The measured initial component maxima are |u|=2.44600 and |v|=2.85429. The
most negative initial neighboring coefficient is about -50.3774, giving a
negative Euler update weight -0.0314859 at the chosen dt. Reducing dt alone
does not change that coefficient's sign. This explains why switching the
time integrator does not make this spatial scheme positivity-preserving.

## Mean-field startup diagnostic

Slurm job **11431039** completed its diagnostic in **9 minutes 52 seconds**
on `htc-n74` (one Turin CPU). It attempted one complete *candidate* mean-field
step for each Da using the production stepper and unchanged gates. It records
stage diagnostics and explicitly labels rejected candidates. Slurm exit 0
means the diagnostic completed, not that the simulated steps passed.
No full mean-field production jobs have been submitted.

**All three candidates failed pressure relaxation after 48,000 Euler
iterations:** relative residual 2.506764e-8 versus tolerance 1e-8. No step
was accepted. The predictor coherent-eigenstate defects were 1.460049e-5
for Da=1 and 10, and 2.174508e-5 for Da=100, already above the final-state
gate 1e-5. No gate was relaxed and no classical pressure solve was substituted.
Euler substep/convergence refinement is needed before a validated full MF
trajectory can be claimed; this check does not establish which refinement
will suffice.

Probe reports: [Da=1](./mf_euler_probe_da1_substeps8.json),
[Da=10](./mf_euler_probe_da10_substeps8.json),
[Da=100](./mf_euler_probe_da100_substeps8.json).
This startup failure is a separate issue from the completed DNS negativity
checks. No accepted full-time mean-field result or MF/DNS species plots are
claimed. The failed candidate probes should not be used as validated
trajectories or as successful dependencies for full production.

## Verification and reproduction

63 Python tests pass, including literal Euler matrix-action tests, one
derivative per substep, method selection in every MF stage, first-order
convergence tests, a short weak-perturbation Euler MF/DNS trajectory with
method metadata checks, and the existing reaction/regression tests. A saved legacy
16×16 nonreacting trajectory also still validates under the reader.

```bash
OPENBLAS_NUM_THREADS=1 python3 reaction_dns_preflight.py outputs/reaction_64x64_re100_pe100_euler_series --re 100 --pe 100 --time-integrator forward-euler
sbatch --clusters=htc hpc/reaction_euler_probe.sbatch
```

The probe is diagnostic only, not a way to bypass the numerical gates. See
[the reaction equations and operator implementation](../../REACTING_MIXING_LAYER.md).
