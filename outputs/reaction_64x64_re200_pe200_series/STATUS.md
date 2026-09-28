# Reaction series startup checks — 2026-09-23

Historical record: the user subsequently requested forward Euler and
Re=Pe=100. See [the newer Euler checks](../reaction_64x64_re100_pe100_euler_series/STATUS.md).
The numerical results and original startup decision below describe the
earlier midpoint/RK4 implementation, not the current reaction preset.

The three-species mean-field implementation, independent reacting DNS,
checkpoint/validation pipeline, and c1/c2/c3 comparison plots are implemented.
All 56 Python tests pass, including the original 42 tests, reaction algebra,
analytic homogeneous reaction, restart, and end-to-end small-grid plotting.
Shell syntax and diff whitespace checks pass.

**No 64×64 reacting mean-field Slurm jobs have been submitted.** Long-run
submission is awaiting the user's choice about positivity-preserving scalar
transport versus retaining the previous centered scheme for comparison.
No full-size MF comparison plots or completed MF trajectories exist yet.

## Independent DNS preflights

Each requested Da completed all 1,040 DNS steps at 64×64, Re=Pe=200,
dt=0.000625, t_end=0.65, with thickness 0.01875 and the existing KH seeds.
c2(0)=1-c1(0), c3(0)=0. Initial fields were measured from the actual initial
local Fock kets. This tests DNS only, not mean-field time evolution.

| Da | Min c1/c2 | Max c1/c2 | Min c3 | Max c3 | Minimum Da c1 c2 |
|---|---|---|---|---|---|
| 1 | -0.0571488 | 1.06085 | -0.00256596 | 0.101689 | -0.0606192 |
| 10 | -0.0448116 | 1.07658 | -0.0200378 | 0.348773 | -0.479111 |
| 100 | -0.0224394 | 1.11634 | -0.0576930 | 0.468927 | -2.11492 |

These extrema span the entire trajectory, including initialization. Maximum
relative drift of the c1+c3 / c2+c3 integral invariants is 2.22e-16 in every
case. Divergence, walls, pressure gauge, and timestep safety checks pass.
However, positivity fails in every case. Negative c3 and negative reaction
rates are numerical artifacts of the inherited centered transport, not
physical reverse chemistry. Conserving the reaction invariants does not
make these concentrations physically admissible.

Full precision and source/configuration fingerprints are in
[Da=1](./dns_preflight_da1.json), [Da=10](./dns_preflight_da10.json), and
[Da=100](./dns_preflight_da100.json). No clipping or mass correction was used.
The preflight script can be rerun with:

```bash
OPENBLAS_NUM_THREADS=1 python3 reaction_dns_preflight.py outputs/reaction_64x64_re200_pe200_series
```

The options are to retain this scheme explicitly as a numerical comparison,
or to change scalar transport consistently in both MF and DNS to preserve
nonnegative concentrations before launching production. Any such change must
retain explicit single-site operators for all MF dynamical stages; no
classical scalar or reaction update may be inserted into MF.

See [the reaction derivation and workflow](../../REACTING_MIXING_LAYER.md).
