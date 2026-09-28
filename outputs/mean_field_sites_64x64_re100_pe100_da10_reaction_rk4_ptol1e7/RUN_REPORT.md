# Reacting 64×64 explicit single-site mean-field mixing layer

Completed and validated 1040 steps to t=0.65. Slurm job 11453851; solver time 31.451 hours after resuming from step 10 (earlier work additional).

Re=100, Pe=100, Da=10; thickness 0.01875, dt=0.000625, cutoff 12. Periodic x, free-slip/no-flux y. Initial c2=1-c1, c3=0; all species have the same diffusivity. The nondimensional mass-action rate is Da c1 c2, with sources (-R,-R,+R).

Reaction and transport evolve explicit single-site Fock vectors together in every predictor rk4 substep. Pressure relaxation and correction use the same integrator and local bosonic operators. The projected rk4 DNS is a separate benchmark, never a mean-field substep. All three initial DNS species and both velocities match the measured mean-field fields exactly. No concentration clipping, mass repair, or coherent resets.

## Species comparison plots

- c1: [Mean field](./mean_field_c1.png), [DNS](./dns_c1.png), [side-by-side and signed differences](./mean_field_vs_dns_c1.png)
- c2: [Mean field](./mean_field_c2.png), [DNS](./dns_c2.png), [side-by-side and signed differences](./mean_field_vs_dns_c2.png)
- c3: [Mean field](./mean_field_c3.png), [DNS](./dns_c3.png), [side-by-side and signed differences](./mean_field_vs_dns_c3.png)
- [Reaction, invariant, and comparison diagnostics](./reaction_diagnostics.png)
- [MF vs DNS vorticity](./mean_field_vs_dns_vorticity.png)

## Validation and limitations

Maximum relative drift of the integrals of c1+c3 and c2+c3: 6.661e-14. Maximum relative divergence: 2.525e-10. Individual species are not conserved. The unweighted sum c1+c2+c3 is not a reaction invariant; c1+c2+2c3 is.

| Species | MF all-step range | DNS all-step range | Final MF/DNS relative L2 |
|---|---|---|---|
| c1 | [-0.00700891, 1.01179] | [-0.00707095, 1.00988] | 1.268% |
| c2 | [-0.00700891, 1.01179] | [-0.00707095, 1.00988] | 1.268% |
| c3 | [-0.00188654, 0.333736] | [-0.00193001, 0.343935] | 1.876% |

## Positive and negative concentrations

[Every-step CSV, including t=0](./concentration_sign_history.csv) · [Signed-amount and cell-fraction plot](./concentration_signs.png). Counts/fractions distinguish c>0, c<0, c=0, and c<-1e-12. Positive/negative integrals are the separate signed contributions over the unit-area domain; they sum to each species amount. Negative integrals are nonpositive, not absolute magnitudes. No clipping or repair is applied.

| Method | Species | Peak c<0 fraction | Peak c<-1e-12 fraction | Most negative integral |
|---|---|---|---|---|
| MF | c1 | 0.00268555 | 0.00268555 | -6.17834e-06 |
| MF | c2 | 0.00268555 | 0.00268555 | -6.17834e-06 |
| MF | c3 | 0.00585938 | 0.00585938 | -3.04028e-06 |
| DNS | c1 | 0.00268555 | 0.00268555 | -6.41089e-06 |
| DNS | c2 | 0.00268555 | 0.00268555 | -6.41089e-06 |
| DNS | c3 | 0.00585938 | 0.00585938 | -3.24612e-06 |

Centered transport is not positivity-preserving. Negative concentrations and negative Da c1 c2 rates, if present, are numerical artifacts, not physical reverse chemistry. The displayed bounds include overshoots; data are not clipped. Positivity preserved within 1e-12: MF=False, DNS=False. Passing the conservation and local-state gates does not certify physical positivity or grid/timestep convergence. The two solvers retain different temporal splitting. RK4 in the MF substeps does not establish fourth-order accuracy of the complete split MF algorithm. For RK4 DNS, stored pressure is the stage-weighted step average.

Final velocity/vorticity MF/DNS relative L2 differences: 1.442% / 3.373%. Inspect spatial fields for roll-up/pairing; Fourier-mode ratios alone do not prove merger.

[Full summary](./summary.json) · [MF validation](./validation.json) · [DNS validation](./dns_matched_snapshots.validation.json) · [Comparison table](./comparison_DNS.csv)
