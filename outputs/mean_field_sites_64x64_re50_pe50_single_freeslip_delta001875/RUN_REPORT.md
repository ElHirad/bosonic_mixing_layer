# 64×64 explicit single-site mean-field mixing layer

Completed and validated 1040 steps to t=0.65. Production Slurm job: 11217985. This job resumed from step 3; its measured solver time is 1560.62 minutes, excluding earlier checkpointed work.

Re=50, Pe=50; free-slip y boundaries, periodic x. One shear at y=0.5, tanh thickness 0.01875, KH width 0.12. Mode 2 amplitude 2.5; mode 1 amplitude 0.5. Physical dt=0.000625; boson cutoff=12.

Predictor, pressure relaxation, and correction all evolve explicit single-site bosonic operators. DNS below is a separate projected-midpoint benchmark, initialized from the exact measured mean-field initial fields; it is not a stage of the mean-field solver. No scalar-mass repair or clipping is used.

## Plots

- [Vorticity and velocity](./mean_field_vorticity.png)
- [Concentration](./mean_field_concentration.png)
- [Pairing and numerical diagnostics](./mean_field_diagnostics.png)
- [MF vs DNS vorticity](./mean_field_vs_dns_vorticity.png)
- [MF vs DNS concentration](./mean_field_vs_dns_concentration.png)
- [MF vs DNS errors, pairing, and conservation](./mean_field_vs_dns_diagnostics.png)

## Validation and limitations

Maximum raw relative scalar-mass drift: 1.732e-14. Maximum relative divergence: 2.502e-11. All-step concentration range: [3.37915e-34, 1.00482]. Centered transport is not bound-preserving; any overshoots are retained.

Final MF/DNS relative L2 differences: velocity 0.2963%, vorticity 0.6759%, concentration 0.1732%. The two methods use different temporal splitting.

Final mode-1/mode-2 ratio: 4.7174. Inspect the spatial evolution to assess roll-up and pairing: subharmonic dominance alone can also result from faster decay of the primary mode. These large initial perturbations do not test linear KH instability. This run by itself does not establish grid or timestep convergence.

[Full numerical summary](./summary.json) · [Mean-field validation](./validation.json) · [DNS validation](./dns_matched_snapshots.validation.json)

## Post-run spatial inspection

The mean-field snapshots show early roll-up followed by diffuse broadening
and weakening. They do not demonstrate a clear strong pairing or completed
merger, despite the final mode-1/mode-2 ratio exceeding one. The small MF/DNS
differences do not by themselves establish grid or timestep convergence.

This assessment was added after reviewing the generated plots. The production
Slurm allocation lasted 26 hours 1 minute 55 seconds, including validation,
DNS comparison, and plotting; earlier preflight work is additional.
